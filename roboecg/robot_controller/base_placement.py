"""UR3 base-placement search from measured chest targets.

The UR3 has a 500 mm reach, so where its base sits decides whether the
contralateral precordial targets are reachable at all.  This module searches a
grid of base poses (relative to the chest frame origin), evaluates the planned
tool poses with the Lula IK and the patient/table clearance, and returns the
best placement.  Extracted from the M0 verification script so both M0 and M1
use the same code path.
"""
from __future__ import annotations

import numpy as np

from roboecg.robot_controller.reach_plan import link_world_positions
from roboecg.robot_controller.safety import (
    CLEARANCE_MARGIN_M,
    box_clearance,
    link_clearance,
)
from roboecg.task_manager.supine_pose import rotation_z

# Fine grid: the coarse grid (0.15 m steps) left the UR3 (500 mm reach) with
# no placement covering all six targets; the fine grid finds 5/6-IK bases from
# which the approach-tilt ladder completes the plan.
DEFAULT_DX_M = (-0.30, -0.225, -0.15, -0.075, 0.0, 0.075, 0.15, 0.225, 0.30)
DEFAULT_DY_M = (0.40, 0.45, 0.50, 0.55, 0.60)
DEFAULT_DZ_M = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_YAW_DEG = (90.0, 135.0, 180.0, 225.0, 270.0)
PEDESTAL_MARGIN_M = 0.0  # engineering: base centre must clear the table edge
# Diverse IK seeds: the placement search must not report a false negative
# just because one seed converges to a limit/singularity (the planning stage
# uses a warm start plus these).
IK_SEEDS = (
    np.zeros(6),
    np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0]),
    np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),
    np.array([0.0, -2.2, 2.2, -1.6, -1.57, 0.0]),
    np.array([np.pi, -1.2, 1.2, -1.5, -1.5, 0.0]),
)


def base_matrix(position, yaw_deg: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_z(yaw_deg)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def apply_base_placement(stage, position, yaw_deg: float, prim_path="/World/UR3"):
    """Move the UR3 root prim to the chosen placement."""
    from roboecg.task_manager.supine_pose import (
        set_prim_rotate_xyz,
        set_prim_translate,
    )

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    set_prim_translate(prim, position)
    set_prim_rotate_xyz(prim, (0.0, 0.0, float(yaw_deg)))


def search_base_placement(
    ik,
    frame,
    tool_poses,
    capsules,
    table_box,
    base_link_local=None,
    dxs=DEFAULT_DX_M,
    dys=DEFAULT_DY_M,
    dzs=DEFAULT_DZ_M,
    yaws=DEFAULT_YAW_DEG,
):
    """Search base placements; `tool_poses` is a list of (position, rotation)."""
    from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation

    base_link_local = (
        np.eye(4) if base_link_local is None else np.asarray(base_link_local)
    )
    table_center, table_half = (np.asarray(table_box[0]), np.asarray(table_box[1]))
    candidates = []
    skipped_footprint = 0
    for dx in dxs:
        for dy in dys:
            for dz in dzs:
                for yaw in yaws:
                    position = frame.origin + np.array([dx, dy, dz])
                    # The base stands on a floor pedestal beside the table: a
                    # position whose footprint is over the table would have no
                    # support (the link-vs-box check alone does not catch it).
                    outside = (
                        abs(position[0] - table_center[0]) >= table_half[0] + PEDESTAL_MARGIN_M
                        or abs(position[1] - table_center[1]) >= table_half[1] + PEDESTAL_MARGIN_M
                    )
                    if not outside:
                        skipped_footprint += 1
                        continue
                    matrix = base_matrix(position, yaw) @ base_link_local
                    candidates.append(
                        {"position": position.tolist(), "yaw_deg": float(yaw), "matrix": matrix}
                    )

    results = []
    for candidate in candidates:
        matrix = candidate["matrix"]
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        success = 0
        min_clearance = None
        min_box = None
        max_reach = 0.0
        for position_world, rotation_world in tool_poses:
            max_reach = max(
                max_reach,
                float(np.linalg.norm(np.asarray(position_world) - translation)),
            )
            local = rotation.T @ (np.asarray(position_world) - translation)
            quaternion = quat_wxyz_from_rotation(rotation.T @ np.asarray(rotation_world))
            # The fixed IK seeds reach different branches; the first success is
            # not necessarily a collision-free configuration (the screening then
            # reports a collision and drops an otherwise usable base).  Keep
            # the solution with the best link clearance over all seeds.
            best_joints = None
            best_clearance = None
            for seed in IK_SEEDS:
                joints, ok = ik.solve_pose(local, quaternion, seed)
                if not (ok and np.all(np.isfinite(joints))):
                    continue
                joints = np.asarray(joints, dtype=float)
                positions = link_world_positions(ik, matrix, joints)
                clearance = link_clearance(positions, capsules)
                if best_clearance is None or clearance[0] > best_clearance:
                    best_clearance = float(clearance[0])
                    best_joints = joints
                if best_clearance >= CLEARANCE_MARGIN_M:
                    break
            if best_joints is None:
                continue
            joints = best_joints
            success += 1
            if min_clearance is None or best_clearance < min_clearance:
                min_clearance = float(best_clearance)
            if table_box is not None:
                center, half = table_box
                box = box_clearance(
                    link_world_positions(ik, matrix, joints), center, half
                )
                if min_box is None or box < min_box:
                    min_box = float(box)
        results.append(
            {
                "position": candidate["position"],
                "yaw_deg": candidate["yaw_deg"],
                "ik_success": success,
                "targets": len(tool_poses),
                "min_clearance_m": min_clearance,
                "table_clearance_m": min_box,
                "max_reach_m": max_reach,
            }
        )

    def effective_clearance(item):
        """Worst of the body-capsule and table-box clearances (metres)."""
        body = item["min_clearance_m"]
        table = item["table_clearance_m"]
        if body is None:
            return table
        if table is None:
            return body
        return min(body, table)

    def score(item):
        clearance = effective_clearance(item)
        # Reach margin first: the search evaluates one IK solution per pose, so
        # its collision/IK verdicts are optimistic/pessimistic in places; the
        # planning stage (with the approach-tilt ladder and trajectory checks)
        # is the real arbiter.  A geometrically comfortable base is far more
        # likely to admit a full plan than one at the reach limit.
        return (
            -float(item.get("max_reach_m", 0.0)),
            item["ik_success"],
            -1.0 if clearance is None else clearance,
        )

    # A colliding placement is unusable regardless of its IK count (this
    # includes bases under the table): rank collision-free candidates first.
    collision_free = [
        item for item in results if (effective_clearance(item) or 0.0) >= 0.0
    ]
    pool = collision_free if collision_free else results
    ranked = sorted(pool, key=score, reverse=True)
    best = ranked[0] if ranked else None
    passed = sum(1 for item in results if item["ik_success"] == len(tool_poses))
    clearance = effective_clearance(best) if best else None
    clearance_ok = clearance is not None and clearance >= CLEARANCE_MARGIN_M
    return {
        "status": (
            "PASS"
            if best and best["ik_success"] == len(tool_poses) and clearance_ok
            else "PARTIAL"
        ),
        "collision_free_candidates": len(collision_free),
        "skipped_over_table": skipped_footprint,
        "targets": len(tool_poses),
        "candidates_total": len(results),
        "candidates_passed": passed,
        "clearance_margin_m": CLEARANCE_MARGIN_M,
        "best_clearance_ok": bool(clearance_ok),
        "best": best,
        "top10": ranked[:10],
        "top30": ranked[:30],
    }
