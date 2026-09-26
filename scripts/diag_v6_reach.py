"""Diagnose why V6 (midaxillary) resists reaching in the M1 scene.

Builds the scene, generates the targets, then for the best base placements
reports, per approach candidate: the tool0 goal, the IK result, and the
trajectory clearance.  Run:

    ./scripts/run_headless.sh scripts/diag_v6_reach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    app = boot(headless=True)
    try:
        from isaacsim.core.prims import Articulation
        from isaacsim.core.api import World

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import (
            apply_base_placement,
            search_base_placement,
        )
        from roboecg.robot_controller.reach_plan import (
            link_world_positions,
            link_clearance,
            plan_reach,
        )
        from roboecg.robot_controller.ur3_lula import (
            UR3LulaIK,
            find_articulation_roots,
        )
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.m1_demo import (
            APPROACH_CANDIDATES,
            table_box,
            tool0_pose_for_target,
        )
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.robot_controller.reach_plan import body_capsules

        print("diag: imports ok", flush=True)
        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        rules = load_ecg_rules()
        points = ecg_scene.mesh_world_points(stage, "/World/Human")
        result = generate_v1_v6(landmarks, frame, points, rules)
        targets = {t.name: t for t in result.targets}

        roots = find_articulation_roots(stage)
        robot = Articulation([p for p in roots if "UR3" in p][0])
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joint_positions)
        box_center, box_half = table_box()
        offset = ecg_scene.LAYOUT["tool_electrode_offset_m"]
        q_start = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        print("diag: scene built", flush=True)
        print(f"electrode offset = {offset:.3f} m", flush=True)

        v6 = targets["V6"]
        print(
            f"V6 contact={np.round(v6.position, 4).tolist()} "
            f"normal={np.round(v6.normal, 3).tolist()} "
            f"frame_coords={np.round(v6.frame_coords, 4).tolist()}",
            flush=True,
        )
        for name in ("V4", "V5", "V6"):
            t = targets[name]
            print(
                f"  {name}: contact={np.round(t.position, 4).tolist()} "
                f"normal={np.round(t.normal, 3).tolist()}",
                flush=True,
            )

        tool_poses = [
            tool0_pose_for_target(targets[n], frame, offset)
            for n in ("V1", "V2", "V3", "V4", "V5", "V6")
        ]
        placement = search_base_placement(
            ik, frame, tool_poses, capsules, (box_center, box_half)
        )
        candidates = []
        seen = set()
        for item in [placement["best"], *placement["top10"]]:
            if item is None:
                continue
            key = (tuple(np.round(item["position"], 4)), float(item["yaw_deg"]))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
        print(f"candidates: {len(candidates)}", flush=True)

        # include hand-picked bases that are close to V6 for reference
        for extra in (
            {"position": [-0.413, 0.45, 1.021], "yaw_deg": 180.0},
            {"position": [-0.263, 0.45, 1.021], "yaw_deg": 180.0},
            {"position": [-0.113, 0.45, 1.021], "yaw_deg": 180.0},
        ):
            candidates.insert(0, extra)
        for candidate in candidates[:4]:
            apply_base_placement(stage, candidate["position"], candidate["yaw_deg"])
            from roboecg.task_manager.ecg_scene import world_matrix

            base_matrix = world_matrix(stage, "/World/UR3/base_link")
            print(
                f"\nbase={np.round(candidate['position'], 3).tolist()} "
                f"yaw={candidate['yaw_deg']:.0f}",
                flush=True,
            )
            for tilt, azimuth in APPROACH_CANDIDATES:
                tool0, rotation = tool0_pose_for_target(
                    v6, frame, offset, tilt_deg=tilt, azimuth_deg=azimuth
                )
                base_rot = base_matrix[:3, :3]
                local = base_rot.T @ (np.asarray(tool0) - base_matrix[:3, 3])
                from roboecg.coordinate_transform.frames import (
                    quat_wxyz_from_rotation,
                )

                quaternion = quat_wxyz_from_rotation(base_rot.T @ np.asarray(rotation))
                solved = None
                for seed in (
                    np.asarray(q_start, dtype=float),
                    np.zeros(6),
                    np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0]),
                ):
                    joints, ok = ik.solve_pose(
                        local, quaternion, seed, orientation_tolerance=0.2
                    )
                    if ok and np.all(np.isfinite(joints)):
                        solved = np.asarray(joints, dtype=float)
                        break
                if solved is None:
                    print(
                        f"  tilt={tilt:4.0f} az={azimuth:3.0f}: IK FAIL "
                        f"goal={np.round(tool0, 3).tolist()} "
                        f"|goal-base|={np.linalg.norm(np.asarray(tool0) - base_matrix[:3,3]):.3f}",
                        flush=True,
                    )
                    continue
                clearance = link_clearance(
                    link_world_positions(ik, base_matrix, solved), capsules
                )
                print(
                    f"  tilt={tilt:4.0f} az={azimuth:3.0f}: IK ok  "
                    f"|goal-base|={np.linalg.norm(np.asarray(tool0) - base_matrix[:3,3]):.3f}  "
                    f"goal_clearance={clearance[0]:+.4f} ({clearance[1]} vs {clearance[2]})",
                    flush=True,
                )
                try:
                    plan_reach(
                        ik,
                        base_matrix,
                        tool0,
                        rotation,
                        q_seed=q_start,
                        capsules=capsules,
                        pre_distances=(0.06, 0.08, 0.10),
                        steps=60,
                        ground_box=(box_center, box_half),
                    )
                    print("        trajectory: OK", flush=True)
                except RuntimeError as error:
                    print(f"        trajectory: FAIL ({error})", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_diag_v6_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
