# Copied from farus_thyroid_isaac/farus/robot_controller/reach_plan.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Branch-consistent, collision-checked reach planning for the UR3."""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation
from roboecg.robot_controller.safety import (
    CLEARANCE_MARGIN_M,
    body_capsules,
    box_clearance,
    link_clearance,
)

UR3_LINK_FRAMES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "flange",
    "tool0",
)

DEFAULT_PRE_DISTANCES_M = (0.06, 0.08, 0.10, 0.12)
MAX_BRANCH_JOINT_DISTANCE_RAD = 1.5


def link_world_positions(ik, base_matrix, joints):
    """FK world positions for the UR3 link frames at the given joint values."""
    base_rotation = base_matrix[:3, :3]
    base_translation = base_matrix[:3, 3]
    positions = {}
    for frame in UR3_LINK_FRAMES:
        position, _ = ik.solver.compute_forward_kinematics(frame, joints)
        positions[frame] = base_rotation @ np.asarray(position, dtype=float) + base_translation
    return positions


def solve_tool_pose(
    ik,
    base_matrix,
    position_world,
    rotation_world,
    warm_start,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.2,
):
    """Solve IK for a tool0 world pose; returns (joints, success).

    The tolerance defaults follow the planning path (5 mm / 0.2 rad); tight
    loops (e.g. the compliant-press hold) pass smaller values.
    """
    base_rotation = base_matrix[:3, :3]
    base_translation = base_matrix[:3, 3]
    position_base = base_rotation.T @ (
        np.asarray(position_world, dtype=float) - base_translation
    )
    quaternion_base = quat_wxyz_from_rotation(base_rotation.T @ rotation_world)
    q, ok = ik.solve_pose(
        position_base,
        quaternion_base,
        warm_start,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
    )
    if not ok:
        q, ok = ik.solve_position(
            position_base, warm_start, position_tolerance=position_tolerance
        )
    return np.asarray(q, dtype=float), bool(ok)


def _worst_clearance_over_trajectory(
    ik, base_matrix, trajectory, capsules, stride=4, margin=CLEARANCE_MARGIN_M
):
    """Fast clearance scan over a trajectory (every `stride` waypoints)."""
    worst = None
    for q in trajectory[:: max(stride, 1)]:
        clearance = link_clearance(link_world_positions(ik, base_matrix, q), capsules)
        if worst is None or clearance[0] < worst[0]:
            worst = clearance
        if worst[0] < margin:
            return worst
    return worst


def plan_reach(
    ik,
    base_matrix,
    goal_tool0_world,
    goal_rotation_world,
    q_seed,
    capsules=None,
    margin=CLEARANCE_MARGIN_M,
    pre_distances=DEFAULT_PRE_DISTANCES_M,
    steps=120,
    max_branch_joint_distance=MAX_BRANCH_JOINT_DISTANCE_RAD,
    max_backoff=0.15,
    backoff_step=0.005,
    ground_box=None,
):
    """Plan a safe, branch-consistent joint trajectory to the requested goal.

    The goal is backed off along -approach in small steps until the goal pose
    and the whole joint trajectory (including the retreat pose) satisfy the
    patient clearance margin.  The retreat pose is solved with the goal
    solution as the warm start so both stay in the same IK branch.
    """
    requested_tool0_world = np.asarray(goal_tool0_world, dtype=float)
    approach = np.asarray(goal_rotation_world, dtype=float)[:, 2]
    warm = np.asarray(q_seed, dtype=float)

    best = None
    for backoff in np.arange(0.0, max_backoff + 1e-9, backoff_step):
        candidate_goal = requested_tool0_world - approach * float(backoff)
        q_goal, goal_ok = solve_tool_pose(
            ik, base_matrix, candidate_goal, goal_rotation_world, warm
        )
        if not goal_ok:
            continue
        warm = q_goal
        goal_clearance = None
        if capsules is not None:
            goal_clearance = link_clearance(
                link_world_positions(ik, base_matrix, q_goal), capsules
            )
            if goal_clearance[0] < margin:
                continue

        passing = []
        for pre_distance in pre_distances:
            pre_world = candidate_goal - approach * pre_distance
            q_pre, pre_ok = solve_tool_pose(
                ik, base_matrix, pre_world, goal_rotation_world, q_goal
            )
            if not pre_ok:
                continue
            joint_distance = float(np.linalg.norm(q_pre - q_goal))
            trajectory = [
                q_pre + (q_goal - q_pre) * s for s in np.linspace(0.0, 1.0, steps + 1)
            ]
            if capsules is not None:
                trajectory_clearance = _worst_clearance_over_trajectory(
                    ik, base_matrix, trajectory, capsules, margin=margin
                )
                if trajectory_clearance[0] < margin:
                    continue
                if ground_box is not None:
                    center, half = ground_box
                    box = min(
                        box_clearance(
                            link_world_positions(ik, base_matrix, q), center, half
                        )
                        for q in trajectory[:: max(steps // 8, 1)]
                    )
                    if box < 0.0:
                        continue
            passing.append(
                {
                    "joint_distance": joint_distance,
                    "pre_distance": pre_distance,
                    "q_pre": q_pre,
                    "q_goal": q_goal,
                    "trajectory": trajectory,
                    "pre_world": pre_world,
                    "goal_world": candidate_goal,
                    "goal_clearance": goal_clearance,
                    "backoff": float(backoff),
                }
            )

        if passing:
            best = min(passing, key=lambda item: item["joint_distance"])
            break

    if best is None:
        raise RuntimeError(
            f"no safe trajectory found within {max_backoff} m backoff "
            f"(requested goal {requested_tool0_world.tolist()})"
        )

    joint_distance = best["joint_distance"]
    return {
        "q_pre": best["q_pre"],
        "q_goal": best["q_goal"],
        "trajectory": best["trajectory"],
        "pre_tool0_world": best["pre_world"],
        "goal_tool0_world": best["goal_world"],
        "requested_tool0_world": requested_tool0_world,
        "goal_backoff_m": best["backoff"],
        "target_clamped": bool(best["backoff"] > 1e-9),
        "goal_clearance": best["goal_clearance"],
        "pre_distance_m": best["pre_distance"],
        "branch_joint_distance_rad": joint_distance,
        "same_branch": bool(joint_distance <= max_branch_joint_distance),
    }


def trajectory_clearance(
    ik, base_matrix, trajectory, joints, margin=CLEARANCE_MARGIN_M, ground_box=None
):
    """Check every joint waypoint against the patient and an optional scene box."""
    capsules = body_capsules(joints)
    worst = None
    worst_index = None
    for index, q in enumerate(trajectory):
        positions = link_world_positions(ik, base_matrix, q)
        clearance = link_clearance(positions, capsules)
        if worst is None or clearance[0] < worst[0]:
            worst = clearance
            worst_index = index
    box = None
    if ground_box is not None:
        center, half = ground_box
        box = min(
            box_clearance(link_world_positions(ik, base_matrix, q), center, half)
            for q in trajectory
        )
    return {
        "min_clearance_m": float(worst[0]),
        "link": worst[1],
        "capsule": worst[2],
        "waypoint": worst_index,
        "ground_box_clearance_m": None if box is None else float(box),
        "margin_m": margin,
        "ok": bool(worst[0] >= margin and (box is None or box >= 0.0)),
    }
