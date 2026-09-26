"""Evaluate specific base placements: per-target IK + collision pair."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

BASES = [
    (-0.188, 0.40, 0.971, 180.0),
    (-0.188, 0.40, 1.021, 180.0),
    (-0.263, 0.40, 0.971, 180.0),
    (-0.113, 0.40, 0.971, 180.0),
    (-0.188, 0.45, 0.971, 180.0),
    (-0.413, 0.45, 1.071, 180.0),
]


def main() -> None:
    app = boot(headless=True)
    try:
        from isaacsim.core.api import World

        from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import base_matrix
        from roboecg.robot_controller.reach_plan import (
            body_capsules,
            link_clearance,
            link_world_positions,
        )
        from roboecg.robot_controller.ur3_lula import UR3LulaIK
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.m1_demo import tool0_pose_for_target
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        rules = load_ecg_rules()
        points = ecg_scene.mesh_world_points(stage, "/World/Human")
        targets = {
            t.name: t for t in generate_v1_v6(landmarks, frame, points, rules).targets
        }
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joint_positions)
        offset = ecg_scene.LAYOUT["tool_electrode_offset_m"]
        order = ("V1", "V2", "V3", "V4", "V5", "V6")
        poses = [tool0_pose_for_target(targets[n], frame, offset) for n in order]

        seeds = (
            np.zeros(6),
            np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0]),
            np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),
            np.array([0.0, -2.2, 2.2, -1.6, -1.57, 0.0]),
        )
        for bx, by, bz, yaw in BASES:
            matrix = base_matrix(np.array([bx, by, bz]), yaw)
            rot, trans = matrix[:3, :3], matrix[:3, 3]
            print(f"\nbase=({bx},{by},{bz}) yaw={yaw}", flush=True)
            for name, (pw, rw) in zip(order, poses):
                dist = float(np.linalg.norm(np.asarray(pw) - trans))
                local = rot.T @ (np.asarray(pw) - trans)
                quat = quat_wxyz_from_rotation(rot.T @ np.asarray(rw))
                solved = None
                for seed in seeds:
                    joints, ok = ik.solve_pose(local, quat, seed, orientation_tolerance=0.2)
                    if ok and np.all(np.isfinite(joints)):
                        solved = np.asarray(joints, dtype=float)
                        break
                if solved is None:
                    print(f"  {name}: IK FAIL  dist={dist:.3f}", flush=True)
                    continue
                clearance = link_clearance(
                    link_world_positions(ik, matrix, solved), capsules
                )
                print(
                    f"  {name}: IK ok  dist={dist:.3f}  "
                    f"clear={clearance[0]:+.4f} ({clearance[1]} vs {clearance[2]})",
                    flush=True,
                )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_diag_specific_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
