"""Print the base-placement search ranking with per-target IK results."""
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

        from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import (
            search_base_placement,
            base_matrix,
        )
        from roboecg.robot_controller.reach_plan import body_capsules
        from roboecg.robot_controller.ur3_lula import (
            UR3LulaIK,
            find_articulation_roots,
        )
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.m1_demo import table_box, tool0_pose_for_target
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
        box_center, box_half = table_box()
        offset = ecg_scene.LAYOUT["tool_electrode_offset_m"]
        order = ("V1", "V2", "V3", "V4", "V5", "V6")
        tool_poses = [
            tool0_pose_for_target(targets[n], frame, offset) for n in order
        ]
        placement = search_base_placement(
            ik, frame, tool_poses, capsules, (box_center, box_half)
        )
        print(
            f"total={placement['candidates_total']} "
            f"over_table_skipped={placement.get('skipped_over_table')} "
            f"collision_free={placement.get('collision_free_candidates')} "
            f"passed={placement['candidates_passed']}",
            flush=True,
        )
        for rank, item in enumerate(placement["top10"], 1):
            pos = np.round(item["position"], 3).tolist()
            # per-target IK for this base
            matrix = base_matrix(np.asarray(item["position"]), item["yaw_deg"])
            rot = matrix[:3, :3]
            trans = matrix[:3, 3]
            flags = []
            for name, (pw, rw) in zip(order, tool_poses):
                local = rot.T @ (np.asarray(pw) - trans)
                quat = quat_wxyz_from_rotation(rot.T @ np.asarray(rw))
                ok_any = False
                for seed in (
                    np.zeros(6),
                    np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0]),
                ):
                    _, ok = ik.solve_pose(local, quat, seed, orientation_tolerance=0.2)
                    if ok:
                        ok_any = True
                        break
                flags.append(name if ok_any else f"{name}X")
            print(
                f"{rank:2d}. {pos} yaw={item['yaw_deg']:.0f} "
                f"ik={item['ik_success']}/6 clear={item['min_clearance_m']:+.3f} "
                f"table={item['table_clearance_m']:+.3f}  {' '.join(flags)}",
                flush=True,
            )
        print("(X = IK failed; the search uses zero-seed IK only)", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_diag_base_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
