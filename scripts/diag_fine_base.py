"""Fine base-placement grid search: is there one base covering all six targets?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    app = boot(headless=True)
    try:
        from isaacsim.core.api import World

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import search_base_placement
        from roboecg.robot_controller.reach_plan import body_capsules
        from roboecg.robot_controller.ur3_lula import UR3LulaIK
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
        tool_poses = [tool0_pose_for_target(targets[n], frame, offset) for n in order]

        grids = {
            "coarse": (None, None, None, None),
            "fine": (
                (-0.30, -0.225, -0.15, -0.075, 0.0, 0.075, 0.15, 0.225, 0.30),
                (0.40, 0.45, 0.50, 0.55, 0.60),
                (0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
                (90.0, 135.0, 180.0, 225.0, 270.0),
            ),
        }
        for name, grid in grids.items():
            if grid[0] is None:
                placement = search_base_placement(
                    ik, frame, tool_poses, capsules, (box_center, box_half)
                )
            else:
                dxs, dys, dzs, yaws = grid
                placement = search_base_placement(
                    ik, frame, tool_poses, capsules, (box_center, box_half),
                    dxs=dxs, dys=dys, dzs=dzs, yaws=yaws,
                )
            print(
                f"[{name}] total={placement['candidates_total']} "
                f"skipped_over_table={placement.get('skipped_over_table')} "
                f"collision_free={placement.get('collision_free_candidates')} "
                f"passed={placement['candidates_passed']}",
                flush=True,
            )
            for rank, item in enumerate(placement["top10"][:5], 1):
                print(
                    f"  {rank}. {np.round(item['position'], 3).tolist()} "
                    f"yaw={item['yaw_deg']:.0f} ik={item['ik_success']}/6 "
                    f"clear={item['min_clearance_m']:+.3f}",
                    flush=True,
                )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_diag_fine_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
