"""M0: body-shape family sweep (uniform scale 0.90-1.10).

For each body size the scene is re-placed, the chest landmarks are re-measured
from the rig, the chest surface grid is re-extracted from the asset mesh and
the UR3 base-placement reachability search is repeated.  This is the M0
"parametric body family" acceptance item; uniform scaling is a first
approximation of body-size variation (non-uniform BMI variation is a
follow-up).

Run:
    ./scripts/run_headless.sh scripts/m0_body_family.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import RUNS_DIR, boot, write_json  # noqa: E402
from m0_check_ecg_scene import (  # noqa: E402
    base_matrix,
    chest_surface_grid,
    reachability_search,
)
from roboecg.task_manager.ecg_scene import mesh_world_points  # noqa: E402

BODY_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10)


def main() -> None:
    app = boot(headless=True, width=640, height=480)
    try:
        from isaacsim.core.api import World

        from roboecg.perception.chest_landmarks import (
            audit_chest_landmarks,
            read_chest_landmarks,
        )
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.safety import body_capsules
        from roboecg.robot_controller.ur3_lula import UR3LulaIK
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.supine_pose import (
            place_supine,
            set_prim_scale,
        )

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        human_prim = stage.GetPrimAtPath("/World/Human")

        ik = UR3LulaIK(frame="tool0")
        from roboecg.task_manager.ecg_scene import world_matrix

        analytic_nominal = base_matrix(
            ecg_scene.LAYOUT["ur3_nominal_base_position"],
            ecg_scene.LAYOUT["ur3_nominal_base_yaw_deg"],
        )
        usd_nominal = world_matrix(stage, "/World/UR3/base_link")
        base_link_local = np.linalg.inv(analytic_nominal) @ usd_nominal

        table_center = [
            ecg_scene.LAYOUT["table_center_xy"][0],
            ecg_scene.LAYOUT["table_center_xy"][1],
            ecg_scene.LAYOUT["table_top_z"] - ecg_scene.LAYOUT["table_size_xyz"][2] / 2,
        ]
        table_half = [
            ecg_scene.LAYOUT["table_size_xyz"][0] / 2,
            ecg_scene.LAYOUT["table_size_xyz"][1] / 2,
            ecg_scene.LAYOUT["table_size_xyz"][2] / 2,
        ]

        results = []
        for scale in BODY_SCALES:
            set_prim_scale(human_prim, scale)
            supine = place_supine(
                stage,
                human_root_path="/World/Human",
                root_xy=ecg_scene.LAYOUT["human_root_xy"],
                table_top_z=ecg_scene.LAYOUT["table_top_z"],
            )
            joint_positions = read_joint_world_positions(stage)
            landmarks = read_chest_landmarks(joint_positions)
            frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
            audit = audit_chest_landmarks(landmarks)

            u_min = float(frame.to_frame(landmarks.spine_lower)[0])
            u_max = float(frame.to_frame(landmarks.clavicle_left)[0])
            v_min = 0.0
            v_max = float(frame.to_frame(landmarks.shoulder_left)[1])
            points = mesh_world_points(stage, "/World/Human")
            grid = chest_surface_grid(points, frame, u_min, u_max, v_min, v_max)
            ok_cells = [cell for cell in grid if cell.get("status") == "ok"]

            capsules = body_capsules(joint_positions)
            search = reachability_search(
                ik,
                frame,
                grid,
                frame.origin,
                base_link_local,
                capsules,
                (table_center, table_half),
                ecg_scene.LAYOUT["tool_electrode_offset_m"],
            )
            record = {
                "body_scale": scale,
                "body_height_m": float(
                    supine["bbox_max_world"][0] - supine["bbox_min_world"][0]
                ),
                "shoulder_width_m": audit["shoulder_width_m"],
                "clavicle_separation_m": audit["clavicle_separation_m"],
                "chest_origin_world": frame.origin.tolist(),
                "surface_cells_ok": len(ok_cells),
                "surface_cells_total": len(grid),
                "reachability_status": search["status"],
                "candidates_passed": search["candidates_passed"],
                "candidates_total": search["candidates_total"],
                "best_position": search["best"]["position"] if search["best"] else None,
                "best_yaw_deg": search["best"]["yaw_deg"] if search["best"] else None,
                "best_clearance_m": (
                    search["best"]["min_clearance_m"] if search["best"] else None
                ),
                "clearance_ok": search["best_clearance_ok"],
                "clearance_margin_m": search["clearance_margin_m"],
            }
            results.append(record)
            print(
                f"M0 body scale={scale:.2f} height={record['body_height_m']:.3f} "
                f"shoulder={record['shoulder_width_m']:.3f} "
                f"cells={record['surface_cells_ok']}/{record['surface_cells_total']} "
                f"passed={record['candidates_passed']}/{record['candidates_total']} "
                f"clearance={record['best_clearance_m']} "
                f"clearance_ok={record['clearance_ok']}"
            )

        report = {
            "status": (
                "PASS"
                if all(r["reachability_status"] == "PASS" for r in results)
                else "PARTIAL"
            ),
            "note_clearance": (
                "The patient collision capsules scale with the measured body "
                "size (body_scale_from_joints: |Spine3-pelvis| / 0.2228 m), so "
                "small bodies keep their safety margin; clearance_ok flags "
                "configurations below the 2 cm margin."
            ),
            "body_scales": list(BODY_SCALES),
            "note": (
                "Uniform scale sweep of the M_Medical_01 asset. Body dimensions "
                "are asset measurements; no clinical statistics are implied."
            ),
            "results": results,
        }
        path = write_json("m0_body_family.json", report)
        print(f"M0 body family: {report['status']} -> {path}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
