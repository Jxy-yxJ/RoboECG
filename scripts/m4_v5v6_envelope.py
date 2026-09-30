#!/usr/bin/env python3
"""V5/V6 drop-error envelope probe (v3).

The robust-base study showed that with V4-V6 shifted by +-30 mm of drop error,
V5/V6 cannot be planned from any of 101 candidate bases, with the standard
approach ladder capped at 30 deg tilt (docs/ECG_V3_SOLUTION_PLAN.md section
2.5).  This probe asks whether the binding constraint is the clinical tilt cap:
for a few bases it scans approach tilts up to 60 deg and eight azimuths,
reporting the first (tilt, azimuth) that plans and the goal-level static
feasibility.

Writes runs/m4/v5v6_envelope_report.json.
Usage: ./scripts/run_headless.sh scripts/m4_v5v6_envelope.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs" / "m4"

BIASES_MM = (-30.0, 30.0)
SHIFTED_TARGETS = ("V5", "V6")
TILTS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
AZIMUTHS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

BASES = [
    ("nominal", [-0.188383, 0.45, 1.020932], 90.0),
    ("grid_side", [-0.263, 0.5, 1.071], 90.0),
    ("grid_low", [-0.113, 0.5, 0.871], 90.0),
]


def main() -> None:
    app = boot(headless=True, width=640, height=360)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import estimate_surface_normal
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import apply_base_placement
        from roboecg.robot_controller.reach_plan import (
            CLEARANCE_MARGIN_M,
            body_capsules,
            link_world_positions,
            plan_reach,
            solve_tool_pose,
        )
        from roboecg.robot_controller.safety import link_clearance
        from roboecg.robot_controller.ur3_lula import (
            UR3LulaIK,
            find_articulation_roots,
        )
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import (
            ElectrodeTarget,
            generate_v1_v6,
        )
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.ecg_scene import world_matrix
        from roboecg.task_manager.m1_demo import (
            LAYOUT_M1,
            table_box,
            tool0_pose_for_target,
        )

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("envelope: scene built", flush=True)

        rules = load_ecg_rules()
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        targets = {
            t.name: t
            for t in generate_v1_v6(
                landmarks, frame, mesh_points, rules
            ).targets
        }
        offset = float(ecg_scene.LAYOUT["tool_electrode_offset_m"])

        def shifted(name, bias_mm):
            target = targets[name]
            bias_m = float(bias_mm) / 1000.0
            position = (
                np.asarray(target.position, dtype=float) - frame.up * bias_m
            )
            normal, _ = estimate_surface_normal(
                mesh_points,
                position,
                radius=0.03,
                orient_toward=position + frame.anterior,
            )
            if normal is None:
                normal = np.asarray(target.normal, dtype=float)
            return ElectrodeTarget(
                name=name,
                position=position,
                normal=normal / (np.linalg.norm(normal) + 1e-12),
                frame_coords=target.frame_coords,
            )

        roots = find_articulation_roots(stage)
        ur3_roots = [path for path in roots if "UR3" in path]
        robot = Articulation(ur3_roots[0])
        q_home = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joint_positions)
        box_center, box_half = table_box()

        rows = []
        for base_name, base_position, base_yaw in BASES:
            apply_base_placement(
                stage, np.asarray(base_position, dtype=float), base_yaw
            )
            matrix = world_matrix(stage, "/World/UR3/base_link")
            for bias_mm in BIASES_MM:
                for name in SHIFTED_TARGETS:
                    target = shifted(name, bias_mm)
                    static_feasible = []
                    first_plan = None
                    for tilt in TILTS:
                        azimuths = (0.0,) if tilt == 0.0 else AZIMUTHS
                        for azimuth in azimuths:
                            tool0_world, rotation_world = tool0_pose_for_target(
                                target,
                                frame,
                                offset,
                                tilt_deg=tilt,
                                azimuth_deg=azimuth,
                            )
                            q_goal, goal_ok = solve_tool_pose(
                                ik, matrix, tool0_world, rotation_world, q_home
                            )
                            if not goal_ok:
                                continue
                            clearance = link_clearance(
                                link_world_positions(ik, matrix, q_goal),
                                capsules,
                            )
                            if clearance[0] < CLEARANCE_MARGIN_M:
                                continue
                            static_feasible.append((tilt, azimuth))
                            if first_plan is not None:
                                continue
                            try:
                                plan = plan_reach(
                                    ik,
                                    matrix,
                                    tool0_world,
                                    rotation_world,
                                    q_seed=q_home,
                                    capsules=capsules,
                                    pre_distances=(
                                        LAYOUT_M1["pre_approach_distance_m"],
                                        0.08,
                                        0.10,
                                    ),
                                    steps=LAYOUT_M1["motion_steps"],
                                    ground_box=(box_center, box_half),
                                )
                            except RuntimeError:
                                continue
                            first_plan = {
                                "tilt_deg": float(tilt),
                                "azimuth_deg": float(azimuth),
                                "backoff_mm": float(
                                    plan.get("goal_backoff_m", 0.0)
                                )
                                * 1000.0,
                                "goal_clearance_mm": float(
                                    plan["goal_clearance"][0] * 1000.0
                                )
                                if plan.get("goal_clearance")
                                else None,
                            }
                    rows.append(
                        {
                            "base": base_name,
                            "position": [float(v) for v in base_position],
                            "yaw_deg": base_yaw,
                            "bias_mm": bias_mm,
                            "target": name,
                            "shifted_position": [
                                float(v) for v in target.position
                            ],
                            "static_feasible_count": len(static_feasible),
                            "static_feasible_first": static_feasible[:6],
                            "first_plan": first_plan,
                        }
                    )
                    print(
                        f"envelope: {base_name:<9} {bias_mm:+.0f}mm {name}: "
                        f"static {len(static_feasible)} combos, plan "
                        + (
                            f"tilt {first_plan['tilt_deg']:.0f}"
                            f"/az {first_plan['azimuth_deg']:.0f} "
                            f"backoff {first_plan['backoff_mm']:.1f} mm"
                            if first_plan
                            else "NONE up to 60 deg"
                        ),
                        flush=True,
                    )

        report = {
            "provenance": (
                "simulation: approach-envelope probe for the +-30 mm shifted "
                "V5/V6; static = goal IK + capsule clearance at the goal pose; "
                "plan = full plan_reach with 0.15 m backoff"
            ),
            "settings": {
                "biases_mm": list(BIASES_MM),
                "tilts_deg": list(TILTS),
                "azimuths_deg": list(AZIMUTHS),
                "clearance_margin_m": float(CLEARANCE_MARGIN_M),
            },
            "rows": rows,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "v5v6_envelope_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"envelope: wrote {out}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_v5v6_envelope_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
