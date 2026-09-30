#!/usr/bin/env python3
"""Drop-regression uncertainty evaluation (v3).

The V1->V4 drop regression is a population prior; the one real patient sits
~26.5 mm above the prediction (docs/ECG_P0B_FINDINGS.md, and its absolute value
is discussed in docs/ECG_V3_SOLUTION_PLAN.md).  This script shifts the V4-V6
targets by -30/0/+30 mm of additional drop along the chest frame, re-fits the
surface normal on the mesh at the shifted point, and re-plans every electrode
with the standard approach ladder from the nominal M4 base, reporting IK
success, safety backoff and trajectory clearance per configuration.

Caveat (documented): the base is NOT re-searched per bias - the nominal M4
winner is reused, so the numbers answer "can the nominal setup absorb the rule
error", not "what is the best setup for a biased rule".

Usage: ./scripts/run_headless.sh scripts/m4_drop_bias_eval.py
Writes runs/m4/drop_bias_report.json
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

BIASES_MM = (-30.0, 0.0, 30.0)
SHIFTED = ("V4", "V5", "V6")
NOMINAL_REPORTS = ("m4_report.json", "m4_report_perception.json")


def load_nominal_base():
    for name in NOMINAL_REPORTS:
        path = RUNS_DIR / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            base = data.get("base_placement")
            if base:
                return (
                    np.asarray(base["position"], dtype=float),
                    float(base["yaw_deg"]),
                    name,
                )
    raise RuntimeError("no nominal M4 report found to reuse a base from")


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
            body_capsules,
            trajectory_clearance,
        )
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
            _plan_target_with_approach_ladder,
            table_box,
        )

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("drop-bias: scene built", flush=True)

        rules = load_ecg_rules()
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        base_targets = {
            t.name: t
            for t in generate_v1_v6(
                landmarks, frame, mesh_points, rules
            ).targets
        }

        base_position, base_yaw, source = load_nominal_base()
        print(
            f"drop-bias: nominal base {np.round(base_position, 3).tolist()} "
            f"yaw {base_yaw} (from {source})",
            flush=True,
        )
        apply_base_placement(stage, base_position, base_yaw)
        base_matrix = world_matrix(stage, "/World/UR3/base_link")
        roots = find_articulation_roots(stage)
        ur3_roots = [path for path in roots if "UR3" in path]
        if not ur3_roots:
            raise RuntimeError(f"no UR3 articulation root among {roots}")
        robot = Articulation(ur3_roots[0])
        q_home = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joint_positions)
        box_center, box_half = table_box()
        electrode_offset = float(ecg_scene.LAYOUT["tool_electrode_offset_m"])

        rows = []
        for bias_mm in BIASES_MM:
            bias_m = bias_mm / 1000.0
            per_target = []
            q_seed = q_home
            for name in ("V1", "V2", "V3", "V4", "V5", "V6"):
                target = base_targets[name]
                position = np.asarray(target.position, dtype=float)
                if name in SHIFTED and bias_m != 0.0:
                    # additional drop: toward the feet = -frame.up
                    position = position - frame.up * bias_m
                    normal, inliers = estimate_surface_normal(
                        mesh_points,
                        position,
                        radius=0.03,
                        orient_toward=position + frame.anterior,
                    )
                    if normal is None:
                        normal = np.asarray(target.normal, dtype=float)
                    target = ElectrodeTarget(
                        name=name,
                        position=position,
                        normal=normal / (np.linalg.norm(normal) + 1e-12),
                        frame_coords=target.frame_coords,
                    )
                plan = _plan_target_with_approach_ladder(
                    ik,
                    base_matrix,
                    target,
                    frame,
                    electrode_offset,
                    q_seed,
                    capsules,
                    (box_center, box_half),
                )
                if plan is None:
                    per_target.append({"target": name, "ok": False})
                    continue
                clearance = trajectory_clearance(
                    ik,
                    base_matrix,
                    plan["trajectory"],
                    joint_positions,
                    ground_box=(box_center, box_half),
                )
                per_target.append(
                    {
                        "target": name,
                        "ok": True,
                        "shift_mm": float(bias_mm) if name in SHIFTED else 0.0,
                        "backoff_mm": float(plan.get("goal_backoff_m", 0.0))
                        * 1000.0,
                        "tilt_deg": float(plan.get("approach_tilt_deg", 0.0)),
                        "trajectory_clearance_m": float(
                            clearance["min_clearance_m"]
                        ),
                    }
                )
                q_seed = np.asarray(plan["q_goal"], dtype=float)
            planned_ok = sum(1 for row in per_target if row["ok"])
            rows.append(
                {
                    "bias_mm": bias_mm,
                    "planned_ok": planned_ok,
                    "targets": per_target,
                }
            )
            print(f"drop-bias {bias_mm:+.0f} mm: planned {planned_ok}/6", flush=True)

        report = {
            "provenance": (
                "simulation planning evaluation: V4-V6 shifted by additional "
                "drop along the chest frame; normal re-fitted on the mesh; "
                "standard approach ladder; nominal M4 base reused (no re-search)"
            ),
            "settings": {
                "biases_mm": list(BIASES_MM),
                "shifted_targets": list(SHIFTED),
                "base": {
                    "position": [float(v) for v in base_position],
                    "yaw_deg": base_yaw,
                    "source": source,
                },
            },
            "rows": rows,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "drop_bias_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"drop-bias: wrote {out}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_drop_bias_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
