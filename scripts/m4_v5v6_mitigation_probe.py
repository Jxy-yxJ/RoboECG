#!/usr/bin/env python3
"""V5/V6 +-30 mm drop error: mitigation probe.

The envelope probes showed the blocker is the robot wrist (``wrist_2_link``)
against the patient ``torso`` capsule, not the patient's arm: at the lateral
wall the wrist sits right behind the electrode, and a +-30 mm vertical target
shift pushes it into the chest.  A longer electrode tool moves the wrist away
from the torso, so this probe scans:

  A. tool (electrode) offset 0.088/0.12/0.15 m: best goal clearance for the
     nominal and the +-30 mm shifted V5/V6;
  B. the vertical-offset envelope at the nominal tool length: delta in
     (0, +-10, +-15, +-30) mm (where does the curve cross the margin).

Writes runs/m4/v5v6_mitigation_report.json.
Usage: ./scripts/run_headless.sh scripts/m4_v5v6_mitigation_probe.py
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

TOOL_OFFSETS_M = (0.088, 0.12, 0.15)
SHIFT_CASES_MM = (-30.0, 30.0)
DELTA_CURVE_MM = (0.0, -30.0, -15.0, -10.0, 10.0, 15.0, 30.0)
TARGETS_TO_PROBE = ("V5", "V6")
TILTS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)


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
        print("mitigation: scene built", flush=True)

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

        def shifted(name, offset_mm):
            target = targets[name]
            position = (
                np.asarray(target.position, dtype=float)
                - frame.up * (float(offset_mm) / 1000.0)
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

        base_position = np.array([-0.188383, 0.45, 1.020932])
        apply_base_placement(stage, base_position, 90.0)
        matrix = world_matrix(stage, "/World/UR3/base_link")

        def seed_bank(offset):
            seeds = [("home", q_home)]
            for name in TARGETS_TO_PROBE:
                plan = None
                for tilt in TILTS:
                    azimuths = (0.0,) if tilt == 0.0 else AZIMUTHS
                    for azimuth in azimuths:
                        tool0_world, rotation_world = tool0_pose_for_target(
                            targets[name], frame, offset,
                            tilt_deg=tilt, azimuth_deg=azimuth,
                        )
                        try:
                            plan = plan_reach(
                                ik, matrix, tool0_world, rotation_world,
                                q_seed=q_home, capsules=capsules,
                                pre_distances=(
                                    LAYOUT_M1["pre_approach_distance_m"],
                                    0.08,
                                    0.10,
                                ),
                                steps=LAYOUT_M1["motion_steps"],
                                ground_box=(box_center, box_half),
                            )
                            break
                        except RuntimeError:
                            continue
                    if plan:
                        break
                if plan:
                    q_ref = np.asarray(plan["q_goal"], dtype=float)
                    seeds.append((f"nominal_{name}", q_ref))
                    for joint in (1, 2, 3):
                        for delta in (-0.2, 0.2):
                            perturbed = q_ref.copy()
                            perturbed[joint] += delta
                            seeds.append((f"{name}_j{joint}_{delta:+.1f}", perturbed))
            return seeds

        def scan(target, offset, seeds):
            best_clearance = None
            best_combo = None
            best_blocker = None
            hits = 0
            for tilt in TILTS:
                azimuths = (0.0,) if tilt == 0.0 else AZIMUTHS
                for azimuth in azimuths:
                    tool0_world, rotation_world = tool0_pose_for_target(
                        target, frame, offset,
                        tilt_deg=tilt, azimuth_deg=azimuth,
                    )
                    for seed_name, seed in seeds:
                        q_goal, goal_ok = solve_tool_pose(
                            ik, matrix, tool0_world, rotation_world, seed
                        )
                        if not goal_ok:
                            continue
                        clearance = link_clearance(
                            link_world_positions(ik, matrix, q_goal), capsules
                        )
                        if best_clearance is None or clearance[0] > best_clearance:
                            best_clearance = float(clearance[0])
                            best_combo = {
                                "tilt_deg": float(tilt),
                                "azimuth_deg": float(azimuth),
                                "seed": seed_name,
                            }
                            best_blocker = {
                                "link": str(clearance[1]),
                                "capsule": str(clearance[2]),
                            }
                        if clearance[0] >= CLEARANCE_MARGIN_M:
                            hits += 1
            return {
                "hits": hits,
                "best_clearance_m": best_clearance,
                "best_combo": best_combo,
                "best_blocker": best_blocker,
            }

        rows_a = []
        seeds_by_offset = {}
        for tool_offset in TOOL_OFFSETS_M:
            seeds = seed_bank(tool_offset)
            seeds_by_offset[tool_offset] = seeds
            cases = [("nominal", targets[name]) for name in TARGETS_TO_PROBE]
            cases += [
                (f"{shift:+.0f}mm", shifted(name, shift))
                for shift in SHIFT_CASES_MM
                for name in TARGETS_TO_PROBE
            ]
            for label, target in cases:
                result = scan(target, tool_offset, seeds)
                rows_a.append(
                    {
                        "tool_offset_m": tool_offset,
                        "case": label,
                        **result,
                    }
                )
                print(
                    f"mitigation: tool {tool_offset:.3f} m {label:<8}: "
                    f"hits {result['hits']:>3}, best "
                    + (
                        f"{result['best_clearance_m'] * 1000:+7.1f} mm "
                        f"({result['best_blocker']['capsule']})"
                        if result["best_clearance_m"] is not None
                        else "n/a"
                    ),
                    flush=True,
                )

        rows_b = []
        nominal_seeds = seeds_by_offset[0.088]
        for name in TARGETS_TO_PROBE:
            for delta in DELTA_CURVE_MM:
                target = shifted(name, delta)
                result = scan(target, 0.088, nominal_seeds)
                rows_b.append({"target": name, "delta_mm": delta, **result})
                print(
                    f"mitigation: curve {name} {delta:+6.1f} mm: hits "
                    f"{result['hits']:>3}, best "
                    + (
                        f"{result['best_clearance_m'] * 1000:+7.1f} mm"
                        if result["best_clearance_m"] is not None
                        else "n/a"
                    ),
                    flush=True,
                )

        report = {
            "provenance": (
                "simulation: mitigation probe; A = tool-length scan for the "
                "nominal/+-30 mm V5/V6; B = vertical-offset envelope at the "
                "nominal 0.088 m tool; static hit = clearance >= margin"
            ),
            "settings": {
                "tool_offsets_m": list(TOOL_OFFSETS_M),
                "shift_cases_mm": list(SHIFT_CASES_MM),
                "delta_curve_mm": list(DELTA_CURVE_MM),
                "clearance_margin_m": float(CLEARANCE_MARGIN_M),
            },
            "tool_length_scan": rows_a,
            "offset_curve": rows_b,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "v5v6_mitigation_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"mitigation: wrote {out}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_v5v6_mitigation_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
