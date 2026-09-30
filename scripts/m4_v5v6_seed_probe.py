#!/usr/bin/env python3
"""V5/V6 seed-sensitivity probe for the +-30 mm shifted targets (v3).

The envelope probe reported zero statically feasible poses for the shifted
V5/V6, but it used a single fixed IK warm start (q_home); Lula IK is branch
sensitive, so that may understate reachability.  This probe re-checks the
shifted targets with a seed bank (home, the nominal V5 plan, joint-space
perturbations and interpolants) across the tilt/azimuth range, and runs the
full plan for the feasible poses.

Writes runs/m4/v5v6_seed_probe.json.
Usage: ./scripts/run_headless.sh scripts/m4_v5v6_seed_probe.py
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

BIAS_MM = -30.0
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
        print("seed-probe: scene built", flush=True)

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

        def shifted(name):
            target = targets[name]
            position = (
                np.asarray(target.position, dtype=float)
                - frame.up * (BIAS_MM / 1000.0)
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

        # nominal base
        base_position = np.array([-0.188383, 0.45, 1.020932])
        apply_base_placement(stage, base_position, 90.0)
        matrix = world_matrix(stage, "/World/UR3/base_link")

        # seed bank: home + the nominal V5/V6 solutions (planned normally)
        seeds = [("home", q_home)]
        nominal_plan_q = {}
        for name in ("V5", "V6"):
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
                            pre_distances=(LAYOUT_M1["pre_approach_distance_m"], 0.08, 0.10),
                            steps=LAYOUT_M1["motion_steps"],
                            ground_box=(box_center, box_half),
                        )
                        break
                    except RuntimeError:
                        continue
                if plan:
                    break
            if plan:
                nominal_plan_q[name] = np.asarray(plan["q_goal"], dtype=float)
                seeds.append((f"nominal_{name}", nominal_plan_q[name]))
        if nominal_plan_q:
            keys = list(nominal_plan_q)
            q_ref = nominal_plan_q[keys[0]]
            for i in range(1, 4):
                seeds.append(
                    (f"interp_{i}", q_home + (q_ref - q_home) * (i / 4.0))
                )
            for joint in range(len(q_ref)):
                for delta in (-0.2, 0.2):
                    perturbed = q_ref.copy()
                    perturbed[joint] += delta
                    seeds.append((f"ref_j{joint}_{delta:+.1f}", perturbed))

        rows = []
        scan_targets = [
            (f"{name}_shifted", shifted(name)) for name in TARGETS_TO_PROBE
        ] + [(f"{name}_nominal", targets[name]) for name in TARGETS_TO_PROBE]
        for label, target in scan_targets:
            static_hits = []
            plan_result = None
            ik_ok_count = 0
            best_clearance = None
            best_combo = None
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
                        ik_ok_count += 1
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
                        if clearance[0] < CLEARANCE_MARGIN_M:
                            continue
                        static_hits.append(
                            {
                                "tilt_deg": float(tilt),
                                "azimuth_deg": float(azimuth),
                                "seed": seed_name,
                                "clearance_mm": float(clearance[0] * 1000.0),
                            }
                        )
                        if plan_result is None:
                            try:
                                plan = plan_reach(
                                    ik, matrix, tool0_world, rotation_world,
                                    q_seed=seed, capsules=capsules,
                                    pre_distances=(
                                        LAYOUT_M1["pre_approach_distance_m"], 0.08, 0.10
                                    ),
                                    steps=LAYOUT_M1["motion_steps"],
                                    ground_box=(box_center, box_half),
                                )
                            except RuntimeError:
                                continue
                            plan_result = {
                                "tilt_deg": float(tilt),
                                "azimuth_deg": float(azimuth),
                                "seed": seed_name,
                                "backoff_mm": float(
                                    plan.get("goal_backoff_m", 0.0)
                                )
                                * 1000.0,
                            }
            rows.append(
                {
                    "target": label,
                    "static_hits": static_hits[:12],
                    "n_static_hits": len(static_hits),
                    "ik_ok_count": ik_ok_count,
                    "best_clearance_m": best_clearance,
                    "best_clearance_combo": best_combo,
                    "first_plan": plan_result,
                }
            )
            print(
                f"seed-probe: {label:<16}: ik_ok {ik_ok_count}, "
                f"best clearance "
                + (
                    f"{best_clearance * 1000:.1f} mm at "
                    f"tilt{best_combo['tilt_deg']:.0f}"
                    f"/az{best_combo['azimuth_deg']:.0f}"
                    f"/{best_combo['seed']}"
                    if best_clearance is not None
                    else "n/a"
                )
                + f", static hits {len(static_hits)}, plan "
                + (
                    f"tilt{plan_result['tilt_deg']:.0f}"
                    f"/az{plan_result['azimuth_deg']:.0f}"
                    f" backoff {plan_result['backoff_mm']:.1f} mm"
                    if plan_result
                    else "NONE"
                ),
                flush=True,
            )

        report = {
            "provenance": (
                f"simulation: seed-sensitivity probe, V4-V6 shifted by "
                f"{BIAS_MM:+.0f} mm drop; static = goal IK + capsule "
                f"clearance over a seed bank; plan = full plan_reach"
            ),
            "settings": {
                "bias_mm": BIAS_MM,
                "n_seeds": len(seeds),
                "seed_names": [name for name, _ in seeds],
            },
            "rows": rows,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "v5v6_seed_probe.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"seed-probe: wrote {out}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_v5v6_seed_probe_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
