#!/usr/bin/env python3
"""V5/V6 robustness to the V1->V4 drop rule error: base-selection study.

The drop-bias evaluation (docs/ECG_V3_SOLUTION_PLAN.md section 2.4) showed that
with V4-V6 shifted by +-30 mm of additional drop, V5/V6 cannot be planned FROM
THE NOMINAL base.  This script answers two follow-up questions:

  A. is each biased target set plannable at all, with a re-searched base?
  B. does one base exist that is feasible across the whole uncertainty interval
     (-30, 0, +30 mm)?  That base is the conservative choice when the drop rule
     may be off by the real-patient-counterexample magnitude.

For every candidate base the standard approach ladder plans all six electrodes
for each bias (shifted contact points + mesh-refitted normals); a configuration
is feasible when all six plan with a collision-free trajectory.

Writes runs/m4/robust_base_report.json (its best_base section can be fed to
scripts/m4_ecg_place.py --base-from for a full execution demo).

Usage: ./scripts/run_headless.sh scripts/m4_robust_base_eval.py
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
ORDER = ("V1", "V2", "V3", "V4", "V5", "V6")
POOL_PER_SEARCH = 6  # best + first 5 of top10 per search


def plan_with_reason(
    ik, base_matrix, target, frame, offset, q_seed, capsules, ground_box,
    joint_positions,
):
    """Approach ladder with the failure reason of the last candidate kept."""
    from roboecg.robot_controller.reach_plan import (
        plan_reach,
        trajectory_clearance,
    )
    from roboecg.task_manager.m1_demo import (
        APPROACH_CANDIDATES,
        LAYOUT_M1,
        tool0_pose_for_target,
    )

    reasons = []
    for tilt_deg, azimuth_deg in APPROACH_CANDIDATES:
        tool0_world, rotation_world = tool0_pose_for_target(
            target, frame, offset, tilt_deg=tilt_deg, azimuth_deg=azimuth_deg
        )
        try:
            plan = plan_reach(
                ik,
                base_matrix,
                tool0_world,
                rotation_world,
                q_seed=q_seed,
                capsules=capsules,
                pre_distances=(LAYOUT_M1["pre_approach_distance_m"], 0.08, 0.10),
                steps=LAYOUT_M1["motion_steps"],
                ground_box=ground_box,
            )
        except RuntimeError as error:
            reasons.append(f"tilt{tilt_deg:.0f}: {error}")
            continue
        clearance = trajectory_clearance(
            ik,
            base_matrix,
            plan["trajectory"],
            joint_positions,
            ground_box=ground_box,
        )
        if not clearance["ok"]:
            reasons.append(
                f"tilt{tilt_deg:.0f}: clearance "
                f"{clearance['min_clearance_m'] * 1000:.1f} mm"
            )
            continue
        plan["approach_tilt_deg"] = float(tilt_deg)
        plan["trajectory_clearance_m"] = float(clearance["min_clearance_m"])
        return plan, None
    return None, " | ".join(reasons)


def main() -> None:
    app = boot(headless=True, width=640, height=360)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import Articulation

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import estimate_surface_normal
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import (
            apply_base_placement,
            search_base_placement,
        )
        from roboecg.robot_controller.reach_plan import body_capsules
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
            table_box,
            tool0_pose_for_target,
        )

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("robust-base: scene built", flush=True)

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
        offset = float(ecg_scene.LAYOUT["tool_electrode_offset_m"])

        def shifted_targets(bias_mm):
            bias_m = float(bias_mm) / 1000.0
            out = {}
            for name, target in base_targets.items():
                if name in SHIFTED and bias_m:
                    position = (
                        np.asarray(target.position, dtype=float)
                        - frame.up * bias_m
                    )
                    normal, _ = estimate_surface_normal(
                        mesh_points,
                        position,
                        radius=0.03,
                        orient_toward=position + frame.anterior,
                    )
                    if normal is None:
                        normal = np.asarray(target.normal, dtype=float)
                    out[name] = ElectrodeTarget(
                        name=name,
                        position=position,
                        normal=normal / (np.linalg.norm(normal) + 1e-12),
                        frame_coords=target.frame_coords,
                    )
                else:
                    out[name] = target
            return out

        variants = {bias: shifted_targets(bias) for bias in BIASES_MM}

        roots = find_articulation_roots(stage)
        ur3_roots = [path for path in roots if "UR3" in path]
        robot = Articulation(ur3_roots[0])
        q_home = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joint_positions)
        box_center, box_half = table_box()

        # ---- candidate pool: per-bias base searches + the nominal M4 base ----
        pool = []
        seen = set()

        def add_candidate(item, source):
            key = (
                tuple(np.round(item["position"], 4)),
                round(float(item["yaw_deg"]), 3),
            )
            if key in seen:
                return
            seen.add(key)
            pool.append(
                {
                    "position": [float(v) for v in item["position"]],
                    "yaw_deg": float(item["yaw_deg"]),
                    "ik_success": item.get("ik_success"),
                    "screening_clearance_m": item.get("min_clearance_m"),
                    "source": source,
                }
            )

        nominal_report = RUNS_DIR / "m4_report.json"
        if nominal_report.is_file():
            base = json.loads(
                nominal_report.read_text(encoding="utf-8")
            ).get("base_placement")
            if base:
                add_candidate(base, "nominal(m4_report)")

        # Coarse manual grid over the default search ranges: the screening
        # ranks only its top candidates, so sample the space directly.
        from roboecg.robot_controller.base_placement import search_base_placement
        for dx in (-0.30, -0.15, 0.0, 0.15, 0.30):
            for dy in (0.40, 0.50, 0.60):
                for dz in (0.0, 0.10, 0.20):
                    for yaw in (90.0, 180.0, 270.0):
                        position = frame.origin + np.array([dx, dy, dz])
                        if (
                            abs(position[0] - box_center[0])
                            < box_half[0]
                            and abs(position[1] - box_center[1]) < box_half[1]
                        ):
                            continue  # footprint over the table (no pedestal)
                        add_candidate(
                            {
                                "position": position,
                                "yaw_deg": yaw,
                                "ik_success": None,
                                "min_clearance_m": None,
                            },
                            "grid",
                        )
        print(f"robust-base: grid -> pool {len(pool)}", flush=True)

        for bias in BIASES_MM:
            tool_poses = [
                tool0_pose_for_target(variants[bias][name], frame, offset)
                for name in ORDER
            ]
            placement = search_base_placement(
                ik, frame, tool_poses, capsules, (box_center, box_half)
            )
            items = []
            if placement["best"] is not None:
                items.append(placement["best"])
            items.extend(placement["top10"][: POOL_PER_SEARCH - 1])
            for item in items:
                add_candidate(item, f"search@{bias:+.0f}mm")
            print(
                f"robust-base: search@{bias:+.0f} mm -> pool {len(pool)}",
                flush=True,
            )

        # ---- stage 1: static prefilter (goal IK + goal clearance, tilt 0) ----
        from roboecg.robot_controller.reach_plan import (
            CLEARANCE_MARGIN_M,
            link_world_positions,
            solve_tool_pose,
        )
        from roboecg.robot_controller.safety import link_clearance

        static_scores = {}
        for candidate in pool:
            apply_base_placement(
                stage, candidate["position"], candidate["yaw_deg"]
            )
            matrix = world_matrix(stage, "/World/UR3/base_link")
            counts = {}
            for bias in BIASES_MM:
                count = 0
                for name in ORDER:
                    position, rotation = tool0_pose_for_target(
                        variants[bias][name], frame, offset
                    )
                    q, ok = solve_tool_pose(
                        ik, matrix, position, rotation, q_home
                    )
                    if not ok:
                        continue
                    clearance = link_clearance(
                        link_world_positions(ik, matrix, q), capsules
                    )
                    if clearance[0] >= CLEARANCE_MARGIN_M:
                        count += 1
                counts[bias] = count
            static_scores[
                (tuple(np.round(candidate["position"], 4)),
                 round(candidate["yaw_deg"], 3))
            ] = counts

        def static_total(candidate):
            key = (
                tuple(np.round(candidate["position"], 4)),
                round(candidate["yaw_deg"], 3),
            )
            return sum(static_scores[key].values())

        ranked = sorted(pool, key=static_total, reverse=True)
        stage2 = ranked[:14]
        for candidate in pool:
            if candidate["source"].startswith("nominal"):
                if candidate not in stage2:
                    stage2.append(candidate)
        print(
            "robust-base: static prefilter top5 "
            + "; ".join(
                f"{np.round(c['position'], 3).tolist()}/yaw{c['yaw_deg']:.0f}"
                f":{static_total(c)}/18"
                for c in ranked[:5]
            ),
            flush=True,
        )

        # ---- stage 2: full approach ladder on the shortlist -------------------
        evaluations = []
        for candidate in stage2:
            apply_base_placement(
                stage, candidate["position"], candidate["yaw_deg"]
            )
            base_matrix = world_matrix(stage, "/World/UR3/base_link")
            per_bias = {}
            for bias in BIASES_MM:
                q_seed = q_home
                rows = []
                for name in ORDER:
                    plan, reason = plan_with_reason(
                        ik,
                        base_matrix,
                        variants[bias][name],
                        frame,
                        offset,
                        q_seed,
                        capsules,
                        (box_center, box_half),
                        joint_positions,
                    )
                    if plan is None:
                        rows.append(
                            {"target": name, "ok": False, "reason": reason}
                        )
                        continue
                    rows.append(
                        {
                            "target": name,
                            "ok": True,
                            "backoff_mm": float(
                                plan.get("goal_backoff_m", 0.0)
                            )
                            * 1000.0,
                            "tilt_deg": float(plan.get("approach_tilt_deg", 0.0)),
                            "trajectory_clearance_m": float(
                                plan["trajectory_clearance_m"]
                            ),
                        }
                    )
                    q_seed = np.asarray(plan["q_goal"], dtype=float)
                per_bias[bias] = rows
            full = sum(
                1
                for bias in BIASES_MM
                if all(row["ok"] for row in per_bias[bias])
            )
            total_ok = sum(
                1
                for bias in BIASES_MM
                for row in per_bias[bias]
                if row["ok"]
            )
            total_backoff = sum(
                row.get("backoff_mm", 0.0)
                for bias in BIASES_MM
                for row in per_bias[bias]
                if row["ok"]
            )
            evaluations.append(
                {
                    "candidate": candidate,
                    "full_biases": full,
                    "total_ok": total_ok,
                    "total_backoff_mm": total_backoff,
                    "per_bias": per_bias,
                }
            )
            print(
                f"robust-base: {candidate['source']:<18} "
                f"{np.round(candidate['position'], 3).tolist()}"
                f"/yaw{candidate['yaw_deg']:.0f} -> full {full}/3, "
                f"ok {total_ok}/18, backoff {total_backoff:.1f} mm",
                flush=True,
            )

        # ---- summaries ---------------------------------------------------------
        def best_for_bias(bias):
            subset = [
                item
                for item in evaluations
                if all(row["ok"] for row in item["per_bias"][bias])
            ]
            if not subset:
                return None
            return max(
                subset,
                key=lambda item: (
                    -item["total_backoff_mm"],
                    min(
                        row.get("trajectory_clearance_m", 0.0)
                        for row in item["per_bias"][bias]
                        if row["ok"]
                    ),
                ),
            )

        per_bias_best = {}
        for bias in BIASES_MM:
            item = best_for_bias(bias)
            per_bias_best[str(bias)] = (
                {
                    **item["candidate"],
                    "total_backoff_mm": item["total_backoff_mm"],
                }
                if item
                else None
            )

        best = max(
            evaluations,
            key=lambda item: (
                item["full_biases"],
                item["total_ok"],
                -item["total_backoff_mm"],
            ),
        )
        best_base = {
            **best["candidate"],
            "n_biases_full": int(best["full_biases"]),
            "total_ok": int(best["total_ok"]),
            "total_backoff_mm": best["total_backoff_mm"],
            "per_bias_ok": {
                str(bias): sum(1 for row in best["per_bias"][bias] if row["ok"])
                for bias in BIASES_MM
            },
            "provenance": (
                "simulation: base selected for feasibility across the +-30 mm "
                "drop-uncertainty interval (standard approach ladder, mesh "
                "targets)"
            ),
        }

        nominal = next(
            (
                item
                for item in evaluations
                if item["candidate"]["source"].startswith("nominal")
            ),
            None,
        )
        report = {
            "provenance": (
                "planning study: per-bias base search + interval-robust base "
                "selection; feasibility = all six electrodes plan with a "
                "collision-free trajectory from that base"
            ),
            "settings": {
                "biases_mm": list(BIASES_MM),
                "shifted_targets": list(SHIFTED),
                "pool_size": len(pool),
            },
            "targets": {
                str(bias): {
                    name: [float(v) for v in variants[bias][name].position]
                    for name in ORDER
                }
                for bias in BIASES_MM
            },
            "static_prefilter": [
                {
                    "candidate": candidate,
                    "static_ok_per_bias": {
                        str(bias): static_scores[
                            (
                                tuple(np.round(candidate["position"], 4)),
                                round(candidate["yaw_deg"], 3),
                            )
                        ][bias]
                        for bias in BIASES_MM
                    },
                }
                for candidate in ranked
            ],
            "per_bias_best": per_bias_best,
            "best_base": best_base,
            "nominal_base_diagnostic": (
                {
                    "candidate": nominal["candidate"],
                    "per_bias_ok": {
                        str(bias): sum(
                            1 for row in nominal["per_bias"][bias] if row["ok"]
                        )
                        for bias in BIASES_MM
                    },
                    "per_bias_rows": {
                        str(bias): nominal["per_bias"][bias]
                        for bias in BIASES_MM
                    },
                }
                if nominal
                else None
            ),
            "evaluations": [
                {
                    "candidate": item["candidate"],
                    "full_biases": item["full_biases"],
                    "total_ok": item["total_ok"],
                    "total_backoff_mm": item["total_backoff_mm"],
                }
                for item in evaluations
            ],
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "robust_base_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"robust-base: wrote {out}", flush=True)
        print(
            "robust-base: best "
            f"{np.round(best_base['position'], 3).tolist()}/yaw"
            f"{best_base['yaw_deg']:.0f} full {best_base['n_biases_full']}/3 "
            f"ok {best_base['total_ok']}/18",
            flush=True,
        )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_robust_base_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
