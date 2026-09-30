#!/usr/bin/env python3
"""Probe-corrected ICS row end-to-end (v3, I4-a).

Chain: rib phantom (construction truth) -> light-touch probe scan ->
``roboecg.target_localization.rib_probe.estimate_ics_rows`` ->
``generate_v1_v6(u_5ics_override=...)`` -> V1-V6 targets.

Reports the V4-row placement error (u of V4/V5/V6) against the construction
truth for:
  * the population drop regression (current default), and
  * the probe-corrected chain,
plus the press plan of the corrected targets (plan_cycle, pure geometry).

Writes runs/m4/probe_correction_demo.json.
Usage: ./scripts/run_headless.sh scripts/m4_probe_correction_demo.py
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

DROP_TRUE_M = 0.048  # the real-patient row spacing (P0B counterexample)
KAPPA = 3.0
STROKE_MM = 2.0
NOISE_FRACTION = 0.05
PROBE_PITCH_M = 0.005


def main() -> None:
    app = boot(headless=True, width=320, height=180)
    try:
        from isaacsim.core.api import World

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.press_plan import (
            plan_cycle,
            press_settings_from_rules,
        )
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.target_localization.rib_probe import (
            estimate_ics_rows,
            synthetic_probe_profile,
        )
        from roboecg.task_manager import ecg_scene

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("probe-demo: scene built", flush=True)

        rules = load_ecg_rules()
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")

        anatomy = rules["anatomy"]
        u_ics4 = -float(anatomy["sternal_notch_to_nipple"]["value"])
        u_ics5_true = u_ics4 - DROP_TRUE_M
        v_mcl = float(
            frame.to_frame(
                0.5 * (landmarks.clavicle_left + landmarks.shoulder_left)
            )[1]
        )

        # phantom rib construction (same as scripts/m4_rib_probe_study.py)
        ribs = [
            u_ics4 + 3 * DROP_TRUE_M / 2,
            u_ics4 + DROP_TRUE_M / 2,
            u_ics4 - DROP_TRUE_M / 2,
            u_ics4 - 3 * DROP_TRUE_M / 2,
        ]

        # light-touch probe scan (analytic contact model + mild noise)
        u_samples = np.arange(
            u_ics4 + 0.10, u_ics4 - 0.10 - 1e-9, -PROBE_PITCH_M
        )
        profile = synthetic_probe_profile(
            u_samples, ribs, kappa=KAPPA, stroke_m=STROKE_MM / 1000.0
        )
        contrast = float(profile.max() - profile.min())
        rng = np.random.default_rng(0)
        profile = profile + rng.normal(0.0, NOISE_FRACTION * contrast,
                                      profile.shape)

        estimate = estimate_ics_rows(u_samples, profile, u_ics4)

        # targets: regression vs probe-corrected
        default = generate_v1_v6(landmarks, frame, mesh_points, rules)
        corrected = None
        if estimate["ics5_m"] is not None:
            corrected = generate_v1_v6(
                landmarks, frame, mesh_points, rules,
                u_5ics_override=estimate["ics5_m"],
            )

        def row_errors(targets_result):
            errors = {}
            for target in targets_result.targets:
                u = float(frame.to_frame(target.position)[0])
                if target.name in ("V1", "V2"):
                    reference = u_ics4
                elif target.name == "V3":
                    # V3 is defined as midpoint(V2, V4) in the frame, so its
                    # nominal reference is the midpoint of the two true rows.
                    reference = 0.5 * (u_ics4 + u_ics5_true)
                else:
                    reference = u_ics5_true
                errors[target.name] = float((u - reference) * 1000.0)
            return errors

        default_errors = row_errors(default)
        corrected_errors = (
            row_errors(corrected) if corrected is not None else None
        )

        plan = None
        if corrected is not None:
            settings = press_settings_from_rules(rules)
            cycle = plan_cycle(
                list(corrected.targets),
                frame,
                rules,
                settings=settings,
            )
            plan = {
                "order": cycle["sequence"]["order"],
                "cycle_length_m": float(cycle["sequence"]["length_m"]),
                "estimated_cycle_time_s": float(cycle["estimated_cycle_time_s"]),
                "n_presses": len(cycle["presses"]),
            }

        report = {
            "provenance": (
                "end-to-end probe-correction chain on the rib phantom: "
                "construction truth ICS4 at the SNND level, ICS5 exactly 48 mm "
                "below; probe = analytic contact model (kappa=3, stroke 2 mm, "
                "5% noise); correction via estimate_ics_rows + "
                "generate_v1_v6(u_5ics_override=...)"
            ),
            "phantom": {
                "u_ics4_m": float(u_ics4),
                "u_ics5_true_m": float(u_ics5_true),
                "v_mcl_m": float(v_mcl),
                "rib_centers_m": [float(v) for v in ribs],
            },
            "probe": {
                "n_points": int(len(u_samples)),
                "estimate": estimate,
                "drop_error_mm": (
                    None
                    if estimate["drop_m"] is None
                    else float((estimate["drop_m"] - DROP_TRUE_M) * 1000.0)
                ),
            },
            "row_errors_mm": {
                "regression": default_errors,
                "probe_corrected": corrected_errors,
            },
            "press_plan_corrected": plan,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "probe_correction_demo.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"probe-demo: wrote {out}", flush=True)
        print(
            "probe-demo: estimated ICS5 "
            f"{estimate['ics5_m']} m (true {u_ics5_true:.4f}); drop error "
            f"{report['probe']['drop_error_mm']} mm",
            flush=True,
        )
        for name in ("V1", "V2", "V3", "V4", "V5", "V6"):
            print(
                f"probe-demo: row error {name}: regression "
                f"{default_errors[name]:+.1f} mm"
                + (
                    f", probe-corrected {corrected_errors[name]:+.1f} mm"
                    if corrected_errors
                    else ", probe-corrected n/a"
                ),
                flush=True,
            )
        if plan:
            print(
                f"probe-demo: corrected press plan OK ({plan['n_presses']} "
                f"presses, {plan['cycle_length_m']:.3f} m cycle)",
                flush=True,
            )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_probe_correction_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
