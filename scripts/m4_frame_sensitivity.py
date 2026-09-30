#!/usr/bin/env python3
"""Chest-frame sensitivity to landmark errors (v3, roadmap 2c).

`build_chest_frame` uses only three measured landmarks: up = neck_base - chest,
lateral = clavicle_left - clavicle_right, origin = clavicle midpoint.  The M4
true-error metric (detector vs rig joints) gives 0.1 mm (chest), 1.7 mm
(neck_base) and 0.8 mm (clavicles), while the badly matching spine_mid /
neck_top / spine_lower (43-122 mm) are synthetic fills (chest_detector.py)
that the frame never touches.

This script quantifies both facts by Monte-Carlo worst-case perturbation:
  A. the used landmarks at their measured error magnitudes;
  B. a counterfactual frame that used neck_top - spine_mid for the up axis at
     their measured error magnitudes (what the design avoids).

Pure numpy; no Isaac.  Writes runs/m4/frame_sensitivity.json.
Usage: python3 scripts/m4_frame_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.perception.chest_landmarks import ChestLandmarks  # noqa: E402
from roboecg.target_localization.chest_frame import (  # noqa: E402
    build_chest_frame,
)

RUNS_DIR = PROJECT_ROOT / "runs" / "m4"

# A plausible supine chest (short axis = conservative for angular sensitivity).
BASE = dict(
    spine_lower=np.array([0.0, 0.0, 0.90]),
    spine_mid=np.array([0.0, 0.0, 1.05]),
    chest=np.array([0.0, 0.0, 1.15]),
    upper_chest=np.array([0.0, 0.0, 1.28]),
    neck_base=np.array([0.0, 0.0, 1.35]),
    neck_top=np.array([0.0, 0.0, 1.50]),
    clavicle_left=np.array([0.08, 0.02, 1.33]),
    clavicle_right=np.array([0.08, -0.02, 1.33]),
    shoulder_left=np.array([0.05, 0.20, 1.31]),
    shoulder_right=np.array([0.05, -0.20, 1.31]),
    pelvis=np.array([0.0, 0.0, 0.85]),
)

# Landmark error magnitudes measured on the M4 scene (detector vs rig).
ERRORS_USED_MM = {"chest": 0.1, "neck_base": 1.7,
                  "clavicle_left": 0.8, "clavicle_right": 0.8}
ERRORS_BACK_MM = {"neck_top": 43.4, "spine_mid": 50.3}
# A representative lateral target (V6-like) for the implied target shift.
TARGET_UV = (-0.28, 0.15)
# For these synthetic landmarks (up = +Z, lateral = +Y) the anterior axis is
# lateral x up = +X.
ANTERIOR_HINT = (1.0, 0.0, 0.0)


def sphere_directions(count):
    indices = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    return np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


def make_landmarks(data):
    return ChestLandmarks(**{key: np.asarray(value, dtype=float)
                             for key, value in data.items()})


def sample_worst_case(base_data, errors_mm, count=200, seed=0):
    rng = np.random.default_rng(seed)
    directions = sphere_directions(count)
    nominal = build_chest_frame(make_landmarks(base_data), ANTERIOR_HINT)
    nominal_target = nominal.from_frame(*TARGET_UV)
    worst = {"up_angle_deg": 0.0, "origin_shift_mm": 0.0,
             "target_shift_mm": 0.0}
    samples = []
    for name, error_mm in errors_mm.items():
        error_m = error_mm / 1000.0
        for direction in directions:
            data = dict(base_data)
            data[name] = np.asarray(base_data[name], dtype=float) + direction * error_m
            frame = build_chest_frame(make_landmarks(data), ANTERIOR_HINT)
            up_angle = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(frame.up, nominal.up), -1.0, 1.0))
                )
            )
            origin_shift = float(
                np.linalg.norm(frame.origin - nominal.origin) * 1000.0
            )
            target_shift = float(
                np.linalg.norm(frame.from_frame(*TARGET_UV) - nominal_target)
                * 1000.0
            )
            samples.append(
                {"landmark": name, "up_angle_deg": up_angle,
                 "origin_shift_mm": origin_shift,
                 "target_shift_mm": target_shift}
            )
            worst["up_angle_deg"] = max(worst["up_angle_deg"], up_angle)
            worst["origin_shift_mm"] = max(worst["origin_shift_mm"], origin_shift)
            worst["target_shift_mm"] = max(worst["target_shift_mm"], target_shift)
    mean_target = float(np.mean([s["target_shift_mm"] for s in samples]))
    return worst, mean_target


def counterfactual_back_axis(base_data, count=200, seed=1):
    """Frame whose up axis would be neck_top - spine_mid (what we avoid)."""
    directions = sphere_directions(count)
    nominal = build_chest_frame(make_landmarks(base_data), ANTERIOR_HINT)
    nominal_target = nominal.from_frame(*TARGET_UV)
    worst_target = 0.0
    for direction in directions:
        data = dict(base_data)
        for name, error_mm in ERRORS_BACK_MM.items():
            # worst case: the two axis endpoints move apart (independent errors)
            sign = 1.0 if name == "neck_top" else -1.0
            data[name] = (
                np.asarray(base_data[name], dtype=float)
                + sign * direction * (error_mm / 1000.0)
            )
        landmarks = make_landmarks(data)
        # counterfactual up axis, same construction otherwise
        up = landmarks.neck_top - landmarks.spine_mid
        up = up / np.linalg.norm(up)
        lateral = landmarks.clavicle_left - landmarks.clavicle_right
        lateral = lateral - np.dot(lateral, up) * up
        lateral = lateral / np.linalg.norm(lateral)
        anterior = np.cross(lateral, up)
        anterior = anterior / np.linalg.norm(anterior)
        if np.dot(anterior, ANTERIOR_HINT) < 0.0:
            anterior = -anterior
        origin = 0.5 * (landmarks.clavicle_left + landmarks.clavicle_right)
        target = (
            origin + up * TARGET_UV[0] + lateral * TARGET_UV[1]
        )
        worst_target = max(
            worst_target,
            float(np.linalg.norm(target - nominal_target) * 1000.0),
        )
    return worst_target


def main() -> None:
    used_worst, used_mean = sample_worst_case(BASE, ERRORS_USED_MM)
    back_worst = counterfactual_back_axis(BASE)
    report = {
        "provenance": (
            "pure-logic Monte-Carlo: worst-case landmark perturbations at the "
            "measured detector errors (M4 scene); frame = build_chest_frame; "
            "target shift evaluated at a V6-like (u, v) = %.2f, %.2f"
            % TARGET_UV
        ),
        "used_landmarks": {
            "errors_mm": ERRORS_USED_MM,
            "worst_up_angle_deg": used_worst["up_angle_deg"],
            "worst_origin_shift_mm": used_worst["origin_shift_mm"],
            "worst_target_shift_mm": used_worst["target_shift_mm"],
            "mean_target_shift_mm": used_mean,
        },
        "counterfactual_back_axis": {
            "errors_mm": ERRORS_BACK_MM,
            "worst_target_shift_mm": back_worst,
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "frame_sensitivity.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"frame-sensitivity: wrote {out}")
    print(
        "frame-sensitivity: used landmarks -> worst target shift "
        f"{report['used_landmarks']['worst_target_shift_mm']:.2f} mm "
        f"(mean {report['used_landmarks']['mean_target_shift_mm']:.2f}), "
        f"worst up-angle {report['used_landmarks']['worst_up_angle_deg']:.3f} deg"
    )
    print(
        "frame-sensitivity: counterfactual up = neck_top - spine_mid -> worst "
        f"target shift {back_worst:.1f} mm (the design avoids this)"
    )


if __name__ == "__main__":
    main()
