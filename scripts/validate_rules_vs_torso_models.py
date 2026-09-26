"""Validate the V1-V6 rules against an independent electrode-position dataset.

Dataset: "Spatial Distribution of Wilson's Central Terminal (WCT) on the Body
Surface" (Bender et al., Zenodo 20086105, CC-BY-4.0) -- 25 torso models derived
from statistical shape models, each with the 10 standard 12-lead electrode
positions (4 limb + V1-V6) on the torso surface.

This is an *independent* ground truth: the electrode positions were not
produced by this project's rules, so it can expose errors that a
self-consistent (rule-defined) ground truth cannot.

Run:
    $ISAACSIM_ENV/bin/python scripts/validate_rules_vs_torso_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
REPORT_PATH = PROJECT_ROOT / "runs" / "gt_validation" / "torso_models_report.json"

ELECTRODE_ORDER = (
    "arm_left", "arm_right", "leg_right", "V1", "V2", "V3", "V4", "V5", "V6",
    "leg_left",
)


def load_vtk_points(path: Path) -> np.ndarray:
    with path.open() as handle:
        count = 0
        for line in handle:
            if line.startswith("POINTS"):
                count = int(line.split()[1])
                break
        values = []
        for line in handle:
            values.extend(float(token) for token in line.split())
            if len(values) >= count * 3:
                break
    return np.asarray(values[: count * 3]).reshape(count, 3)


def load_models():
    models = []
    for index in range(1, 26):
        vtk = DATA_DIR / f"T_{index:02d}_torso_coarse_surface.vtk"
        csv = DATA_DIR / f"T_{index:02d}_electrodes.csv"
        if not (vtk.is_file() and csv.is_file()):
            continue
        models.append(
            {
                "id": index,
                "surface": load_vtk_points(vtk),
                "electrodes": np.loadtxt(csv, delimiter=",", skiprows=1),
            }
        )
    return models


def measurements(model: dict) -> dict:
    surface = model["surface"]
    electrodes = model["electrodes"]
    v = electrodes[3:9]
    extent = surface.max(axis=0) - surface.min(axis=0)
    return {
        "id": model["id"],
        "torso_height_mm": float(extent[2]),
        "torso_width_mm": float(extent[0]),
        "torso_depth_mm": float(extent[1]),
        "v1_v2_mm": float(np.linalg.norm(v[0] - v[1])),
        "v1_to_v4_vertical_mm": float(v[0, 2] - v[3, 2]),
        "v2_to_v4_vertical_mm": float(v[1, 2] - v[3, 2]),
        "v4_v5_mm": float(np.linalg.norm(v[3] - v[4])),
        "v5_v6_mm": float(np.linalg.norm(v[4] - v[5])),
        "v4_v6_mm": float(np.linalg.norm(v[3] - v[5])),
        "v4_v6_lateral_mm": float(abs(v[5, 0] - v[3, 0])),
        "v6_depth_mm": float(v[5, 1]),
        "v3_midpoint_error_mm": float(
            np.linalg.norm(v[2] - 0.5 * (v[1] + v[3]))
        ),
        "electrodes": v.tolist(),
    }


def summarise(rows, key: str) -> dict:
    values = np.asarray([row[key] for row in rows])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    models = load_models()
    if len(models) < 20:
        raise RuntimeError(f"only {len(models)} torso models found in {DATA_DIR}")
    rows = [measurements(model) for model in models]

    keys = (
        "torso_height_mm", "torso_width_mm", "torso_depth_mm",
        "v1_v2_mm", "v1_to_v4_vertical_mm", "v2_to_v4_vertical_mm",
        "v4_v5_mm", "v5_v6_mm", "v4_v6_mm", "v4_v6_lateral_mm",
        "v3_midpoint_error_mm",
    )
    summary = {key: summarise(rows, key) for key in keys}
    correlations = {}
    for key in keys:
        if key.startswith("torso_"):
            continue
        for size in ("torso_height_mm", "torso_width_mm", "torso_depth_mm"):
            values = np.asarray([row[key] for row in rows])
            sizes = np.asarray([row[size] for row in rows])
            correlations[f"{key}~{size}"] = float(np.corrcoef(values, sizes)[0, 1])

    report = {
        "models": len(rows),
        "summary": summary,
        "correlations": correlations,
        "per_model": rows,
        "provenance": (
            "Bender et al., Spatial Distribution of WCT on the Body Surface, "
            "Zenodo 10.5281/zenodo.20086105, CC-BY-4.0: 25 statistical-shape "
            "torso models with the 10 standard 12-lead electrode positions"
        ),
        "note": (
            "Coordinates are millimetres in the shared shape-model space; the "
            "electrode column order is inferred geometrically (rows 4-9 are "
            "V1-V6: parasternal -> midclavicular -> midaxillary, with the "
            "expected 4th/5th intercostal vertical pattern)."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"GT validation: {len(rows)} torso models")
    for key in keys:
        s = summary[key]
        print(
            f"  {key:26s} mean={s['mean']:7.1f} sd={s['std']:6.1f} "
            f"range=[{s['min']:.1f}, {s['max']:.1f}]"
        )
    print("strong correlations (|r| > 0.7):")
    for name, value in correlations.items():
        if abs(value) > 0.7:
            print(f"  {name}: r={value:+.2f}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
