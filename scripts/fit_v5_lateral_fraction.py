"""Fit the V5 lateral position rule on the independent GT (25 torso models).

Current rule (v1): V5 sits at the lateral midpoint between the midclavicular
line (V4) and the midaxillary line (V6).  The GT shows this is wrong: the
clinical V5 (anterior axillary line) sits at ~74 % of the V4->V6 lateral span,
because the chest wall wraps towards the axilla (the arc midpoint projects
laterally close to V6).

This script measures, for each of the 25 statistical-shape torso models
(Zenodo 10.5281/zenodo.20086105), the fraction

    alpha = (V5_x - V4_x) / (V6_x - V4_x)

and leave-one-out compares three predictors of the V5 lateral coordinate:

    A. alpha = 0.5   (current midpoint rule)
    B. alpha = alpha_hat (constant fitted on the other 24 models)
    C. arc midpoint of the anterior envelope between V4 and V6

Run:
    $ISAACSIM_ENV/bin/python scripts/fit_v5_lateral_fraction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
REPORT_PATH = PROJECT_ROOT / "runs" / "gt_validation" / "v5_fraction_fit.json"


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


def anterior_arc_mid_x(surface: np.ndarray, v4: np.ndarray, v6: np.ndarray):
    """x of the arc midpoint of the anterior envelope between V4 and V6."""
    z_level = 0.5 * (v4[2] + v6[2])
    band = surface[np.abs(surface[:, 2] - z_level) < 20.0]
    edges = np.linspace(min(v4[0], v6[0]) - 5, max(v4[0], v6[0]) + 5, 31)
    xs, ys = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        cell = band[(band[:, 0] >= low) & (band[:, 0] < high)]
        if len(cell) >= 1:
            xs.append(0.5 * (low + high))
            ys.append(float(cell[:, 1].min()))
    if len(xs) < 5:
        return None
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    segment = np.hypot(np.diff(xs), np.diff(ys))
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    return float(np.interp(0.5 * cumulative[-1], cumulative, xs))


def main() -> None:
    alphas, records = [], []
    for index in range(1, 26):
        csv = DATA_DIR / f"T_{index:02d}_electrodes.csv"
        vtk = DATA_DIR / f"T_{index:02d}_torso_coarse_surface.vtk"
        if not csv.is_file():
            continue
        electrodes = np.loadtxt(csv, delimiter=",", skiprows=1)[3:9]
        v4, v5, v6 = electrodes[3], electrodes[4], electrodes[5]
        span = float(v6[0] - v4[0])
        alpha = float((v5[0] - v4[0]) / span)
        alphas.append(alpha)
        arc_x = anterior_arc_mid_x(load_vtk_points(vtk), v4, v6)
        records.append(
            {
                "id": index,
                "v4_x": float(v4[0]),
                "v5_x": float(v5[0]),
                "v6_x": float(v6[0]),
                "lateral_span_mm": span,
                "alpha": alpha,
                "arc_mid_x": arc_x,
            }
        )

    alphas = np.asarray(alphas)
    alpha_hat = float(alphas.mean())
    n = len(alphas)

    # leave-one-out errors of each predictor (in mm)
    err_mid, err_fit, err_arc = [], [], []
    for i, row in enumerate(records):
        v4x, v5x, v6x = row["v4_x"], row["v5_x"], row["v6_x"]
        span = v6x - v4x
        # A: midpoint
        err_mid.append(abs(v4x + 0.5 * span - v5x))
        # B: constant fitted on the other models
        loo_alpha = float(np.delete(alphas, i).mean())
        err_fit.append(abs(v4x + loo_alpha * span - v5x))
        # C: arc midpoint
        if row["arc_mid_x"] is not None:
            err_arc.append(abs(row["arc_mid_x"] - v5x))

    report = {
        "models": n,
        "alpha": {
            "mean": alpha_hat,
            "std": float(alphas.std()),
            "min": float(alphas.min()),
            "max": float(alphas.max()),
        },
        "loo_errors_mm": {
            "midpoint_0.5": float(np.mean(err_mid)),
            "fitted_alpha": float(np.mean(err_fit)),
            "arc_midpoint": float(np.mean(err_arc)) if err_arc else None,
        },
        "per_model": records,
        "provenance": (
            "Bender et al., Zenodo 10.5281/zenodo.20086105 (CC-BY-4.0): 25 "
            "statistical-shape torso models with standard 12-lead electrode "
            "positions; V5 sits at alpha of the V4->V6 lateral span."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"alpha over {n} models: mean={alpha_hat:.4f} std={alphas.std():.4f} "
          f"range=[{alphas.min():.3f}, {alphas.max():.3f}]")
    print(f"leave-one-out mean error (mm): midpoint={np.mean(err_mid):.2f}  "
          f"fitted alpha={np.mean(err_fit):.2f}  "
          f"arc midpoint={np.mean(err_arc) if err_arc else float('nan'):.2f}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
