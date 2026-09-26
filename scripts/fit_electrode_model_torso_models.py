"""Leave-one-out comparison of electrode-position models on the 25 GT torsos.

Answers: can a learned surface->electrode mapping beat the fixed clinical rule?

Methods compared (all trained on 24 models, tested on the held-out one):
  A. population mean offset in a bbox-normalised frame (what a fixed rule is)
  B. linear model on the torso dimensions (height/width/depth)
  C. ridge regression on anterior-surface shape samples + dimensions

Features are registration-free: the torso bounding box gives the scale and the
transverse axes; the supine pose gives the vertical axis. This mimics what a
robot could measure from a depth cloud.

Run:
    $ISAACSIM_ENV/bin/python scripts/fit_electrode_model_torso_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
REPORT_PATH = PROJECT_ROOT / "runs" / "gt_validation" / "electrode_model_loo.json"

GRID = 8  # surface samples per axis
ELECTRODES = ("V1", "V2", "V3", "V4", "V5", "V6")


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


def model_features(surface: np.ndarray):
    """Registration-free canonical frame + surface samples (no labels used)."""
    lo, hi = surface.min(axis=0), surface.max(axis=0)
    size = hi - lo
    centre = 0.5 * (lo + hi)
    # u: 0 at the shoulder end (z high), 1 at the pelvis; v: -0.5..0.5 across
    u = (hi[2] - surface[:, 2]) / size[2]
    v = (surface[:, 0] - centre[0]) / size[0]
    # anterior surface height (y) sampled on a grid; the anterior side is -y
    grid = np.full((GRID, GRID), np.nan)
    u_edges = np.linspace(0.0, 1.0, GRID + 1)
    v_edges = np.linspace(-0.5, 0.5, GRID + 1)
    for i in range(GRID):
        for j in range(GRID):
            mask = (
                (u >= u_edges[i]) & (u < u_edges[i + 1])
                & (v >= v_edges[j]) & (v < v_edges[j + 1])
            )
            if np.count_nonzero(mask) >= 3:
                grid[i, j] = np.percentile(surface[mask, 1], 2.0)
    # fill empty cells with the column mean
    for j in range(GRID):
        column = grid[:, j]
        if np.isnan(column).any():
            fill = np.nanmean(column) if not np.isnan(column).all() else 0.0
            column[np.isnan(column)] = fill
    samples = (grid - centre[1]) / size[1]
    features = np.concatenate([size, samples.ravel()])
    return features, centre, size


def main() -> None:
    surfaces, electrodes = [], []
    for index in range(1, 26):
        vtk = DATA_DIR / f"T_{index:02d}_torso_coarse_surface.vtk"
        csv = DATA_DIR / f"T_{index:02d}_electrodes.csv"
        if not (vtk.is_file() and csv.is_file()):
            continue
        surfaces.append(load_vtk_points(vtk))
        electrodes.append(np.loadtxt(csv, delimiter=",", skiprows=1)[3:9])

    feature_rows, targets, centres, sizes = [], [], [], []
    for surface, points in zip(surfaces, electrodes):
        features, centre, size = model_features(surface)
        feature_rows.append(features)
        centres.append(centre)
        sizes.append(size)
        # target: electrode offset from the bbox centre, in metres
        targets.append((points - centre) / 1000.0)
    X = np.asarray(feature_rows)
    Y = np.asarray(targets).reshape(len(targets), -1)
    centres = np.asarray(centres)
    sizes = np.asarray(sizes)
    print(f"models={len(X)} features={X.shape[1]} targets={Y.shape[1]}")

    # standardise features for the ridge variant
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xz = (X - mean) / std

    n = len(X)
    errors = {"mean": [], "size_linear": [], "surface_ridge": []}
    per_electrode = {
        name: {key: [] for key in errors} for name in ELECTRODES
    }
    for held in range(n):
        train = np.array([i for i in range(n) if i != held])
        # A: population mean offset
        pred_mean = Y[train].mean(axis=0)
        # B: linear on torso dimensions
        design_b = np.column_stack([np.ones(len(train)), sizes[train]])
        coef_b, *_ = np.linalg.lstsq(design_b, Y[train], rcond=None)
        pred_b = np.column_stack(
            [np.ones(1), sizes[held][None, :]]
        ) @ coef_b
        # C: ridge on standardised surface features
        design_c = np.column_stack([np.ones(len(train)), Xz[train]])
        penalty = np.eye(design_c.shape[1]) * 5.0
        penalty[0, 0] = 0.0
        coef_c = np.linalg.solve(
            design_c.T @ design_c + penalty, design_c.T @ Y[train]
        )
        pred_c = np.column_stack([np.ones(1), Xz[held][None, :]]) @ coef_c

        for key, pred in (
            ("mean", pred_mean), ("size_linear", pred_b), ("surface_ridge", pred_c)
        ):
            residual = (pred.reshape(6, 3) - Y[held].reshape(6, 3)) * 1000.0
            distance = np.linalg.norm(residual, axis=1)
            errors[key].append(distance.mean())
            for name, value in zip(ELECTRODES, distance):
                per_electrode[name][key].append(float(value))

    report = {
        "models": n,
        "features": int(X.shape[1]),
        "protocol": "leave-one-out",
        "mean_error_mm": {
            key: float(np.mean(values)) for key, values in errors.items()
        },
        "std_error_mm": {
            key: float(np.std(values)) for key, values in errors.items()
        },
        "per_electrode_mm": {
            name: {
                key: float(np.mean(values)) for key, values in methods.items()
            }
            for name, methods in per_electrode.items()
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print("\nleave-one-out mean error per electrode (mm):")
    header = f"{'electrode':>10} " + " ".join(
        f"{key:>14}" for key in errors
    )
    print(header)
    for name in ELECTRODES:
        row = " ".join(
            f"{np.mean(per_electrode[name][key]):14.1f}" for key in errors
        )
        print(f"{name:>10} {row}")
    print(
        "\noverall: "
        + "  ".join(
            f"{key}={np.mean(values):.1f} mm" for key, values in errors.items()
        )
    )
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
