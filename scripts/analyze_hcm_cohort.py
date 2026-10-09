"""HCM cohort geometry analysis (I5 extension, item C).

Reads the 17 HCM patient torso meshes (Malik et al., Zenodo 10.5281/zenodo.18890229,
CC-BY-4.0; CARP `.pts` node files in um, volume tetrahedra so the node cloud
is a filled volume) and compares their coarse torso geometry with the 25 SSM
models (Bender et al., arm-free surfaces) on metrics that need no electrodes:

  * torso dimensions in thirds of the height (low / chest / upper bands),
  * the lateral-wall wrap fraction at the extreme torso |x| per band
    (|y(x_extreme) - y_front| / y_span),
  * the wall steepness there: the local surface normal's angle to the A-P
    axis (sign-free: arccos(|n . y|)), so ~90 deg = a fully wrapped tangent
    and small = a flat wall.

The HCM meshes include arm stumps (flat cuts at |x| ~ 0.25 m), so the chest
band is contaminated by the arms; the low band (25-40%) is arm-free and is
the primary comparison, with the chest band reported for reference only.
The sim asset's V6 wall steepness (53.8 deg against the 5th-ICS row) is
quoted from the v3 plan for the final comparison.

Run:
    python3 scripts/analyze_hcm_cohort.py --hcm-dir /media/jxy/E1/opencode_hcm/pts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.perception.torso_mesh import parse_vtk_polydata  # noqa: E402

SSM_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
OUT = PROJECT_ROOT / "runs" / "gt_validation" / "hcm_cohort_geometry.json"


def load_points(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".pts":
        with path.open() as handle:
            handle.readline()
            values = np.loadtxt(handle, dtype=np.float64)
        return values * scale
    points, _ = parse_vtk_polydata(path)
    return points * scale


def wall_metrics(pts: np.ndarray, z_lo: float, z_hi: float) -> dict:
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    height = z1 - z0
    band = pts[(pts[:, 2] >= z0 + z_lo * height) & (pts[:, 2] < z0 + z_hi * height)]
    if band.shape[0] < 50:
        return {}
    y_span = float(band[:, 1].max() - band[:, 1].min())
    # extreme torso |x|: robust 99.5th percentile of |x| (arm-free cohorts give
    # their wall; the HCM arm stumps inflate it -- kept and labelled)
    x_extreme = float(np.percentile(np.abs(band[:, 0]), 99.5))
    shell = band[np.abs(np.abs(band[:, 0]) - x_extreme) <= 0.015]
    if shell.shape[0] < 10:
        return {}
    # sign-free wrappedness: where the extreme-|x| wall sits inside the A-P
    # range (0 = mid-depth = wrapped tangent, 1 = at an end = flat wall)
    y_shell = shell[:, 1]
    y_mid = 0.5 * (band[:, 1].min() + band[:, 1].max())
    y_extreme = float(y_shell[0])
    wrap = abs(y_extreme - y_mid) / max(0.5 * y_span, 1e-9)
    front = shell[np.argmin(np.abs(y_shell - y_mid))]
    normal = _local_normal(shell, front)
    steepness = None
    if normal is not None:
        steepness = float(
            np.degrees(np.arccos(np.clip(abs(float(normal[1])), 0.0, 1.0)))
        )
    return {
        "y_span_mm": y_span * 1000.0,
        "x_extreme_mm": x_extreme * 1000.0,
        "wrap_fraction": float(wrap),
        "wall_steepness_deg": steepness,
    }


def _local_normal(points: np.ndarray, query: np.ndarray, k: int = 40):
    d = np.linalg.norm(points - query, axis=1)
    k = max(min(k, points.shape[0] - 1), 3)
    idx = np.argpartition(d, k)[: k + 1]
    patch = points[idx]
    centroid = patch.mean(axis=0)
    _, _, vt = np.linalg.svd(patch - centroid)
    return vt[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcm-dir", type=Path, default=Path("/media/jxy/E1/opencode_hcm/pts"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cohorts = {}
    # SSM cohort (arm-free POLYDATA, mm)
    ssm = {}
    for index in range(1, 26):
        path = SSM_DIR / f"T_{index:02d}_torso_coarse_surface.vtk"
        pts = load_points(path, 1e-3)
        ssm[f"T_{index:02d}"] = {
            "extent_m": [float(e) for e in pts.max(0) - pts.min(0)],
            "low": wall_metrics(pts, 0.25, 0.40),
            "chest": wall_metrics(pts, 0.55, 0.80),
        }
    cohorts["ssm_25"] = ssm

    hcm = {}
    for path in sorted(args.hcm_dir.glob("*.pts")):
        pts = load_points(path, 1e-6)
        hcm[path.stem] = {
            "nodes": int(pts.shape[0]),
            "extent_m": [float(e) for e in pts.max(0) - pts.min(0)],
            "low": wall_metrics(pts, 0.25, 0.40),
            "chest": wall_metrics(pts, 0.55, 0.80),
        }
    cohorts["hcm_17"] = hcm

    summary = {}
    for band in ("low", "chest"):
        for cohort in ("ssm_25", "hcm_17"):
            values = [
                row[band]["wall_steepness_deg"]
                for row in cohorts[cohort].values()
                if row.get(band) and row[band].get("wall_steepness_deg") is not None
            ]
            wraps = [
                row[band]["wrap_fraction"]
                for row in cohorts[cohort].values()
                if row.get(band) and row[band].get("wrap_fraction") is not None
            ]
            summary[f"{cohort}_{band}"] = {
                "n": len(values),
                "steepness_mean": float(np.mean(values)) if values else None,
                "steepness_sd": float(np.std(values)) if values else None,
                "wrap_mean": float(np.mean(wraps)) if wraps else None,
                "wrap_sd": float(np.std(wraps)) if wraps else None,
            }

    report = {
        "provenance": (
            "HCM 17 patient torso meshes (Malik et al., Zenodo 10.5281/zenodo."
            "18890229, CC-BY-4.0; CARP .pts, um, volume mesh so the node cloud "
            "is filled) vs the 25 SSM models (Bender et al.); arm stumps in "
            "the HCM chest band are flagged; sim asset V6 wall steepness "
            "53.8 deg quoted from the v3 plan section 3.3"
        ),
        "summary": summary,
        "cohorts": cohorts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, value in summary.items():
        print(
            f"{key:>14}: n={value['n']:>2} steepness "
            f"{value['steepness_mean']:.1f}±{value['steepness_sd']:.1f} deg"
            if value["steepness_mean"] is not None
            else f"{key}: n={value['n']}",
            flush=True,
        )
    print(f"report -> {args.out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
