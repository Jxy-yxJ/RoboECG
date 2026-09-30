"""I5 transferability test: geometry-sensitive quantities on real vs sim torsos.

Compares the 25 SSM torso models (assets/external/torso_models, CC-BY-4.0;
independent GT: real electrode positions) with the stylized biped asset on the
three items from docs/ECG_V3_SOLUTION_PLAN.md section 3.3:

  a. alpha_V5        V5 lateral fraction (|x5|-|x4|)/(|x6|-|x4|);
                     real: GT electrode rows; sim: the rule is fitted from the
                     same distribution, so the comparison is rule-side only.
  b. wrap_ratio      V4->V6 depth/lateral ratio (how far the lateral wall wraps
                     backwards relative to the sideways displacement);
                     real: GT electrodes (Y depth, X lateral);
                     sim: runs/m4/m4_report.json press_plan contacts (the m4
                     scene's anterior is +Z; the patient is yawed, so the
                     lateral axis is derived from the V1->V2 parasternal pair,
                     which is pure lateral by construction).
  c. wall steepness  local surface normal angle vs the anterior axis at V5/V6;
                     real: kNN plane fit on the VTK surface;
                     sim: press_plan normal_world vs +Z.

Output: runs/gt_validation/ssm_i5_transfer.json + a printed table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
GT_REPORT = PROJECT_ROOT / "runs" / "gt_validation" / "ssm_i5_transfer.json"
SIM_REPORT = PROJECT_ROOT / "runs" / "m4" / "m4_report.json"

sys.path.insert(0, str(PROJECT_ROOT))
from roboecg.perception.torso_mesh import parse_vtk_polydata  # noqa: E402

ANTERIOR_AXIS = np.array([0.0, 1.0, 0.0])


def surface_normal(points: np.ndarray, query: np.ndarray, k: int = 40):
    distances = np.linalg.norm(points - query, axis=1)
    index = np.argpartition(distances, k)[:k]
    patch = points[index]
    centroid = patch.mean(axis=0)
    _, _, vt = np.linalg.svd(patch - centroid)
    return centroid, vt[-1]


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b)) / (
        float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-12
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def model_row(model_id: str) -> dict:
    surface = parse_vtk_polydata(
        DATA_DIR / f"{model_id}_torso_coarse_surface.vtk"
    )[0]
    electrodes = np.loadtxt(
        DATA_DIR / f"{model_id}_electrodes.csv", delimiter=",", skiprows=1
    )
    v = electrodes[3:9]  # V1..V6
    v4, v5, v6 = v[3], v[4], v[5]

    alpha = abs(v5[0] - v4[0]) / max(abs(v6[0] - v4[0]), 1e-9)
    wrap = abs(v6[1] - v4[1]) / max(abs(v6[0] - v4[0]), 1e-9)

    # Anterior sign in the model space: the chest electrodes sit on the side of
    # the torso centroid that the electrode cloud's Y offset points to.
    anterior = ANTERIOR_AXIS * np.sign(
        float(np.mean(v[:, 1]) - surface[:, 1].mean())
    )
    steepness = {}
    for label, position in (("V5", v5), ("V6", v6)):
        _, normal = surface_normal(surface, position)
        if float(np.dot(normal, anterior)) < 0:
            normal = -normal
        steepness[label] = angle_deg(normal, anterior)
    return {
        "id": model_id,
        "alpha_v5": float(alpha),
        "wrap_ratio": float(wrap),
        "steepness_v5_deg": steepness["V5"],
        "steepness_v6_deg": steepness["V6"],
    }


def sim_row(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = {step["target"]: step for step in report["press_plan"]}
    v1 = np.asarray(plan["V1"]["contact_world"], dtype=float)
    v2 = np.asarray(plan["V2"]["contact_world"], dtype=float)
    v4 = np.asarray(plan["V4"]["contact_world"], dtype=float)
    v6 = np.asarray(plan["V6"]["contact_world"], dtype=float)
    # The patient is yawed in the scene, so the lateral axis must come from the
    # parasternal pair (V1->V2 is pure lateral by construction: same u, +-v).
    anterior = np.array([0.0, 0.0, 1.0])
    lateral = v2 - v1
    lateral = lateral - anterior * float(np.dot(lateral, anterior))
    lateral = lateral / (np.linalg.norm(lateral) + 1e-12)
    delta = v6 - v4
    lateral_span = abs(float(np.dot(delta, lateral)))
    depth_span = abs(float(np.dot(delta, anterior)))
    wrap = depth_span / max(lateral_span, 1e-9)
    rows = {}
    for label, step in (("V5", plan["V5"]), ("V6", plan["V6"])):
        normal = np.asarray(step["normal_world"], dtype=float)
        if float(np.dot(normal, anterior)) < 0:
            normal = -normal
        rows[label] = angle_deg(normal, anterior)
    return {
        "source": str(report_path.relative_to(PROJECT_ROOT)),
        "lateral_axis_world": [float(c) for c in lateral],
        "wrap_ratio": float(wrap),
        "steepness_v5_deg": rows["V5"],
        "steepness_v6_deg": rows["V6"],
        "lateral_span_v4_v6_m": float(lateral_span),
        "depth_span_v4_v6_m": float(depth_span),
    }


def summarise(rows, key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def zscore(summary: dict, value: float) -> float | None:
    if summary["std"] <= 1e-9:
        return None
    return float((value - summary["mean"]) / summary["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-report", type=Path, default=SIM_REPORT)
    parser.add_argument("--out", type=Path, default=GT_REPORT)
    args = parser.parse_args()

    rows = [model_row(f"T_{i:02d}") for i in range(1, 26)]
    sim = sim_row(args.sim_report)

    items = {}
    for key in ("wrap_ratio", "steepness_v5_deg", "steepness_v6_deg"):
        gt = summarise(rows, key)
        items[key] = {
            "gt": gt,
            "sim": sim[key],
            "sim_z": zscore(gt, sim[key]),
        }
    items["alpha_v5"] = {
        "gt": summarise(rows, "alpha_v5"),
        "sim": "rule-imposed (fitted from the same distribution: 0.736)",
        "sim_z": None,
    }

    print(f"{'item':>18} {'GT mean+-sd':>18} {'sim':>10} {'z':>7}")
    for key in ("alpha_v5", "wrap_ratio", "steepness_v5_deg", "steepness_v6_deg"):
        item = items[key]
        gt = item["gt"]
        sim_text = (
            item["sim"] if isinstance(item["sim"], str) else f"{item['sim']:.2f}"
        )
        z = item["sim_z"]
        print(
            f"{key:>18} {gt['mean']:>10.2f} +- {gt['std']:<5.2f} "
            f"{sim_text:>10} {('-' if z is None else f'{z:+.2f}'):>7}"
        )

    report = {
        "provenance": (
            "I5 transferability: 25 SSM torso models (Bender et al., Zenodo "
            "10.5281/zenodo.20086105, CC-BY-4.0) vs the stylized biped; sim "
            f"row from {sim['source']} (GT path press plan)"
        ),
        "models": len(rows),
        "items": items,
        "sim": sim,
        "per_model": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ssm_i5_transfer: report -> {args.out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
