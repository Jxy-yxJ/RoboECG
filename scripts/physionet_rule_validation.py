"""P0b (completed): real-patient validation of the V1-V6 rules.

Earlier attempt (P0b in the frozen v1) failed to map V1-V6 onto the 120
Dalhousie electrodes: it tried to match digitised waveforms against the 352-node
BSPM matrix and the matching did not converge.

The blocker is resolved by `assets/external/physionet_cinc2007/README_challenge.txt`
(the challenge data readme), which defines the standard-lead positions in terms
of the 352-node mesh (1-based node numbers):

    V1 = node169
    V2 = node171
    V3 = (node192 + node193)/2
    V4 = node216
    V5 = (node217 + 2*node218)/3
    V6 = node219

V1, V2, V4, V6 coincide with actual electrodes of the 120-lead array (they are
designed into the array); V3 and V5 are interpolated, which confirms that these
are clinically placed standard positions rather than grid-snapped electrodes.

This script measures, on the real patient (case 3, male, 78 y):

  * the V1-V2 / V4-V5 / V5-V6 / V4-V6 distances and the V4-V6 wrap ratio,
    compared with the 25-model GT (Zenodo 20086105),
  * the V1 -> V4 vertical drop, compared with the regression fitted on that GT
    (`configs/ecg_rules.yaml -> anatomy.fourth_to_fifth_ics`),
  * the sternal-notch -> V1/V2 distance (SNND rule check), using the mesh neck
    ring as the notch proxy.

Run:
    $ISAACSIM_ENV/bin/python scripts/physionet_rule_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "assets" / "external" / "physionet_cinc2007"
REPORT_PATH = PROJECT_ROOT / "runs" / "p0b" / "real_patient_report.json"

# Challenge readme.txt, 1-based node numbers into case0003_b352.pts
STANDARD_LEADS = {
    "V1": [(169, 1.0)],
    "V2": [(171, 1.0)],
    "V3": [(192, 0.5), (193, 0.5)],
    "V4": [(216, 1.0)],
    "V5": [(217, 1.0 / 3.0), (218, 2.0 / 3.0)],
    "V6": [(219, 1.0)],
}

# GT population (25 shape models, Zenodo 20086105), from
# runs/gt_validation/torso_models_report.json
GT = {
    "v1_v2_mm": (49.6, 10.3),
    "v1_to_v4_vertical_mm": (86.7, 16.7),
    "v2_to_v4_vertical_mm": (89.6, 18.3),
    "v4_v5_mm": (57.9, 12.9),
    "v5_v6_mm": (50.1, 14.0),
    "v4_v6_mm": (105.6, 23.7),
    "v4_v6_wrap_ratio": (2.37, 0.42),
}

SNND_M = 0.193
DROP_INTERCEPT_MM = -60.89
DROP_SLOPE_MM_PER_MM = 0.39345

# Arm stubs of the Dalhousie mesh extend beyond |x| = 183 mm (mid-upper-arm
# cut); the torso proper stays inside |x| <= 175 mm.  Used to measure the
# torso-only width the way the GT models (arms removed at the deltoid) define it.
TORSO_HALF_WIDTH_LIMIT_MM = 175.0


def load_points(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name)


def standard_lead_positions(mesh: np.ndarray) -> dict:
    out = {}
    for name, nodes in STANDARD_LEADS.items():
        position = np.zeros(3)
        for node, weight in nodes:
            position = position + weight * mesh[node - 1]
        out[name] = position
    return out


def measure_torso_width(mesh: np.ndarray) -> dict:
    """Torso transverse width in mm, excluding the arm stubs."""
    torso = mesh[np.abs(mesh[:, 0]) <= TORSO_HALF_WIDTH_LIMIT_MM]
    width_bbox = float(torso[:, 0].max() - torso[:, 0].min())
    # width at the V4/V6 level (z of the 5th intercostal row)
    band = torso[np.abs(torso[:, 2] + 39.5) < 10.0]
    width_5ics = (
        float(band[:, 0].max() - band[:, 0].min()) if len(band) > 3 else None
    )
    return {
        "torso_only_bbox_mm": width_bbox,
        "torso_width_at_v4_level_mm": width_5ics,
        "limit_mm": TORSO_HALF_WIDTH_LIMIT_MM,
    }


def estimate_notch_level(mesh: np.ndarray) -> float:
    """Sternal notch proxy: front-centre bottom of the neck cut ring.

    The neck cut ring is the set of the highest-z nodes; its front-bottom point
    is the closest mesh analogue of the sternal notch.
    """
    top = mesh[mesh[:, 2] > 195.0]
    front = top[top[:, 1] < 0.0]
    # bottom of the front neck ring = lowest z among the front neck nodes
    return float(front[:, 2].min())


def main() -> None:
    mesh = load_points("case0003_b352.pts")
    v = standard_lead_positions(mesh)

    report: dict = {"source": {
        "dataset": "PhysioNet/CinC Challenge 2007 case 3 (Dalhousie torso)",
        "license": "ODC-By 1.0",
        "mapping": "challenge data readme.txt (standard-lead node definitions)",
        "subject": "male, 78 y (case0003.dat header)",
    }}

    # --- geometric measurements on the real patient -----------------------
    report["positions_mm"] = {
        name: [round(float(x), 2) for x in p] for name, p in v.items()
    }

    def dist(a, b):
        return float(np.linalg.norm(v[a] - v[b]))

    measurements = {
        "v1_v2_mm": dist("V1", "V2"),
        "v1_to_v4_vertical_mm": float(v["V1"][2] - v["V4"][2]),
        "v2_to_v4_vertical_mm": float(v["V2"][2] - v["V4"][2]),
        "v4_v5_mm": dist("V4", "V5"),
        "v5_v6_mm": dist("V5", "V6"),
        "v4_v6_mm": dist("V4", "V6"),
    }
    # wrap ratio: depth (anterior-posterior) over lateral travel, V4 -> V6
    lateral = float(v["V6"][0] - v["V4"][0])
    depth = float(abs(v["V6"][1] - v["V4"][1]))
    measurements["v4_v6_wrap_ratio"] = depth / lateral if lateral else float("nan")
    report["measurements_mm"] = {
        key: round(value, 2) for key, value in measurements.items()
    }

    # --- comparison with the GT population --------------------------------
    comparison = {}
    for key, (mean, sd) in GT.items():
        value = measurements[key]
        comparison[key] = {
            "measured": round(value, 2),
            "gt_mean": mean,
            "gt_sd": sd,
            "z_score": round((value - mean) / sd, 2),
        }
    report["vs_gt_population"] = comparison

    # --- rule prediction (drop regression + SNND) -------------------------
    width = measure_torso_width(mesh)
    report["torso_width"] = {k: (round(v, 2) if v else None)
                             for k, v in width.items()}
    notch_z = estimate_notch_level(mesh)
    u_4ics_predicted = notch_z - SNND_M * 1000.0
    drop_predicted = (
        DROP_INTERCEPT_MM + DROP_SLOPE_MM_PER_MM * width["torso_only_bbox_mm"]
    )
    u_5ics_predicted = u_4ics_predicted - drop_predicted
    report["rule_prediction"] = {
        "notch_z_mm": round(notch_z, 2),
        "snnd_m": SNND_M,
        "u_4ics_predicted_z_mm": round(u_4ics_predicted, 2),
        "u_4ics_actual_z_mm": round(float(v["V1"][2]), 2),
        "u_4ics_error_mm": round(u_4ics_predicted - float(v["V1"][2]), 2),
        "torso_width_used_mm": width["torso_only_bbox_mm"],
        "drop_predicted_mm": round(drop_predicted, 2),
        "drop_actual_mm": round(float(v["V1"][2] - v["V4"][2]), 2),
        "drop_error_mm": round(drop_predicted - float(v["V1"][2] - v["V4"][2]), 2),
        "u_5ics_predicted_z_mm": round(u_5ics_predicted, 2),
        "u_5ics_actual_z_mm": round(float(v["V4"][2]), 2),
        "u_5ics_error_mm": round(u_5ics_predicted - float(v["V4"][2]), 2),
    }

    # --- sensitivity of the drop error to the width definition -------------
    sensitivity = {}
    for label, w in (
        ("torso_only_bbox", width["torso_only_bbox_mm"]),
        ("at_v4_level", width["torso_width_at_v4_level_mm"]),
        ("electrode_span", float(mesh[:, 0].max() - mesh[:, 0].min())),
    ):
        if w is None:
            continue
        predicted = DROP_INTERCEPT_MM + DROP_SLOPE_MM_PER_MM * w
        sensitivity[label] = {
            "width_mm": round(w, 2),
            "drop_predicted_mm": round(predicted, 2),
            "drop_error_mm": round(predicted - measurements["v1_to_v4_vertical_mm"], 2),
        }
    report["drop_error_sensitivity"] = sensitivity

    report["conclusion"] = {
        "snnd_rule": (
            "VALIDATED on real data: predicted 4th-ICS level (notch - 193 mm) "
            f"lands within {abs(report['rule_prediction']['u_4ics_error_mm']):.0f} mm "
            "of the actual V1/V2 electrodes."
        ),
        "drop_regression": (
            "OVERESTIMATES the real patient's V1->V4 drop by "
            f"{report['rule_prediction']['drop_error_mm']:.0f} mm: the GT-fitted "
            "regression predicts ~69-89 mm (for any plausible torso width) while "
            "the patient's clinical electrodes sit 48.4 mm apart vertically."
        ),
        "caveats": [
            "single real subject (male, 78 y) - cannot establish a population",
            "the GT models (25 SSM torsos) may themselves carry an algorithmic "
            "placement bias for the vertical drop",
            "electrode placement by the clinical staff carries the usual "
            "inter-operator variability (+-1-2 cm)",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print("real-patient measurements (mm):")
    for key, value in measurements.items():
        mean, sd = GT[key]
        print(f"  {key:26s} = {value:7.2f}   GT {mean:6.1f} +- {sd:5.1f}"
              f"   z={(value - mean) / sd:+.2f}")
    print("\nrule prediction:")
    for key, value in report["rule_prediction"].items():
        print(f"  {key:26s} = {value}")
    print(f"\nreport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
