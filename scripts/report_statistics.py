"""Statistical summary of the v2 evaluation reports (mean +- SD, 95 % CI).

The detector evaluation is a fixed held-out split of 60 samples (360 electrode
placements); the disturbance evaluation is 11 deterministic configurations.
This script reports the sampling uncertainty of those distributions instead of
single-point numbers.

Run:
    $ISAACSIM_ENV/bin/python scripts/report_statistics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "runs" / "statistics_report.json"
ELECTRODES = ("V1", "V2", "V3", "V4", "V5", "V6")


def ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(values.size)
    return mean - half, mean + half


def summarise(values: np.ndarray) -> dict:
    low, high = ci95(values)
    return {
        "mean_mm": float(values.mean()),
        "std_mm": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.percentile(values, 90)),
        "max_mm": float(values.max()),
        "ci95_mm": [low, high],
        "n": int(values.size),
    }


def main() -> None:
    report: dict = {"provenance": "computed from the v2 run reports"}

    # --- M3b detector: per-placement errors over the 60-sample split --------
    m3b = json.loads((PROJECT_ROOT / "runs" / "m3b" / "eval_report.json").read_text())
    errors = np.asarray(
        [sample["target_errors_mm"] for sample in m3b["per_sample"]], dtype=float
    )
    report["m3b_detector"] = {
        "overall": summarise(errors.reshape(-1)),
        "per_electrode": {
            name: summarise(errors[:, index])
            for index, name in enumerate(ELECTRODES)
        },
    }

    # --- P0 direct regression (same split) ----------------------------------
    p0 = json.loads((PROJECT_ROOT / "runs" / "p0" / "eval_report.json").read_text())
    report["p0_direct_vs_rules"] = {
        "direct_mean_mm": p0["direct"]["error_mm_mean"],
        "rules_mean_mm": p0["rules"]["error_mm_mean"],
        "same_val_split": p0["rules"].get("same_val_split"),
    }

    # --- M4 disturbance evaluation -----------------------------------------
    disturbance_path = PROJECT_ROOT / "runs" / "m4" / "disturbance_report.json"
    if disturbance_path.is_file():
        disturbance = json.loads(disturbance_path.read_text())
        means = np.asarray(
            [config["mean_mm"] for config in disturbance["configs"]], dtype=float
        )
        report["m4_disturbance"] = {
            "configs": len(means),
            "overall": summarise(means),
            "per_config": {
                config["config"]: {
                    "mean_mm": config["mean_mm"],
                    "max_mm": config["max_mm"],
                }
                for config in disturbance["configs"]
            },
        }

    # --- M4 placement runs (verification errors) ----------------------------
    placements = {}
    for name in ("m4_report.json", "m4_report_perception.json",
                 "m4_report_breathing.json"):
        path = PROJECT_ROOT / "runs" / "m4" / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        errors = np.asarray(
            [row["error_mm"] for row in data["verification"]
             if row.get("error_mm") is not None],
            dtype=float,
        )
        placements[name] = {
            "status": data["status"],
            "source": data["target_source"],
            "execution_error": summarise(errors) if errors.size else None,
            "min_clearance_m": data["execution"]["min_clearance_m"],
            "success_rate": data["acceptance"]["first_attempt_success_rate"],
        }
    report["m4_placements"] = placements

    # --- P0b real patient ---------------------------------------------------
    p0b_path = PROJECT_ROOT / "runs" / "p0b" / "real_patient_report.json"
    if p0b_path.is_file():
        p0b = json.loads(p0b_path.read_text())
        report["p0b_real_patient"] = {
            "measurements_mm": p0b["measurements_mm"],
            "rule_prediction": p0b["rule_prediction"],
        }

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print("M3b detector over 60 held-out samples (360 placements):")
    overall = report["m3b_detector"]["overall"]
    print(
        f"  overall mean {overall['mean_mm']:.2f} mm "
        f"(95% CI {overall['ci95_mm'][0]:.2f}-{overall['ci95_mm'][1]:.2f}), "
        f"p90 {overall['p90_mm']:.2f}, max {overall['max_mm']:.2f}"
    )
    for name, stats in report["m3b_detector"]["per_electrode"].items():
        print(
            f"  {name}: {stats['mean_mm']:5.2f} mm "
            f"(CI {stats['ci95_mm'][0]:.2f}-{stats['ci95_mm'][1]:.2f})"
        )
    if "m4_disturbance" in report:
        overall = report["m4_disturbance"]["overall"]
        print(
            f"M4 disturbance over {report['m4_disturbance']['configs']} configs: "
            f"mean {overall['mean_mm']:.2f} mm "
            f"(CI {overall['ci95_mm'][0]:.2f}-{overall['ci95_mm'][1]:.2f}), "
            f"max {overall['max_mm']:.2f}"
        )
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
