"""I5-c close-out: leave-one-out population calibration of the SSM transfer.

Reads the per-model I5-b/e reports (runs/m4/m4_report_ssm_torso_T_XX.json) and
computes, with honest leave-one-out fitting over the 25-model population:

  * raw               - no calibration
  * translation       - mean target offset fitted on the other 24 models
  * similarity        - scale/rotation/translation fitted on the other 24
  * per-electrode     - per-electrode 3D offset table fitted on the other 24

Protocol note: the calibration constants are always fitted on the *other*
models, so the reported errors for a held-out model are honest.  This mirrors
the project's existing LOO methodology for the intercostal-drop regression and
the V5 lateral fraction.

Output: runs/gt_validation/ssm_loo_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "runs" / "m4"
OUT = PROJECT_ROOT / "runs" / "gt_validation" / "ssm_loo_calibration.json"


def similarity_fit(points: np.ndarray, targets: np.ndarray):
    centre_p, centre_t = points.mean(0), targets.mean(0)
    x, y = points - centre_p, targets - centre_t
    u, s, vt = np.linalg.svd(x.T @ y)
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T
    scale = float(s.sum() / (x**2).sum())
    translation = centre_t - scale * rotation @ centre_p
    return scale, rotation, translation


def main() -> None:
    models, perceived, truth = [], [], []
    for index in range(1, 26):
        path = RUNS / f"m4_report_ssm_torso_T_{index:02d}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        models.append(f"T_{index:02d}")
        perceived.append(
            np.array([row["position_world"] for row in report["electrodes"]])
        )
        truth.append(np.array([row["gt_world"] for row in report["electrodes"]]))
    if len(models) < 3:
        raise RuntimeError("need at least three model reports")
    perceived = np.asarray(perceived)
    truth = np.asarray(truth)
    n = len(models)

    errs_raw = np.linalg.norm(perceived - truth, axis=2) * 1000.0
    rows = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        p_o = perceived[others].reshape(-1, 3)
        t_o = truth[others].reshape(-1, 3)
        delta = (t_o - p_o).mean(axis=0)
        err_t = np.linalg.norm(perceived[i] + delta - truth[i], axis=1) * 1000.0
        scale, rotation, translation = similarity_fit(p_o, t_o)
        err_s = (
            np.linalg.norm(
                scale * (rotation @ perceived[i].T).T + translation - truth[i],
                axis=1,
            )
            * 1000.0
        )
        offsets = (t_o - p_o).reshape(-1, 6, 3).mean(axis=0)
        err_p = np.linalg.norm(perceived[i] + offsets - truth[i], axis=1) * 1000.0
        rows.append(
            {
                "model": models[i],
                "raw_mm": float(errs_raw[i].mean()),
                "translation_mm": float(err_t.mean()),
                "similarity_mm": float(err_s.mean()),
                "per_electrode_mm": float(err_p.mean()),
            }
        )

    summary = {}
    for key in ("raw_mm", "translation_mm", "similarity_mm", "per_electrode_mm"):
        values = np.array([row[key] for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "under_20mm": int((values <= 20.0).sum()),
            "under_25mm": int((values <= 25.0).sum()),
        }

    report = {
        "models": models,
        "provenance": (
            "I5-c: SSM fine-tuned detector + LOO notch anchor; calibration "
            "constants fitted on the other 24 models and applied to the "
            "held-out one (honest LOO); canonical training-matched placement"
        ),
        "summary": summary,
        "per_model": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'variant':>14} {'mean':>7} {'min':>7} {'max':>7} {'<=20':>5} {'<=25':>5}")
    for key, label in (
        ("raw_mm", "raw"),
        ("translation_mm", "translation"),
        ("similarity_mm", "similarity"),
        ("per_electrode_mm", "per-electrode"),
    ):
        s = summary[key]
        print(
            f"{label:>14} {s['mean']:>7.1f} {s['min']:>7.1f} {s['max']:>7.1f} "
            f"{s['under_20mm']:>3}/{n} {s['under_25mm']:>3}/{n}"
        )
    print(f"report -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
