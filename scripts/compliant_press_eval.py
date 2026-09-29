#!/usr/bin/env python3
"""Evaluate the compliant electrode press against the position baseline.

Writes ``runs/m4/compliant_press_report.json`` and a force/depth figure
``runs/m4/compliant_press.png``.  Pure simulation (no Isaac): the contact
model is the documented engineering linear skin stiffness used by the M4
breathing evaluation.

Usage::

    python3 scripts/compliant_press_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.robot_controller.compliant_press import (  # noqa: E402
    CompliantPressConfig,
    run_scenarios,
    simulate_hold,
)
from roboecg.robot_controller.press_plan import (  # noqa: E402
    press_settings_from_rules,
)
from roboecg.target_localization.ecg_rules import load_ecg_rules  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "runs" / "m4"


def main() -> int:
    settings = press_settings_from_rules(load_ecg_rules())
    report = run_scenarios(settings)
    report["settings"]["contact_stiffness_n_m"] = settings.contact_stiffness_n_m
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "compliant_press_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"{'scenario':<38} {'maxF(N)':>8} {'maxDepth(mm)':>12} "
          f"{'RMSE(N)':>8} {'violation':>10}")
    for row in report["scenarios"]:
        print(
            f"{row['tag']:<38} {row['max_force_n']:>8.3f} "
            f"{row['max_indentation_m'] * 1000.0:>12.2f} "
            f"{row['force_rmse_n']:>8.3f} "
            f"{str(row['violation']):>10}"
        )
    print("summary:", json.dumps(report["summary"]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"plot skipped: {exc}")
        return 0

    base = CompliantPressConfig()
    pos = simulate_hold(settings, "position", 150.0, base)
    com = simulate_hold(settings, "compliant", 150.0, base)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(pos["time_s"], pos["force_n"], label="position (fixed depth)")
    axes[0].plot(com["time_s"], com["force_n"], label="compliant (force loop)")
    axes[0].axhline(settings.force_limit_n, color="crimson", ls="--",
                    label="force cap")
    axes[0].axhline(settings.force_target_n, color="grey", ls=":", label="target")
    axes[0].set_ylabel("contact force [N]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("15 breaths/min, +-8 mm chest motion, k=150 N/m")
    axes[1].plot(pos["time_s"], pos["indentation_m"] * 1000.0,
                 label="position")
    axes[1].plot(com["time_s"], com["indentation_m"] * 1000.0,
                 label="compliant")
    axes[1].axhline(settings.max_press_depth_m * 1000.0, color="crimson",
                    ls="--", label="depth cap")
    axes[1].set_ylabel("indentation [mm]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    png = RUNS_DIR / "compliant_press.png"
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
