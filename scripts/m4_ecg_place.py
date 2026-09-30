"""M4 entry point: full V1-V6 placement cycle (press + verification + re-place).

Run headless (writes report + video):
    ./scripts/run_headless.sh scripts/m4_ecg_place.py                 # GT path
    ./scripts/run_headless.sh scripts/m4_ecg_place.py --perception    # full path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    parser.add_argument("--no-video", action="store_true", help="skip frame recording")
    parser.add_argument(
        "--perception",
        action="store_true",
        help="use the detector + rules + fusion path instead of the GT path",
    )
    parser.add_argument(
        "--breathing",
        action="store_true",
        help="move the patient chest with a 15/min +-8 mm breathing motion",
    )
    parser.add_argument(
        "--force-tracking",
        action="store_true",
        help=(
            "compliant force-feedback hold instead of the fixed press depth "
            "(implies --breathing; see docs/ECG_V3_SOLUTION_PLAN.md)"
        ),
    )
    parser.add_argument(
        "--multiview",
        action="store_true",
        help=(
            "perception path with the lateral second view for the V5/V6 wall "
            "(see docs/ECG_V3_SOLUTION_PLAN.md section 1.4)"
        ),
    )
    args = parser.parse_args()

    app = boot(headless=not args.gui, width=1280, height=720)
    try:
        import json

        from roboecg.task_manager.m4_demo import RUNS_DIR, run_m4

        report = run_m4(
            app,
            gui=args.gui,
            video=not args.no_video,
            perception=args.perception,
            breathing=args.breathing,
            force_tracking=args.force_tracking,
            multiview=args.multiview,
        )
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        if args.force_tracking:
            report_name = "m4_report_force_tracking.json"
        elif args.perception and args.multiview:
            report_name = "m4_report_perception_multiview.json"
        elif args.perception:
            report_name = "m4_report_perception.json"
        elif args.breathing:
            report_name = "m4_report_breathing.json"
        else:
            report_name = "m4_report.json"
        (RUNS_DIR / report_name).write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(f"M4: status = {report['status']} ({report['target_source']})")
        print(
            f"M4: verified {report['acceptance']['verified_ok']}"
            f"/{report['acceptance']['targets']}"
        )
        print(
            f"M4: min clearance = {report['execution']['min_clearance_m']} m"
        )
        if report.get("perception"):
            print(
                "M4: perception mean diff vs rules-on-cloud = "
                f"{report['perception']['target_diff_vs_cloud_rules_mean_mm'] * 1000:.2f} mm, "
                "vs rules-on-mesh = "
                f"{report['perception']['target_diff_vs_mesh_rules_mean_mm'] * 1000:.2f} mm, "
                "landmark error = "
                f"{report['perception']['detector_landmark_error_mean_mm'] * 1000:.2f} mm"
            )
        if report.get("video"):
            print(f"M4: video = {report['video']}")
    except BaseException:
        # SimulationApp.close() terminates the process, so the traceback must be
        # captured before the finally block runs.
        import traceback

        with open("/tmp/roboecg_m4_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
