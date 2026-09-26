"""M1 entry point: measured V1-V6 targets -> UR3 reach demo.

Run headless (writes report + video):
    ./scripts/run_headless.sh scripts/m1_ecg_reach.py

Run with the Isaac Sim window:
    ./scripts/run_headless.sh scripts/m1_ecg_reach.py --gui
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
    args = parser.parse_args()

    app = boot(headless=not args.gui, width=1280, height=720)
    try:
        from roboecg.task_manager.m1_demo import run_m1

        report = run_m1(app, gui=args.gui, video=not args.no_video)
        print(f"M1: status = {report['status']}")
        print(f"M1: min clearance = {report['safety']['min_clearance_m']} m")
        print(f"M1: video = {report['execution']['video']}")
    except BaseException:
        # SimulationApp.close() terminates the process, so the traceback must be
        # captured before the finally block runs.
        import traceback

        with open("/tmp/roboecg_m1_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        with open("/tmp/roboecg_m1_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
