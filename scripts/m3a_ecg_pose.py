"""M3a entry point: MediaPipe perceptual chest frame and V1-V6 targets.

Run headless:
    ./scripts/run_headless.sh scripts/m3a_ecg_pose.py
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
    args = parser.parse_args()

    app = boot(headless=not args.gui, width=1280, height=720)
    try:
        from roboecg.task_manager.m3a_demo import run_m3a

        report = run_m3a(app, gui=args.gui)
        print(f"M3a: status = {report['status']}")
        print(
            f"M3a: frame origin error = "
            f"{report['frame_difference']['origin_distance_m'] * 1000:.1f} mm"
        )
        if report.get("fused_errors"):
            print(
                f"M3a: fused error mean/max = "
                f"{report['fused_errors']['position_error_m']['mean'] * 1000:.1f}/"
                f"{report['fused_errors']['position_error_m']['max'] * 1000:.1f} mm"
            )
        else:
            print(f"M3a: target generation failed: {report.get('target_generation_error')}")
    except BaseException:
        import traceback

        with open("/tmp/roboecg_m3a_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
