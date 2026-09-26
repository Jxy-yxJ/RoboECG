"""M2 entry point: RGB-D depth fusion of V1-V6 with the local normal prior.

Run headless (writes report + screenshots):
    ./scripts/run_headless.sh scripts/m2_ecg_depth.py
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
        from roboecg.task_manager.m2_demo import run_m2

        report = run_m2(app, gui=args.gui)
        print(f"M2: status = {report['status']}")
        evaluation = report["evaluation"]
        print(f"M2: hit rate = {evaluation['depth_hit_rate']:.2f}")
        print(
            f"M2: position error mean/max = "
            f"{evaluation['position_error_m']['mean'] * 1000:.2f}/"
            f"{evaluation['position_error_m']['max'] * 1000:.2f} mm"
        )
        measured = evaluation["measured_normal_angle_vs_mesh_deg"]
        print(
            f"M2: measured normal vs mesh mean/max = "
            f"{measured['mean']:.2f}/{measured['max']:.2f} deg"
        )
        print(f"M2: normal acceptance rate = {evaluation['normal_acceptance_rate']:.2f}")
        print(
            f"M2: fused normal vs mesh mean/max = "
            f"{evaluation['fused_normal_angle_vs_mesh_deg']['mean']:.2f}/"
            f"{evaluation['fused_normal_angle_vs_mesh_deg']['max']:.2f} deg"
        )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_m2_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
