"""Visual re-detection of the placed electrodes (closed-loop verification, sim).

Places a marker disc at each planned contact from an M4 report, renders the
overhead RGB-D, detects the markers from the image only (colour segmentation +
connected components + deprojection) and scores the re-detection error.

This is the simulation counterpart of the real visual verification route
(El Ghebouli 2025 / Bayer 2023): the robot does not trust its own pose, it
looks at the placed electrodes.

Run:
    ./scripts/run_headless.sh scripts/m4_visual_verify.py
    ./scripts/run_headless.sh scripts/m4_visual_verify.py --report runs/m4/m4_report_perception.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "runs" / "m4" / "m4_report.json"
REPORT_PATH = PROJECT_ROOT / "runs" / "m4" / "visual_verify.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    app = boot(headless=not args.gui, width=640, height=360)
    try:
        from isaacsim.core.api import World

        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.visual_verify import verify_placed_electrodes

        m4_report = json.loads(Path(args.report).read_text())
        planned = {
            row["target"]: {
                "contact_world": row["contact_world"],
                "normal_world": row["normal_world"],
            }
            for row in m4_report["press_plan"]
        }

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        result = verify_placed_electrodes(stage, planned, width=640, height=360)

        result["source_report"] = str(args.report)
        result["provenance"] = (
            "visual re-detection of marker discs placed at the M4 planned "
            "contacts; detection uses only the rendered overhead RGB-D"
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n")

        print(
            f"visual verify: detected {result['detected_count']}"
            f"/{result['targets']} markers"
        )
        for row in result["per_target"]:
            error = row["error_mm"]
            print(
                f"  {row['target']}: error = "
                f"{'n/a' if error is None else f'{error:.2f} mm'} "
                f"({row['marker_pixels']} px)"
            )
        print(
            f"visual verify: mean = {result['error_mm']['mean']:.2f} mm, "
            f"max = {result['error_mm']['max']:.2f} mm"
        )
        print(f"report: {REPORT_PATH}")
    except BaseException:
        import traceback

        with open("/tmp/roboecg_visual_verify_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
