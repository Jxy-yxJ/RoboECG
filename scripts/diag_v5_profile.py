"""Print the chest cross-section profile at the V4-V6 level to fix V5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    app = boot(headless=True)
    try:
        from isaacsim.core.api import World

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import _frame_components
        from roboecg.task_manager import ecg_scene

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        joints = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joints)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        points = ecg_scene.mesh_world_points(stage, "/World/Human")
        along, lateral, normal = _frame_components(points, frame)
        for band in (0.015, 0.03):
            mask = (np.abs(along + 0.283) <= band) & (normal > -0.10)
            v, n = lateral[mask], normal[mask]
            print(f"\n=== u=-0.283 +/-{band*1000:.0f} mm : {np.count_nonzero(mask)} pts ===")
            for lo in np.arange(0.10, 0.17, 0.01):
                cell = (v >= lo) & (v < lo + 0.01)
                if np.count_nonzero(cell) == 0:
                    print(f"  v={lo:.2f}-{lo+0.01:.2f}: (empty)")
                    continue
                print(
                    f"  v={lo:.2f}-{lo+0.01:.2f}: n pts={np.count_nonzero(cell):3d} "
                    f"n_max={n[cell].max():+.4f} n_p95={np.percentile(n[cell],95):+.4f} "
                    f"n_min={n[cell].min():+.4f}"
                )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_diag_v5_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
