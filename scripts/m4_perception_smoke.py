"""Smoke test: run the perception path once and print the targets vs GT.

Run:
    ./scripts/run_headless.sh scripts/m4_perception_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    app = boot(headless=True, width=320, height=180)
    try:
        from isaacsim.core.api import World

        from roboecg.perception.chest_detector import ChestLandmarkDetector
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.perception_pipeline import perceive_targets

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        rules = load_ecg_rules()

        detector = ChestLandmarkDetector()
        perceived = perceive_targets(stage, detector, rules)

        joints = read_joint_world_positions(stage)
        gt_landmarks = read_chest_landmarks(joints)
        gt_frame = build_chest_frame(gt_landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        gt_targets = {
            t.name: t
            for t in generate_v1_v6(gt_landmarks, gt_frame, mesh_points, rules).targets
        }

        print("prior fit rms = %.2f mm" % (perceived["prior"].fit_rms_m * 1000))
        for target in perceived["fused"]:
            gt = gt_targets[target.name]
            error = np.linalg.norm(
                np.asarray(target.position) - np.asarray(gt.position)
            )
            print(
                f"  {target.name}: source={target.source:13s} "
                f"err_vs_gt={error * 1000:6.2f} mm "
                f"incidence={target.info.get('incidence_deg', float('nan')):5.1f} deg "
                f"normal_vs_prior={target.info.get('normal_angle_vs_prior_deg')} "
                f"reason={target.info.get('reason')}"
            )
        errors = [
            float(np.linalg.norm(np.asarray(t.position) - np.asarray(gt_targets[t.name].position)))
            for t in perceived["fused"]
        ]
        print(f"mean error vs GT = {np.mean(errors) * 1000:.2f} mm "
              f"(max {np.max(errors) * 1000:.2f})")
    finally:
        app.close()


if __name__ == "__main__":
    main()
