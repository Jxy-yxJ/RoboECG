"""M4 disturbance evaluation: detector-path accuracy under 5 disturbances.

For each configuration the scene is disturbed, the overhead depth is rendered,
the M3b detector predicts the chest landmarks (with the training-split
calibration), the rules generate V1-V6, the M2 fusion produces the final
contact targets, and the result is compared against the same chain applied to
the undisturbed-sensor ground-truth landmarks of that configuration (so the
metric isolates the perception error introduced by the disturbance).

Both paths share one code path (`perception_pipeline`); the previous version of
this evaluation fed the rules the *raw* detector landmarks while the frame was
calibrated, which inflated the error.

Dimensions (docs/ECG_PIPELINE.md 5.7): body size, arm pose, breathing, camera
mounting, patient placement.

Run:
    ./scripts/run_headless.sh scripts/m4_ecg_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "runs" / "m4" / "disturbance_report.json"

WIDTH, HEIGHT, FOV_DEG = 320, 180, 90.0
MULTIVIEW = "--multiview" in sys.argv
# Same lateral view as the deployed pipeline (m4_demo.LATERAL_CAMERA): with
# --multiview both views are captured per config and pooled (anchored snap +
# per-target view routing), matching the current perception chain.
LATERAL_CAMERA = {
    "prim_path": "/World/Cameras/PerceptionLateral",
    "position": (0.00, 0.75, 0.95),
    "look_at": (-0.133, 0.14, 0.89),
}

CONFIGS = [
    ("baseline", {}),
    ("scale-0.90", {"scale": 0.90}),
    ("scale-1.10", {"scale": 1.10}),
    ("arm-60deg", {"abduction": 60.0}),
    ("arm-90deg", {"abduction": 90.0}),
    ("breath-+8mm", {"breath_mm": 8.0}),
    ("breath--8mm", {"breath_mm": -8.0}),
    ("camera-+4cm", {"camera_dxyz": (0.04, 0.03, 0.02)}),
    ("camera--4cm", {"camera_dxyz": (-0.04, -0.03, -0.02)}),
    ("patient-+2cm", {"patient_dxy": (0.02, 0.02)}),
    ("patient--2cm", {"patient_dxy": (-0.02, -0.02)}),
]


def main() -> None:
    app = boot(headless=True, width=WIDTH, height=HEIGHT)
    try:
        from isaacsim.core.api import World

        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
        )
        from roboecg.perception.chest_detector import (
            ChestLandmarkDetector,
            landmarks_from_points,
        )
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import depth_to_world_points
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.perception.torso_prior import fit_chest_surface_prior
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import (
            generate_v1_v6,
            measure_torso_width,
        )
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.target_localization.fusion import (
            best_view_index,
            fuse_target,
        )
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.rendering import capture_depth
        from roboecg.task_manager.supine_pose import (
            apply_supine_pose,
            place_supine,
            set_prim_scale,
            set_prim_translate,
        )

        detector = ChestLandmarkDetector()
        rules = load_ecg_rules()
        fusion_settings = rules["depth_fusion"]
        intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_DEG)
        camera_path = "/World/Cameras/PerceptionRGBD"
        base_camera = list(ecg_scene.CAMERAS["PerceptionRGBD"]["position"])
        base_root = list(ecg_scene.LAYOUT["human_root_xy"])

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)

        lateral_path = None
        if MULTIVIEW:
            lateral_path = LATERAL_CAMERA["prim_path"]
            ecg_scene.add_camera(
                stage,
                lateral_path,
                position=LATERAL_CAMERA["position"],
                look_at=LATERAL_CAMERA["look_at"],
            )
            for _ in range(3):
                world.step(render=True)
            print("M4 eval: multiview enabled (overhead + lateral pooled)",
                  flush=True)

        rows = []
        for label, config in CONFIGS:
            scale = float(config.get("scale", 1.0))
            abduction = float(config.get("abduction", 75.0))
            breath = float(config.get("breath_mm", 0.0)) / 1000.0
            camera_d = np.asarray(config.get("camera_dxyz", (0.0, 0.0, 0.0)))
            patient_d = np.asarray(config.get("patient_dxy", (0.0, 0.0)))

            apply_supine_pose(
                stage,
                pose=(
                    (("L_UpArm",), (0.0, 0.0, 1.0), 90.0 - abduction),
                    (("R_UpArm",), (0.0, 0.0, 1.0), -90.0),
                ),
            )
            set_prim_scale(stage.GetPrimAtPath("/World/Human"), scale)
            place_supine(
                stage,
                human_root_path="/World/Human",
                root_xy=(
                    base_root[0] + patient_d[0],
                    base_root[1] + patient_d[1],
                ),
                table_top_z=ecg_scene.LAYOUT["table_top_z"] + breath,
            )
            set_prim_translate(
                stage.GetPrimAtPath(camera_path),
                (
                    base_camera[0] + camera_d[0],
                    base_camera[1] + camera_d[1],
                    base_camera[2] + camera_d[2],
                ),
            )
            joints = read_joint_world_positions(stage)
            gt_landmarks = read_chest_landmarks(joints)
            gt_frame = build_chest_frame(gt_landmarks, anterior_hint=(0.0, 0.0, 1.0))

            camera_matrix = ecg_scene.world_matrix(stage, camera_path)
            camera_position = camera_matrix[:3, 3]
            cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
            depth = capture_depth(camera_path, WIDTH, HEIGHT)

            # -- perception path (shared detector module, calibrated) --------
            prediction = detector.predict(
                depth, intrinsics, camera_position, cv_rotation
            )
            pred_landmarks = prediction["landmarks"]
            pred_frame = prediction["frame"]

            # Surface source for the rules: the depth cloud (the real pipeline
            # path), not the mesh, so this evaluation matches the M3b setup.
            points = depth_to_world_points(
                depth, intrinsics, camera_position, cv_rotation, stride=1
            )
            views = [
                {
                    "depth": depth,
                    "intrinsics": intrinsics,
                    "camera_position": camera_position,
                    "cv_rotation": cv_rotation,
                }
            ]
            if lateral_path is not None:
                lateral_matrix = ecg_scene.world_matrix(stage, lateral_path)
                lateral_position = lateral_matrix[:3, 3]
                lateral_rotation = cv_rotation_from_usd(lateral_matrix[:3, :3])
                lateral_depth = capture_depth(lateral_path, WIDTH, HEIGHT)
                views.append(
                    {
                        "depth": lateral_depth,
                        "intrinsics": intrinsics,
                        "camera_position": lateral_position,
                        "cv_rotation": lateral_rotation,
                    }
                )
                points = np.concatenate(
                    [
                        points,
                        depth_to_world_points(
                            lateral_depth,
                            intrinsics,
                            lateral_position,
                            lateral_rotation,
                            stride=1,
                        ),
                    ],
                    axis=0,
                )
            prior = fit_chest_surface_prior(points, pred_frame)
            prior_gt = fit_chest_surface_prior(points, gt_frame)
            gt_targets = generate_v1_v6(
                gt_landmarks, gt_frame, points, rules, prior=prior_gt
            )
            pred_targets = generate_v1_v6(
                pred_landmarks, pred_frame, points, rules, prior=prior
            )

            # -- M2 fusion on both paths (the deployed chain), per-view ------
            camera_positions = [view["camera_position"] for view in views]

            def _fuse(targets, frame, path_prior):
                fused = []
                for target in targets:
                    view = views[0]
                    if len(views) > 1:
                        view = views[
                            best_view_index(
                                target.normal, target.position, camera_positions
                            )
                        ]
                    fused.append(
                        fuse_target(
                            target,
                            frame,
                            path_prior,
                            view["depth"],
                            view["intrinsics"],
                            view["camera_position"],
                            view["cv_rotation"],
                            fusion_settings,
                        )
                    )
                return fused

            fused_gt = _fuse(gt_targets.targets, gt_frame, prior_gt)
            fused_pred = _fuse(pred_targets.targets, pred_frame, prior)
            errors = [
                float(np.linalg.norm(np.asarray(p.position) - np.asarray(g.position)))
                for p, g in zip(fused_pred, fused_gt)
            ]
            # diagnostics: the u/v of both target sets and the rule inputs
            def _uv(target, fr):
                p = np.asarray(target.position, dtype=float)
                d = p - np.asarray(fr.origin)
                return [float(d @ fr.up), float(d @ fr.lateral)]

            def _rule_inputs(landmarks_lm, fr):
                v_sh = abs(float(fr.to_frame(landmarks_lm.shoulder_left)[1]))
                width = measure_torso_width(points, fr, half_width_limit=v_sh)
                return {
                    "v_shoulder_mm": v_sh * 1000.0,
                    "width_mm": width * 1000.0,
                    "drop_mm": -60.89 + 0.39345 * width * 1000.0,
                }

            diag = {
                "gt": {
                    t.name: _uv(t, gt_frame) for t in gt_targets.targets
                },
                "pred": {
                    t.name: _uv(t, pred_frame) for t in pred_targets.targets
                },
                "gt_rules": _rule_inputs(gt_landmarks, gt_frame),
                "pred_rules": _rule_inputs(pred_landmarks, pred_frame),
                "fusion_sources_pred": [t.source for t in fused_pred],
            }
            rows.append(
                {
                    "config": label,
                    "target_errors_mm": [round(e * 1000.0, 2) for e in errors],
                    "mean_mm": float(np.mean(errors) * 1000.0),
                    "max_mm": float(np.max(errors) * 1000.0),
                    "diagnostics": diag,
                    "frame_origin_mm": float(
                        np.linalg.norm(
                            np.asarray(pred_frame.origin)
                            - np.asarray(gt_frame.origin)
                        )
                        * 1000.0
                    ),
                }
            )
            print(
                f"M4 eval: {label:>12} mean={rows[-1]['mean_mm']:6.2f} mm "
                f"max={rows[-1]['max_mm']:6.2f} mm",
                flush=True,
            )

        baseline = rows[0]["mean_mm"]
        report = {
            "multiview": bool(MULTIVIEW),
            "configs": rows,
            "baseline_mean_mm": baseline,
            "worst_mean_mm": max(row["mean_mm"] for row in rows),
            "worst_max_mm": max(row["max_mm"] for row in rows),
            "mean_mm_overall": float(np.mean([row["mean_mm"] for row in rows])),
            "acceptance": {
                "target_mean_mm": 10.0,
                "pass": max(row["mean_mm"] for row in rows) <= 10.0,
            },
            "provenance": (
                "synthetic; detector trained on Isaac renders; disturbances are "
                "engineering ranges from docs/ECG_PIPELINE.md 5.7; "
                "prior-anchored snap; "
                + ("overhead + lateral views pooled" if MULTIVIEW else "overhead only")
            ),
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"M4 eval: worst mean={report['worst_mean_mm']:.2f} mm "
            f"(baseline {baseline:.2f}), report {REPORT_PATH}",
            flush=True,
        )
    except BaseException:
        import traceback

        with open("/tmp/roboecg_m4_eval_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
