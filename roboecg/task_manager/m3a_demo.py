"""M3a demo: perceptual chest frame from MediaPipe keypoints.

Pipeline:
    overhead RGB render -> MediaPipe Pose (external venv)
      -> model-based 3D lifting with the depth anchor
      -> perceptual chest frame (shoulders/hips + one-time calibration)
      -> V1-V6 rules on the depth point cloud
      -> M2 depth fusion (local prior fitted from the depth cloud)
      -> comparison against the ground-truth (rig landmark) targets

The perceptual path uses no rig joints and no asset mesh: the surface comes
from the RGB-D render, as on the real robot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboecg.coordinate_transform.camera import CameraIntrinsics
from roboecg.perception.body_tracking import run_pose_estimator
from roboecg.perception.chest_frame_from_pose import (
    build_perceptual_chest,
    fix_left_right,
    frame_difference,
    measure_shoulder_to_clavicle_offset,
)
from roboecg.perception.chest_landmarks import read_chest_landmarks
from roboecg.perception.depth import (
    depth_to_world_points,
    sample_depth_closest,
)
from roboecg.perception.isaac_skeleton import read_joint_world_positions
from roboecg.perception.pose2d import lift_keypoints, lift_world_landmarks
from roboecg.perception.torso_prior import fit_chest_surface_prior
from roboecg.target_localization.chest_frame import build_chest_frame
from roboecg.target_localization.ecg import generate_v1_v6
from roboecg.target_localization.ecg_rules import load_ecg_rules
from roboecg.target_localization.fusion import fuse_target
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.rendering import capture_depth, capture_rgb, save_png

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "m3a"
CAMERA_FOV_DEG = 90.0


def camera_pose(stage, prim_path):
    from roboecg.coordinate_transform.camera import cv_rotation_from_usd
    from roboecg.task_manager.ecg_scene import world_matrix

    matrix = world_matrix(stage, prim_path)
    return matrix[:3, 3], cv_rotation_from_usd(matrix[:3, :3])


def target_errors(targets, reference_targets) -> dict:
    by_name = {t.name: t for t in reference_targets}
    rows = []
    for target in targets:
        reference = by_name[target.name]
        rows.append(
            {
                "name": target.name,
                "position_error_m": float(
                    np.linalg.norm(
                        np.asarray(target.position) - np.asarray(reference.position)
                    )
                ),
                "normal_angle_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(
                                    np.asarray(target.normal) / np.linalg.norm(target.normal),
                                    np.asarray(reference.normal)
                                    / np.linalg.norm(reference.normal),
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
            }
        )
    errors = [row["position_error_m"] for row in rows]
    return {
        "per_target": rows,
        "position_error_m": {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
        },
    }


def run_m3a(app, gui: bool = False, width: int = 1280, height: int = 720) -> dict:
    from isaacsim.core.api import World

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0)
    stage, scene_report = ecg_scene.build_scene(world)
    print("M3a: scene built", flush=True)

    joint_positions = read_joint_world_positions(stage)
    gt_landmarks = read_chest_landmarks(joint_positions)
    gt_frame = build_chest_frame(gt_landmarks, anterior_hint=(0.0, 0.0, 1.0))
    rules = load_ecg_rules()

    mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
    gt_nominal = generate_v1_v6(gt_landmarks, gt_frame, mesh_points, rules)

    calibration = measure_shoulder_to_clavicle_offset(gt_landmarks, gt_frame)
    print(
        f"M3a: calibration du={calibration['du_m']:.4f} dv={calibration['dv_m']:.4f} "
        f"dn={calibration['dn_m']:.4f} m",
        flush=True,
    )

    camera_position, cv_rotation = camera_pose(stage, "/World/Cameras/PerceptionRGBD")
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, CAMERA_FOV_DEG)
    rgb = capture_rgb("/World/Cameras/PerceptionRGBD", width, height)
    save_png(RUNS_DIR / "pose_input.png", rgb)
    depth = capture_depth("/World/Cameras/PerceptionRGBD", width, height)
    depth_points = depth_to_world_points(depth, intrinsics, camera_position, cv_rotation)
    print(f"M3a: depth points={depth_points.shape[0]}", flush=True)

    pose_report = run_pose_estimator(
        RUNS_DIR / "pose_input.png",
        RUNS_DIR / "pose_keypoints.json",
        overlay_path=RUNS_DIR / "pose_overlay.png",
    )
    visibility = {
        name: keypoint["visibility"]
        for name, keypoint in pose_report["keypoints"].items()
    }
    # On the overhead supine view MediaPipe's 3D world landmarks are
    # unreliable (measured: hip width 47% narrow, shoulder-hip height error
    # 8.5 cm), but the required keypoints (shoulders/hips/nose) are unoccluded,
    # so per-pixel depth lifting is the appropriate 3D source here.
    world_lifted, anchor = lift_world_landmarks(
        pose_report.get("world_keypoints", {}),
        pose_report["keypoints"],
        depth,
        intrinsics,
        camera_position,
        cv_rotation,
    )
    lifted = lift_keypoints(
        pose_report["keypoints"],
        depth,
        intrinsics,
        camera_position,
        cv_rotation,
    )
    # The shoulder keypoints can sit a few pixels off the silhouette, where a
    # median depth picks the table; the closest valid depth is the body.
    from roboecg.coordinate_transform.camera import deproject_pixel

    for name, keypoint in pose_report["keypoints"].items():
        closest = sample_depth_closest(
            depth, keypoint["u"], keypoint["v"], window=6
        )
        if closest is not None:
            lifted[name] = deproject_pixel(
                keypoint["u"],
                keypoint["v"],
                closest,
                intrinsics,
                camera_position,
                cv_rotation,
            )
    print(
        f"M3a: per-pixel lifted={len(lifted)} (world-model lifted={len(world_lifted)}, "
        f"anchor={anchor})",
        flush=True,
    )

    # M3a mitigation: MediaPipe swaps left/right on the overhead supine view
    # (measured lateral-axis error ~160 deg); the bed setup knows the patient's
    # left direction (world +Y here).
    lifted, swapped = fix_left_right(lifted, patient_left_axis=(0.0, 1.0, 0.0))
    print(f"M3a: left/right swapped by MediaPipe = {swapped}", flush=True)
    if swapped:
        visibility = dict(visibility)
    perceptual = build_perceptual_chest(
        lifted,
        visibility,
        calibration,
        anterior_hint=(0.0, 0.0, 1.0),
    )
    frame_diff = frame_difference(perceptual.frame, gt_frame)
    print(
        f"M3a: frame origin error={frame_diff['origin_distance_m'] * 1000:.1f} mm "
        f"up={frame_diff['up_angle_deg']:.1f}deg "
        f"lateral={frame_diff['lateral_angle_deg']:.1f}deg",
        flush=True,
    )

    # perceptual rules on the depth cloud (no mesh, no rig joints)
    try:
        perceptual_nominal = generate_v1_v6(
            perceptual.landmarks, perceptual.frame, depth_points, rules
        )
        nominal_errors = target_errors(
            perceptual_nominal.targets, gt_nominal.targets
        )
        target_generation_error = None
        print(
            f"M3a: perceptual nominal error mean="
            f"{nominal_errors['position_error_m']['mean'] * 1000:.1f} mm",
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - reported, not hidden
        perceptual_nominal = None
        nominal_errors = None
        target_generation_error = repr(error)
        print(f"M3a: target generation failed: {error}", flush=True)

    # M2 depth fusion with a prior fitted from the depth cloud (real path)
    fused = []
    fused_errors = None
    prior = None
    if perceptual_nominal is not None:
        prior = fit_chest_surface_prior(depth_points, perceptual.frame)
        settings = rules["depth_fusion"]
        fused = [
            fuse_target(
                target,
                perceptual.frame,
                prior,
                depth,
                intrinsics,
                camera_position,
                cv_rotation,
                settings,
            )
            for target in perceptual_nominal.targets
        ]
        fused_errors = target_errors(fused, gt_nominal.targets)
        print(
            f"M3a: fused error mean="
            f"{fused_errors['position_error_m']['mean'] * 1000:.1f} mm "
            f"max={fused_errors['position_error_m']['max'] * 1000:.1f} mm",
            flush=True,
        )

    from roboecg.task_manager.ecg_scene import add_frame_axes, add_line, add_marker

    add_frame_axes(stage, "/World/Markers/GTFrame", gt_frame)
    add_frame_axes(stage, "/World/Markers/PoseFrame", perceptual.frame)
    for target in gt_nominal.targets:
        add_marker(
            stage, f"/World/Markers/GT_{target.name}", target.position, 0.008, (0.25, 0.45, 0.95)
        )
    for fused_target in fused:
        color = (0.15, 0.85, 0.35) if fused_target.source == "depth" else (0.95, 0.35, 0.15)
        add_marker(
            stage,
            f"/World/Markers/M3a_{fused_target.name}",
            fused_target.position,
            0.011,
            color,
        )
        add_line(
            stage,
            f"/World/Markers/M3a_{fused_target.name}_normal",
            fused_target.position,
            np.asarray(fused_target.position) + np.asarray(fused_target.normal) * 0.04,
            color=color,
            width=0.003,
        )
    save_png(RUNS_DIR / "fusion_closeup.png", capture_rgb("/World/Cameras/ChestCloseup", width, height))

    report = {
        "status": (
            "PASS"
            if fused_errors is not None
            and fused_errors["position_error_m"]["mean"] <= 0.02
            and fused_errors["position_error_m"]["max"] <= 0.03
            else "PARTIAL"
        ),
        "target_generation_error": target_generation_error,
        "scene": scene_report,
        "calibration": calibration,
        "pose": {
            "keypoints": {
                name: {
                    "visibility": keypoint.get("visibility"),
                    "u": keypoint.get("u"),
                    "v": keypoint.get("v"),
                }
                for name, keypoint in pose_report["keypoints"].items()
            },
            "lifted_count": len(lifted),
            "anchor": anchor,
            "left_right_swapped_by_mediapipe": bool(swapped),
        },
        "frame_difference": frame_diff,
        "gt_frame": gt_frame.describe(),
        "perceptual_frame": perceptual.frame.describe(),
        "perceptual_info": perceptual.info,
        "prior": None if prior is None else prior.describe(),
        "nominal_errors": nominal_errors,
        "fused_errors": fused_errors,
        "targets": {
            target.name: {
                "position_world": np.asarray(target.position).tolist(),
                "source": target.source,
                "info": target.info,
            }
            for target in fused
        },
        "screenshots": {
            "pose_input": str(RUNS_DIR / "pose_input.png"),
            "pose_overlay": str(RUNS_DIR / "pose_overlay.png"),
            "closeup": str(RUNS_DIR / "fusion_closeup.png"),
        },
    }
    (RUNS_DIR / "m3a_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report
