"""Full perception path: overhead depth -> detector -> rules -> M2 fusion.

Shared by the M4 placement demo (`--perception`) and the disturbance
evaluation, so the deployed chain and the evaluated chain are the same:

    depth render (PerceptionRGBD)
      -> M3b detector: 7 chest landmarks (calibrated, training-split offsets)
      -> ChestFrame
      -> depth point cloud (stride=1)
      -> V1-V6 rules (SNND + torso-width drop regression + fitted V5 fraction)
      -> M2 fusion: median-depth skin point + PCA normal, local prior gate,
         grazing-incidence fallback
      -> fused contact targets (position + normal) for press planning

Everything the perception layer consumes is measurable on a real robot: one
depth image and the camera extrinsics; the mesh is never touched.
"""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.camera import (
    CameraIntrinsics,
    cv_rotation_from_usd,
)
from roboecg.perception.chest_detector import ChestLandmarkDetector
from roboecg.perception.depth import depth_to_world_points
from roboecg.perception.torso_prior import fit_chest_surface_prior
from roboecg.target_localization.ecg import generate_v1_v6
from roboecg.target_localization.fusion import best_view_index, fuse_target
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.rendering import capture_depth

DEFAULT_CAMERA_PATH = "/World/Cameras/PerceptionRGBD"


def perceive_targets(
    stage,
    detector: ChestLandmarkDetector,
    rules: dict,
    camera_path: str = DEFAULT_CAMERA_PATH,
    width: int = 320,
    height: int = 180,
    fov_deg: float = 90.0,
    surface_stride: int = 1,
    lateral_camera: dict | None = None,
    allow_snap_fallback: bool = False,
    world=None,
) -> dict:
    """Run the perception path once and return fused V1-V6 targets.

    With ``lateral_camera`` (a dict with position/look_at, plus an optional
    prim_path) a second fixed view is rendered, its points are merged into the
    surface cloud, and each electrode is routed to the view that sees its patch
    most frontally (lateral wall V5/V6 use the side view; V1-V4 the overhead
    one).  See docs/ECG_V3_SOLUTION_PLAN.md section 1.4.

    Returns a dict with the depth image, the depth point cloud, the calibrated
    landmarks/frame, the fused targets (list, V1..V6 order) and diagnostics
    (prior fit quality, per-target fusion info, rule inputs).
    """
    camera_matrix = ecg_scene.world_matrix(stage, camera_path)
    camera_position = camera_matrix[:3, 3]
    cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, fov_deg)

    depth = capture_depth(camera_path, width, height)
    prediction = detector.predict(depth, intrinsics, camera_position, cv_rotation)
    landmarks = prediction["landmarks"]
    frame = prediction["frame"]

    points = depth_to_world_points(
        depth, intrinsics, camera_position, cv_rotation, stride=surface_stride
    )

    views = [
        {
            "name": "overhead",
            "depth": depth,
            "intrinsics": intrinsics,
            "camera_position": camera_position,
            "cv_rotation": cv_rotation,
        }
    ]
    lateral_depth = None
    if lateral_camera is not None:
        lateral_path = str(
            lateral_camera.get("prim_path", "/World/Cameras/PerceptionLateral")
        )
        if not stage.GetPrimAtPath(lateral_path).IsValid():
            ecg_scene.add_camera(
                stage,
                lateral_path,
                position=lateral_camera["position"],
                look_at=lateral_camera["look_at"],
            )
            if world is not None:
                for _ in range(3):
                    world.step(render=True)
        lateral_matrix = ecg_scene.world_matrix(stage, lateral_path)
        lateral_position = lateral_matrix[:3, 3]
        lateral_rotation = cv_rotation_from_usd(lateral_matrix[:3, :3])
        lateral_depth = capture_depth(lateral_path, width, height)
        views.append(
            {
                "name": "lateral",
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
                    stride=surface_stride,
                ),
            ],
            axis=0,
        )

    prior = fit_chest_surface_prior(points, frame)
    generated = generate_v1_v6(
        landmarks,
        frame,
        points,
        rules,
        prior=prior,
        allow_snap_fallback=allow_snap_fallback,
    )

    fusion_settings = rules["depth_fusion"]
    camera_positions = [view["camera_position"] for view in views]
    fused = []
    for target in generated.targets:
        if len(views) == 1:
            view = views[0]
        else:
            view = views[
                best_view_index(
                    target.normal, target.position, camera_positions
                )
            ]
        result = fuse_target(
            target,
            frame,
            prior,
            view["depth"],
            view["intrinsics"],
            view["camera_position"],
            view["cv_rotation"],
            fusion_settings,
        )
        result.info["view"] = view["name"]
        fused.append(result)

    return {
        "depth": depth,
        "points": points,
        "views": [view["name"] for view in views],
        "lateral_depth": lateral_depth,
        "intrinsics": intrinsics,
        "camera_position": camera_position,
        "cv_rotation": cv_rotation,
        "landmarks": landmarks,
        "frame": frame,
        "generated": generated,
        "fused": fused,
        "prior": prior,
        "detector_pixels": prediction["pixels"],
        "detector_raw_points": prediction["raw_points"],
        "detector_points": prediction["points"],
    }
