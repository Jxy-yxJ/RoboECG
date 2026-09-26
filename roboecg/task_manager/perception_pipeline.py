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
from roboecg.target_localization.fusion import fuse_target
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.rendering import capture_depth

DEFAULT_CAMERA_PATH = "/World/Cameras/PerceptionRGBD"


def model_normals_for_grazing(targets, frame, prior, camera_position,
                              max_incidence_deg: float = 65.0):
    """Substitute the smooth model normal where the depth surface is unusable.

    On the lateral chest wall (V5/V6) the overhead camera sees the surface at
    a grazing angle: the depth adds nothing and the normal of a cloud-snapped
    target is unreliable.  The fitted prior supplies a usable approach
    direction; the fusion then falls back to these model targets verbatim.
    """
    import dataclasses

    out = []
    for target in targets:
        position = np.asarray(target.position, dtype=float)
        normal = np.asarray(target.normal, dtype=float)
        to_camera = np.asarray(camera_position, dtype=float) - position
        to_camera = to_camera / (np.linalg.norm(to_camera) + 1e-12)
        incidence = float(
            np.degrees(
                np.arccos(np.clip(float(np.dot(normal, to_camera)), -1.0, 1.0))
            )
        )
        if incidence > max_incidence_deg:
            u, v = float(target.frame_coords[0]), float(target.frame_coords[1])
            normal = np.asarray(prior.normal_world(frame, u, v), dtype=float)
        out.append(dataclasses.replace(target, normal=normal))
    return out


def perceive_targets(
    stage,
    detector: ChestLandmarkDetector,
    rules: dict,
    camera_path: str = DEFAULT_CAMERA_PATH,
    width: int = 320,
    height: int = 180,
    fov_deg: float = 90.0,
    surface_stride: int = 1,
) -> dict:
    """Run the perception path once and return fused V1-V6 targets.

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
    generated = generate_v1_v6(landmarks, frame, points, rules)

    prior = fit_chest_surface_prior(points, frame)
    fusion_settings = rules["depth_fusion"]
    model_targets = model_normals_for_grazing(
        generated.targets,
        frame,
        prior,
        camera_position,
        fusion_settings.get("max_incidence_deg", 65.0),
    )
    fused = [
        fuse_target(
            target,
            frame,
            prior,
            depth,
            intrinsics,
            camera_position,
            cv_rotation,
            fusion_settings,
        )
        for target in model_targets
    ]

    return {
        "depth": depth,
        "points": points,
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
