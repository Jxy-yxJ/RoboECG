"""Fuse the mesh targets with RGB-D depth measurements (M2).

Design (from the M0 findings):
  * position: the nominal target is projected into the depth image; the
    deprojected skin point replaces it (radial correction along the camera
    ray), but only when the hit is valid and within a bounded distance of the
    nominal target;
  * normal: the PCA normal of the local depth patch is accepted only when it
    agrees with the *position-dependent* chest surface prior
    (`torso_prior.ChestSurfacePrior`), replacing the thyroid's global-anterior
    25 deg rule which does not hold on the chest (measured deviation up to
    62 deg);
  * any rejected measurement falls back to the nominal mesh target / prior, so
    a bad depth hit can never drag the electrode into the body.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roboecg.coordinate_transform.camera import project_world_to_pixel
from roboecg.coordinate_transform.frames import angle_between_deg
from roboecg.perception.depth import (
    deproject_window,
    estimate_surface_normal,
    surface_point_from_depth,
)
from roboecg.perception.torso_prior import ChestSurfacePrior
from roboecg.target_localization.chest_frame import ChestFrame


@dataclass(frozen=True)
class FusedTarget:
    name: str
    position: np.ndarray
    normal: np.ndarray
    source: str  # "depth" | "mesh_fallback"
    info: dict = field(default_factory=dict)


def fuse_target(
    target,
    frame: ChestFrame,
    prior: ChestSurfacePrior,
    depth,
    intrinsics,
    camera_position,
    cv_rotation,
    settings: dict,
) -> FusedTarget:
    """Fuse one mesh target with the depth image."""
    info = {
        "pixel": None,
        "depth_m": None,
        "skin_point_distance_m": None,
        "normal_angle_vs_prior_deg": None,
        "normal_inliers": 0,
        "skin_point_rejected": False,
        "normal_rejected": False,
        "normal_accepted": False,
        "measured_normal_world": None,
    }
    u_target, v_target = float(target.frame_coords[0]), float(target.frame_coords[1])
    prior_normal = prior.normal_world(frame, u_target, v_target)
    prior_point = frame.from_frame(u_target, v_target, prior.height(u_target, v_target))

    position = np.asarray(target.position, dtype=float)
    normal = np.asarray(target.normal, dtype=float)
    source = "mesh_fallback"

    pixel, _ = project_world_to_pixel(
        position, camera_position, cv_rotation, intrinsics
    )
    if pixel is None:
        info["reason"] = "projection_failed"
        return FusedTarget(target.name, position, normal, source, info)
    info["pixel"] = [float(pixel[0]), float(pixel[1])]

    # Grazing incidence: on the lateral chest wall (V6 midaxillary) the camera
    # ray meets the surface at ~70 deg, so a small depth error displaces the
    # skin point by centimetres and the local PCA normal is unreliable.  The
    # depth measurement adds nothing there; keep the model target instead.
    to_camera = np.asarray(camera_position, dtype=float) - position
    to_camera = to_camera / (np.linalg.norm(to_camera) + 1e-12)
    incidence_deg = float(
        np.degrees(
            np.arccos(np.clip(float(np.dot(normal, to_camera)), -1.0, 1.0))
        )
    )
    info["incidence_deg"] = incidence_deg
    grazing = incidence_deg > float(settings.get("max_incidence_deg", 65.0))
    if grazing:
        # At grazing incidence the depth measurement adds nothing: keep the
        # model target supplied by the caller.  A deployment pipeline that
        # builds its model targets from the depth cloud should substitute the
        # smooth prior normal for such targets beforehand
        # (see task_manager.perception_pipeline.model_normals_for_grazing).
        info["reason"] = "grazing_incidence"
        info["measured_normal_world"] = None
        return FusedTarget(target.name, position, normal, source, info)

    skin_point, depth_m = surface_point_from_depth(
        depth,
        pixel[0],
        pixel[1],
        int(settings["depth_median_window"]),
        intrinsics,
        camera_position,
        cv_rotation,
        max_depth=float(settings["max_depth_m"]),
    )
    info["depth_m"] = None if depth_m is None else float(depth_m)

    if skin_point is not None:
        distance = float(np.linalg.norm(skin_point - position))
        info["skin_point_distance_m"] = distance
        if distance <= float(settings["max_skin_offset_m"]):
            position = skin_point
            source = "depth"
        else:
            info["skin_point_rejected"] = True

    points, _ = deproject_window(
        depth,
        pixel[0],
        pixel[1],
        int(settings["normal_window"]),
        intrinsics,
        camera_position,
        cv_rotation,
    )
    measured_normal, inliers = estimate_surface_normal(
        points,
        position,
        radius=float(settings["normal_radius_m"]),
        orient_toward=camera_position,
    )
    info["normal_inliers"] = int(inliers)
    if measured_normal is not None:
        measured_normal = np.asarray(measured_normal, dtype=float)
        info["measured_normal_world"] = measured_normal.tolist()
        angle = angle_between_deg(measured_normal, prior_normal)
        info["normal_angle_vs_prior_deg"] = float(angle)
        if angle <= float(settings["normal_max_angle_deg"]):
            normal = measured_normal
            info["normal_accepted"] = True
        else:
            info["normal_rejected"] = True

    info["prior_normal_world"] = prior_normal.tolist()
    info["prior_point_world"] = prior_point.tolist()
    return FusedTarget(target.name, position, normal, source, info)


def best_view_index(normal, position, camera_positions) -> int:
    """Index of the camera with the smallest incidence angle on the surface.

    Used by the multi-view perception path: each electrode is routed to the
    view that sees its patch most frontally (the lateral wall V5/V6 is grazing
    in the overhead view but frontal in the side view).
    """
    normal = np.asarray(normal, dtype=float)
    position = np.asarray(position, dtype=float)
    best_index = 0
    best_incidence = None
    for index, camera_position in enumerate(camera_positions):
        ray = np.asarray(camera_position, dtype=float) - position
        ray = ray / (np.linalg.norm(ray) + 1e-12)
        incidence = float(
            np.degrees(
                np.arccos(np.clip(float(np.dot(normal, ray)), -1.0, 1.0))
            )
        )
        if best_incidence is None or incidence < best_incidence:
            best_incidence = incidence
            best_index = index
    return best_index


def evaluate_fusion(fused_targets, mesh_targets, frame: ChestFrame) -> dict:
    """Position/normal errors of the fused targets against the mesh targets."""
    by_name = {t.name: t for t in mesh_targets}
    rows = []
    for fused in fused_targets:
        reference = by_name[fused.name]
        measured = fused.info.get("measured_normal_world")
        rows.append(
            {
                "name": fused.name,
                "source": fused.source,
                "position_error_m": float(
                    np.linalg.norm(fused.position - reference.position)
                ),
                "fused_normal_angle_vs_mesh_deg": angle_between_deg(
                    fused.normal, reference.normal
                ),
                "measured_normal_angle_vs_mesh_deg": (
                    None
                    if measured is None
                    else angle_between_deg(np.asarray(measured), reference.normal)
                ),
                "normal_angle_vs_prior_deg": fused.info.get(
                    "normal_angle_vs_prior_deg"
                ),
                "normal_accepted": bool(fused.info.get("normal_accepted")),
                "depth_hit": fused.source == "depth",
            }
        )
    position_errors = [row["position_error_m"] for row in rows]
    measured_angles = [
        row["measured_normal_angle_vs_mesh_deg"]
        for row in rows
        if row["measured_normal_angle_vs_mesh_deg"] is not None
    ]
    return {
        "per_target": rows,
        "position_error_m": {
            "mean": float(np.mean(position_errors)),
            "max": float(np.max(position_errors)),
        },
        "fused_normal_angle_vs_mesh_deg": {
            "mean": float(np.mean([r["fused_normal_angle_vs_mesh_deg"] for r in rows])),
            "max": float(np.max([r["fused_normal_angle_vs_mesh_deg"] for r in rows])),
        },
        "measured_normal_angle_vs_mesh_deg": {
            "mean": float(np.mean(measured_angles)) if measured_angles else None,
            "max": float(np.max(measured_angles)) if measured_angles else None,
        },
        "normal_acceptance_rate": float(
            np.mean([1.0 if row["normal_accepted"] else 0.0 for row in rows])
        ),
        "depth_hit_rate": float(
            np.mean([1.0 if row["depth_hit"] else 0.0 for row in rows])
        ),
        # Every target must yield a usable pose: either a depth measurement or
        # a justified model fallback (grazing incidence / no valid depth).
        "valid_rate": float(
            np.mean(
                [
                    1.0
                    if row["source"] in ("depth", "mesh_fallback")
                    else 0.0
                    for row in rows
                ]
            )
        ),
        "fallback_targets": [
            row["name"] for row in rows if row["source"] != "depth"
        ],
    }
