# Copied from farus_thyroid_isaac/farus/coordinate_transform/camera.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Pinhole camera math (pure numpy, no Isaac dependency).

USD cameras look along -Z with +Y up; the functions here use the computer
vision convention (x right, y down, z forward) so depth can be treated as
positive z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_horizontal_fov(
        cls, width: int, height: int, horizontal_fov_deg: float
    ) -> "CameraIntrinsics":
        fx = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fx),
            cx=width / 2.0,
            cy=height / 2.0,
        )


def cv_rotation_from_usd(usd_rotation) -> np.ndarray:
    """Convert a USD camera rotation into the CV camera frame."""
    return np.asarray(usd_rotation, dtype=float) @ np.diag([1.0, -1.0, -1.0])


def world_to_camera(point_world, camera_position, cv_rotation) -> np.ndarray:
    return cv_rotation.T @ (
        np.asarray(point_world, dtype=float) - np.asarray(camera_position, dtype=float)
    )


def camera_to_world(point_camera, camera_position, cv_rotation) -> np.ndarray:
    return (
        np.asarray(cv_rotation, dtype=float) @ np.asarray(point_camera, dtype=float)
        + np.asarray(camera_position, dtype=float)
    )


def project_world_to_pixel(point_world, camera_position, cv_rotation, intrinsics):
    """Return ((u, v), point_camera) or (None, point_camera) if behind."""
    point_camera = world_to_camera(point_world, camera_position, cv_rotation)
    if point_camera[2] <= 1e-6:
        return None, point_camera
    u = intrinsics.fx * point_camera[0] / point_camera[2] + intrinsics.cx
    v = intrinsics.fy * point_camera[1] / point_camera[2] + intrinsics.cy
    return (float(u), float(v)), point_camera


def deproject_pixel(u, v, depth, intrinsics, camera_position, cv_rotation):
    point_camera = np.array(
        [
            (u - intrinsics.cx) / intrinsics.fx * depth,
            (v - intrinsics.cy) / intrinsics.fy * depth,
            depth,
        ]
    )
    return camera_to_world(point_camera, camera_position, cv_rotation)
