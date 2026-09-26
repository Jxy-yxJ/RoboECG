# Copied from farus_thyroid_isaac/farus/perception/depth.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Depth image utilities (pure numpy, no Isaac dependency)."""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.camera import deproject_pixel


def sample_depth(depth, u, v, window=2):
    """Median of finite positive depth values around pixel (u, v)."""
    height, width = depth.shape
    u0 = max(0, int(round(u)) - window)
    u1 = min(width, int(round(u)) + window + 1)
    v0 = max(0, int(round(v)) - window)
    v1 = min(height, int(round(v)) + window + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject_window(depth, u, v, window, intrinsics, camera_position, cv_rotation):
    """Deproject valid pixels around (u, v) into world points.

    Returns (points_world (N, 3), pixel_coords (us, vs, zs)).
    """
    height, width = depth.shape
    u0 = max(0, int(np.floor(u)) - window)
    u1 = min(width, int(np.ceil(u)) + window + 1)
    v0 = max(0, int(np.floor(v)) - window)
    v1 = min(height, int(np.ceil(v)) + window + 1)

    patch = depth[v0:v1, u0:u1]
    vv, uu = np.mgrid[v0:v1, u0:u1]
    valid = np.isfinite(patch) & (patch > 0)
    zs = patch[valid].astype(float)
    us = uu[valid].astype(float)
    vs = vv[valid].astype(float)

    xs = (us - intrinsics.cx) / intrinsics.fx * zs
    ys = (vs - intrinsics.cy) / intrinsics.fy * zs
    points_camera = np.stack([xs, ys, zs], axis=1)
    points_world = points_camera @ np.asarray(cv_rotation, dtype=float).T + np.asarray(
        camera_position, dtype=float
    )
    return points_world, (us, vs, zs)


def estimate_surface_normal(points, query, radius=None, orient_toward=None):
    """PCA surface normal of the neighborhood around `query`.

    Returns (normal, inlier_count) or (None, 0) when there are too few points.
    """
    points = np.asarray(points, dtype=float)
    if radius is not None:
        mask = np.linalg.norm(points - np.asarray(query, dtype=float), axis=1) <= radius
        points = points[mask]
    if points.shape[0] < 3:
        return None, int(points.shape[0])

    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]

    if orient_toward is not None:
        direction = np.asarray(orient_toward, dtype=float) - np.asarray(query, dtype=float)
        if np.dot(normal, direction) < 0:
            normal = -normal
    return normal / np.linalg.norm(normal), int(points.shape[0])


def surface_point_from_depth(
    depth, u, v, window, intrinsics, camera_position, cv_rotation, max_depth=3.0
):
    """Depth-based skin point at pixel (u, v) using the median window depth."""
    z = sample_depth(depth, u, v, window)
    if z is None or z > max_depth:
        return None, None
    return deproject_pixel(u, v, z, intrinsics, camera_position, cv_rotation), z


def depth_to_world_points(
    depth,
    intrinsics,
    camera_position,
    cv_rotation,
    max_depth=3.0,
    stride=2,
):
    """Deproject the whole depth image into world points (person range only)."""
    height, width = depth.shape
    vs, us = np.mgrid[0:height:stride, 0:width:stride]
    zs = depth[0:height:stride, 0:width:stride]
    valid = np.isfinite(zs) & (zs > 0) & (zs <= max_depth)
    us = us[valid].astype(float)
    vs = vs[valid].astype(float)
    zs = zs[valid].astype(float)
    xs = (us - intrinsics.cx) / intrinsics.fx * zs
    ys = (vs - intrinsics.cy) / intrinsics.fy * zs
    points_camera = np.stack([xs, ys, zs], axis=1)
    return (
        points_camera @ np.asarray(cv_rotation, dtype=float).T
        + np.asarray(camera_position, dtype=float)
    )

def sample_depth_closest(depth, u, v, window=5, max_depth=3.0):
    """Closest valid depth in a pixel window (local addition).

    With an overhead camera the body is the nearest surface, so the minimum
    valid depth in the window is robust to keypoints that sit slightly off the
    body silhouette (where a median would pick the table/background).
    """
    import numpy as np

    height, width = depth.shape
    u0 = max(0, int(round(u)) - window)
    u1 = min(width - 1, int(round(u)) + window)
    v0 = max(0, int(round(v)) - window)
    v1 = min(height - 1, int(round(v)) + window)
    patch = depth[v0 : v1 + 1, u0 : u1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 0) & (patch <= max_depth)]
    if valid.size == 0:
        return None
    return float(valid.min())
