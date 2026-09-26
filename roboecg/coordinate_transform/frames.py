# Copied from farus_thyroid_isaac/farus/coordinate_transform/frames.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Frame and orientation helpers (pure numpy, no Isaac dependency)."""
from __future__ import annotations

import numpy as np


def normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def rotation_from_axes(x_axis, z_axis) -> np.ndarray:
    """Rotation matrix with columns (x, y, z); x is re-orthogonalized to z."""
    z = normalize(z_axis)
    x = np.asarray(x_axis, dtype=float)
    x = x - np.dot(x, z) * z
    x = normalize(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def look_at_rotation(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera rotation looking from `position` at `target` with world up kept."""
    forward = normalize(np.asarray(target, dtype=float) - np.asarray(position, dtype=float))
    world_up = normalize(up)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right = normalize(right)
    true_up = np.cross(right, forward)
    # USD cameras look along -Z with +Y up.
    return np.column_stack([right, true_up, -forward])


def quat_wxyz_from_rotation(rotation) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return normalize(quat)


def transform_point(matrix_4x4, point) -> np.ndarray:
    matrix = np.asarray(matrix_4x4, dtype=float)
    point = np.asarray(point, dtype=float)
    return matrix[:3, :3] @ point + matrix[:3, 3]


def point_segment_distance(point, start, end) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def angle_between_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def rotation_from_quat_wxyz(quat) -> np.ndarray:
    """3x3 rotation matrix from a (w, x, y, z) quaternion (local addition)."""
    q = np.asarray(quat, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp_wxyz(q0, q1, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (w, x, y, z) quaternions."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return normalize(a + (b - a) * float(t))
    theta = float(np.arccos(dot))
    return normalize(
        (np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)
    )
