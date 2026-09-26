"""Visual re-detection of placed electrodes (sim counterpart of the real loop).

The real closed loop re-detects the placed electrode from a camera (the
El Ghebouli / Bayer route) instead of trusting the robot pose.  This module
implements the sim counterpart:

    placed electrode markers on the skin (distinct colour)
      -> overhead RGB-D render
      -> colour segmentation + connected components
      -> deprojection to 3D
      -> nearest matching against the planned contacts
      -> per-electrode re-detection error

The marker is a small disc on the skin (the placed dry electrode); detection
uses only the rendered RGB-D, so the measured error contains the render /
deprojection / segmentation error a real system would also have.
"""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.camera import (
    CameraIntrinsics,
    cv_rotation_from_usd,
    deproject_pixel,
)
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.rendering import capture_depth, capture_rgb

MARKER_COLOR = (0.05, 0.95, 0.10)  # bright green, unlike any scene material
# V5 and V6 are ~18 mm apart on the 5th ICS row: the marker must be small
# enough that their blobs stay separate (10 mm disc).
MARKER_RADIUS_M = 0.0035
MARKER_HEIGHT_M = 0.002


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def add_marker_disc(stage, prim_path: str, position, normal, radius=MARKER_RADIUS_M,
                    height=MARKER_HEIGHT_M, color=MARKER_COLOR):
    """A thin disc lying on the skin at `position` (marker of a placed electrode)."""
    from pxr import Gf, UsdGeom

    from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation

    prim = UsdGeom.Cylinder.Define(stage, prim_path)
    prim.CreateRadiusAttr(float(radius))
    prim.CreateHeightAttr(float(height))
    prim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    # cylinder axis (+Z local) aligned with the surface normal
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, normal)
    angle = float(np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0)))
    rotation = (
        np.eye(3) if np.linalg.norm(axis) < 1e-9 else _rodrigues(axis, angle)
    )
    quat = quat_wxyz_from_rotation(rotation)
    prim.AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in (np.asarray(position) + normal * height * 0.5)])
    )
    prim.AddOrientOp().Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))
    )
    return prim


def detect_marker_pixels(rgb: np.ndarray, color=MARKER_COLOR, tol: float = 0.18,
                         dominance: float = 0.05, min_brightness: float = 0.3):
    """Boolean mask of marker pixels.

    The renderer's tonemapping maps the pure display colour to a pale tint
    (measured: the green marker renders as ~(0.76, 0.83, 0.72)), so an absolute
    colour distance fails.  Detect by channel dominance of the marker colour
    instead, with a brightness floor to reject dark noise.
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.size and rgb.max() > 1.5:
        rgb = rgb / 255.0
    target = np.asarray(color, dtype=float)
    channel = int(np.argmax(target))
    others = [index for index in range(3) if index != channel]
    mask = np.ones(rgb.shape[:2], dtype=bool)
    for other in others:
        mask &= rgb[:, :, channel] > rgb[:, :, other] + dominance
    mask &= rgb[:, :, channel] > min_brightness
    # also accept pixels close to the nominal colour (untonemapped renders)
    distance = np.linalg.norm(rgb - target, axis=2)
    return mask | (distance < tol)


def connected_components(mask: np.ndarray, min_pixels: int = 4,
                         max_pixels: int = 60):
    """Simple 4-neighbour component labelling (no scipy dependency)."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components = []
    current = 0
    for y0 in range(height):
        for x0 in range(width):
            if not mask[y0, x0] or labels[y0, x0]:
                continue
            current += 1
            stack = [(y0, x0)]
            labels[y0, x0] = current
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
            # size gate: the markers are ~10 mm discs (a handful of pixels);
            # larger blobs are lighting/colour false positives
            if min_pixels <= len(pixels) <= max_pixels:
                components.append(np.asarray(pixels, dtype=float))
    return components


def detect_markers(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_position,
    cv_rotation,
    color=MARKER_COLOR,
    tol: float = 0.18,
    min_pixels: int = 4,
    max_pixels: int = 60,
):
    """Detected marker world positions (one per connected component)."""
    mask = detect_marker_pixels(rgb, color=color, tol=tol)
    detections = []
    for pixels in connected_components(
        mask, min_pixels=min_pixels, max_pixels=max_pixels
    ):
        ys, xs = pixels[:, 0], pixels[:, 1]
        u = float(xs.mean())
        v = float(ys.mean())
        depths = depth[ys.astype(int), xs.astype(int)]
        valid = depths[depths > 0.0]
        z = float(np.median(valid)) if valid.size else 0.0
        if z <= 0.0:
            continue
        point = deproject_pixel(u, v, z, intrinsics, camera_position, cv_rotation)
        detections.append(
            {
                "pixel": [u, v],
                "depth_m": z,
                "position": np.asarray(point, dtype=float),
                "pixels": int(len(pixels)),
            }
        )
    return detections


def verify_placed_electrodes(
    stage,
    planned_contacts: dict,
    camera_path: str = "/World/Cameras/PerceptionRGBD",
    width: int = 320,
    height: int = 180,
    fov_deg: float = 90.0,
) -> dict:
    """Place markers at the planned contacts, render, detect, and score.

    Returns per-electrode detection errors (mm) and the raw detections.
    """
    for name, entry in planned_contacts.items():
        add_marker_disc(
            stage,
            f"/World/PlacedElectrodes/{name}",
            np.asarray(entry["contact_world"], dtype=float),
            np.asarray(entry["normal_world"], dtype=float),
        )
    camera_matrix = ecg_scene.world_matrix(stage, camera_path)
    camera_position = camera_matrix[:3, 3]
    cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, fov_deg)
    rgb = capture_rgb(camera_path, width, height)
    depth = capture_depth(camera_path, width, height)
    detections = detect_markers(
        rgb, depth, intrinsics, camera_position, cv_rotation
    )

    # Greedy unique assignment: a target and a detection are used at most once,
    # so two merged markers cannot both claim the same blob.
    pairs = []
    for name, entry in planned_contacts.items():
        planned = np.asarray(entry["contact_world"], dtype=float)
        for index, detection in enumerate(detections):
            pairs.append(
                (float(np.linalg.norm(detection["position"] - planned)), name, index)
            )
    pairs.sort(key=lambda item: item[0])
    assigned = {}
    used_names, used_detections = set(), set()
    for distance, name, index in pairs:
        if name in used_names or index in used_detections:
            continue
        used_names.add(name)
        used_detections.add(index)
        assigned[name] = (distance, index)

    rows = []
    for name, entry in planned_contacts.items():
        planned = np.asarray(entry["contact_world"], dtype=float)
        if name in assigned:
            error, index = assigned[name]
            detected = detections[index]["position"]
            pixels = detections[index]["pixels"]
        else:
            error, detected, pixels = None, None, 0
        rows.append(
            {
                "target": name,
                "planned_contact": planned.tolist(),
                "detected_contact": None if detected is None else detected.tolist(),
                "error_mm": None if error is None else error * 1000.0,
                "marker_pixels": pixels,
            }
        )
    errors = [row["error_mm"] for row in rows if row["error_mm"] is not None]
    return {
        "detections": [
            {
                "pixel": d["pixel"],
                "depth_m": d["depth_m"],
                "position": d["position"].tolist(),
                "pixels": d["pixels"],
            }
            for d in detections
        ],
        "per_target": rows,
        "detected_count": len(detections),
        "targets": len(planned_contacts),
        "error_mm": {
            "mean": float(np.mean(errors)) if errors else None,
            "max": float(np.max(errors)) if errors else None,
        },
    }
