"""Shared helpers for the electrode-position experiments (P0).

Loads the M3b synthetic dataset and computes the ground-truth electrode
positions with the clinical-rule path, so both the rule-based and the direct
regression experiments use exactly the same labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.coordinate_transform.camera import (  # noqa: E402
    CameraIntrinsics,
    project_world_to_pixel,
)
from roboecg.perception.chest_landmarks import ChestLandmarks  # noqa: E402
from roboecg.perception.depth import depth_to_world_points  # noqa: E402
from roboecg.target_localization.chest_frame import build_chest_frame  # noqa: E402
from roboecg.target_localization.ecg import generate_v1_v6  # noqa: E402
from roboecg.target_localization.ecg_rules import load_ecg_rules  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset"
ELECTRODE_ORDER = ("V1", "V2", "V3", "V4", "V5", "V6")


def load_manifest() -> dict:
    return json.loads((DATASET_DIR / "manifest.json").read_text())


def load_sample(index: int) -> dict:
    return dict(np.load(DATASET_DIR / f"sample_{index:04d}.npz"))


def landmarks_from_points(points: dict) -> ChestLandmarks:
    """Build ChestLandmarks from the seven detected points."""
    clavicle_left = np.asarray(points["clavicle_left"], dtype=float)
    chest = np.asarray(points["chest"], dtype=float)
    neck_base = np.asarray(points["neck_base"], dtype=float)
    up = neck_base - chest
    up = up / (np.linalg.norm(up) + 1e-12)
    return ChestLandmarks(
        spine_lower=np.asarray(points["pelvis"], dtype=float) - up * 0.05,
        spine_mid=np.asarray(points["pelvis"], dtype=float) + up * 0.10,
        chest=chest,
        upper_chest=chest + up * 0.07,
        neck_base=neck_base,
        neck_top=neck_base + up * 0.06,
        clavicle_left=clavicle_left,
        clavicle_right=np.asarray(points["clavicle_right"], dtype=float),
        shoulder_left=np.asarray(points["shoulder_left"], dtype=float),
        shoulder_right=np.asarray(points["shoulder_right"], dtype=float),
        pelvis=np.asarray(points["pelvis"], dtype=float),
        provenance="simulation GT: rig joint landmarks",
    )


def gt_electrodes(sample: dict, rules: dict | None = None):
    """Rule-based ground-truth electrodes (world positions + image pixels)."""
    rules = rules or load_ecg_rules()
    landmark_order = load_manifest()["landmark_order"]
    points_world = sample["world_points"]
    landmarks = landmarks_from_points(
        {name: points_world[i] for i, name in enumerate(landmark_order)}
    )
    frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))

    depth_m = sample["depth"].astype(np.float32) / 1000.0
    width = sample["depth"].shape[1]
    height = sample["depth"].shape[0]
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, 90.0)
    camera_position = sample["camera_position"]
    cv_rotation = sample["cv_rotation"]
    points = depth_to_world_points(
        depth_m, intrinsics, camera_position, cv_rotation, stride=1
    )

    result = generate_v1_v6(landmarks, frame, points, rules)
    by_name = {target.name: target for target in result.targets}
    world = np.stack([np.asarray(by_name[name].position) for name in ELECTRODE_ORDER])
    normals = np.stack([np.asarray(by_name[name].normal) for name in ELECTRODE_ORDER])

    pixels = []
    for name in ELECTRODE_ORDER:
        pixel, _ = project_world_to_pixel(
            np.asarray(by_name[name].position),
            camera_position,
            cv_rotation,
            intrinsics,
        )
        if pixel is None:
            raise RuntimeError(f"{name} not visible in sample")
        pixels.append(pixel)
    return {
        "world": world,
        "normals": normals,
        "pixels": np.asarray(pixels, dtype=np.float32),
        "intrinsics": intrinsics,
        "camera_position": camera_position,
        "cv_rotation": cv_rotation,
        "frame": frame,
    }
