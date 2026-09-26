"""M3b learned chest-landmark detector: overhead depth -> 7 landmarks (world).

One code path for training, the M4 placement demo and the disturbance
evaluation:

    depth (normalised)  ->  heatmap U-Net  ->  soft-argmax (sub-pixel)
      ->  per-pixel 3D lifting  ->  per-landmark skin/joint offset calibration
      ->  rotation calibration  ->  ChestFrame

The calibration (per-landmark offsets in the raw frame + a rotation matrix) is
estimated on the M3b TRAINING split only (`assets/models/m3b_calibration.json`,
written by `scripts/m3b_ecg_detector_eval.py`) and is applied to every
prediction here, so evaluation and deployment share the same numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from roboecg.coordinate_transform.camera import deproject_pixel
from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.target_localization.chest_frame import ChestFrame, build_chest_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "chest_landmark_heatmap.pt"
CALIBRATION_PATH = PROJECT_ROOT / "assets" / "models" / "m3b_calibration.json"

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 3.0
HEATMAP_STRIDE = 2
LANDMARK_ORDER = (
    "clavicle_left",
    "clavicle_right",
    "shoulder_left",
    "shoulder_right",
    "chest",
    "neck_base",
    "pelvis",
)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ChestLandmarkHeatmapNet(nn.Module):
    def __init__(self, n_points: int):
        super().__init__()
        self.enc1 = conv_block(1, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 96)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.Conv2d(96, 64, 3, padding=1)
        self.dec3 = conv_block(64 + 64, 64)
        self.up2 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = conv_block(32 + 32, 32)
        self.head = nn.Conv2d(32, n_points, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        up3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(up3), e3], dim=1))
        up2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(up2), e2], dim=1))
        return self.head(d2)


def soft_argmax(heatmaps: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
    batch, points, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, points, -1)
    probs = torch.softmax(flat * beta, dim=2)
    coords = torch.arange(height * width, device=heatmaps.device, dtype=torch.float32)
    xs = (coords % width).float()
    ys = torch.div(coords, width, rounding_mode="floor").float()
    x = (probs * xs).sum(dim=2)
    y = (probs * ys).sum(dim=2)
    return torch.stack([x, y], dim=2)


def landmarks_from_points(points: dict, provenance: str = "detector") -> ChestLandmarks:
    """Build the ChestLandmarks bundle from the seven detector points."""
    up = np.asarray(points["neck_base"]) - np.asarray(points["chest"])
    up = up / (np.linalg.norm(up) + 1e-12)
    return ChestLandmarks(
        spine_lower=np.asarray(points["pelvis"]) - up * 0.05,
        spine_mid=np.asarray(points["pelvis"]) + up * 0.10,
        chest=np.asarray(points["chest"]),
        upper_chest=np.asarray(points["chest"]) + up * 0.07,
        neck_base=np.asarray(points["neck_base"]),
        neck_top=np.asarray(points["neck_base"]) + up * 0.06,
        clavicle_left=np.asarray(points["clavicle_left"]),
        clavicle_right=np.asarray(points["clavicle_right"]),
        shoulder_left=np.asarray(points["shoulder_left"]),
        shoulder_right=np.asarray(points["shoulder_right"]),
        pelvis=np.asarray(points["pelvis"]),
        provenance=provenance,
    )


class ChestLandmarkDetector:
    """Heatmap U-Net + soft-argmax + training-split calibration."""

    def __init__(self, model_path: Path | str = MODEL_PATH,
                 calibration_path: Path | str | None = None):
        model_path = Path(model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.order = tuple(checkpoint["landmark_order"])
        self.model = ChestLandmarkHeatmapNet(len(self.order))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.stride = int(checkpoint["heatmap_stride"])
        self.depth_min_m = float(checkpoint.get("depth_min_m", DEPTH_MIN_M))
        self.depth_max_m = float(checkpoint.get("depth_max_m", DEPTH_MAX_M))
        calibration_path = Path(calibration_path) if calibration_path else (
            model_path.parent / "m3b_calibration.json"
        )
        calibration = json.loads(calibration_path.read_text())
        self.rotation_calibration = np.asarray(
            calibration["rotation_matrix"], dtype=float
        )
        self.landmark_offset = np.asarray(
            [calibration["landmark_offset_uvn_m"][name] for name in self.order],
            dtype=float,
        )
        self.calibration = calibration

    # -- raw prediction -----------------------------------------------------
    def predict_pixels(self, depth: np.ndarray) -> dict:
        """Per-landmark sub-pixel pixel coordinates from a depth image."""
        height, width = depth.shape
        normalised = np.clip(depth, self.depth_min_m, self.depth_max_m)
        normalised = (normalised - self.depth_min_m) / (
            self.depth_max_m - self.depth_min_m
        )
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(normalised[None, None, :, :]).float()
            )
            pixels = soft_argmax(output)[0].numpy() * float(self.stride)
        result = {}
        for index, name in enumerate(self.order):
            u, v = pixels[index]
            result[name] = (
                float(np.clip(u, 0.0, width - 1.0)),
                float(np.clip(v, 0.0, height - 1.0)),
            )
        return result

    # -- calibrated 3D landmarks + frame ------------------------------------
    def predict(self, depth: np.ndarray, intrinsics, camera_position,
                cv_rotation) -> dict:
        """Calibrated landmarks and chest frame from one depth image."""
        pixels = self.predict_pixels(depth)
        raw = {}
        for name, (u, v) in pixels.items():
            z = float(depth[int(round(v)), int(round(u))])
            if z <= 0.0:
                z = float(np.median(depth[depth > 0]))
            raw[name] = deproject_pixel(
                u, v, z, intrinsics, camera_position, cv_rotation
            )
        raw_frame = build_chest_frame(
            landmarks_from_points(raw), anterior_hint=(0.0, 0.0, 1.0)
        )
        corrected = {}
        for index, name in enumerate(self.order):
            offset = self.landmark_offset[index]
            corrected[name] = (
                np.asarray(raw[name], dtype=float)
                - offset[0] * raw_frame.up
                - offset[1] * raw_frame.lateral
                - offset[2] * raw_frame.anterior
            )
        frame = build_chest_frame(
            landmarks_from_points(corrected, provenance="detector_calibrated"),
            anterior_hint=(0.0, 0.0, 1.0),
        )
        axes = np.column_stack([frame.up, frame.lateral, frame.anterior])
        axes = axes @ self.rotation_calibration.T
        frame = ChestFrame(
            origin=np.asarray(frame.origin, dtype=float),
            up=axes[:, 0] / np.linalg.norm(axes[:, 0]),
            lateral=axes[:, 1] / np.linalg.norm(axes[:, 1]),
            anterior=axes[:, 2] / np.linalg.norm(axes[:, 2]),
            provenance=dict(frame.provenance, calibrated=True),
        )
        return {
            "pixels": pixels,
            "raw_points": raw,
            "points": corrected,
            "landmarks": landmarks_from_points(corrected, provenance="detector_calibrated"),
            "frame": frame,
        }
