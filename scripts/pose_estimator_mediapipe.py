"""Standalone MediaPipe pose estimator (run with the pose venv).

This script is the simulation stand-in for the Azure Kinect Body Tracking SDK
used by FARUS.  It reads one RGB PNG and writes 2D keypoints as JSON; the
Isaac Sim side lifts them to 3D with the depth image.

Run:
    /home/jxy/.venvs/farus_pose/bin/python scripts/pose_estimator_mediapipe.py \
        --image runs/m3/pose_input.png \
        --model assets/models/pose_landmarker_full.task \
        --out runs/m3/pose_keypoints.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

KEYPOINT_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

SKELETON_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (7, 8),
    (7, 11),
    (8, 12),
    (0, 7),
    (0, 8),
]


def draw_overlay(image: np.ndarray, keypoints: dict) -> Image.Image:
    overlay = Image.fromarray(image.astype("uint8"), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    for start, end in SKELETON_EDGES:
        a = keypoints.get(KEYPOINT_NAMES[start])
        b = keypoints.get(KEYPOINT_NAMES[end])
        if a and b:
            draw.line([(a["u"], a["v"]), (b["u"], b["v"])], fill=(0, 255, 0), width=3)
    for name, point in keypoints.items():
        color = (255, 80, 0) if name in ("LEFT_SHOULDER", "RIGHT_SHOULDER") else (0, 160, 255)
        if name in ("LEFT_EAR", "RIGHT_EAR", "NOSE"):
            color = (255, 0, 255)
        radius = 5
        draw.ellipse(
            [point["u"] - radius, point["v"] - radius, point["u"] + radius, point["v"] + radius],
            fill=color,
        )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/models/pose_landmarker_full.task"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, default=None)
    args = parser.parse_args()

    image = np.ascontiguousarray(np.asarray(Image.open(args.image).convert("RGB")))
    height, width = image.shape[:2]

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = landmarker.detect(mp_image)

    report = {"image": str(args.image), "width": width, "height": height, "detected": False}
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        keypoints = {}
        for name, landmark in zip(KEYPOINT_NAMES, landmarks):
            keypoints[name] = {
                "u": float(landmark.x * width),
                "v": float(landmark.y * height),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility),
                "presence": float(landmark.presence),
            }
        report["detected"] = True
        report["keypoints"] = keypoints
        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks[0]
            report["world_keypoints"] = {
                name: {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(landmark.visibility),
                    "presence": float(landmark.presence),
                }
                for name, landmark in zip(KEYPOINT_NAMES, world_landmarks)
            }
        if args.overlay is not None:
            args.overlay.parent.mkdir(parents=True, exist_ok=True)
            draw_overlay(image, keypoints).save(args.overlay)
            report["overlay"] = str(args.overlay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"POSE_ESTIMATOR detected={report['detected']} out={args.out}")
    if report["detected"]:
        for name in ("NOSE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"):
            point = report["keypoints"][name]
            print(
                f"  {name}: u={point['u']:.1f} v={point['v']:.1f} "
                f"visibility={point['visibility']:.3f}"
            )


if __name__ == "__main__":
    main()
