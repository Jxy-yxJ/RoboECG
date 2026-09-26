"""Perceptual chest frame from MediaPipe keypoints (M3a).

MediaPipe Pose gives acromia (shoulders), hips, nose and ears -- not the
sternoclavicular joints or the sternal notch that the chest frame needs.  This
module extrapolates them:

  * up       : shoulder midpoint -> hip midpoint (torso long axis)
  * lateral  : right shoulder -> left shoulder (clavicle direction proxy)
  * anterior : lateral x up, sign fixed by a hint
  * origin   : shoulder midpoint + a one-time calibration offset
               (acromion -> sternoclavicular joint), expressed in torso-frame
               coordinates and measured from the asset

The calibration offset is the geometric extrapolation's weak point: M3a uses a
one-time asset measurement (documented), M3b replaces it with a learned
anatomical landmark detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roboecg.coordinate_transform.frames import normalize
from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.target_localization.chest_frame import ChestFrame

MIN_VISIBILITY = 0.5
REQUIRED_KEYPOINTS = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
)
OPTIONAL_KEYPOINTS = ("NOSE", "LEFT_EAR", "RIGHT_EAR")


@dataclass(frozen=True)
class PoseChestResult:
    frame: ChestFrame
    landmarks: ChestLandmarks
    info: dict = field(default_factory=dict)


def measure_shoulder_to_clavicle_offset(landmarks: ChestLandmarks, frame: ChestFrame) -> dict:
    """One-time calibration: SC-joint midpoint relative to the shoulder midpoint."""
    shoulder_mid = 0.5 * (landmarks.shoulder_left + landmarks.shoulder_right)
    clavicle_mid = 0.5 * (landmarks.clavicle_left + landmarks.clavicle_right)
    delta = np.asarray(clavicle_mid, dtype=float) - np.asarray(
        shoulder_mid, dtype=float
    )
    return {
        "du_m": float(np.dot(delta, frame.up)),
        "dv_m": float(np.dot(delta, frame.lateral)),
        "dn_m": float(np.dot(delta, frame.anterior)),
        "provenance": (
            "asset_measurement: one-time acromion->SC-joint calibration "
            "(simulation-internal); M3b replaces it with a learned detector"
        ),
    }


def build_perceptual_chest(
    lifted: dict,
    visibility: dict,
    calibration: dict,
    anterior_hint,
    min_visibility: float = MIN_VISIBILITY,
) -> PoseChestResult:
    """Build the chest frame and rule landmarks from lifted MediaPipe keypoints."""
    missing = [name for name in REQUIRED_KEYPOINTS if name not in lifted]
    if missing:
        raise RuntimeError(f"missing lifted keypoints: {missing}")
    low = {
        name: float(visibility.get(name, 0.0))
        for name in REQUIRED_KEYPOINTS
        if float(visibility.get(name, 0.0)) < min_visibility
    }
    if low:
        raise RuntimeError(f"keypoints below visibility threshold: {low}")

    shoulder_left = np.asarray(lifted["LEFT_SHOULDER"], dtype=float)
    shoulder_right = np.asarray(lifted["RIGHT_SHOULDER"], dtype=float)
    hip_left = np.asarray(lifted["LEFT_HIP"], dtype=float)
    hip_right = np.asarray(lifted["RIGHT_HIP"], dtype=float)

    shoulder_mid = 0.5 * (shoulder_left + shoulder_right)
    hip_mid = 0.5 * (hip_left + hip_right)

    up = normalize(shoulder_mid - hip_mid)
    # Lateral axis: average the shoulder line and the hip line.  The shoulder
    # points sit at the body edge next to the arms and are the noisiest
    # landmarks (M3b: 12 deg axis error from shoulders alone); the hip line is
    # more stable, so averaging reduces the axis error.
    lateral = normalize(
        (shoulder_left - shoulder_right) + (hip_left - hip_right)
    )
    lateral = normalize(lateral - np.dot(lateral, up) * up)
    anterior = normalize(np.cross(lateral, up))
    hint = normalize(anterior_hint)
    if float(np.dot(hint, anterior)) < 0.0:
        anterior = -anterior

    origin = (
        shoulder_mid
        + up * float(calibration["du_m"])
        + lateral * float(calibration["dv_m"])
        + anterior * float(calibration["dn_m"])
    )
    frame = ChestFrame(
        origin=origin,
        up=up,
        lateral=lateral,
        anterior=anterior,
        provenance={
            "up": "pose_estimation: MediaPipe shoulder midpoint -> hip midpoint",
            "lateral": "pose_estimation: MediaPipe shoulders (acromia)",
            "anterior": "clinical_definition: lateral x up, sign from hint",
            "origin": (
                "pose_estimation: shoulder midpoint + one-time calibration "
                f"(du={calibration['du_m']:.4f}, dv={calibration['dv_m']:.4f}, "
                f"dn={calibration['dn_m']:.4f}) m"
            ),
        },
    )

    # Rule landmarks: the SC joints are extrapolated from the shoulders along
    # the calibration; the spine chain is placed on the torso axis (the rules
    # only use clavicle_left and shoulder_left, the rest keep the dataclass
    # complete and are derived, not measured).
    clavicle_left = shoulder_left + (
        frame.up * float(calibration["du_m"])
        + frame.lateral * float(calibration["dv_m"])
        + frame.anterior * float(calibration["dn_m"])
    )
    clavicle_right = shoulder_right + (
        frame.up * float(calibration["du_m"])
        - frame.lateral * float(calibration["dv_m"])
        + frame.anterior * float(calibration["dn_m"])
    )
    landmarks = ChestLandmarks(
        spine_lower=origin - up * 0.25,
        spine_mid=origin - up * 0.18,
        chest=origin - up * 0.10,
        upper_chest=origin - up * 0.03,
        neck_base=origin + up * 0.02,
        neck_top=origin + up * 0.08,
        clavicle_left=clavicle_left,
        clavicle_right=clavicle_right,
        shoulder_left=shoulder_left,
        shoulder_right=shoulder_right,
        pelvis=hip_mid,
        provenance="pose_estimation: MediaPipe keypoints + geometric extrapolation",
    )
    info = {
        "required_visibility": {
            name: float(visibility.get(name, 0.0)) for name in REQUIRED_KEYPOINTS
        },
        "optional_visibility": {
            name: float(visibility.get(name, 0.0))
            for name in OPTIONAL_KEYPOINTS
            if name in visibility
        },
        "calibration": calibration,
    }
    return PoseChestResult(frame=frame, landmarks=landmarks, info=info)


def frame_difference(perceptual: ChestFrame, reference: ChestFrame) -> dict:
    """Origin distance and axis angles between two chest frames."""
    from roboecg.coordinate_transform.frames import angle_between_deg

    return {
        "origin_distance_m": float(
            np.linalg.norm(perceptual.origin - reference.origin)
        ),
        "up_angle_deg": angle_between_deg(perceptual.up, reference.up),
        "lateral_angle_deg": angle_between_deg(perceptual.lateral, reference.lateral),
        "anterior_angle_deg": angle_between_deg(
            perceptual.anterior, reference.anterior
        ),
    }

LEFT_RIGHT_PAIRS = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_EAR", "RIGHT_EAR"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
)


def fix_left_right(lifted: dict, patient_left_axis) -> tuple:
    """Swap MediaPipe left/right labels when they disagree with the known side.

    Measured in M3a: on the overhead supine view MediaPipe swaps the
    anatomical left/right (lateral axis error ~160 deg).  The patient's left
    direction is known from the bed setup (here world +Y), so the labels are
    corrected before building the frame.  The learned detector (M3b) must not
    need this correction.
    """
    left = lifted.get("LEFT_SHOULDER")
    right = lifted.get("RIGHT_SHOULDER")
    if left is None or right is None:
        return lifted, False
    axis = np.asarray(patient_left_axis, dtype=float)
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if float(np.dot(delta, axis)) >= 0.0:
        return lifted, False
    swapped = dict(lifted)
    for left_name, right_name in LEFT_RIGHT_PAIRS:
        left_value = lifted.get(left_name)
        right_value = lifted.get(right_name)
        if left_value is not None:
            swapped[right_name] = left_value
        if right_value is not None:
            swapped[left_name] = right_value
    return swapped, True
