# Copied from farus_thyroid_isaac/farus/perception/pose2d.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""2D pose keypoints -> 3D FARUS landmarks (pure numpy).

Simulation approximation of the Azure Kinect Body Tracking SDK: MediaPipe
Pose gives 2D keypoints on the RGB image, the depth image supplies the third
coordinate.  The paper never names its exact joints, so the mapping below is
a documented choice.
"""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.camera import camera_to_world, deproject_pixel
from roboecg.coordinate_transform.frames import normalize
from roboecg.perception.depth import sample_depth
from roboecg.perception.landmarks import BodyLandmarks

MIN_VISIBILITY = 0.5
DEFAULT_DEPTH_WINDOW = 5
MAX_PERSON_DEPTH_M = 3.0

ANCHOR_PREFERENCE = (
    "NOSE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
)


def lift_world_landmarks(
    world_keypoints,
    keypoints_2d,
    depth,
    intrinsics,
    camera_position,
    cv_rotation,
    reference_names=ANCHOR_PREFERENCE,
    window=DEFAULT_DEPTH_WINDOW,
    max_depth=MAX_PERSON_DEPTH_M,
):
    """Model-based lifting of MediaPipe's metric world landmarks.

    MediaPipe world landmarks are metric and camera-axis-aligned (x right,
    y down, z away from the camera) with the hip center as origin.  One
    visible keypoint (preferably the nose) is matched to its depth-deprojected
    camera position to fix the translation, which keeps the correct relative
    depth of occluded joints such as the far shoulder.

    Returns (lifted_world_points, anchor_name) or ({}, None).
    """
    anchor_name = None
    offset = None
    for name in reference_names:
        keypoint = keypoints_2d.get(name)
        world_point = world_keypoints.get(name)
        if keypoint is None or world_point is None:
            continue
        z = sample_depth(depth, keypoint["u"], keypoint["v"], window)
        if z is None or z > max_depth:
            continue
        anchor_camera = np.array(
            [
                (keypoint["u"] - intrinsics.cx) / intrinsics.fx * z,
                (keypoint["v"] - intrinsics.cy) / intrinsics.fy * z,
                z,
            ]
        )
        offset = anchor_camera - np.array(
            [world_point["x"], world_point["y"], world_point["z"]]
        )
        anchor_name = name
        break

    if offset is None:
        return {}, None

    lifted = {}
    for name, world_point in world_keypoints.items():
        if world_point is None:
            continue
        point_camera = (
            np.array([world_point["x"], world_point["y"], world_point["z"]]) + offset
        )
        lifted[name] = camera_to_world(point_camera, camera_position, cv_rotation)
    return lifted, anchor_name


def lift_keypoints(
    keypoints_2d,
    depth,
    intrinsics,
    camera_position,
    cv_rotation,
    window=DEFAULT_DEPTH_WINDOW,
    max_depth=MAX_PERSON_DEPTH_M,
):
    """Deproject each 2D keypoint with the depth image; skip invalid ones."""
    lifted = {}
    for name, keypoint in keypoints_2d.items():
        u = float(keypoint["u"])
        v = float(keypoint["v"])
        z = sample_depth(depth, u, v, window)
        if z is None or z > max_depth:
            continue
        lifted[name] = deproject_pixel(u, v, z, intrinsics, camera_position, cv_rotation)
    return lifted


def landmarks_from_pose(lifted, visibility=None):
    """Map lifted MediaPipe keypoints to FARUS body landmarks."""
    visibility = visibility or {}

    def valid(name):
        point = lifted.get(name)
        if point is None:
            return None
        if visibility.get(name, 1.0) < MIN_VISIBILITY:
            return None
        return np.asarray(point, dtype=float)

    left_shoulder = valid("LEFT_SHOULDER")
    right_shoulder = valid("RIGHT_SHOULDER")
    if left_shoulder is None or right_shoulder is None:
        raise ValueError("pose landmarks require both shoulders with valid depth")

    left_ear = valid("LEFT_EAR")
    right_ear = valid("RIGHT_EAR")
    nose = valid("NOSE")
    if left_ear is not None and right_ear is not None:
        neck_top = 0.5 * (left_ear + right_ear)
    elif nose is not None:
        neck_top = nose
    else:
        raise ValueError("pose landmarks require ears or nose")

    neck_base = 0.5 * (left_shoulder + right_shoulder)
    neck_axis = normalize(neck_top - neck_base)
    head = nose if nose is not None else neck_top

    return BodyLandmarks(
        head=head,
        neck_top=neck_top,
        neck_base=neck_base,
        chest=neck_base - 0.10 * neck_axis,
        spine=neck_base - 0.25 * neck_axis,
        clavicle_left=left_shoulder,
        clavicle_right=right_shoulder,
    )


def landmark_errors(estimated: BodyLandmarks, reference: BodyLandmarks) -> dict:
    """Per-landmark Euclidean error (meters) between estimate and reference."""
    names = (
        "head",
        "neck_top",
        "neck_base",
        "clavicle_left",
        "clavicle_right",
    )
    return {
        name: float(
            np.linalg.norm(
                np.asarray(getattr(estimated, name), dtype=float)
                - np.asarray(getattr(reference, name), dtype=float)
            )
        )
        for name in names
    }


def _joint_by_leaf(joint_positions, leaf):
    for name, position in joint_positions.items():
        if name.rsplit("/", 1)[-1] == leaf:
            return np.asarray(position, dtype=float)
    return None


def comparison_reference_landmarks(joint_positions) -> BodyLandmarks:
    """Ground-truth landmarks with anatomically matching points for MediaPipe.

    MediaPipe's shoulder keypoint is the acromion (near the upper-arm joint),
    while the FARUS landmark set uses the clavicle root; the eye midpoint is a
    better match for the ear midpoint than the neck-top joint.  Comparing
    against this reference separates definitional offsets from measurement
    error.
    """
    upperarm_left = _joint_by_leaf(joint_positions, "L_Upperarm")
    upperarm_right = _joint_by_leaf(joint_positions, "R_Upperarm")
    if upperarm_left is None or upperarm_right is None:
        raise KeyError("comparison reference requires L_Upperarm/R_Upperarm")
    eye_left = _joint_by_leaf(joint_positions, "L_Eye")
    eye_right = _joint_by_leaf(joint_positions, "R_Eye")
    neck_base = 0.5 * (upperarm_left + upperarm_right)
    eyes = (
        0.5 * (eye_left + eye_right)
        if eye_left is not None and eye_right is not None
        else neck_base
    )
    neck_axis = normalize(eyes - neck_base)
    return BodyLandmarks(
        head=eyes,
        neck_top=eyes,
        neck_base=neck_base,
        chest=neck_base - 0.10 * neck_axis,
        spine=neck_base - 0.25 * neck_axis,
        clavicle_left=upperarm_left,
        clavicle_right=upperarm_right,
    )
