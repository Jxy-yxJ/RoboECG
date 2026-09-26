"""Pure-logic tests for the chest frame (no Isaac Sim required)."""
from __future__ import annotations

import numpy as np
import pytest

from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.target_localization.chest_frame import build_chest_frame


def make_landmarks() -> ChestLandmarks:
    """A synthetic upright torso: up +Z, patient left +Y, anterior +X."""
    return ChestLandmarks(
        spine_lower=np.array([0.0, 0.0, 1.00]),
        spine_mid=np.array([0.0, 0.0, 1.10]),
        chest=np.array([0.0, 0.0, 1.20]),
        upper_chest=np.array([0.0, 0.0, 1.30]),
        neck_base=np.array([0.0, 0.0, 1.40]),
        neck_top=np.array([0.0, 0.0, 1.50]),
        clavicle_left=np.array([0.08, 0.08, 1.36]),
        clavicle_right=np.array([0.08, -0.08, 1.36]),
        shoulder_left=np.array([0.05, 0.20, 1.34]),
        shoulder_right=np.array([0.05, -0.20, 1.34]),
        pelvis=np.array([0.0, 0.0, 0.95]),
    )


def test_frame_axes_are_orthonormal_and_right_handed():
    frame = build_chest_frame(make_landmarks(), anterior_hint=(1.0, 0.0, 0.0))
    # Right-handed triple is (lateral, up, anterior): lateral x up = anterior.
    basis = np.column_stack([frame.lateral, frame.up, frame.anterior])
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(basis), 1.0, atol=1e-9)
    assert np.allclose(np.cross(frame.lateral, frame.up), frame.anterior, atol=1e-9)


def test_anterior_sign_follows_hint():
    landmarks = make_landmarks()
    forward = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    backward = build_chest_frame(landmarks, anterior_hint=(-1.0, 0.0, 0.0))
    assert np.allclose(forward.anterior, -backward.anterior, atol=1e-9)
    assert np.isclose(np.dot(forward.anterior, [1.0, 0.0, 0.0]), 1.0, atol=1e-9)


def test_origin_is_clavicle_midpoint():
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    expected = 0.5 * (landmarks.clavicle_left + landmarks.clavicle_right)
    assert np.allclose(frame.origin, expected)


def test_to_from_frame_round_trip():
    frame = build_chest_frame(make_landmarks(), anterior_hint=(1.0, 0.0, 0.0))
    point = frame.from_frame(0.05, -0.03, 0.12)
    assert np.allclose(frame.to_frame(point), [0.05, -0.03, 0.12], atol=1e-9)


def test_lower_landmark_is_below_clavicle_in_frame_coordinates():
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    assert frame.to_frame(landmarks.spine_lower)[0] < 0.0
    assert frame.to_frame(landmarks.chest)[0] < 0.0


def test_degenerate_landmarks_raise():
    landmarks = make_landmarks()
    with pytest.raises(ValueError):
        build_chest_frame(landmarks, anterior_hint=(0.0, 1.0, 0.0))


def test_chest_landmarks_read_and_missing_joint():
    from roboecg.perception.chest_landmarks import (
        CHEST_LANDMARK_JOINTS,
        read_chest_landmarks,
    )

    joints = {
        f"RL_BoneRoot/{leaf}": [0.0, 0.0, float(index)]
        for index, leaf in enumerate(CHEST_LANDMARK_JOINTS.values())
    }
    landmarks = read_chest_landmarks(joints)
    expected = float(
        list(CHEST_LANDMARK_JOINTS.values()).index(CHEST_LANDMARK_JOINTS["chest"])
    )
    assert landmarks.chest.tolist() == [0.0, 0.0, expected]

    broken = dict(joints)
    broken.pop(f"RL_BoneRoot/{CHEST_LANDMARK_JOINTS['chest']}")
    with pytest.raises(KeyError):
        read_chest_landmarks(broken)
