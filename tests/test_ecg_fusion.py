"""Pure-logic tests for the M2 depth fusion (synthetic depth image)."""
from __future__ import annotations

import numpy as np
import pytest

from roboecg.coordinate_transform.camera import CameraIntrinsics
from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.perception.torso_prior import ChestSurfacePrior
from roboecg.target_localization.chest_frame import build_chest_frame
from roboecg.target_localization.fusion import (
    best_view_index,
    evaluate_fusion,
    fuse_target,
)

SETTINGS = {
    "depth_median_window": 3,
    "normal_window": 8,
    "normal_radius_m": 0.05,
    "max_depth_m": 3.0,
    "max_skin_offset_m": 0.05,
    "normal_max_angle_deg": 25.0,
}

INTRINSICS = CameraIntrinsics(width=64, height=64, fx=50.0, fy=50.0, cx=32.0, cy=32.0)
# camera at +Z looking toward -Z (CV convention: forward = +Z camera axis)
CAMERA_POSITION = np.array([0.0, 0.0, 1.0])
CV_ROTATION = np.diag([1.0, -1.0, -1.0])


def make_landmarks() -> ChestLandmarks:
    """Synthetic torso with up=+Y, lateral=+X, anterior=+Z.

    Frame origin (clavicle midpoint) sits at z=0, so the synthetic chest plane
    n=0 is the world plane z=0 and the camera at +Z sees it at depth 1.0.
    """
    return ChestLandmarks(
        spine_lower=np.array([0.0, -0.30, 0.0]),
        spine_mid=np.array([0.0, -0.20, 0.0]),
        chest=np.array([0.0, -0.10, 0.0]),
        upper_chest=np.array([0.0, 0.0, 0.0]),
        neck_base=np.array([0.0, 0.10, 0.0]),
        neck_top=np.array([0.0, 0.20, 0.0]),
        clavicle_left=np.array([0.04, 0.05, 0.0]),
        clavicle_right=np.array([-0.04, 0.05, 0.0]),
        shoulder_left=np.array([0.20, 0.04, 0.0]),
        shoulder_right=np.array([-0.20, 0.04, 0.0]),
        pelvis=np.array([0.0, -0.40, 0.0]),
    )


class SimpleTarget:
    def __init__(self, name, position, normal, frame_coords):
        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.frame_coords = np.asarray(frame_coords, dtype=float)


def make_frame():
    # frame with up=+Z, lateral=+Y, anterior=+X  -> synthetic landmarks map to it
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
    return landmarks, frame


def flat_prior():
    """Flat chest model n = 0 (the synthetic depth plane sits at n = 0)."""
    return ChestSurfacePrior(
        coefficients=np.zeros(6),
        u_range=(-0.4, 0.2),
        v_limit=0.3,
        fit_rms_m=0.0,
        samples=1,
    )


def synthetic_depth(value=1.0, size=64):
    return np.full((size, size), float(value))


@pytest.fixture()
def setup():
    landmarks, frame = make_frame()
    prior = flat_prior()
    return frame, prior


def test_depth_hit_refines_position(setup):
    frame, prior = setup
    # a target on the plane at n = 0, slightly off the optical axis
    point = frame.from_frame(0.0, 0.03, 0.0)
    target = SimpleTarget("Vx", point, frame.anterior, [0.0, 0.03, 0.0])
    fused = fuse_target(
        target,
        frame,
        prior,
        synthetic_depth(1.0),
        INTRINSICS,
        CAMERA_POSITION,
        CV_ROTATION,
        SETTINGS,
    )
    assert fused.source == "depth"
    assert fused.info["depth_m"] == pytest.approx(1.0)
    assert np.linalg.norm(fused.position - point) < 5e-3
    # flat plane normal must match the prior -> accepted, angle ~0
    assert fused.info["normal_angle_vs_prior_deg"] is not None
    assert fused.info["normal_angle_vs_prior_deg"] < 0.01
    assert not fused.info["normal_rejected"]


def test_invalid_depth_falls_back_to_mesh(setup):
    frame, prior = setup
    point = frame.from_frame(0.0, 0.0, 0.0)
    target = SimpleTarget("Vx", point, frame.anterior, [0.0, 0.0, 0.0])
    fused = fuse_target(
        target,
        frame,
        prior,
        synthetic_depth(0.0),
        INTRINSICS,
        CAMERA_POSITION,
        CV_ROTATION,
        SETTINGS,
    )
    assert fused.source == "mesh_fallback"
    assert np.allclose(fused.position, point)


def test_far_skin_point_is_rejected(setup):
    frame, prior = setup
    point = frame.from_frame(0.0, 0.0, 0.0)
    target = SimpleTarget("Vx", point, frame.anterior, [0.0, 0.0, 0.0])
    fused = fuse_target(
        target,
        frame,
        prior,
        synthetic_depth(1.5),  # 0.5 m behind the target -> outside the 5 cm bound
        INTRINSICS,
        CAMERA_POSITION,
        CV_ROTATION,
        SETTINGS,
    )
    assert fused.info["skin_point_rejected"] is True
    assert np.allclose(fused.position, point)


def test_normal_far_from_prior_is_rejected(setup):
    frame, prior = setup
    # prior tilted ~40 deg away from the flat measured normal
    tilt = np.radians(40.0)
    tilted = ChestSurfacePrior(
        coefficients=np.array([0.0, 0.0, np.tan(tilt), 0.0, 0.0, 0.0]),
        u_range=(-0.4, 0.2),
        v_limit=0.3,
        fit_rms_m=0.0,
        samples=1,
    )
    point = frame.from_frame(0.0, 0.0, 0.0)
    target = SimpleTarget("Vx", point, frame.anterior, [0.0, 0.0, 0.0])
    fused = fuse_target(
        target,
        frame,
        tilted,
        synthetic_depth(1.0),
        INTRINSICS,
        CAMERA_POSITION,
        CV_ROTATION,
        SETTINGS,
    )
    assert fused.info["normal_rejected"] is True
    # on rejection the fusion keeps the nominal (upstream) normal; the prior is
    # only used as the acceptance reference
    assert np.allclose(fused.normal, target.normal)


def test_evaluate_fusion_reports_errors(setup):
    frame, prior = setup
    point = frame.from_frame(0.0, 0.02, 0.0)
    target = SimpleTarget("Vx", point, frame.anterior, [0.0, 0.02, 0.0])
    fused = fuse_target(
        target,
        frame,
        prior,
        synthetic_depth(1.0),
        INTRINSICS,
        CAMERA_POSITION,
        CV_ROTATION,
        SETTINGS,
    )
    report = evaluate_fusion([fused], [target], frame)
    assert report["depth_hit_rate"] == 1.0
    assert report["position_error_m"]["max"] < 5e-3
    assert report["fused_normal_angle_vs_mesh_deg"]["max"] < 0.01
    assert report["normal_acceptance_rate"] == 1.0


def test_best_view_index_picks_the_frontal_camera():
    normal = np.array([0.0, 1.0, 0.0])  # surface faces +Y
    position = np.zeros(3)
    overhead = np.array([0.0, 0.0, 1.0])  # grazing on this surface
    lateral = np.array([0.0, 1.0, 0.0])  # frontal
    assert best_view_index(normal, position, [overhead, lateral]) == 1
    assert best_view_index(normal, position, [lateral, overhead]) == 0
    assert best_view_index(normal, position, [overhead]) == 0
