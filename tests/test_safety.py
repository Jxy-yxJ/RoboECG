"""Pure-logic tests for the patient collision proxies (capsule scaling)."""
from __future__ import annotations

import numpy as np
import pytest

from roboecg.robot_controller.safety import (
    NOMINAL_SPINE3_TO_PELVIS_M,
    body_capsules,
    body_scale_from_joints,
)


def make_joints(scale: float = 1.0) -> dict:
    def p(z: float):
        return np.array([0.0, 0.0, z * scale])

    return {
        "Pelvis": p(0.60),
        "Spine3": p(0.60 + NOMINAL_SPINE3_TO_PELVIS_M),
        "Chest": p(0.60 + NOMINAL_SPINE3_TO_PELVIS_M + 0.10),
        "Neck1": p(0.60 + NOMINAL_SPINE3_TO_PELVIS_M + 0.21),
        "Neck2": p(0.60 + NOMINAL_SPINE3_TO_PELVIS_M + 0.29),
        "Head": p(0.60 + NOMINAL_SPINE3_TO_PELVIS_M + 0.39),
        "L_UpArm": p(0.95),
        "L_LoArm": p(0.75),
        "L_Wrist": p(0.55),
        "R_UpArm": p(0.95),
        "R_LoArm": p(0.75),
        "R_Wrist": p(0.55),
    }


def test_body_scale_is_measured_from_the_joints():
    assert body_scale_from_joints(make_joints(1.0)) == pytest.approx(1.0, abs=1e-6)
    assert body_scale_from_joints(make_joints(0.9)) == pytest.approx(0.9, abs=1e-6)


def test_capsule_radii_scale_with_the_body():
    full = {name: radius for name, _, _, radius in body_capsules(make_joints(1.0))}
    small = {name: radius for name, _, _, radius in body_capsules(make_joints(0.9))}
    assert small["torso"] == pytest.approx(0.14 * 0.9, abs=1e-9)
    assert small["forearm_l"] == pytest.approx(0.045 * 0.9, abs=1e-9)
    assert small["torso"] < full["torso"]


def test_explicit_scale_overrides_the_measurement():
    capsules = {
        name: radius
        for name, _, _, radius in body_capsules(make_joints(1.0), scale=1.1)
    }
    assert capsules["torso"] == pytest.approx(0.14 * 1.1, abs=1e-9)
