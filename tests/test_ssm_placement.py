"""Tests for the SSM torso anatomical axes used by the I5-b scene placement."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roboecg.perception.torso_mesh import anatomical_axes, parse_vtk_polydata

DATA_DIR = Path(__file__).resolve().parents[1] / "assets" / "external" / "torso_models"


def _load(model_id: str):
    points_mm, _ = parse_vtk_polydata(
        DATA_DIR / f"{model_id}_torso_coarse_surface.vtk"
    )
    electrodes = np.loadtxt(
        DATA_DIR / f"{model_id}_electrodes.csv", delimiter=",", skiprows=1
    )
    return points_mm / 1000.0, electrodes[3:9] / 1000.0


def test_anatomical_axes_are_orthonormal_and_signed():
    points_m, electrodes_m = _load("T_01")
    a, b, c = anatomical_axes(points_m, electrodes_m)
    for axis in (a, b, c):
        assert np.isclose(float(np.linalg.norm(axis)), 1.0, atol=1e-9)
    assert abs(float(np.dot(a, b))) < 1e-9
    np.testing.assert_allclose(np.cross(a, b), c, atol=1e-9)
    # model frame: X ~ lateral (V1->V2), anterior ~ -Y (chest normals), down ~ -Z
    assert a[0] > 0.9
    assert b[1] < -0.9
    assert c[2] < -0.9


def test_placement_rotation_is_proper():
    """left/anterior/down -> +Y/+Z/+X (the m4 scene) must give det(R) = 1."""
    points_m, electrodes_m = _load("T_02")
    a_m, b_m, c_m = anatomical_axes(points_m, electrodes_m)
    a_w = np.array([0.0, 1.0, 0.0])
    b_w = np.array([0.0, 0.0, 1.0])
    c_w = np.cross(a_w, b_w)
    rotation = np.column_stack([a_w, b_w, c_w]) @ np.column_stack(
        [a_m, b_m, c_m]
    ).T
    assert np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6)
    # anatomical checks in world coordinates
    world_electrodes = electrodes_m @ rotation.T
    # V4-V6 run towards the patient's left (+Y) and below V1-V2 (+X = feet)
    assert world_electrodes[5, 1] > world_electrodes[0, 1]
    assert world_electrodes[5, 0] > world_electrodes[0, 0]
    # the chest faces up (+Z): electrodes are above the model's depth centroid
    assert world_electrodes[:, 2].mean() > (points_m @ rotation.T)[:, 2].mean()
