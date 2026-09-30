"""Pure-logic tests for V1-V6 target generation."""
from __future__ import annotations

import numpy as np
import pytest

from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.target_localization.chest_frame import (
    ChestFrame,
    build_chest_frame,
    cloud_blended_frame,
)
from roboecg.target_localization.ecg import (
    V5_LATERAL_FRACTION_DEFAULT,
    generate_v1_v6,
    measure_lateral_extent,
    measure_thorax_circumference,
    measure_lateral_extent,
    measure_torso_width,
    robust_lateral_extent,
    robust_torso_width,
    snap_to_surface,
)

RULES = {
    "anatomy": {
        "sternal_notch_to_nipple": {"value": 0.193, "citation": "test-citation"},
        "nipple_fourth_ics": {"value": 0.75, "citation": "test-citation"},
        "fourth_to_fifth_ics": {
            "value": None,
            "provenance": "published_regression",
            "citation": "test-fixture",
            "coefficients": {
                "input": "torso_width_mm",
                "intercept_mm": -60.89,
                "slope_mm_per_mm": 0.39345,
            },
            "fallback_m": 0.0867,
            "fit_quality": "test-fixture",
        },
    }
}


def make_landmarks() -> ChestLandmarks:
    return ChestLandmarks(
        spine_lower=np.array([0.0, 0.0, 1.00]),
        spine_mid=np.array([0.0, 0.0, 1.10]),
        chest=np.array([0.0, 0.0, 1.20]),
        upper_chest=np.array([0.0, 0.0, 1.30]),
        neck_base=np.array([0.0, 0.0, 1.40]),
        neck_top=np.array([0.0, 0.0, 1.50]),
        clavicle_left=np.array([0.08, 0.02, 1.36]),
        clavicle_right=np.array([0.08, -0.02, 1.36]),
        shoulder_left=np.array([0.05, 0.20, 1.34]),
        shoulder_right=np.array([0.05, -0.20, 1.34]),
        pelvis=np.array([0.0, 0.0, 0.95]),
    )


def make_chest_points(frame, u_range=(-0.30, 0.05), v_range=(-0.25, 0.25)):
    """Synthetic chest surface: a closed elliptic cylinder in frame coordinates."""
    us = np.linspace(u_range[0], u_range[1], 60)
    vs = np.linspace(v_range[0], v_range[1], 60)
    points = []
    for u in us:
        for v in vs:
            # elliptic cross-section that narrows toward the head and the waist
            half_width = 0.22
            if abs(v) > half_width:
                continue
            bulge = 0.12 * np.sqrt(max(0.0, 1.0 - (v / half_width) ** 2))
            for n in (bulge, -bulge):
                points.append(frame.from_frame(u, v, n))
    return np.array(points)


@pytest.fixture()
def setup():
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    points = make_chest_points(frame)
    return landmarks, frame, points


def test_measures_are_positive_and_plausible(setup):
    _, frame, points = setup
    circumference = measure_thorax_circumference(points, frame, -0.15)
    lateral = measure_lateral_extent(points, frame, -0.20)
    assert 0.3 < circumference < 1.5
    assert 0.15 < lateral < 0.30


def test_snap_anchor_selects_the_anchored_sheet():
    """A second surface sheet at the same (u, v) must not capture the snap."""
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    us = np.linspace(-0.04, 0.04, 9)
    vs = np.linspace(-0.04, 0.04, 9)
    sheet_a = [frame.from_frame(u, v, 0.0) for u in us for v in vs]
    sheet_b = [
        frame.from_frame(u, v, 0.07)
        for u in np.linspace(-0.01, 0.01, 5)
        for v in np.linspace(-0.01, 0.01, 5)
    ]
    points = np.array(sheet_a + sheet_b)
    point, normal, info = snap_to_surface(
        points, frame, 0.0, 0.0, anchor_height=0.07
    )
    assert info["status"] == "ok"
    assert point is not None
    _, _, n = frame.to_frame(point)
    assert abs(n - 0.07) < 5e-3, n


def test_snap_two_sheet_cloud_rank_deficient_patch():
    """A two-sheet patch whose quadratic design is rank-deficient must not
    crash the linear fallback (regression: the truncated design was multiplied
    by the zero-padded six-vector)."""
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    sheet_a = [
        frame.from_frame(u, v, 0.0)
        for u in np.linspace(-0.04, 0.04, 9)
        for v in np.linspace(-0.04, 0.04, 9)
    ]
    sheet_b = [
        frame.from_frame(u, v, 0.07)
        for u in np.linspace(-0.01, 0.01, 5)
        for v in np.linspace(-0.01, 0.01, 5)
    ]
    points = np.array(sheet_a + sheet_b)
    point, normal, info = snap_to_surface(points, frame, 0.0, 0.0)
    assert info["status"] == "ok"
    assert point is not None
    _, _, n = frame.to_frame(point)
    assert abs(n) < 0.08, n


def test_snap_anchor_band_falls_back_when_empty():
    """An anchor far from every sheet must not break the snap."""
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    points = np.array(
        [
            frame.from_frame(u, v, 0.0)
            for u in np.linspace(-0.04, 0.04, 9)
            for v in np.linspace(-0.04, 0.04, 9)
        ]
    )
    point, normal, info = snap_to_surface(
        points, frame, 0.0, 0.0, anchor_height=0.5
    )
    assert info["status"] == "ok"
    _, _, n = frame.to_frame(point)
    assert abs(n) < 5e-3


def test_snap_returns_surface_point_and_outward_normal(setup):
    _, frame, points = setup
    point, normal, info = snap_to_surface(points, frame, -0.16, 0.0)
    assert info["status"] == "ok"
    assert point is not None and normal is not None
    assert np.dot(normal, frame.anterior) > 0.0
    # the realised point must sit on the surface at the requested (u, v),
    # not at the nearest mesh vertex
    u, v, n = frame.to_frame(point)
    assert abs(u + 0.16) < 1e-6
    assert abs(v) < 1e-6
    # and it must lie on the synthetic cylinder (n = 0.12 at v = 0)
    assert abs(n - 0.12) < 5e-3
    assert info["fit_rms_m"] < 5e-3


def test_snap_stays_at_nominal_uv_on_the_lateral_wall(setup):
    """A lateral target (near the silhouette) must not slide towards the midline."""
    _, frame, points = setup
    for v in (0.0, 0.10, 0.18, 0.21):
        point, normal, info = snap_to_surface(points, frame, -0.20, v)
        assert info["status"] == "ok"
        u, actual_v, _ = frame.to_frame(point)
        assert abs(u + 0.20) < 2e-3, (u, v)
        # tolerance covers the quadratic-fit curvature error on the synthetic
        # cylinder; the failure mode this guards against is a ~20 mm collapse
        assert abs(actual_v - v) < 5e-3, (actual_v, v)
        assert info["dv_error_m"] == pytest.approx(actual_v - v, abs=1e-9)
        # the normal must point outward (away from the torso axis), also at the
        # lateral silhouette where an anterior-facing test is degenerate
        radial = point - (frame.origin + (-0.20) * frame.up)
        assert np.dot(normal, radial) > 0.0, (v, normal)


def test_v1_v6_topology_and_provenance(setup):
    landmarks, frame, points = setup
    result = generate_v1_v6(landmarks, frame, points, RULES)
    targets = {t.name: t for t in result.targets}
    assert set(targets) == {"V1", "V2", "V3", "V4", "V5", "V6"}

    # vertical grouping: V1/V2 on the 4th ICS, V4/V5/V6 on the 5th
    assert np.isclose(targets["V1"].frame_coords[0], targets["V2"].frame_coords[0])
    for name in ("V5", "V6"):
        assert np.isclose(
            targets["V4"].frame_coords[0], targets[name].frame_coords[0]
        )
    # the 4th->5th ICS drop is predicted from the measured torso width
    coefficients = RULES["anatomy"]["fourth_to_fifth_ics"]["coefficients"]
    v_shoulder = abs(float(frame.to_frame(landmarks.shoulder_left)[1]))
    width_m = robust_torso_width(points, frame, landmark_limit=v_shoulder)
    expected_drop = (
        coefficients["intercept_mm"]
        + coefficients["slope_mm_per_mm"] * width_m * 1000.0
    ) / 1000.0
    assert np.isclose(
        targets["V4"].frame_coords[0], targets["V2"].frame_coords[0] - expected_drop
    )
    # and the drop must be of the published order (8-9 cm), not 2 cm
    assert 0.05 < expected_drop < 0.12

    # V3 is the midpoint of V2 and V4
    expected_v3 = 0.5 * (
        targets["V2"].frame_coords + targets["V4"].frame_coords
    )
    assert np.allclose(targets["V3"].frame_coords[:2], expected_v3[:2], atol=1e-9)

    # V1/V2 are symmetric about the sternum midline
    assert np.isclose(
        targets["V1"].frame_coords[1], -targets["V2"].frame_coords[1]
    )

    # horizontal ordering: V4 medial of V5 medial of V6
    assert (
        targets["V4"].frame_coords[1]
        < targets["V5"].frame_coords[1]
        < targets["V6"].frame_coords[1]
    )

    # every target carries the published citations
    for target in result.targets:
        assert "published_statistic" in target.provenance["vertical"][
            "sternal_notch_to_nipple"
        ]
        assert target.provenance["snap"]["status"] == "ok"


def test_lateral_extent_rejects_the_table_layer(setup):
    """A depth cloud sees the table beyond the body; the extent must not."""
    landmarks, frame, points = setup
    u_level = -0.28
    # table: a flat far layer 10 cm behind the deepest body point, wider than
    # the body contour but inside the shoulder limit
    table = []
    for v in np.linspace(-0.30, 0.30, 200):
        for n in (-0.22, -0.23):
            table.append(frame.from_frame(u_level, v, n))
    cloud = np.vstack([points, np.asarray(table)])
    extent = measure_lateral_extent(
        cloud, frame, u_level, half_width_limit=0.30
    )
    clean = measure_lateral_extent(
        points, frame, u_level, half_width_limit=0.30
    )
    # the synthetic body contour is ~0.22 m; without the guard the table
    # would report ~0.29 m
    assert extent < 0.24, extent
    assert abs(extent - clean) < 0.02, (extent, clean)


def test_v5_uses_the_fitted_lateral_fraction(setup):
    """V5 sits lateral of the V4-V6 midpoint (GT-fitted fraction, not 0.5)."""
    landmarks, frame, points = setup
    result = generate_v1_v6(landmarks, frame, points, RULES)
    targets = {t.name: t for t in result.targets}
    alpha = float(
        RULES["anatomy"].get("anterior_axillary_line", {}).get(
            "fraction", V5_LATERAL_FRACTION_DEFAULT
        )
    )
    v4 = targets["V4"].frame_coords[1]
    v6 = targets["V6"].frame_coords[1]
    v5 = targets["V5"].frame_coords[1]
    expected = v4 + alpha * (v6 - v4)
    # the surface snap keeps a small (u, v) residual; the guarded failure mode
    # is the old midpoint rule, which sits ~10 mm medial of the fit
    assert abs(v5 - expected) < 5e-3, (v5, expected)
    assert v5 > 0.6 * (v4 + v6) - 0.2 * v4  # lateral of the midpoint


def test_spacing_sensitivity_shifts_the_fifth_ics_row(setup):
    landmarks, frame, points = setup
    base = generate_v1_v6(landmarks, frame, points, RULES, spacing_m=0.020)
    tight = generate_v1_v6(landmarks, frame, points, RULES, spacing_m=0.015)
    base_v4 = {t.name: t for t in base.targets}["V4"].frame_coords[0]
    tight_v4 = {t.name: t for t in tight.targets}["V4"].frame_coords[0]
    assert np.isclose(tight_v4 - base_v4, 0.005, atol=1e-6)
    # V1/V2 must not move with the spacing
    base_v1 = {t.name: t for t in base.targets}["V1"].frame_coords[0]
    tight_v1 = {t.name: t for t in tight.targets}["V1"].frame_coords[0]
    assert np.isclose(base_v1, tight_v1)


def test_generate_u_5ics_override_replaces_the_regression(setup):
    """I4-a: a probed ICS5 row replaces the population drop regression."""
    landmarks, frame, points = setup
    override = -0.193 - 0.048
    default = generate_v1_v6(landmarks, frame, points, RULES)
    corrected = generate_v1_v6(
        landmarks, frame, points, RULES, u_5ics_override=override
    )
    by_default = {t.name: t for t in default.targets}
    by_corrected = {t.name: t for t in corrected.targets}
    u_default = frame.to_frame(by_default["V4"].position)[0]
    u_corrected = frame.to_frame(by_corrected["V4"].position)[0]
    assert abs(u_corrected - override) < 5e-3
    assert abs(u_default - u_corrected) > 5e-3
    provenance = by_corrected["V4"].provenance["vertical"]["intercostal_spacing"]
    assert "probe_corrected" in provenance


def _profile_cloud(frame, half_width, rows=12, n_v=41, arm=0.0, gap=0.18):
    """Front-facing surface rows with an optional axilla gap + arm lobe."""
    points = []
    for u in np.linspace(-0.30, 0.05, rows):
        for v in np.linspace(-half_width, half_width, n_v):
            points.append(frame.from_frame(u, v, -0.01 * abs(v)))
        if arm > half_width:
            for v in np.linspace(gap, arm, 9):
                points.append(frame.from_frame(u, +v, -0.05))
                points.append(frame.from_frame(u, -v, -0.05))
    return np.array(points)


def test_robust_width_recovers_from_clipped_landmark():
    """A landmark far inside the true silhouette must not clip the width."""
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    points = _profile_cloud(frame, half_width=0.20)
    clipped = measure_torso_width(points, frame, half_width_limit=0.10)
    robust = robust_torso_width(points, frame, landmark_limit=0.10)
    assert clipped < 0.25
    assert 0.34 < robust < 0.44


def test_robust_width_keeps_the_axilla_gap():
    """With a rest-pose arm lobe the measurement must stop at the chest edge."""
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    points = _profile_cloud(frame, half_width=0.18, arm=0.29, gap=0.21)
    robust = robust_torso_width(points, frame, landmark_limit=0.22)
    assert 0.30 < robust < 0.42


def test_robust_lateral_extent_uses_the_silhouette_edge():
    landmarks = make_landmarks()
    frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    points = _profile_cloud(frame, half_width=0.20)
    extent = robust_lateral_extent(points, frame, u_level=-0.30, landmark_limit=0.10)
    clipped = measure_lateral_extent(
        points, frame, u_level=-0.30, half_width_limit=0.10
    )
    assert clipped < 0.15
    assert 0.15 < extent < 0.25


def _rotate(vec, axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    vec = np.asarray(vec, dtype=float)
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * float(np.dot(axis, vec)) * (1.0 - np.cos(angle))
    )


def test_cloud_blended_frame_recovers_axes():
    """A tilted landmark frame must be straightened by the cloud's PCA."""
    landmarks = make_landmarks()
    true_frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    points = np.array(
        [
            true_frame.from_frame(u, v, n)
            for u, v, n in zip(
                rng.uniform(-0.35, 0.15, 4000),
                rng.uniform(-0.15, 0.15, 4000),
                rng.uniform(-0.05, 0.05, 4000),
            )
        ]
    )
    angle = np.radians(25.0)
    tilted = ChestFrame(
        origin=true_frame.origin,
        up=_rotate(true_frame.up, true_frame.anterior, angle),
        lateral=_rotate(true_frame.lateral, true_frame.anterior, angle),
        anterior=true_frame.anterior,
        provenance={},
    )
    blended = cloud_blended_frame(tilted, points)
    for axis, reference in (("up", true_frame.up), ("lateral", true_frame.lateral)):
        got = getattr(blended, axis)
        got = got if np.dot(got, reference) >= 0 else -got
        cosine = float(np.clip(np.dot(got, reference), -1.0, 1.0))
        assert np.degrees(np.arccos(cosine)) < 3.0, axis
    assert np.isclose(
        float(np.dot(blended.up, blended.lateral)), 0.0, atol=1e-9
    )


def test_cloud_blended_frame_falls_back_when_sparse():
    landmarks = make_landmarks()
    true_frame = build_chest_frame(landmarks, anterior_hint=(1.0, 0.0, 0.0))
    sparse = np.zeros((10, 3))
    blended = cloud_blended_frame(true_frame, sparse)
    assert blended is true_frame
