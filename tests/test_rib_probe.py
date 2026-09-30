"""Pure-logic tests for the intercostal probe decision layer (I4-a)."""
from __future__ import annotations

import numpy as np

from roboecg.target_localization.rib_probe import (
    classify_rib_bands,
    estimate_ics_rows,
    synthetic_probe_profile,
)

U_ICS4 = -0.193
DROP_TRUE = 0.048
RIB_CENTERS = [
    U_ICS4 + 3 * DROP_TRUE / 2,
    U_ICS4 + DROP_TRUE / 2,
    U_ICS4 - DROP_TRUE / 2,
    U_ICS4 - 3 * DROP_TRUE / 2,
]
U_SAMPLES = np.arange(-0.10, -0.29, -0.005)


def profile(kappa=3.0, stroke_mm=2.0, noise_scale=0.0, seed=0):
    force = synthetic_probe_profile(
        U_SAMPLES, RIB_CENTERS, kappa=kappa, stroke_m=stroke_mm / 1000.0
    )
    if noise_scale > 0.0:
        rng = np.random.default_rng(seed)
        contrast = force.max() - force.min()
        force = force + rng.normal(0.0, noise_scale * contrast, force.shape)
    return force


def test_classify_recovers_band_centers():
    result = classify_rib_bands(U_SAMPLES, profile())
    assert len(result["band_centers"]) == len(RIB_CENTERS)
    for center, truth in zip(result["band_centers"], sorted(RIB_CENTERS, reverse=True)):
        assert abs(center - truth) < 0.006, (center, truth)


def test_estimate_ics_rows_clean():
    estimate = estimate_ics_rows(U_SAMPLES, profile(), U_ICS4)
    assert estimate["ics4_m"] is not None
    assert abs(estimate["ics4_m"] - U_ICS4) < 0.005
    assert abs(estimate["ics5_m"] - (U_ICS4 - DROP_TRUE)) < 0.005
    assert abs(estimate["drop_m"] - DROP_TRUE) < 0.005


def test_estimate_survives_noise_and_low_contrast():
    for kappa in (1.5, 3.0, 10.0):
        for seed in (0, 1, 2):
            estimate = estimate_ics_rows(
                U_SAMPLES, profile(kappa=kappa, noise_scale=0.2, seed=seed),
                U_ICS4,
            )
            assert estimate["drop_m"] is not None, (kappa, seed)
            assert abs(estimate["drop_m"] - DROP_TRUE) < 0.006, (kappa, seed)


def test_flat_profile_returns_none():
    estimate = estimate_ics_rows(U_SAMPLES, profile(kappa=1.0), U_ICS4)
    assert estimate["ics4_m"] is None
    assert estimate["ics5_m"] is None


def test_prior_outside_tolerance_returns_none():
    estimate = estimate_ics_rows(U_SAMPLES, profile(), U_ICS4 + 0.10)
    assert estimate["ics4_m"] is None
