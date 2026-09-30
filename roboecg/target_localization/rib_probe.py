"""Intercostal-space estimation from a light-touch probe profile (I4-a).

The V1->V4 vertical drop is a population regression: it is fitted on 25
statistical-shape torsos and overestimates the one real patient by 26.5 mm
(docs/ECG_P0B_FINDINGS.md).  The phantom study (docs/ECG_V3_SOLUTION_PLAN.md
section 2.6) showed that a light-touch scan along the midclavicular line
recovers the true row spacing to 0.75 mm, because over a rib band the contact
stiffness is a multiple of the soft-tissue stiffness while the working depth
resolution cannot resolve the 5 mm ridges at all.

This module is the pure decision layer:

  * ``synthetic_probe_profile`` - the analytic contact model of a scan over
    rib bands (used by tests and the phantom demo);
  * ``classify_rib_bands`` - weighted centres of the high-force bands;
  * ``estimate_ics_rows`` - ICS4 = the band midpoint nearest the SNND prior,
    ICS5 = the next midpoint downwards, i.e. the row that replaces the
    regression.

Protocol assumptions (documented): fixed light stroke (1-2 mm), probe points
on the midclavicular line (v = v_mcl), rib/skin stiffness contrast kappa, and
rib bands at least as wide as the probe pitch.
"""
from __future__ import annotations

import numpy as np


def synthetic_probe_profile(
    u_samples,
    rib_centers,
    *,
    half_width_m: float = 0.011,
    kappa: float = 3.0,
    stroke_m: float = 0.002,
    k_skin_n_m: float = 150.0,
) -> np.ndarray:
    """Contact force of a light-touch scan over rib bands (analytic model)."""
    u = np.asarray(u_samples, dtype=float)
    centers = np.asarray(rib_centers, dtype=float).reshape(-1)
    distance = np.abs(u[:, None] - centers[None, :]).min(axis=1)
    overlap = np.maximum(0.0, 1.0 - distance / half_width_m)
    return k_skin_n_m * stroke_m * (1.0 + (kappa - 1.0) * (overlap > 0.0))


def classify_rib_bands(
    u_samples, profile, threshold_fraction: float = 0.5
) -> dict:
    """High-force bands in a probe profile.

    Returns ``{"band_centers": [...], "ics_midpoints": [...]}`` (u descending:
    head -> feet); empty lists when the profile has no contrast.
    """
    profile = np.asarray(profile, dtype=float)
    u_samples = np.asarray(u_samples, dtype=float)
    low = float(np.percentile(profile, 50))
    high = float(np.percentile(profile, 95))
    if high - low < 1e-9:
        return {"band_centers": [], "ics_midpoints": []}
    threshold = low + threshold_fraction * (high - low)
    mask = profile >= threshold
    bands = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            bands.append((start, index - 1))
            start = None
    if start is not None:
        bands.append((start, len(mask) - 1))
    centers = []
    for first, last in bands:
        weight = np.maximum(profile[first:last + 1] - low, 0.0)
        centers.append(
            float(np.average(u_samples[first:last + 1], weights=weight + 1e-9))
        )
    centers.sort(reverse=True)
    midpoints = [
        0.5 * (centers[i] + centers[i + 1]) for i in range(len(centers) - 1)
    ]
    return {"band_centers": centers, "ics_midpoints": midpoints}


def estimate_ics_rows(
    u_samples,
    profile,
    u_ics4_prior: float,
    tolerance_m: float = 0.025,
    threshold_fraction: float = 0.5,
) -> dict:
    """Estimate the ICS4/ICS5 rows from a probe profile.

    ICS4 is the band midpoint nearest ``u_ics4_prior`` (the SNND rule level)
    within ``tolerance_m``; ICS5 is the next midpoint downwards (towards the
    feet).  Fields are None when the probe cannot resolve the rows.
    """
    classification = classify_rib_bands(u_samples, profile, threshold_fraction)
    midpoints = classification["ics_midpoints"]
    result = {
        "ics4_m": None,
        "ics5_m": None,
        "drop_m": None,
        **classification,
    }
    if len(midpoints) < 2:
        return result
    distances = [abs(m - float(u_ics4_prior)) for m in midpoints]
    nearest = int(np.argmin(distances))
    if distances[nearest] > tolerance_m:
        return result
    if nearest + 1 >= len(midpoints):
        return result
    ics4 = float(midpoints[nearest])
    ics5 = float(midpoints[nearest + 1])
    result.update(
        {"ics4_m": ics4, "ics5_m": ics5, "drop_m": ics4 - ics5}
    )
    return result
