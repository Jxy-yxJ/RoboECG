"""V1-V6 target generation from measured chest geometry.

Every coordinate of every electrode carries a provenance tag:

  vertical (u)
    - 4th ICS level  = sternal notch level - sternal-notch-to-nipple distance
      (published statistic, DOI 10.1097/SAP.0000000000002018) combined with the
      nipple-in-4th-ICS statistic (Beer 2001, DOI 10.1097/00006534-200112000-00015)
    - 5th ICS level  = 4th ICS level - vertical drop predicted from the measured
      torso width (published regression fitted on 25 statistical-shape torso
      models with real 12-lead electrode positions, DOI 10.5281/zenodo.20086105;
      width rather than depth because a single-view cloud cannot see the back).
      The drop includes the downward rib slope towards the midclavicular line,
      which a constant intercostal spacing misses by a factor of ~4.
  horizontal (v)
    - V1/V2 parasternal : |v| of the sternoclavicular joints (asset measurement)
    - V4 midclavicular  : v of the clavicle midpoint (asset measurement)
    - V6 midaxillary    : lateral-most chest contour at the V4 level (asset measurement)
    - V5                : v_mcl + alpha x (v_midax - v_mcl), alpha fitted on the
                          25-model GT (alpha = 0.736 +- 0.070; the chest wraps
                          towards the axilla, so the clinical V5 is lateral of
                          the midpoint)
  clinical topology
    - V3 = midpoint(V2, V4)

The nominal (u, v) points are snapped onto the measured chest surface and the
local surface normal is estimated by PCA, so downstream planning gets a real
contact pose rather than a free-floating point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roboecg.perception.chest_landmarks import ChestLandmarks
from roboecg.perception.depth import estimate_surface_normal
from roboecg.target_localization.chest_frame import ChestFrame

NORMAL_FIT_RADIUS_M = 0.03
SURFACE_FIT_RADIUS_M = 0.035
SURFACE_FIT_MIN_POINTS = 8
# Multi-sheet guard: a multi-view cloud can contain a second surface sheet at
# the same (u, v) (measured ~75 mm apart at V5); the patch is restricted to the
# sheet within this band of the anchor height when enough points remain.
SURFACE_ANCHOR_BAND_M = 0.040
TORSO_HALF_WIDTH_LIMIT_M = 0.50  # excludes rest-pose arms from chest contours
# Fallback V5 lateral fraction (see configs/ecg_rules.yaml
# -> anatomy.anterior_axillary_line); fitted on 25 GT torso models.
V5_LATERAL_FRACTION_DEFAULT = 0.7356


@dataclass(frozen=True)
class ElectrodeTarget:
    name: str
    position: np.ndarray
    normal: np.ndarray
    frame_coords: np.ndarray  # (u, v, n)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class V1V6Result:
    targets: tuple
    vertical: dict
    horizontal: dict
    provenance: dict


def _frame_components(points, frame: ChestFrame):
    delta = np.asarray(points, dtype=float) - frame.origin
    return delta @ frame.up, delta @ frame.lateral, delta @ frame.anterior


def _convex_hull_perimeter(uv: np.ndarray) -> float:
    """Perimeter of the 2D convex hull (Andrew's monotone chain)."""
    points = np.unique(np.round(uv, 6), axis=0)
    if points.shape[0] < 3:
        return 0.0
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return 0.0
    closed = np.vstack([hull, hull[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def measure_thorax_circumference(points, frame: ChestFrame, u_level: float,
                                 band: float = 0.01) -> float:
    """Convex-hull perimeter of the chest cross-section at `u_level` (metres)."""
    along, lateral, normal = _frame_components(points, frame)
    mask = (np.abs(along - u_level) <= band) & (
        np.abs(lateral) <= TORSO_HALF_WIDTH_LIMIT_M
    )
    if np.count_nonzero(mask) < 8:
        return 0.0
    return _convex_hull_perimeter(np.column_stack([lateral[mask], normal[mask]]))


def _front_layer_extent(
    lateral: np.ndarray,
    normal: np.ndarray,
    bin_m: float = 0.02,
    min_layer_gap_m: float = 0.06,
) -> float | None:
    """Lateral extent of the near (chest) depth layer of a cross-section band.

    A depth cloud from the overhead camera contains a second, far layer beyond
    the body silhouette (the table, ~17 cm below the chest surface); a plain
    percentile of |v| therefore reports the table edge instead of the
    midaxillary line (measured on the sim body: 190 mm vs the true 150 mm).

    The two layers are separated by an empty band in the surface-height (n)
    histogram.  Cut at the largest empty run (if it exceeds
    `min_layer_gap_m`) and take the 98th percentile of |v| in the near layer.
    A sparse mesh without a far layer has no such gap and is left untouched.
    """
    if lateral.size < 8:
        return None
    low, high = float(normal.min()), float(normal.max())
    n_bins = max(int(np.ceil((high - low) / bin_m)), 1)
    counts, edges = np.histogram(normal, bins=n_bins)
    populated = np.where(counts > 0)[0]
    if populated.size < 2:
        return None
    keep = np.ones(lateral.size, dtype=bool)
    gaps = np.diff(populated)
    gap_bins = int(np.ceil(min_layer_gap_m / bin_m))
    if int(gaps.max()) >= gap_bins:
        last_near_bin = populated[int(np.argmax(gaps))]
        threshold = float(edges[last_near_bin + 1])
        keep = normal >= threshold
    if int(np.count_nonzero(keep)) < 8:
        return None
    return float(np.percentile(np.abs(lateral[keep]), 98.0))


def measure_lateral_extent(
    points, frame: ChestFrame, u_level: float, band: float = 0.01,
    half_width_limit: float | None = None,
) -> float:
    """Largest |v| of the chest contour at `u_level` (midaxillary line).

    `half_width_limit` should be the measured shoulder |v|: the arms lie
    alongside the torso in the supine pose, so a depth cloud would otherwise
    report the arm's outer edge as the torso contour.  The near-layer guard
    additionally rejects the table/back layers of a depth cloud (see
    `_front_layer_extent`).
    """
    limit = (
        TORSO_HALF_WIDTH_LIMIT_M
        if half_width_limit is None
        else float(half_width_limit)
    )
    along, lateral, normal = _frame_components(points, frame)
    mask = (np.abs(along - u_level) <= band) & (np.abs(lateral) <= limit)
    if np.count_nonzero(mask) < 8:
        return 0.0
    extent = _front_layer_extent(lateral[mask], normal[mask])
    if extent is not None:
        return extent
    # fallback: 98th percentile (isolated depth pixels must not define the
    # contour, but the table layer is only handled by the guard above)
    return float(np.percentile(np.abs(lateral[mask]), 98.0))


def measure_torso_width(
    points, frame: ChestFrame, half_width_limit: float,
    u_top: float = 0.02, u_bottom: float = -0.42,
) -> float:
    """Lateral extent of the torso (metres).

    Matches the GT dataset's `torso_width_mm` (bounding-box width of the torso
    surface).  Unlike the anterior-posterior depth, the width is measurable
    from a single-view (overhead) depth cloud, because the body silhouette
    exposes the lateral extremes; `half_width_limit` (measured shoulder |v|)
    excludes the rest-pose arms and the table.
    """
    along, lateral, _ = _frame_components(points, frame)
    mask = (
        (along <= u_top)
        & (along >= u_bottom)
        & (np.abs(lateral) <= float(half_width_limit))
    )
    if np.count_nonzero(mask) < 32:
        return 0.0
    # 98th percentile: isolated depth pixels must not define the silhouette.
    return float(2.0 * np.percentile(np.abs(lateral[mask]), 98.0))


def arc_midpoint_v(
    points, frame: ChestFrame, u_level: float, v_start: float, v_end: float,
    band: float = 0.015, bins: int = 40,
) -> float:
    """Lateral coordinate of the arc-length midpoint of the chest contour.

    V5 lies between V4 (midclavicular) and V6 (midaxillary); on the real chest
    the wrap towards the axilla is much longer than the lateral span, so the
    midpoint must be taken along the surface (the GT dataset has V4-V5 =
    57.9 mm and V5-V6 = 50.1 mm, i.e. near-equal arc lengths, while a lateral
    midpoint gives 29.6 / 74.4 mm on the simulated body: -2.2 SD).
    """
    along, lateral, normal = _frame_components(points, frame)
    mask = (
        (np.abs(along - u_level) <= band)
        & (normal > -0.10)
        & (lateral >= min(v_start, v_end))
        & (lateral <= max(v_start, v_end))
    )
    if np.count_nonzero(mask) < 8:
        return 0.5 * (v_start + v_end)
    v = lateral[mask]
    n = normal[mask]
    edges = np.linspace(min(v_start, v_end), max(v_start, v_end), bins + 1)
    centres, envelope = [], []
    for i in range(bins):
        cell = (v >= edges[i]) & (v < edges[i + 1])
        if np.count_nonzero(cell) >= 1:
            centres.append(0.5 * (edges[i] + edges[i + 1]))
            envelope.append(float(np.percentile(n[cell], 95.0)))
    if len(centres) < 3:
        return 0.5 * (v_start + v_end)
    centres = np.asarray(centres)
    envelope = np.asarray(envelope)
    order = np.argsort(centres)
    centres, envelope = centres[order], envelope[order]
    segment = np.hypot(np.diff(centres), np.diff(envelope))
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    target = 0.5 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target))
    index = min(max(index, 1), len(centres) - 1)
    span = cumulative[index] - cumulative[index - 1]
    fraction = 0.0 if span <= 0 else (target - cumulative[index - 1]) / span
    return float(
        centres[index - 1] + fraction * (centres[index] - centres[index - 1])
    )


def snap_to_surface(points, frame: ChestFrame, u: float, v: float,
                    v_limit: float | None = None,
                    anchor_height: float | None = None):
    """Contact point and normal at nominal (u, v) via a local surface fit.

    The fit lives in a local PCA frame (not in (u, v)): parameterising the
    surface as n = f(u, v) is ill-conditioned on the steep lateral chest wall
    (V5/V6, where the surface turns towards the back, fit RMS 16-25 mm), while
    a quadratic in the local tangent frame stays well-posed everywhere on the
    torso.  Evaluating it at the nominal point removes the mesh-vertex
    quantisation of a nearest-vertex snap (~2 cm on the biped asset).

    ``anchor_height`` (the frame normal coordinate, e.g. from the smooth
    surface prior at (u, v)) restricts the patch to the sheet within
    SURFACE_ANCHOR_BAND_M of it whenever that sheet keeps enough points; with
    a multi-view cloud a second sheet at the same (u, v) can otherwise capture
    the nearest-point height guess and displace the contact by ~75 mm.
    Note (measured, 2026-10-01): making the anchor conditional (only when the
    free guess disagrees with the prior by more than the band, to avoid
    injecting the prior's local bias at V6) was implemented and reverted --
    it left V6 unchanged and worsened the same-cloud term at V4 (+0.9 mm) and
    V5 (+0.3 mm), because even a correct guess benefits from excluding the
    far sheet from the surface fit.
    """
    limit = TORSO_HALF_WIDTH_LIMIT_M if v_limit is None else float(v_limit)
    along, lateral, normal = _frame_components(points, frame)
    delta_u = along - u
    delta_v = lateral - v
    distance_sq = delta_u**2 + delta_v**2
    mask = None
    count = 0
    for radius in (
        SURFACE_FIT_RADIUS_M,
        2.0 * SURFACE_FIT_RADIUS_M,
        3.0 * SURFACE_FIT_RADIUS_M,
    ):
        mask = (
            (distance_sq <= radius**2)
            & (normal > -0.10)
            & (np.abs(lateral) <= limit)
        )
        if anchor_height is not None:
            band = mask & (
                np.abs(normal - float(anchor_height)) <= SURFACE_ANCHOR_BAND_M
            )
            if int(np.count_nonzero(band)) >= SURFACE_FIT_MIN_POINTS:
                mask = band
        count = int(np.count_nonzero(mask))
        if count >= SURFACE_FIT_MIN_POINTS:
            break
    if mask is None or count < 6:
        return None, None, {"status": "missing", "point_count": count}

    patch = np.asarray(points, dtype=float)[mask]
    centroid = patch.mean(axis=0)
    centred = patch - centroid
    covariance = centred.T @ centred / patch.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    e3 = eigenvectors[:, 0]
    if float(np.dot(e3, frame.anterior)) < 0.0:
        e3 = -e3
    e1, e2 = eigenvectors[:, 2], eigenvectors[:, 1]
    x = centred @ e1
    y = centred @ e2
    z = centred @ e3
    design = np.column_stack(
        [np.ones_like(x), x, y, x * x, x * y, y * y]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if int(rank) < design.shape[1]:
        design = design[:, :3]
        coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
        if int(rank) < 3:
            return None, None, {"status": "degenerate", "point_count": count}
        coefficients = np.concatenate([coefficients, np.zeros(3)])
    # design may have been truncated to its linear part above; slice the
    # (zero-padded) coefficients to match before evaluating the residual.
    residual = design @ coefficients[: design.shape[1]] - z
    fit_rms = float(np.sqrt(np.mean(residual**2)))

    # Nominal point: (u, v) at the height of the patch point nearest to the
    # nominal (u, v).  Using the anterior-most height instead pulls lateral
    # targets (V5/V6) towards the patch centroid, because at the chest wall
    # the most anterior point sits at a much smaller |v|.
    nearest = int(np.argmin(distance_sq[mask]))
    n_guess = float(normal[mask][nearest])
    nominal_point = frame.from_frame(u, v, n_guess)
    offset = nominal_point - centroid
    x0 = float(offset @ e1)
    y0 = float(offset @ e2)

    def _surface(x, y):
        z = float(
            coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * y
            + coefficients[3] * x * x
            + coefficients[4] * x * y
            + coefficients[5] * y * y
        )
        return centroid + x * e1 + y * e2 + z * e3

    # Fixed-point iteration: the returned contact must sit at the NOMINAL
    # (u, v).  Evaluating the fit at the nominal point's local coordinates
    # leaves a frame-dependent residual (measured up to 12 mm on the steep
    # lateral wall), which showed up as a target error between two frames.
    # Note (measured, 2026-09-19): forcing the contact to the exact nominal
    # (u, v) via damped Newton on the local fit, and restricting the patch to
    # its front envelope, were both implemented and reverted: each fixed one
    # evaluation path while regressing another (M3b 6.05 -> 8.7 mm on the
    # depth cloud; V5 snapping 32 mm off after the envelope).  The direct
    # evaluation is kept; the residual (u, v) is reported in the snap info.
    z0 = float(
        coefficients[0]
        + coefficients[1] * x0
        + coefficients[2] * y0
        + coefficients[3] * x0 * x0
        + coefficients[4] * x0 * y0
        + coefficients[5] * y0 * y0
    )
    contact = centroid + x0 * e1 + y0 * e2 + z0 * e3
    normal_local = np.array(
        [
            -(coefficients[1] + coefficients[4] * y0 + 2.0 * coefficients[3] * x0),
            -(coefficients[2] + coefficients[4] * x0 + 2.0 * coefficients[5] * y0),
            1.0,
        ]
    )
    norm = float(np.linalg.norm(normal_local))
    if norm < 1e-9:
        return None, None, {"status": "degenerate", "point_count": count}
    normal_local /= norm
    normal_world = (
        normal_local[0] * e1 + normal_local[1] * e2 + normal_local[2] * e3
    )
    # Orient outward, away from the torso axis through the frame origin.  An
    # anterior-facing test is degenerate at the midaxillary line (V6), where
    # the surface normal is perpendicular to the anterior direction and the
    # sign would be arbitrary.
    axis_point = frame.origin + float(u) * frame.up
    radial = contact - axis_point
    if float(np.dot(normal_world, radial)) < 0.0:
        normal_world = -normal_world
    # Robustness guard: a local patch that spans a discontinuity (body edge,
    # arm gap) can fit a surface far from the actual skin.  When the fitted
    # contact departs from the nearest real patch point by more than the
    # patch radius, fall back to that point instead of trusting the fit.
    nearest_index = int(np.argmin(distance_sq[mask]))
    nearest_point = patch[nearest_index]
    if float(np.linalg.norm(contact - nearest_point)) > 0.06:
        contact = nearest_point
        nearest_normal, _ = estimate_surface_normal(
            points, contact, radius=NORMAL_FIT_RADIUS_M, orient_toward=None
        )
        if nearest_normal is not None:
            normal_world = np.asarray(nearest_normal, dtype=float)
        fit_rms = float(np.linalg.norm(contact - nearest_point))
        fallback = True
    else:
        fallback = False

    actual = frame.to_frame(contact)
    info = {
        "status": "ok",
        "point_count": count,
        "fit_order": 2 if coefficients[3] != 0.0 or coefficients[5] != 0.0 else 1,
        "fit_rms_m": fit_rms,
        "du_error_m": float(actual[0] - u),
        "dv_error_m": float(actual[1] - v),
        "nominal_offset_m": float(np.linalg.norm(nominal_point - contact)),
        "patch_fallback": bool(fallback),
        "normal_anterior_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal_world, frame.anterior)), -1.0, 1.0)
                )
            )
        ),
    }
    return contact, normal_world, info


def generate_v1_v6(
    landmarks: ChestLandmarks,
    frame: ChestFrame,
    points,
    rules: dict,
    spacing_m: float | None = None,
    prior=None,
    u_5ics_override: float | None = None,
) -> V1V6Result:
    """Generate the six precordial targets with full provenance.

    `spacing_m` overrides the configured intercostal spacing (used by the
    sensitivity sweep); `prior` (a fitted ChestSurfacePrior) anchors the
    surface snap when the cloud contains several sheets at the same (u, v);
    `u_5ics_override` replaces the population 5th-ICS regression with a row
    measured on the patient (contact probing, I4-a: see
    roboecg/target_localization/rib_probe.py).
    """
    anatomy = rules["anatomy"]
    snnd = float(anatomy["sternal_notch_to_nipple"]["value"])
    drop_rule = anatomy["fourth_to_fifth_ics"]
    coefficients = drop_rule["coefficients"]

    # The torso cannot be wider than the shoulders; this also excludes the arms
    # which lie alongside the body in the supine pose (depth clouds).
    v_shoulder = abs(float(frame.to_frame(landmarks.shoulder_left)[1]))

    u_notch = 0.0  # ChestFrame origin is the clavicle midpoint (notch proxy)
    u_4ics = u_notch - snnd

    if spacing_m is not None:
        vertical_drop = float(spacing_m)
        drop_source = "sensitivity override"
        width_m = float("nan")
    else:
        width_m = measure_torso_width(points, frame, half_width_limit=v_shoulder)
        if width_m > 0.0:
            vertical_drop = (
                float(coefficients["intercept_mm"])
                + float(coefficients["slope_mm_per_mm"]) * width_m * 1000.0
            ) / 1000.0
            drop_source = f"published_regression on measured torso width {width_m:.3f} m"
        else:
            vertical_drop = float(drop_rule["fallback_m"])
            drop_source = "published dataset mean (torso width unavailable)"
    if u_5ics_override is not None:
        # I4-a: the 5th-ICS row measured on the patient (contact probing)
        # replaces the population drop regression.
        u_5ics = float(u_5ics_override)
        drop_source = (
            f"probe_corrected: ICS5 {u_5ics:.4f} m from contact probing "
            "(roboecg/target_localization/rib_probe.py)"
        )
    else:
        u_5ics = u_4ics - vertical_drop

    v_parasternal = abs(float(frame.to_frame(landmarks.clavicle_left)[1]))
    clavicle_mid_left = 0.5 * (landmarks.clavicle_left + landmarks.shoulder_left)
    v_mcl = float(frame.to_frame(clavicle_mid_left)[1])
    v_midax = measure_lateral_extent(
        points, frame, u_5ics, half_width_limit=v_shoulder
    )
    # V5 (anterior axillary line) sits at a FITTED fraction of the V4->V6
    # lateral span, not at the midpoint: the chest wall wraps towards the
    # axilla, so the clinical V5 projects laterally close to V6.  The 25-model
    # GT has (V5 - V4)/(V6 - V4) = 0.736 +- 0.070 (leave-one-out V5 error
    # 2.5 mm vs 9.6 mm for the midpoint; the anterior arc midpoint is worse,
    # 15.8 mm).  The real PhysioNet patient gives 0.653, the same order.
    alpha_v5 = float(
        anatomy.get("anterior_axillary_line", {}).get(
            "fraction", V5_LATERAL_FRACTION_DEFAULT
        )
    )
    v_v5 = v_mcl + alpha_v5 * (v_midax - v_mcl)

    nominal = {
        "V1": (u_4ics, -v_parasternal),
        "V2": (u_4ics, +v_parasternal),
        "V4": (u_5ics, +v_mcl),
        "V5": (u_5ics, +v_v5),
        "V6": (u_5ics, +v_midax),
    }
    nominal["V3"] = (
        0.5 * (nominal["V2"][0] + nominal["V4"][0]),
        0.5 * (nominal["V2"][1] + nominal["V4"][1]),
    )

    vertical_provenance = {
        "notch_level": "asset_measurement: clavicle joint midpoint (sternal notch proxy)",
        "sternal_notch_to_nipple": (
            f"published_statistic {snnd:.3f} m, "
            f"{anatomy['sternal_notch_to_nipple']['citation']}"
        ),
        "nipple_fourth_ics": (
            f"published_statistic {anatomy['nipple_fourth_ics']['value']:.2f}, "
            f"{anatomy['nipple_fourth_ics']['citation']}"
        ),
        "intercostal_spacing": (
            drop_source
            if u_5ics_override is not None
            else (
                f"published_regression {vertical_drop:.4f} m ({drop_source}), "
                f"{drop_rule['citation']}; R2={drop_rule['fit_quality']}"
            )
        ),
        "torso_width_m": (None if np.isnan(width_m) else width_m),
        "u_4ics_m": u_4ics,
        "u_5ics_m": u_5ics,
    }
    horizontal_provenance = {
        "parasternal": "asset_measurement: |v| of sternoclavicular joint",
        "midclavicular": "asset_measurement: v of clavicle midpoint",
        "midaxillary": "asset_measurement: max|v| of chest contour at V4 level",
        "anterior_axillary": (
            f"published_regression: v_mcl + {alpha_v5:.4f} x (v_midax - v_mcl); "
            "fitted on 25 GT torso models (DOI 10.5281/zenodo.20086105), "
            "LOO V5 error 2.5 mm"
        ),
        "V3": "clinical_definition: midpoint(V2, V4)",
        "values_m": {
            "parasternal": v_parasternal,
            "midclavicular": v_mcl,
            "midaxillary": v_midax,
            "anterior_axillary": v_v5,
        },
    }

    targets = []
    for name, (u, v) in nominal.items():
        anchor = None
        if prior is not None:
            anchor = float(prior.height(u, v))
        point, normal, info = snap_to_surface(
            points, frame, u, v, v_limit=v_shoulder, anchor_height=anchor
        )
        if point is None:
            raise RuntimeError(
                f"cannot snap {name} to the chest surface at (u={u:.3f}, v={v:.3f})"
            )
        targets.append(
            ElectrodeTarget(
                name=name,
                position=np.asarray(point, dtype=float),
                normal=np.asarray(normal, dtype=float),
                frame_coords=np.array([u, v, float(frame.to_frame(point)[2])]),
                provenance={
                    "vertical": vertical_provenance,
                    "horizontal": horizontal_provenance,
                    "snap": info,
                },
            )
        )
    return V1V6Result(
        targets=tuple(targets),
        vertical=vertical_provenance,
        horizontal=horizontal_provenance,
        provenance={
            "rule_source": "configs/ecg_rules.yaml",
            "note": (
                "V4-V6 vertical position uses a published regression on the "
                "measured torso depth (independent GT: 25 torso models with real "
                "electrode positions); see docs/ECG_GT_VALIDATION.md."
            ),
        },
    )
