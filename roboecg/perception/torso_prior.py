"""Smooth chest-surface prior fitted to the body geometry.

M0 measured that chest surface normals deviate from a single global anterior
axis by up to 62 deg (lateral chest, clavicle region), so the thyroid-derived
"normal vs global prior <= 25 deg" rule cannot be used as-is.  This module fits
a low-order polynomial surface n = f(u, v) to the anterior chest of the asset
mesh and exposes the analytic normal at any (u, v).  The prior is therefore
position-dependent and comes from a *model of the body surface*, not from the
per-target measurement.

Provenance: asset_measurement (polynomial fit to the M_Medical_01/biped mesh);
the fit residual is reported with the model so its quality is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roboecg.target_localization.chest_frame import ChestFrame

CHEST_U_RANGE = (-0.35, 0.05)
CHEST_V_LIMIT = 0.25
CHEST_N_MIN = -0.05
FIT_GRID = 12
FIT_DEGREE = 4


@dataclass(frozen=True)
class ChestSurfacePrior:
    coefficients: np.ndarray
    u_range: tuple
    v_limit: float
    fit_rms_m: float
    samples: int
    degree: int = 2
    provenance: dict = field(default_factory=dict)

    def height(self, u: float, v: float) -> float:
        """Model surface height n at (u, v)."""
        return float(_basis(float(u), float(v), self.degree) @ self.coefficients)

    def normal_world(self, frame: ChestFrame, u: float, v: float) -> np.ndarray:
        """Analytic normal of the fitted surface in world coordinates."""
        step = 1e-4
        du = (
            self.height(float(u) + step, float(v))
            - self.height(float(u) - step, float(v))
        ) / (2.0 * step)
        dv = (
            self.height(float(u), float(v) + step)
            - self.height(float(u), float(v) - step)
        ) / (2.0 * step)
        normal = frame.anterior * 1.0 - frame.up * du - frame.lateral * dv
        return normal / (np.linalg.norm(normal) + 1e-12)

    def describe(self) -> dict:
        return {
            "degree": self.degree,
            "coefficients": self.coefficients.tolist(),
            "u_range_m": list(self.u_range),
            "v_limit_m": self.v_limit,
            "fit_rms_m": self.fit_rms_m,
            "samples": self.samples,
            "provenance": self.provenance,
        }


def _basis(u: float, v: float, degree: int) -> np.ndarray:
    """Bivariate monomial basis ordered by total degree: [1, u, v, u^2, uv, v^2, ...]."""
    terms = []
    for total in range(degree + 1):
        for i in range(total, -1, -1):
            j = total - i
            terms.append((u**i) * (v**j))
    return np.asarray(terms, dtype=float)


def fit_chest_surface_prior(
    points, frame: ChestFrame, degree: int = FIT_DEGREE, v_limit: float | None = None
) -> ChestSurfacePrior:
    """Fit n = f(u, v) to the anterior-most mesh point of each (u, v) cell."""
    points = np.asarray(points, dtype=float)
    delta = points - frame.origin
    along = delta @ frame.up
    lateral = delta @ frame.lateral
    normal = delta @ frame.anterior

    limit = CHEST_V_LIMIT if v_limit is None else float(v_limit)
    mask = (
        (along >= CHEST_U_RANGE[0])
        & (along <= CHEST_U_RANGE[1])
        & (np.abs(lateral) <= limit)
        & (normal >= CHEST_N_MIN)
    )
    if np.count_nonzero(mask) < 50:
        raise RuntimeError("not enough chest points to fit the surface prior")
    u = along[mask]
    v = lateral[mask]
    n = normal[mask]

    u_edges = np.linspace(CHEST_U_RANGE[0], CHEST_U_RANGE[1], FIT_GRID + 1)
    v_edges = np.linspace(-CHEST_V_LIMIT, CHEST_V_LIMIT, FIT_GRID + 1)
    rows = []
    targets = []
    for i in range(FIT_GRID):
        for j in range(FIT_GRID):
            cell = (
                (u >= u_edges[i])
                & (u < u_edges[i + 1])
                & (v >= v_edges[j])
                & (v < v_edges[j + 1])
            )
            if np.count_nonzero(cell) < 3:
                continue
            rows.append(
                _basis(
                    0.5 * (u_edges[i] + u_edges[i + 1]),
                    0.5 * (v_edges[j] + v_edges[j + 1]),
                    degree,
                )
            )
            targets.append(float(n[cell].max()))
    design = np.asarray(rows, dtype=float)
    values = np.asarray(targets, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = design @ coefficients - values
    rms = float(np.sqrt(np.mean(residual**2)))
    return ChestSurfacePrior(
        coefficients=coefficients,
        u_range=CHEST_U_RANGE,
        v_limit=CHEST_V_LIMIT,
        fit_rms_m=rms,
        samples=int(design.shape[0]),
        degree=degree,
        provenance={
            "source": "asset_measurement: polynomial fit to the anterior chest mesh",
            "model": f"bivariate polynomial of degree {degree} in (u, v)",
        },
    )
