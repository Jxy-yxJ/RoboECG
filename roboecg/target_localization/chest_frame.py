"""Chest coordinate frame built from measured landmarks.

The frame contains no clinical constants: it is a rigid right-handed basis
derived from three measured directions on the asset (or, on the perceptual
path, from landmarks output by a detector).  Clinical rules (which rib level
or which vertical line an electrode sits on) are applied *inside* this frame
by `roboecg.target_localization.ecg`.

Axes:
    up       toward the head
    lateral  toward the patient's left
    anterior out of the chest (right-hand rule: lateral x up)

Sign convention: ``anterior = lateral x up`` is anatomically anterior for a
right-handed frame, but the sign of a cross product is easy to flip, so
``build_chest_frame`` requires an explicit ``anterior_hint`` (e.g. the world
up direction of a supine patient or the camera-to-chest direction) and flips
the computed axis if it disagrees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roboecg.coordinate_transform.frames import normalize
from roboecg.perception.chest_landmarks import ChestLandmarks


@dataclass(frozen=True)
class ChestFrame:
    origin: np.ndarray
    up: np.ndarray
    lateral: np.ndarray
    anterior: np.ndarray
    provenance: dict

    def to_frame(self, point) -> np.ndarray:
        """World point -> (u, v, n) chest-frame coordinates."""
        delta = np.asarray(point, dtype=float) - self.origin
        return np.array(
            [
                float(np.dot(delta, self.up)),
                float(np.dot(delta, self.lateral)),
                float(np.dot(delta, self.anterior)),
            ]
        )

    def from_frame(self, u: float, v: float, n: float = 0.0) -> np.ndarray:
        """(u, v, n) chest-frame coordinates -> world point."""
        return (
            self.origin
            + self.up * float(u)
            + self.lateral * float(v)
            + self.anterior * float(n)
        )

    def rotation(self) -> np.ndarray:
        """Rotation matrix whose columns are (lateral, up, anterior)."""
        return np.column_stack([self.lateral, self.up, self.anterior])

    def describe(self) -> dict:
        return {
            "origin_world": self.origin.tolist(),
            "up_world": self.up.tolist(),
            "lateral_world": self.lateral.tolist(),
            "anterior_world": self.anterior.tolist(),
            "provenance": self.provenance,
        }


def build_chest_frame(landmarks: ChestLandmarks, anterior_hint) -> ChestFrame:
    """Build the chest frame from measured landmarks.

    - up:       Spine02 -> NeckTwist01 (chest long axis, toward the head)
    - lateral:  R_Clavicle -> L_Clavicle, orthogonalized against up
    - anterior: lateral x up, sign fixed by `anterior_hint`
    - origin:   midpoint of the two sternoclavicular joints (rig clavicle
                joints); proxy for the top of the sternum, recorded as an
                asset measurement with its caveat.

    Raises ValueError when the landmarks are degenerate (coincident points or
    a hint perpendicular to the computed anterior axis).
    """
    up = normalize(landmarks.neck_base - landmarks.chest)
    lateral = np.asarray(landmarks.clavicle_left, dtype=float) - np.asarray(
        landmarks.clavicle_right, dtype=float
    )
    lateral = normalize(lateral - np.dot(lateral, up) * up)

    anterior = normalize(np.cross(lateral, up))
    hint = normalize(anterior_hint)
    if abs(float(np.dot(hint, anterior))) < 1e-6:
        raise ValueError("anterior_hint is perpendicular to the chest anterior axis")
    if float(np.dot(hint, anterior)) < 0.0:
        anterior = -anterior

    origin = 0.5 * (
        np.asarray(landmarks.clavicle_left, dtype=float)
        + np.asarray(landmarks.clavicle_right, dtype=float)
    )

    return ChestFrame(
        origin=origin,
        up=up,
        lateral=lateral,
        anterior=anterior,
        provenance={
            "up": "asset_measurement: Spine02 -> NeckTwist01",
            "lateral": "asset_measurement: R_Clavicle -> L_Clavicle (orthogonalized)",
            "anterior": "clinical_definition: lateral x up; sign from anterior_hint",
            "origin": (
                "asset_measurement: midpoint of L/R_Clavicle (sternoclavicular "
                "joints); proxy for the top of the sternum"
            ),
            "caveat": (
                "The rig chest bones are skinning joints, not individual ribs; "
                "clinical rib levels are applied with the documented uncertainty "
                "in configs/ecg_rules.yaml."
            ),
        },
    )


def near_layer_mask(
    normal: np.ndarray, bin_m: float = 0.02, min_layer_gap_m: float = 0.06,
) -> np.ndarray:
    """Boolean mask of the near (chest) depth layer, by the n-histogram gap.

    A depth cloud from the overhead camera contains a second, far layer beyond
    the body silhouette (the table, ~17 cm below the chest surface); the two
    layers are separated by an empty band in the surface-height histogram.  A
    sparse mesh without a far layer has no such gap and the mask keeps
    everything.
    """
    normal = np.asarray(normal, dtype=float)
    if normal.size < 8:
        return np.ones(normal.size, dtype=bool)
    low, high = float(normal.min()), float(normal.max())
    n_bins = max(int(np.ceil((high - low) / bin_m)), 1)
    counts, edges = np.histogram(normal, bins=n_bins)
    populated = np.where(counts > 0)[0]
    if populated.size < 2:
        return np.ones(normal.size, dtype=bool)
    gaps = np.diff(populated)
    gap_bins = int(np.ceil(min_layer_gap_m / bin_m))
    if int(gaps.max()) >= gap_bins:
        last_near_bin = populated[int(np.argmax(gaps))]
        threshold = float(edges[last_near_bin + 1])
        return normal >= threshold
    return np.ones(normal.size, dtype=bool)


def cloud_blended_frame(
    frame: ChestFrame, points, torso_lateral_limit: float = 0.30,
    torso_u_range: tuple = (-0.45, 0.25),
) -> ChestFrame:
    """Chest frame whose axis DIRECTIONS come from the torso cloud (PCA).

    Landmark-based axes amplify the detector's out-of-domain landmark errors
    (measured on real SSM torsos: a 31 deg lateral tilt that displaces the
    V5/V6 rows off the body).  The cloud's principal axes -- up: longest,
    lateral: second (the torso and a rest-pose arm are wider than deep),
    anterior: lateral x up -- are robust to a few noisy landmarks; the signs
    and the origin still come from the landmark frame.  Falls back to the
    landmark frame when the cloud is too sparse or degenerate.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 200:
        return frame
    up0 = np.asarray(frame.up, dtype=float)
    lateral0 = np.asarray(frame.lateral, dtype=float)
    anterior0 = np.asarray(frame.anterior, dtype=float)
    rel = points - np.asarray(frame.origin, dtype=float)
    along = rel @ up0
    lateral = rel @ lateral0
    normal = rel @ anterior0
    region = (
        (np.abs(lateral) <= torso_lateral_limit)
        & (along >= torso_u_range[0])
        & (along <= torso_u_range[1])
    )
    pts = points[region]
    if pts.shape[0] < 200:
        return frame
    near = near_layer_mask(normal[region])
    if int(np.count_nonzero(near)) >= 200:
        pts = pts[near]
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    up = vt[0] if float(vt[0] @ up0) >= 0.0 else -vt[0]
    lateral_axis = vt[1] - float(vt[1] @ up) * up
    norm = float(np.linalg.norm(lateral_axis))
    if norm < 1e-6:
        return frame
    lateral_axis = lateral_axis / norm
    if float(lateral_axis @ lateral0) < 0.0:
        lateral_axis = -lateral_axis
    anterior = np.cross(lateral_axis, up)
    anterior = anterior / (np.linalg.norm(anterior) + 1e-12)
    if float(anterior @ anterior0) < 0.0:
        anterior = -anterior
    provenance = dict(frame.provenance)
    provenance["cloud_blended"] = (
        "axis directions from torso-cloud PCA (up: longest, lateral: second, "
        f"anterior: lateral x up) over {int(pts.shape[0])} points; signs and "
        "origin from the landmark frame"
    )
    return ChestFrame(
        origin=frame.origin, up=up, lateral=lateral_axis, anterior=anterior,
        provenance=provenance,
    )
