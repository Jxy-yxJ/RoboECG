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
