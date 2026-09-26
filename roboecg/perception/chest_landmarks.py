"""Chest landmarks read from the official bare-body ``biped_demo`` rig.

Provenance policy (project rule): every anatomical quantity must be traceable.
The landmarks below are *measured from the simulation asset* (rig joint world
positions), not clinical constants.  They are therefore labelled
``asset_measurement`` and may only be used as simulation ground truth; the
perceptual path (M3a/M3b) must estimate them from RGB-D.

Asset choice (2026-09-18): the earlier ``M_Medical_01`` character wears a lab
coat and scrub shirt, and its skin mesh has no torso surface under them, so a
bare chest cannot be obtained from it.  ``biped_demo`` (Isaac 5.0, also usable
in 6.0) is an official bare-body mesh (single 16.8k-point mesh) with an
81-joint skeleton, which is medically correct for ECG electrode placement.

The rig has no breast or rib joints (``Spine1/3``, ``Chest``, ``Neck1/2``,
``L/R_Clavicle``, ``L/R_UpArm`` only).  Intercostal levels therefore *must*
come from measurement or a learned model; the published nipple/4th-ICS
statistic (DOI 10.1097/00006534-200112000-00015) is kept in the rules config
as a population prior with its uncertainty, not as a fixed position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

# Joint tokens that exist in the biped_demo rig (verified in M0).
CHEST_LANDMARK_JOINTS = {
    "spine_lower": "Spine1",
    "spine_mid": "Spine2",
    "chest": "Spine3",
    "upper_chest": "Chest",
    "neck_base": "Neck1",
    "neck_top": "Neck2",
    "clavicle_left": "L_Clavicle",
    "clavicle_right": "R_Clavicle",
    "shoulder_left": "L_UpArm",
    "shoulder_right": "R_UpArm",
    "pelvis": "Pelvis",
}

LANDMARK_PROVENANCE: ClassVar[str] = (
    "asset_measurement: biped_demo rig joint world positions (simulation GT)"
)


@dataclass(frozen=True)
class ChestLandmarks:
    spine_lower: np.ndarray
    spine_mid: np.ndarray
    chest: np.ndarray
    upper_chest: np.ndarray
    neck_base: np.ndarray
    neck_top: np.ndarray
    clavicle_left: np.ndarray
    clavicle_right: np.ndarray
    shoulder_left: np.ndarray
    shoulder_right: np.ndarray
    pelvis: np.ndarray
    provenance: str = LANDMARK_PROVENANCE
    joint_positions: dict = field(default_factory=dict, repr=False, compare=False)


def read_chest_landmarks(joint_positions: dict) -> ChestLandmarks:
    """Select chest landmarks from a {joint_path: position} mapping."""
    by_leaf: dict[str, np.ndarray] = {}
    for name, position in joint_positions.items():
        leaf = name.rsplit("/", 1)[-1]
        by_leaf.setdefault(leaf, np.asarray(position, dtype=float))

    missing = [
        leaf for leaf in CHEST_LANDMARK_JOINTS.values() if leaf not in by_leaf
    ]
    if missing:
        raise KeyError(f"missing chest landmark joints: {missing}")

    values = {key: by_leaf[leaf] for key, leaf in CHEST_LANDMARK_JOINTS.items()}
    return ChestLandmarks(
        **values,
        joint_positions={
            leaf: by_leaf[leaf].tolist()
            for leaf in CHEST_LANDMARK_JOINTS.values()
        },
    )


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def audit_chest_landmarks(landmarks: ChestLandmarks) -> dict:
    """Measure the asset's chest geometry (no clinical interpretation).

    These are raw asset measurements intended for the M0 findings document;
    they are *not* clinical statistics and must not be quoted as such.
    """
    clavicle_mid_left = 0.5 * (landmarks.clavicle_left + landmarks.shoulder_left)
    clavicle_mid_right = 0.5 * (landmarks.clavicle_right + landmarks.shoulder_right)
    return {
        "provenance": LANDMARK_PROVENANCE,
        "clavicle_length_left_m": _distance(
            landmarks.clavicle_left, landmarks.shoulder_left
        ),
        "clavicle_length_right_m": _distance(
            landmarks.clavicle_right, landmarks.shoulder_right
        ),
        "shoulder_width_m": _distance(
            landmarks.shoulder_left, landmarks.shoulder_right
        ),
        "clavicle_separation_m": _distance(
            landmarks.clavicle_left, landmarks.clavicle_right
        ),
        "clavicle_midpoint_left_world": clavicle_mid_left.tolist(),
        "clavicle_midpoint_right_world": clavicle_mid_right.tolist(),
        "neck_base_to_upper_chest_m": _distance(
            landmarks.neck_base, landmarks.upper_chest
        ),
        "spine_lower_to_upper_chest_m": _distance(
            landmarks.spine_lower, landmarks.upper_chest
        ),
        "chest_to_pelvis_m": _distance(landmarks.chest, landmarks.pelvis),
        "note": (
            "Raw asset measurements from the bare-body biped_demo rig. The rig "
            "has no breast or rib joints, so intercostal levels are not "
            "available here by construction; they must be measured or learned."
        ),
    }
