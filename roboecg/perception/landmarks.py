# Copied from farus_thyroid_isaac/farus/perception/landmarks.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Body landmark selection from skeleton joint positions (pure numpy).

The M1 landmark set follows the FARUS coarse localization description:
"we estimated the thyroid location using the neck and head skeleton joint
points" (Su et al. 2024, Methods "Scan planning").  The paper does not name
the exact joints, so this mapping to the official Isaac human rig is our
documented choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANDMARK_JOINTS = {
    "head": "Head",
    "neck_top": "NeckTwist02",
    "neck_base": "NeckTwist01",
    "chest": "Spine02",
    "spine": "Spine01",
    "clavicle_left": "L_Clavicle",
    "clavicle_right": "R_Clavicle",
}


@dataclass(frozen=True)
class BodyLandmarks:
    head: np.ndarray
    neck_top: np.ndarray
    neck_base: np.ndarray
    chest: np.ndarray
    spine: np.ndarray
    clavicle_left: np.ndarray
    clavicle_right: np.ndarray


def read_landmarks(joint_positions: dict) -> BodyLandmarks:
    """Select landmarks from a {joint_path: position} mapping."""
    by_leaf: dict[str, np.ndarray] = {}
    for name, position in joint_positions.items():
        leaf = name.rsplit("/", 1)[-1]
        by_leaf.setdefault(leaf, np.asarray(position, dtype=float))

    missing = [leaf for leaf in LANDMARK_JOINTS.values() if leaf not in by_leaf]
    if missing:
        raise KeyError(f"missing landmark joints: {missing}")

    values = {key: by_leaf[leaf] for key, leaf in LANDMARK_JOINTS.items()}
    return BodyLandmarks(**values)
