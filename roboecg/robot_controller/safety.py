# Copied from farus_thyroid_isaac/farus/robot_controller/safety.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Robot-human clearance geometry (pure numpy).

Collision proxies for the patient are derived from the skeleton joints
(torso/neck/head/arms capsules) and robot links are approximated by spheres
at the URDF link frames.  These checks run before and during every reach so a
penetration can never be reported as success.
"""
from __future__ import annotations

import numpy as np

from roboecg.coordinate_transform.frames import point_segment_distance

# Upper-spine (Spine3) -> pelvis distance of the nominal biped_demo patient at
# scale 1.0 (M0 landmark audit: `chest_to_pelvis_m` = 0.2228 m, measured on the
# same rig).  NOTE: the capsule's `chest_top` joint is the *upper chest*
# ("Chest"); the scale must be measured on a joint pair whose nominal distance
# is audited (Spine3 -> Pelvis), not on Chest -> Pelvis.
NOMINAL_SPINE3_TO_PELVIS_M = 0.2228

CLEARANCE_MARGIN_M = 0.02

ROBOT_LINK_RADII = {
    "base_link": 0.05,
    "shoulder_link": 0.05,
    "upper_arm_link": 0.045,
    "forearm_link": 0.04,
    "wrist_1_link": 0.035,
    "wrist_2_link": 0.035,
    "wrist_3_link": 0.035,
    "flange": 0.03,
    "tool0": 0.02,
}

# Physical UR3 links span two frames; sampling along each segment catches
# collisions that a single sphere per joint frame would miss.
ROBOT_LINK_SEGMENTS = (
    ("base_link", "shoulder_link", 0.06),
    ("shoulder_link", "upper_arm_link", 0.055),
    ("upper_arm_link", "forearm_link", 0.05),
    ("forearm_link", "wrist_1_link", 0.045),
    ("wrist_1_link", "wrist_2_link", 0.04),
    ("wrist_2_link", "wrist_3_link", 0.04),
    ("wrist_3_link", "tool0", 0.03),
)

# ECG adaptation (2026-09-18): the thyroid scene's hard-coded pedestal box was
# removed.  Scene boxes (table, pedestal) are now passed in explicitly through
# `box_clearance`, so this module stays scene-agnostic.


def point_box_distance(point, center, half) -> float:
    delta = np.abs(np.asarray(point, dtype=float) - np.asarray(center, dtype=float)) - np.asarray(half, dtype=float)
    return float(np.linalg.norm(np.maximum(delta, 0.0)))


def leaf_joint(joints, *candidates) -> np.ndarray:
    """First matching joint position among candidate leaf names.

    Candidate lists keep the collision proxies rig-agnostic (M_Medical_01 uses
    Spine02/NeckTwist01/Upperarm, the bare-body biped uses Spine3/Neck1/UpArm).
    """
    for candidate in candidates:
        for name, position in joints.items():
            if name.rsplit("/", 1)[-1] == candidate:
                return np.asarray(position, dtype=float)
    raise KeyError(f"none of {candidates} found in joints")


def body_scale_from_joints(joints) -> float:
    """Patient body scale relative to the nominal biped_demo (M0 audit).

    Capsule radii are engineering proxies for the nominal adult body; a
    uniformly scaled patient (body family 0.90-1.10) must scale its collision
    proxies with it, otherwise the fixed radii are over-conservative for small
    bodies (M0 finding: clearance < 2 cm at scale 0.90/0.95).
    """
    pelvis = leaf_joint(joints, "Pelvis")
    spine_upper = leaf_joint(joints, "Spine3", "Spine02", "Chest")
    length = float(np.linalg.norm(np.asarray(spine_upper) - np.asarray(pelvis)))
    if length <= 0.0:
        return 1.0
    return length / NOMINAL_SPINE3_TO_PELVIS_M


def body_capsules(joints, scale: float | None = None):
    """Simplified patient collision proxies: (name, start, end, radius).

    The torso is split at the chest top because a single capsule from the
    pelvis to the neck base would be far too wide at the neck.  Radii are
    engineering proxies carried over from the thyroid reproduction, scaled by
    the measured patient size (`scale`, defaulting to the pelvis->neck base
    ratio); they are rig-agnostic approximations, not anatomical measurements.
    """
    if scale is None:
        scale = body_scale_from_joints(joints)
    scale = float(np.clip(scale, 0.7, 1.4))
    pelvis = leaf_joint(joints, "Pelvis")
    chest_top = leaf_joint(joints, "Spine02", "Chest", "Spine3")
    neck_base = leaf_joint(joints, "NeckTwist01", "Neck1")
    neck_top = leaf_joint(joints, "NeckTwist02", "Neck2")
    try:
        head = 0.5 * (leaf_joint(joints, "L_Eye") + leaf_joint(joints, "R_Eye"))
    except KeyError:
        head = leaf_joint(joints, "Head")
    capsules = [
        ("torso", pelvis, chest_top, 0.14 * scale),
        ("upper_chest", chest_top, neck_base, 0.09 * scale),
        ("neck", neck_base, neck_top, 0.06 * scale),
        ("head", head, head, 0.10 * scale),
    ]
    for side in ("L", "R"):
        shoulder = leaf_joint(joints, f"{side}_Upperarm", f"{side}_UpArm")
        elbow = leaf_joint(joints, f"{side}_Forearm", f"{side}_LoArm")
        wrist = leaf_joint(joints, f"{side}_Hand", f"{side}_Wrist")
        capsules.append(
            (f"upper_arm_{side.lower()}", shoulder, elbow, 0.055 * scale)
        )
        capsules.append(
            (f"forearm_{side.lower()}", elbow, wrist, 0.045 * scale)
        )
    return capsules


def robot_sample_points(link_positions, samples_per_meter=40.0, min_samples=2):
    """Yield (link_name, world_point, link_radius) along each robot link."""
    for start_name, end_name, radius in ROBOT_LINK_SEGMENTS:
        start = link_positions.get(start_name)
        end = link_positions.get(end_name)
        if start is None or end is None:
            continue
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(min_samples, int(np.ceil(length * samples_per_meter)))
        for index in range(count + 1):
            point = start + (end - start) * (index / count)
            yield start_name, point, radius


def link_clearance(link_positions, capsules):
    """Minimum signed clearance over all robot link samples and capsules.

    Returns (clearance_m, link_name, capsule_name); negative means penetration.
    """
    worst = None
    for link, point, link_radius in robot_sample_points(link_positions):
        for name, start, end, body_radius in capsules:
            clearance = (
                point_segment_distance(point, start, end)
                - body_radius
                - link_radius
            )
            if worst is None or clearance < worst[0]:
                worst = (clearance, link, name)
    if worst is None:
        raise KeyError("link_positions is missing every robot link segment")
    return worst


def box_clearance(link_positions, center, half) -> float:
    """Minimum distance to a scene box (table/pedestal); 0 means contact.

    Base contact is allowed, so the `base_link` samples are excluded.
    """
    return min(
        point_box_distance(point, center, half)
        for link, point, _ in robot_sample_points(link_positions)
        if link != "base_link"
    )
