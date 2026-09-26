"""M4 press planning: target sequence, approach/press/retreat, safety limits.

Pure logic (no Isaac): the sequence planner orders the six electrodes, and the
press planner turns each target into a waypoint chain along the measured
surface normal:

    transit (safe plane above the chest)
      -> approach (standoff)
      -> press   (commanded indentation, speed-limited)
      -> hold    (contact force target, dwell time)
      -> retreat (standoff)

Safety rules enforced here:
  * press depth <= max_press_depth_m and force <= force_limit_n (hard limits);
  * the press direction is the measured surface normal (the contact angle is a
    separate check owned by the fusion stage);
  * the force target comes from the published pressure x engineering contact
    area (configs/ecg_rules.yaml -> contact), never hand-written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from roboecg.coordinate_transform.frames import rotation_from_axes


@dataclass(frozen=True)
class PressSettings:
    approach_standoff_m: float = 0.06
    press_depth_m: float = 0.004
    press_speed_m_s: float = 0.01
    hold_s: float = 1.0
    retreat_m: float = 0.05
    force_target_n: float = 0.6
    force_limit_n: float = 1.5
    max_press_depth_m: float = 0.008
    transit_height_m: float = 0.30
    # Distance from the flange (tool0) to the electrode tip along the tool
    # axis; comes from the scene layout (engineering).
    electrode_offset_m: float = 0.088
    # Engineering contact model for the simulated skin (documented, not a
    # clinical number): the force rises linearly with indentation, calibrated
    # so that press_depth_m yields force_target_n.
    provenance: str = "engineering: press speeds/limits; force target derived from configs/ecg_rules.yaml contact section"

    @property
    def contact_stiffness_n_m(self) -> float:
        return self.force_target_n / max(self.press_depth_m, 1e-9)


def press_settings_from_rules(rules: dict) -> PressSettings:
    """Build settings, taking the force target from the rule config."""
    from roboecg.target_localization.ecg_rules import force_target_n

    target = force_target_n(rules)
    if target is None:
        raise ValueError("contact.force_target_n cannot be derived from the config")
    contact = rules.get("contact") or {}
    limit = (contact.get("force_limit_n") or {}).get("value")
    return PressSettings(
        force_target_n=float(target),
        force_limit_n=float(limit) if limit else 2.5 * float(target),
    )


def path_length(order, positions) -> float:
    return float(
        sum(
            np.linalg.norm(
                np.asarray(positions[order[i + 1]]) - np.asarray(positions[order[i]])
            )
            for i in range(len(order) - 1)
        )
    )


def plan_sequence(names, positions, start_position=None) -> dict:
    """Exact TSP ordering over the electrodes (n=6 -> 720 permutations).

    The transfer cost is the Euclidean distance between successive press
    standoff points; an optional start (the transit pose) anchors the tour.
    """
    names = list(names)
    if not names:
        return {"order": [], "length_m": 0.0, "brute_force": True}
    if len(names) > 8:
        raise ValueError("exact sequencing is limited to 8 targets")
    anchor = None if start_position is None else np.asarray(start_position, dtype=float)
    best_order, best_length = None, None
    for order in permutations(range(len(names))):
        length = 0.0
        previous = anchor
        if previous is not None:
            length += float(
                np.linalg.norm(np.asarray(positions[names[order[0]]]) - previous)
            )
        for i in range(len(order) - 1):
            length += float(
                np.linalg.norm(
                    np.asarray(positions[names[order[i + 1]]])
                    - np.asarray(positions[names[order[i]]])
                )
            )
        if best_length is None or length < best_length - 1e-12:
            best_order, best_length = order, length
    order_names = [names[i] for i in best_order]
    return {
        "order": order_names,
        "length_m": float(best_length),
        "brute_force": True,
        "start_position": None if anchor is None else anchor.tolist(),
    }


def plan_press(target, frame, settings: PressSettings) -> dict:
    """Waypoint chain for one electrode; raises if a hard limit is violated."""
    if settings.press_depth_m > settings.max_press_depth_m + 1e-12:
        raise ValueError(
            f"press depth {settings.press_depth_m} exceeds the hard limit "
            f"{settings.max_press_depth_m}"
        )
    if settings.force_target_n > settings.force_limit_n + 1e-12:
        raise ValueError(
            f"force target {settings.force_target_n} N exceeds the limit "
            f"{settings.force_limit_n} N"
        )
    normal = np.asarray(target.normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    contact = np.asarray(target.position, dtype=float)
    approach_axis = -normal
    rotation = rotation_from_axes(frame.lateral, approach_axis)

    transit = (
        np.asarray(frame.origin, dtype=float)
        + frame.anterior * settings.transit_height_m
    )
    # Waypoints are given as tool0 (flange) positions: the electrode tip is
    # `electrode_offset_m` in front of the flange along the approach axis.
    offset = float(settings.electrode_offset_m)
    standoff = contact + normal * (offset + settings.approach_standoff_m)
    pressed = contact + normal * (offset - settings.press_depth_m)
    retreat = contact + normal * (offset + settings.retreat_m)

    waypoints = [
        {
            "label": "standoff",
            "tool0_world": standoff.tolist(),
            "dwell_s": 0.0,
            "contact": False,
        },
        {
            "label": "press",
            "tool0_world": pressed.tolist(),
            "dwell_s": 0.0,
            "contact": True,
        },
        {
            "label": "hold",
            "tool0_world": pressed.tolist(),
            "dwell_s": float(settings.hold_s),
            "contact": True,
        },
        {
            "label": "retreat",
            "tool0_world": retreat.tolist(),
            "dwell_s": 0.0,
            "contact": False,
        },
    ]
    press_time = settings.press_depth_m / max(settings.press_speed_m_s, 1e-9)
    return {
        "target": target.name,
        "rotation_world": rotation.tolist(),
        "transit_tool0_world": transit.tolist(),
        "contact_world": contact.tolist(),
        "normal_world": normal.tolist(),
        "waypoints": waypoints,
        "press_depth_m": float(settings.press_depth_m),
        "force_target_n": float(settings.force_target_n),
        "force_limit_n": float(settings.force_limit_n),
        "electrode_offset_m": offset,
        "estimated_press_time_s": float(press_time),
        "estimated_cycle_time_s": float(
            2.0 * press_time + settings.hold_s
        ),
        "provenance": settings.provenance,
    }


def estimated_force_n(indentation_m: float, settings: PressSettings) -> float:
    """Engineering contact model: linear skin stiffness (documented)."""
    return float(settings.contact_stiffness_n_m * max(0.0, indentation_m))


def check_contact(indentation_m: float, settings: PressSettings) -> dict:
    """Verify a measured indentation against the depth/force limits."""
    force = estimated_force_n(indentation_m, settings)
    depth_ok = indentation_m <= settings.max_press_depth_m + 1e-9
    force_ok = force <= settings.force_limit_n + 1e-9
    return {
        "indentation_m": float(indentation_m),
        "force_n": force,
        "depth_ok": bool(depth_ok),
        "force_ok": bool(force_ok),
        "violation": None if (depth_ok and force_ok) else (
            "depth" if not depth_ok else "force"
        ),
    }


def plan_cycle(targets, frame, rules, settings: PressSettings | None = None) -> dict:
    """Sequence + per-target press plans for a full placement cycle."""
    settings = settings or press_settings_from_rules(rules)
    by_name = {t.name: t for t in targets}
    contacts = {name: t.position.tolist() for name, t in by_name.items()}
    transit = (
        np.asarray(frame.origin, dtype=float)
        + frame.anterior * settings.transit_height_m
    )
    sequence = plan_sequence(
        list(by_name.keys()), contacts, start_position=transit
    )
    presses = [
        plan_press(by_name[name], frame, settings) for name in sequence["order"]
    ]
    return {
        "settings": {
            "approach_standoff_m": settings.approach_standoff_m,
            "press_depth_m": settings.press_depth_m,
            "press_speed_m_s": settings.press_speed_m_s,
            "hold_s": settings.hold_s,
            "retreat_m": settings.retreat_m,
            "force_target_n": settings.force_target_n,
            "force_limit_n": settings.force_limit_n,
            "max_press_depth_m": settings.max_press_depth_m,
            "transit_height_m": settings.transit_height_m,
            "provenance": settings.provenance,
        },
        "sequence": sequence,
        "presses": presses,
        "estimated_cycle_time_s": float(
            sum(p["estimated_cycle_time_s"] for p in presses)
        ),
    }
