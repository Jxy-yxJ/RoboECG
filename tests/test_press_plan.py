"""Pure-logic tests for M4 press planning (sequence + press waypoints)."""
from __future__ import annotations

import numpy as np
import pytest

from roboecg.robot_controller.press_plan import (
    PressSettings,
    check_contact,
    estimated_force_n,
    plan_cycle,
    plan_press,
    plan_sequence,
    press_settings_from_rules,
)
from roboecg.target_localization.chest_frame import ChestFrame
from roboecg.target_localization.ecg import ElectrodeTarget
from roboecg.target_localization.ecg_rules import force_target_n, load_ecg_rules


def make_frame() -> ChestFrame:
    return ChestFrame(
        origin=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, 0.0, 1.0]),
        lateral=np.array([0.0, 1.0, 0.0]),
        anterior=np.array([1.0, 0.0, 0.0]),
        provenance={"source": "test"},
    )


def make_target(name, position, normal) -> ElectrodeTarget:
    return ElectrodeTarget(
        name=name,
        position=np.asarray(position, dtype=float),
        normal=np.asarray(normal, dtype=float),
        frame_coords=np.zeros(3),
    )


def test_sequence_is_the_shortest_tour():
    positions = {
        "A": [0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0],
        "C": [0.0, 2.0, 0.0],
        "D": [0.0, 3.0, 0.0],
    }
    result = plan_sequence(list(positions), positions)
    assert result["order"] in (["A", "B", "C", "D"], ["D", "C", "B", "A"])
    assert result["length_m"] == pytest.approx(3.0)
    # a start anchor flips the direction to reach the nearer end first
    anchored = plan_sequence(list(positions), positions, start_position=[0.0, 3.2, 0.0])
    assert anchored["order"][0] == "D"


def test_press_plan_follows_the_normal_and_respects_limits():
    frame = make_frame()
    settings = PressSettings(press_depth_m=0.004, force_target_n=0.6)
    target = make_target("V1", [0.08, 0.0, 1.20], [1.0, 0.0, 0.0])
    plan = plan_press(target, frame, settings)
    labels = [w["label"] for w in plan["waypoints"]]
    assert labels == ["standoff", "press", "hold", "retreat"]
    standoff = np.asarray(plan["waypoints"][0]["tool0_world"])
    pressed = np.asarray(plan["waypoints"][1]["tool0_world"])
    retreat = np.asarray(plan["waypoints"][3]["tool0_world"])
    normal = np.asarray(plan["normal_world"])
    offset = settings.electrode_offset_m
    # tool0 sits `offset` in front of the electrode tip along the approach axis
    assert np.dot(standoff - target.position, normal) == pytest.approx(
        offset + settings.approach_standoff_m
    )
    assert np.dot(pressed - target.position, normal) == pytest.approx(
        offset - settings.press_depth_m
    )
    assert np.dot(retreat - target.position, normal) == pytest.approx(
        offset + settings.retreat_m
    )
    # the electrode tip (flange - offset along the tool axis) reaches the
    # commanded indentation
    tip = pressed - normal * offset
    assert np.dot(tip - target.position, normal) == pytest.approx(
        -settings.press_depth_m
    )
    # the tool axis points along -normal (pressing into the surface)
    rotation = np.asarray(plan["rotation_world"])
    assert np.dot(rotation[:, 2], -normal) == pytest.approx(1.0)


def test_press_plan_rejects_limit_violations():
    frame = make_frame()
    target = make_target("V1", [0.08, 0.0, 1.20], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        plan_press(target, frame, PressSettings(press_depth_m=0.02))
    with pytest.raises(ValueError):
        plan_press(
            target, frame, PressSettings(force_target_n=5.0, force_limit_n=1.5)
        )


def test_force_model_and_contact_check():
    settings = PressSettings(press_depth_m=0.004, force_target_n=0.6, force_limit_n=1.5)
    assert settings.contact_stiffness_n_m == pytest.approx(150.0)
    assert estimated_force_n(0.004, settings) == pytest.approx(0.6)
    ok = check_contact(0.004, settings)
    assert ok["depth_ok"] and ok["force_ok"] and ok["violation"] is None
    deep = check_contact(0.02, settings)
    assert not deep["depth_ok"] and deep["violation"] == "depth"


def test_force_target_comes_from_the_config():
    rules = load_ecg_rules()
    target = force_target_n(rules)
    assert target is not None
    settings = press_settings_from_rules(rules)
    assert settings.force_target_n == pytest.approx(float(target))
    assert settings.force_limit_n >= settings.force_target_n


def test_plan_cycle_orders_all_six_targets():
    rules = load_ecg_rules()
    frame = make_frame()
    targets = [
        make_target(
            name,
            [0.08, 0.02 * i, 1.20 - 0.02 * i],
            [1.0, 0.1 * i, 0.0],
        )
        for i, name in enumerate(("V1", "V2", "V3", "V4", "V5", "V6"))
    ]
    cycle = plan_cycle(targets, frame, rules)
    assert sorted(cycle["sequence"]["order"]) == sorted(
        ["V1", "V2", "V3", "V4", "V5", "V6"]
    )
    assert len(cycle["presses"]) == 6
    assert cycle["estimated_cycle_time_s"] > 0.0
    assert cycle["settings"]["force_target_n"] == pytest.approx(0.6)
