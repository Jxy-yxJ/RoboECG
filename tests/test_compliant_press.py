"""Tests for the compliant (force-feedback) press simulation."""
from __future__ import annotations

import numpy as np

from roboecg.robot_controller.compliant_press import (
    CompliantPressConfig,
    run_scenarios,
    simulate_hold,
)
from roboecg.robot_controller.press_plan import PressSettings


def make_settings() -> PressSettings:
    return PressSettings()


def test_position_control_reproduces_v2_breathing_violation():
    """Fixed-depth press: 4 + 8 mm = 12 mm indentation, 1.8 N force."""
    result = simulate_hold(
        make_settings(), "position", 150.0, CompliantPressConfig()
    )
    metrics = result["metrics"]
    assert np.isclose(metrics["max_indentation_m"], 0.012, atol=1e-6)
    assert np.isclose(metrics["max_force_n"], 1.8, atol=5e-4)
    assert metrics["violation"] == "depth"


def test_compliant_holds_force_within_caps_at_nominal_stiffness():
    settings = make_settings()
    result = simulate_hold(settings, "compliant", 150.0, CompliantPressConfig())
    metrics = result["metrics"]
    assert metrics["violation"] is None
    assert metrics["force_rmse_n"] < 0.15
    assert metrics["max_force_n"] < 0.8
    assert metrics["max_indentation_m"] <= settings.max_press_depth_m + 1e-9


def test_compliant_respects_depth_cap_on_soft_chest():
    """A chest softer than the cap allows: depth-capped, force deficit flagged."""
    settings = make_settings()
    result = simulate_hold(settings, "compliant", 50.0, CompliantPressConfig())
    metrics = result["metrics"]
    assert metrics["violation"] is None
    assert np.isclose(
        metrics["max_indentation_m"], settings.max_press_depth_m, atol=1e-6
    )
    assert metrics["max_force_n"] < settings.force_target_n
    assert metrics["depth_saturated_fraction"] > 0.0


def test_compliant_stable_with_stiffness_mismatch():
    """k_est wrong by up to 4x must stay stable and converge to the target."""
    for k_est in (75.0, 300.0, 600.0):
        config = CompliantPressConfig(k_est_n_m=k_est)
        result = simulate_hold(make_settings(), "compliant", 150.0, config)
        metrics = result["metrics"]
        assert metrics["violation"] is None, k_est
        assert metrics["force_rmse_n"] < 0.3, k_est
        final_force = float(result["force_n"][-1])
        assert abs(final_force - 0.6) < 0.1, k_est


def test_compliant_retracts_against_surface_underestimation():
    """5 mm perception error on a stiff chest: retract, never over-press."""
    config = CompliantPressConfig(surface_error_m=0.005)
    result = simulate_hold(make_settings(), "compliant", 2000.0, config)
    metrics = result["metrics"]
    assert metrics["violation"] is None
    assert abs(float(result["force_n"][-1]) - 0.6) < 0.05
    final_cmd = float(result["indentation_cmd_m"][-1])
    assert final_cmd < -0.003, "must retract above the estimated surface"
    position = simulate_hold(make_settings(), "position", 2000.0, config)
    assert position["metrics"]["violation"] is not None


def test_scenario_grid_outcomes():
    report = run_scenarios(make_settings())
    by_tag = {s["tag"]: s for s in report["scenarios"]}
    compliant = [s for s in report["scenarios"] if s["mode"] == "compliant"]
    assert compliant
    unexpected = [s["tag"] for s in compliant if s["violation"] is not None]
    assert not unexpected, unexpected
    position = [s for s in report["scenarios"] if s["mode"] == "position"]
    assert position
    missed = [s["tag"] for s in position if s["violation"] is None]
    assert not missed, missed
    assert by_tag["compliant_k50"]["max_force_n"] < 0.6
    assert by_tag["position_k150"]["violation"] == "depth"
