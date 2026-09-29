"""Compliant (force-feedback) press for the electrode hold phase.

The v2 breathing report showed that a fixed position press (commanded
indentation = ``press_depth_m`` below the *measured* surface) violates the
depth cap and the force cap when the chest wall moves: with a +-8 mm, 15 /min
breathing motion the worst indentation is 4 + 8 = 12 mm (> 8 mm cap) and the
force is k * 12 mm = 1.8 N (> 1.5 N cap) under the documented engineering
contact model.  A perception error of the surface height makes it worse
(position control turns it directly into extra indentation).

This module simulates a 1-D compliant press loop against the *same* contact
model and quantifies the improvement:

* position mode:  indentation command fixed at the planned depth (v2
  behaviour).  There is no surface tracking during the hold, so the true
  indentation is ``press_depth + surface_error + breathing`` whenever positive.
* compliant mode: the indentation command is integrated from the estimated
  force error (``i_cmd += K * (F_target - F_measured) * dt``) with the gain
  taken from a desired closed-loop time constant at an assumed stiffness
  ``k_est``; the command is clamped to ``[retract_limit, depth_cap]`` so the
  tool may retract when the surface is higher than believed.  The design
  assumes the depth camera keeps tracking the chest surface during the hold
  (``residual_breathing_fraction`` = 1 - tracking quality); the force loop also
  rejects the residual via its own bandwidth.

True indentation model::

    u = clamp(i_cmd, -retract, depth_cap) + surface_error + residual_breathing
    F = k_true * max(0, u)

Model assumptions (engineering, documented, not clinical):

* quasistatic 1-D contact, rigid tool, no joint dynamics or friction;
* linear skin stiffness (the project contact model); ``k_true`` is swept over
  the range suggested by the literature (the nominal 150 N/m is on the very
  soft end, see ``docs/ECG_V3_SOLUTION_PLAN.md``);
* ideal force sensing at the tool axis, optional Gaussian noise;
* breathing = sinusoidal surface displacement, identical to the disturbance
  used by the M4 breathing evaluation.

The real robot would implement the compliant loop with an admittance /
impedance controller fed by a wrist force/torque sensor; this simulation is the
controller-level design study that precedes that integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roboecg.robot_controller.press_plan import PressSettings

PROVENANCE = (
    "engineering simulation: 1-D quasistatic contact loop on the documented "
    "linear skin model; gain from a desired time constant at k_est; "
    "not a clinical number"
)


@dataclass(frozen=True)
class CompliantPressConfig:
    duration_s: float = 4.0
    hold_start_s: float = 1.0
    sample_rate_hz: float = 200.0
    breathing_hz: float = 0.25  # 15 breaths / min
    breathing_amplitude_m: float = 0.008
    tau_s: float = 0.05  # desired closed-loop time constant at k_est (~3 Hz)
    k_est_n_m: float | None = None  # None -> use settings.contact_stiffness_n_m
    surface_error_m: float = 0.0  # +: true surface higher than estimated
    residual_breathing_fraction: float = 0.0  # 0 = perfect surface tracking
    retract_limit_m: float = 0.010  # tool may retract this far past the surface
    sensor_noise_n: float = 0.0
    seed: int = 0


def simulate_hold(
    settings: PressSettings,
    mode: str,
    k_true_n_m: float,
    config: CompliantPressConfig,
) -> dict:
    """Simulate one press/hold episode and return the trace + hold metrics."""
    if mode not in ("position", "compliant"):
        raise ValueError(f"unknown mode: {mode}")
    if k_true_n_m <= 0.0:
        raise ValueError("k_true_n_m must be positive")

    dt = 1.0 / config.sample_rate_hz
    frames = int(round(config.duration_s / dt))
    time_s = np.arange(frames) * dt
    rng = np.random.default_rng(config.seed)
    breathing_m = config.breathing_amplitude_m * np.sin(
        2.0 * np.pi * config.breathing_hz * time_s
    )
    residual_m = (
        config.residual_breathing_fraction * breathing_m
        if mode == "compliant"
        else breathing_m
    )

    k_est = config.k_est_n_m or settings.contact_stiffness_n_m
    gain = 1.0 / (k_est * config.tau_s)  # m / (N s)

    indentation_cmd = np.zeros(frames)
    indentation = np.zeros(frames)
    force = np.zeros(frames)
    depth_saturated = np.zeros(frames, dtype=bool)

    i_cmd = 0.0
    measured_force = 0.0
    for i in range(frames):
        if mode == "position":
            i_cmd = settings.press_depth_m
        else:
            if config.sensor_noise_n > 0.0:
                measured_force += float(rng.normal(0.0, config.sensor_noise_n))
            error = settings.force_target_n - measured_force
            i_cmd = float(
                np.clip(
                    i_cmd + gain * error * dt,
                    -config.retract_limit_m,
                    settings.max_press_depth_m,
                )
            )
        indentation_cmd[i] = i_cmd
        true_indentation = max(
            0.0,
            i_cmd + config.surface_error_m + residual_m[i],
        )
        indentation[i] = true_indentation
        force[i] = k_true_n_m * true_indentation
        measured_force = force[i]
        depth_saturated[i] = bool(
            i_cmd >= settings.max_press_depth_m - 1e-12
            and settings.force_target_n - measured_force > 0.0
        )

    hold = time_s >= config.hold_start_s
    hold_force = force[hold]
    hold_indentation = indentation[hold]

    max_force = float(hold_force.max()) if hold.any() else 0.0
    max_indentation = float(hold_indentation.max()) if hold.any() else 0.0
    force_rmse = float(
        np.sqrt(np.mean((hold_force - settings.force_target_n) ** 2))
    )
    contact_loss = (
        float(np.mean(hold_indentation <= 1e-12)) if hold.any() else 0.0
    )

    depth_ok = max_indentation <= settings.max_press_depth_m + 1e-9
    force_ok = max_force <= settings.force_limit_n + 1e-9

    return {
        "mode": mode,
        "k_true_n_m": float(k_true_n_m),
        "k_est_n_m": float(k_est),
        "time_s": time_s,
        "indentation_cmd_m": indentation_cmd,
        "indentation_m": indentation,
        "force_n": force,
        "breathing_m": breathing_m,
        "depth_saturated": depth_saturated,
        "metrics": {
            "max_force_n": max_force,
            "max_indentation_m": max_indentation,
            "force_rmse_n": force_rmse,
            "contact_loss_fraction": contact_loss,
            "depth_ok": bool(depth_ok),
            "force_ok": bool(force_ok),
            "violation": None
            if (depth_ok and force_ok)
            else ("depth" if not depth_ok else "force"),
            "depth_saturated_fraction": float(np.mean(depth_saturated[hold]))
            if hold.any()
            else 0.0,
        },
    }


def run_scenarios(settings: PressSettings) -> dict:
    """Full comparison grid: position vs compliant across stiffness/errors."""
    rows: list[dict] = []

    def record(tag: str, result: dict, config: CompliantPressConfig) -> None:
        rows.append(
            {
                "tag": tag,
                "mode": result["mode"],
                "k_true_n_m": result["k_true_n_m"],
                "k_est_n_m": result["k_est_n_m"],
                "surface_error_m": config.surface_error_m,
                "residual_breathing_fraction": config.residual_breathing_fraction,
                "breathing_hz": config.breathing_hz,
                "breathing_amplitude_m": config.breathing_amplitude_m,
                **result["metrics"],
            }
        )

    base = CompliantPressConfig()
    for k_true in (150.0, 300.0, 1000.0):
        record(
            f"position_k{k_true:.0f}",
            simulate_hold(settings, "position", k_true, base),
            base,
        )
    for k_true in (50.0, 75.0, 150.0, 300.0, 1000.0, 2000.0):
        record(
            f"compliant_k{k_true:.0f}",
            simulate_hold(settings, "compliant", k_true, base),
            base,
        )
    for k_est in (75.0, 300.0, 600.0):
        cfg = CompliantPressConfig(k_est_n_m=k_est)
        record(
            f"compliant_mismatch_kest{k_est:.0f}",
            simulate_hold(settings, "compliant", 150.0, cfg),
            cfg,
        )
    stress = CompliantPressConfig(
        breathing_hz=1.0 / 3.0, breathing_amplitude_m=0.012
    )
    record(
        "compliant_stress20bpm12mm",
        simulate_hold(settings, "compliant", 150.0, stress),
        stress,
    )
    record(
        "position_stress20bpm12mm",
        simulate_hold(settings, "position", 150.0, stress),
        stress,
    )
    err = CompliantPressConfig(surface_error_m=0.005)
    for mode in ("compliant", "position"):
        record(
            f"{mode}_err5mm_k150",
            simulate_hold(settings, mode, 150.0, err),
            err,
        )
    for mode in ("compliant", "position"):
        record(
            f"{mode}_err5mm_k2000",
            simulate_hold(settings, mode, 2000.0, err),
            err,
        )
    partial = CompliantPressConfig(residual_breathing_fraction=0.5)
    record(
        "compliant_partial_tracking_k150",
        simulate_hold(settings, "compliant", 150.0, partial),
        partial,
    )
    record(
        "compliant_partial_tracking_k2000",
        simulate_hold(settings, "compliant", 2000.0, partial),
        partial,
    )
    noisy = CompliantPressConfig(sensor_noise_n=0.05)
    record(
        "compliant_noise005nk150",
        simulate_hold(settings, "compliant", 150.0, noisy),
        noisy,
    )

    violations = [r for r in rows if r["violation"] is not None]
    return {
        "provenance": PROVENANCE,
        "settings": {
            "press_depth_m": settings.press_depth_m,
            "force_target_n": settings.force_target_n,
            "force_limit_n": settings.force_limit_n,
            "max_press_depth_m": settings.max_press_depth_m,
            "nominal_stiffness_n_m": settings.contact_stiffness_n_m,
        },
        "scenarios": rows,
        "summary": {
            "n_scenarios": len(rows),
            "n_violations": len(violations),
            "violating_tags": [r["tag"] for r in violations],
        },
    }
