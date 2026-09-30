#!/usr/bin/env python3
"""Intercostal-space probing feasibility on a ribbed phantom (v3, I4-lite).

Problem: the V1->V4 vertical drop is a population regression; the one real
patient sits 26.5 mm above the prediction (docs/ECG_P0B_FINDINGS.md), and no
public per-patient position data can fix it (docs/ECG_V3_SOLUTION_PLAN.md 7.1).
In-vivo rib/intercostal detection is the mechanism that would replace the
regression, and the literature has no depth-camera or contact-probe rib-level
classification for ECG rows (section 2 of the v3 plan).

This study builds an engineering rib phantom on the sim torso whose TRUE rows
are known by construction (ICS4 at the SNND level, ICS5 exactly 48 mm below -
the real-patient row spacing), then evaluates:

  1. depth visibility: can the working depth resolution see the ridges at all?
  2. contact-probe classification: with a light stroke (1-2 mm) and a rib/skin
     stiffness contrast kappa, can the rib bands (and therefore the ICS
     midpoints) be recovered from the force profile?
  3. the corrected V4-row error: regression (population) vs probe estimate,
     against the construction ground truth.

Writes runs/m4/rib_probe_study.json + runs/m4/rib_probe_study.png.
Usage: ./scripts/run_headless.sh scripts/m4_rib_probe_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs" / "m4"

DROP_TRUE_M = 0.048  # the real-patient row spacing (P0B counterexample)
RIB_HALF_WIDTH_M = 0.011  # rib band half-width around a rib centre
RIB_PROTRUSION_M = 0.005  # ridge height above the smooth skin
RIB_RADIUS_M = 0.008  # phantom sphere radius
PROBE_SPACING_M = 0.005
STROKES_MM = (1.0, 2.0)
KAPPAS = (1.5, 3.0, 10.0)
WIDTH, HEIGHT, FOV_DEG = 320, 180, 90.0  # the deployed depth resolution
LATERAL_CAMERA = {
    "prim_path": "/World/Cameras/PerceptionLateral",
    "position": (0.00, 0.75, 0.95),
    "look_at": (-0.133, 0.14, 0.89),
}


def rib_centers(u_ics4: float) -> dict:
    """Rib centres so that ICS4 = midpoint(rib4, rib5) and ICS5 = ICS4 - 48 mm."""
    offset = DROP_TRUE_M / 2.0
    return {
        "rib3": u_ics4 + 3.0 * offset,
        "rib4": u_ics4 + offset,
        "rib5": u_ics4 - offset,
        "rib6": u_ics4 - 3.0 * offset,
    }


def classify_ics(u_samples, profile, threshold_fraction=0.5):
    """Rib bands -> ICS midpoints from a probe profile (force or height)."""
    profile = np.asarray(profile, dtype=float)
    low = float(np.percentile(profile, 50))
    high = float(np.percentile(profile, 95))
    if high - low < 1e-9:
        return None
    threshold = low + threshold_fraction * (high - low)
    mask = profile >= threshold
    bands = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            bands.append((start, index - 1))
            start = None
    if start is not None:
        bands.append((start, len(mask) - 1))
    centers = []
    u_samples = np.asarray(u_samples, dtype=float)
    for first, last in bands:
        weight = profile[first:last + 1] - low
        weight = np.maximum(weight, 0.0)
        centers.append(
            float(np.average(u_samples[first:last + 1], weights=weight + 1e-9))
        )
    centers.sort(reverse=True)  # head -> feet (u decreases)
    midpoints = [
        0.5 * (centers[i] + centers[i + 1]) for i in range(len(centers) - 1)
    ]
    return {"band_centers": centers, "ics_midpoints": midpoints}


def main() -> None:
    app = boot(headless=True, width=WIDTH, height=HEIGHT)
    try:
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom

        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import (
            measure_torso_width,
            snap_to_surface,
        )
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager import ecg_scene

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("rib-probe: scene built", flush=True)

        rules = load_ecg_rules()
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")

        anatomy = rules["anatomy"]
        u_ics4 = -float(anatomy["sternal_notch_to_nipple"]["value"])
        v_shoulder = abs(float(frame.to_frame(landmarks.shoulder_left)[1]))
        v_mcl = float(
            frame.to_frame(
                0.5 * (landmarks.clavicle_left + landmarks.shoulder_left)
            )[1]
        )
        ribs = rib_centers(u_ics4)

        # ---- phantom: ridge prims on the chest at the rib centres ----------
        rib_points = []
        smooth_cache = {}
        for name, u_rib in ribs.items():
            for index, v in enumerate(np.linspace(-0.13, 0.13, 27)):
                point, normal, _ = snap_to_surface(
                    mesh_points, frame, u_rib, v, v_limit=v_shoulder
                )
                if point is None:
                    continue
                normal = np.asarray(normal, dtype=float)
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                center = np.asarray(point) + normal * (
                    RIB_PROTRUSION_M - RIB_RADIUS_M
                )
                rib_points.append({"rib": name, "u": u_rib, "v": float(v),
                                   "center": center.tolist()})
                sphere = UsdGeom.Sphere.Define(
                    stage, f"/World/RibPhantom/{name}_{index}"
                )
                sphere.CreateRadiusAttr(RIB_RADIUS_M)
                sphere.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in center]))
        print(f"rib-probe: phantom ribs placed ({len(rib_points)} spheres)",
              flush=True)

        # ---- depth visibility at the working resolution --------------------
        for _ in range(3):
            world.step(render=True)
        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
            project_world_to_pixel,
        )
        from roboecg.task_manager.rendering import capture_depth

        probe_points = np.arange(
            u_ics4 + 0.10, u_ics4 - 0.10 - 1e-9, -PROBE_SPACING_M
        )
        smooth_points = []
        smooth_heights = []
        for u in probe_points:
            point, _, _ = snap_to_surface(
                mesh_points, frame, u, v_mcl, v_limit=v_shoulder
            )
            smooth_points.append(
                np.asarray(point) if point is not None else None
            )
            smooth_heights.append(
                float(frame.to_frame(point)[2]) if point is not None else np.nan
            )
        smooth_heights = np.asarray(smooth_heights)

        intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_DEG)
        overhead_path = "/World/Cameras/PerceptionRGBD"
        overhead_matrix = ecg_scene.world_matrix(stage, overhead_path)
        overhead_position = overhead_matrix[:3, 3]
        overhead_rotation = cv_rotation_from_usd(overhead_matrix[:3, :3])
        overhead_depth = capture_depth(overhead_path, WIDTH, HEIGHT)
        ecg_scene.add_camera(stage, LATERAL_CAMERA["prim_path"],
                             position=LATERAL_CAMERA["position"],
                             look_at=LATERAL_CAMERA["look_at"])
        for _ in range(3):
            world.step(render=True)
        lateral_matrix = ecg_scene.world_matrix(stage, LATERAL_CAMERA["prim_path"])
        lateral_position = lateral_matrix[:3, 3]
        lateral_rotation = cv_rotation_from_usd(lateral_matrix[:3, :3])
        lateral_depth = capture_depth(LATERAL_CAMERA["prim_path"], WIDTH, HEIGHT)

        visibility = {}
        for view_name, depth, cam, rot in (
            ("overhead", overhead_depth, overhead_position, overhead_rotation),
            ("lateral", lateral_depth, lateral_position, lateral_rotation),
        ):
            heights = []
            for point in smooth_points:
                if point is None:
                    heights.append(np.nan)
                    continue
                pixel, _ = project_world_to_pixel(point, cam, rot, intrinsics)
                if pixel is None:
                    heights.append(np.nan)
                    continue
                ui = int(round(pixel[0]))
                vi = int(round(pixel[1]))
                if 0 <= ui < WIDTH and 0 <= vi < HEIGHT:
                    value = depth[vi, ui]
                    heights.append(float(value) if np.isfinite(value) else np.nan)
                else:
                    heights.append(np.nan)
            heights = np.asarray(heights)
            entry = {
                "valid_samples": int(np.isfinite(heights).sum()),
                "p95_minus_p05_mm": (
                    float(np.nanpercentile(heights, 95) - np.nanpercentile(heights, 5))
                    * 1000.0
                    if np.isfinite(heights).any()
                    else None
                ),
                "heights_m": [
                    None if not np.isfinite(value) else float(value)
                    for value in heights
                ],
            }
            valid = np.isfinite(heights)
            if valid.sum() >= 11:
                filled = np.interp(
                    np.arange(len(heights)),
                    np.flatnonzero(valid),
                    heights[valid],
                )
                pad = 4
                padded = np.concatenate(
                    [filled[:pad][::-1], filled, filled[-pad:][::-1]]
                )
                kernel = np.ones(9) / 9.0
                smooth = np.convolve(padded, kernel, mode="valid")
                residual = filled - smooth
                entry["ridge_residual_p95p05_mm"] = float(
                    (np.percentile(residual, 95) - np.percentile(residual, 5))
                    * 1000.0
                )
            visibility[view_name] = entry
            print(
                f"rib-probe: depth visibility ({view_name}): "
                f"{visibility[view_name]}",
                flush=True,
            )

        # ---- probe signals (analytic contact model) ------------------------

        def rib_overlap(u):
            return max(
                0.0,
                1.0 - min(abs(u - u_rib) for u_rib in ribs.values())
                / RIB_HALF_WIDTH_M,
            )

        overlaps = np.array([rib_overlap(u) for u in probe_points])
        bump = overlaps * RIB_PROTRUSION_M

        k_skin = 150.0  # N/m, the project's engineering contact model
        results = {}
        for stroke_mm in STROKES_MM:
            stroke = stroke_mm / 1000.0
            for kappa in KAPPAS:
                # rib bands: stiffness contrast at a fixed light stroke
                force = k_skin * stroke * (1.0 + (kappa - 1.0) * (overlaps > 0.0))
                classification = classify_ics(probe_points, force)
                if classification is None or len(classification["ics_midpoints"]) < 2:
                    results[f"stroke{stroke_mm}mm_kappa{kappa}"] = {
                        "detected_drop_mm": None,
                        "error_mm": None,
                    }
                    continue
                midpoints = classification["ics_midpoints"]
                # pick ICS4 as the midpoint nearest the SNND level, then the
                # next one downwards is ICS5
                nearest = int(np.argmin([abs(m - u_ics4) for m in midpoints]))
                if nearest + 1 >= len(midpoints):
                    results[f"stroke{stroke_mm}mm_kappa{kappa}"] = {
                        "detected_drop_mm": None,
                        "error_mm": None,
                        "midpoints": midpoints,
                    }
                    continue
                detected_drop = midpoints[nearest] - midpoints[nearest + 1]
                results[f"stroke{stroke_mm}mm_kappa{kappa}"] = {
                    "detected_drop_mm": float(detected_drop * 1000.0),
                    "error_mm": float(
                        (detected_drop - DROP_TRUE_M) * 1000.0
                    ),
                    "midpoints": midpoints,
                }

        # height-only variant (depth profile if it could resolve the ridges)
        height_profile = smooth_heights + bump
        height_class = classify_ics(probe_points, height_profile - smooth_heights)
        height_error = None
        if height_class and len(height_class["ics_midpoints"]) >= 2:
            midpoints = height_class["ics_midpoints"]
            nearest = int(np.argmin([abs(m - u_ics4) for m in midpoints]))
            if nearest + 1 < len(midpoints):
                height_error = float(
                    (midpoints[nearest] - midpoints[nearest + 1] - DROP_TRUE_M)
                    * 1000.0
                )

        # ---- population regression baseline --------------------------------
        width_m = measure_torso_width(mesh_points, frame, half_width_limit=v_shoulder)
        drop_rule = anatomy["fourth_to_fifth_ics"]
        coefficients = drop_rule["coefficients"]
        regression_drop = (
            float(coefficients["intercept_mm"])
            + float(coefficients["slope_mm_per_mm"]) * width_m * 1000.0
        ) / 1000.0
        regression_error = float((regression_drop - DROP_TRUE_M) * 1000.0)

        report = {
            "provenance": (
                "engineering rib phantom: ICS4 at the SNND level, ICS5 exactly "
                "48 mm below (the real-patient row spacing); probe signals from "
                "the project's analytic contact model with a rib/skin stiffness "
                "contrast kappa; ground truth = the construction"
            ),
            "phantom": {
                "u_ics4_m": float(u_ics4),
                "u_ics5_true_m": float(u_ics4 - DROP_TRUE_M),
                "drop_true_mm": DROP_TRUE_M * 1000.0,
                "rib_centers_m": {k: float(v) for k, v in ribs.items()},
                "rib_half_width_mm": RIB_HALF_WIDTH_M * 1000.0,
                "rib_protrusion_mm": RIB_PROTRUSION_M * 1000.0,
                "v_probe_m": float(v_mcl),
                "n_probe_points": int(len(probe_points)),
            },
            "depth_visibility": visibility,
            "regression": {
                "torso_width_m": float(width_m),
                "drop_mm": float(regression_drop * 1000.0),
                "error_mm": regression_error,
            },
            "probe": results,
            "height_profile_only": {"error_mm": height_error},
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "rib_probe_study.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"rib-probe: wrote {out}", flush=True)
        print(
            f"rib-probe: regression drop {regression_drop*1000:.1f} mm "
            f"(true {DROP_TRUE_M*1000:.0f}, error {regression_error:+.1f})",
            flush=True,
        )
        for key, value in results.items():
            print(
                f"rib-probe: {key}: detected "
                f"{value['detected_drop_mm']} mm, error {value['error_mm']} mm",
                flush=True,
            )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axes[0].plot(probe_points * 1000.0, bump * 1000.0, label="ridge height")
            axes[0].set_ylabel("ridge [mm]")
            for stroke_mm in STROKES_MM:
                stroke = stroke_mm / 1000.0
                force = k_skin * stroke * (1.0 + (3.0 - 1.0) * (overlaps > 0.0))
                axes[1].plot(
                    probe_points * 1000.0, force * 1000.0,
                    label=f"kappa=3, stroke {stroke_mm:.0f} mm",
                )
            axes[1].set_ylabel("probe force [mN]")
            axes[1].set_xlabel("u [mm] (head -> feet)")
            for axis in axes:
                axis.axvline((u_ics4) * 1000.0, color="green", ls=":", alpha=0.7)
                axis.axvline((u_ics4 - DROP_TRUE_M) * 1000.0, color="green",
                             ls=":", alpha=0.7)
                axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(RUNS_DIR / "rib_probe_study.png", dpi=140)
            print(f"rib-probe: wrote {RUNS_DIR / 'rib_probe_study.png'}",
                  flush=True)
        except Exception as error:  # pragma: no cover
            print(f"rib-probe: plot skipped: {error}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_rib_probe_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
