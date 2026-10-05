"""I5-b: run the ECG perception chain on a real SSM torso inside the Isaac scene.

The 25 SSM torsos (Bender et al., CC-BY-4.0, assets/external/torso_models) come
with real electrode positions, so this is the only non-circular end-to-end test
in the project: the perceived targets are compared with actual electrode
coordinates on real anatomy instead of the project's own rules.

Procedure: build the standard scene, compute the biped reference (rig
landmarks + nominal targets), hide the biped, place T_XX supine where the
biped's chest was (lateral axis from the V1->V2 parasternal pair, anterior from
the model's chest side, back resting on the table, electrode centroid aligned
with the biped's nominal electrode centroid), then run the multiview perception
chain and compare with the transformed electrode ground truth.

Run headless:
    ./scripts/run_headless.sh scripts/m4_ssm_torso_place.py --model T_01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m0_common import boot  # noqa: E402

from roboecg.perception.torso_mesh import (  # noqa: E402
    parse_vtk_polydata,
    subdivide_and_smooth,
)

TORSO_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
NAMES = ("V1", "V2", "V3", "V4", "V5", "V6")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T_01", help="model id, e.g. T_01")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model ids; one Isaac session loops over them",
    )
    parser.add_argument(
        "--notch-to-4ics-mm",
        type=float,
        default=None,
        help=(
            "override the notch->4th-ICS distance (I5-c: on the SSM surfaces "
            "the transferred clavicle proxy sits 84.3 +- 6.1 mm above the "
            "V1/V2 row, while the published constant is 193 mm; pass the "
            "leave-one-out population value for an honest transfer test)"
        ),
    )
    parser.add_argument(
        "--detector-checkpoint",
        default=None,
        help="detector checkpoint (default: the deployed model)",
    )
    parser.add_argument(
        "--subdiv",
        type=int,
        default=0,
        help=(
            "midpoint-subdivide the SSM mesh N times (I5-c stage 2: isolates "
            "the coarse-mesh factor; the models are ~3-4 cm triangles while "
            "real skin reads smooth in depth)"
        ),
    )
    parser.add_argument(
        "--smooth-iter",
        type=int,
        default=0,
        help="Laplacian smoothing iterations applied after subdivision",
    )
    parser.add_argument(
        "--arm-proxies",
        action="store_true",
        help=(
            "add two simple arm cylinders alongside the torso (the SSM models "
            "have no arms; the detector was trained on renders that always "
            "had arms, so this isolates the arm-context factor)"
        ),
    )
    args = parser.parse_args()

    app = boot(headless=True, width=1280, height=720)
    try:
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom

        from roboecg.perception.chest_detector import ChestLandmarkDetector
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import estimate_surface_normal
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.m4_demo import LATERAL_CAMERA, RUNS_DIR
        from roboecg.task_manager.perception_pipeline import perceive_targets

        world = World(stage_units_in_meters=1.0)
        stage, scene_report = ecg_scene.build_scene(world)
        print("ssm_torso: scene built", flush=True)

        rules = load_ecg_rules()
        if args.notch_to_4ics_mm is not None:
            rules["anatomy"]["sternal_notch_to_nipple"] = dict(
                rules["anatomy"]["sternal_notch_to_nipple"]
            )
            rules["anatomy"]["sternal_notch_to_nipple"]["value"] = (
                float(args.notch_to_4ics_mm) / 1000.0
            )
            print(
                "ssm_torso: notch->4ICS override = %.1f mm "
                "(leave-one-out population value)"
                % float(args.notch_to_4ics_mm),
                flush=True,
            )
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        biped_targets = {
            t.name: t
            for t in generate_v1_v6(landmarks, frame, mesh_points, rules).targets
        }
        biped_center = np.mean(
            [np.asarray(biped_targets[n].position) for n in NAMES], axis=0
        )

        models = (
            [m.strip() for m in args.models.split(",") if m.strip()]
            if args.models
            else [args.model]
        )
        for model_id in models:
            points_mm, faces = parse_vtk_polydata(
                TORSO_DIR / f"{model_id}_torso_coarse_surface.vtk"
            )
            electrodes = np.loadtxt(
                TORSO_DIR / f"{model_id}_electrodes.csv",
                delimiter=",",
                skiprows=1,
            )
            points_m = points_mm / 1000.0
            v_m = electrodes[3:9] / 1000.0
            if args.subdiv > 0 or args.smooth_iter > 0:
                points_m, faces = subdivide_and_smooth(
                    points_m,
                    faces,
                    subdivisions=args.subdiv,
                    smooth_iterations=args.smooth_iter,
                )
                print(
                    f"ssm_torso: mesh subdivided x{args.subdiv} + smoothed "
                    f"x{args.smooth_iter} -> {len(points_m)} verts / "
                    f"{len(faces)} faces",
                    flush=True,
                )  # V1..V6

            # Canonical model->world mapping, identical to the fine-tune data
            # generation (scripts/m4_ssm_detector_data.py): +X -> +Y (left),
            # -Y -> +Z (anterior), +Z -> -X (head).  Electrode-derived axes
            # deviate up to 36 deg from this canonical frame across the
            # population while the training used the canonical frame, so the
            # electrode-derived placement was a train/eval orientation
            # mismatch that correlated with the per-model raw error.
            a_m = np.array([1.0, 0.0, 0.0])
            b_m = np.array([0.0, -1.0, 0.0])
            c_m = np.cross(a_m, b_m)
            a_w = np.array([0.0, 1.0, 0.0])
            b_w = np.array([0.0, 0.0, 1.0])
            c_w = np.cross(a_w, b_w)
            rotation = np.column_stack([a_w, b_w, c_w]) @ np.column_stack(
                [a_m, b_m, c_m]
            ).T
            det = float(np.linalg.det(rotation))
            if not np.isclose(det, 1.0, atol=1e-6):
                raise RuntimeError(f"placement rotation not proper (det={det:.6f})")
            # Framing matched to the training generation: the model centroid
            # sits at the biped chest joint + 8 cm anterior (jitter zero).
            chest_anchor = np.asarray(landmarks.chest, dtype=float)
            centroid_m = points_m.mean(axis=0)
            translation = (
                chest_anchor
                + np.array([0.0, 0.0, 0.08])
                - (centroid_m @ rotation.T)
            )
            world_points = points_m @ rotation.T + translation
            gt_world = v_m @ rotation.T + translation
            print(
                "ssm_torso: canonical placement (training-matched), "
                f"centroid -> {np.round(chest_anchor + [0, 0, 0.08], 3)}",
                flush=True,
            )
            print(
                f"ssm_torso: {model_id} placed "
                f"(centroid offset xy = "
                f"{np.linalg.norm(gt_world.mean(axis=0)[:2] - biped_center[:2]) * 1000:.1f} mm, "
                f"back z = {world_points[:, 2].min() * 1000:.0f} mm)",
                flush=True,
            )

            torso = UsdGeom.Mesh.Define(stage, "/World/RealTorso")
            torso.CreatePointsAttr([Gf.Vec3f(*[float(c) for c in p]) for p in world_points])
            torso.CreateFaceVertexCountsAttr([3] * len(faces))
            torso.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            torso.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            torso.CreateDisplayColorAttr([Gf.Vec3f(0.87, 0.72, 0.63)])
            if args.arm_proxies:
                # Arms rest alongside the torso in the training renders; the SSM
                # torsos are arm-free.  Two cylinders just lateral to the flanks
                # restore the silhouette context without touching the anatomy.
                half_width = 0.5 * float(world_points[:, 1].max() - world_points[:, 1].min())
                center_y = 0.5 * float(world_points[:, 1].max() + world_points[:, 1].min())
                arm_length = 0.55
                for side, sign in (("L", +1.0), ("R", -1.0)):
                    cylinder = UsdGeom.Cylinder.Define(stage, f"/World/ArmProxy{side}")
                    cylinder.CreateRadiusAttr(0.05)
                    cylinder.CreateHeightAttr(arm_length)
                    cylinder.CreateAxisAttr(UsdGeom.Tokens.x)
                    cylinder.AddTranslateOp().Set(
                        Gf.Vec3d(
                            -0.39 + arm_length / 2.0,
                            center_y + sign * (half_width + 0.05),
                            scene_report["table_bounds"]["top_z"] + 0.05,
                        )
                    )
                    cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.66, 0.58)])
                print("ssm_torso: arm proxies added", flush=True)
            stage.GetPrimAtPath("/World/Human").SetActive(False)
            world.reset()

            detector = (
                ChestLandmarkDetector(args.detector_checkpoint)
                if args.detector_checkpoint
                else ChestLandmarkDetector()
            )
            perceived = perceive_targets(
                stage,
                detector,
                rules,
                world=world,
                lateral_camera=LATERAL_CAMERA,
                allow_snap_fallback=True,
            )
            fused = {t.name: t for t in perceived["fused"]}
            generated = {t.name: t for t in perceived["generated"].targets}

            rows = []
            for index, name in enumerate(NAMES):
                target = fused[name]
                position = np.asarray(target.position, dtype=float)
                gt_position = gt_world[index]
                gt_normal, _ = estimate_surface_normal(
                    world_points,
                    gt_position,
                    radius=0.03,
                    orient_toward=gt_position + rotation @ b_m,
                )
                normal_error = None
                if gt_normal is not None:
                    cosine = float(
                        np.dot(
                            np.asarray(target.normal, dtype=float) / np.linalg.norm(target.normal),
                            np.asarray(gt_normal, dtype=float) / np.linalg.norm(gt_normal),
                        )
                    )
                    normal_error = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                rows.append(
                    {
                        "target": name,
                        "source": target.source,
                        "view": target.info.get("view"),
                        "incidence_deg": target.info.get("incidence_deg"),
                        "normal_angle_vs_prior_deg": target.info.get(
                            "normal_angle_vs_prior_deg"
                        ),
                        "snap_status": generated[name].provenance.get("snap", {}).get(
                            "status"
                        ),
                        "generated_world": [
                            float(c) for c in np.asarray(generated[name].position)
                        ],
                        "generated_frame_coords": [
                            float(c) for c in np.asarray(generated[name].frame_coords)
                        ],
                        "wrap_refined": bool(
                            generated[name]
                            .provenance.get("snap", {})
                            .get("wrap_refined", False)
                        ),
                        "position_world": [float(c) for c in position],
                        "gt_world": [float(c) for c in gt_position],
                        "error_mm": float(np.linalg.norm(position - gt_position)) * 1000.0,
                        "normal_error_deg": normal_error,
                    }
                )
            errors = [row["error_mm"] for row in rows]
            print("ssm_torso: perceived vs REAL electrodes (mm):", flush=True)
            for row in rows:
                print(
                    f"  {row['target']}: {row['error_mm']:7.2f} "
                    f"(normal {row['normal_error_deg'] if row['normal_error_deg'] is None else round(row['normal_error_deg'], 1)} deg, "
                    f"{row['source']} / {row['view']})",
                    flush=True,
                )
            print(
                f"ssm_torso: mean {float(np.mean(errors)):.2f} mm, "
                f"max {float(np.max(errors)):.2f} mm",
                flush=True,
            )

            report = {
                "model": model_id,
                "provenance": (
                    f"I5-b: {model_id} (Bender et al., CC-BY-4.0) placed supine "
                    "where the biped chest was; the biped is hidden; targets from "
                    "the multiview perception chain; ground truth = the dataset's "
                    "real electrode coordinates (non-circular)"
                ),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
                "biped_center_world": biped_center.tolist(),
                "table_top_z": scene_report["table_bounds"]["top_z"],
                "arm_proxies": bool(args.arm_proxies),
                "notch_to_4ics_mm": (
                    None if args.notch_to_4ics_mm is None else float(args.notch_to_4ics_mm)
                ),
                "mesh_subdivisions": int(args.subdiv),
                "mesh_smooth_iterations": int(args.smooth_iter),
                "perception": {
                    "views": perceived["views"],
                    "prior_fit_rms_m": perceived["prior"].fit_rms_m,
                    "frame": {
                        "origin": [float(c) for c in perceived["frame"].origin],
                        "up": [float(c) for c in perceived["frame"].up],
                        "lateral": [float(c) for c in perceived["frame"].lateral],
                        "anterior": [float(c) for c in perceived["frame"].anterior],
                    },
                    "frame_angles_deg": {
                        "up_vs_head": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        float(
                                            np.dot(
                                                perceived["frame"].up,
                                                np.array([-1.0, 0.0, 0.0]),
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        ),
                        "lateral_vs_left": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        float(
                                            np.dot(
                                                perceived["frame"].lateral,
                                                np.array([0.0, 1.0, 0.0]),
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        ),
                        "anterior_vs_up": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        float(
                                            np.dot(
                                                perceived["frame"].anterior,
                                                np.array([0.0, 0.0, 1.0]),
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        ),
                    },
                    "detector_landmarks": {
                        field: [float(c) for c in np.asarray(getattr(perceived["landmarks"], field))]
                        for field in (
                            "chest",
                            "neck_base",
                            "clavicle_left",
                            "clavicle_right",
                            "shoulder_left",
                            "shoulder_right",
                        )
                    },
                },
                "electrodes": rows,
                "error_mean_mm": float(np.mean(errors)),
                "error_max_mm": float(np.max(errors)),
            }
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = RUNS_DIR / f"m4_report_ssm_torso_{model_id}.json"
            report_path.write_text(
                json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"ssm_torso: report -> {report_path}", flush=True)
    except BaseException:
        # SimulationApp.close() terminates the process, so the traceback must
        # be captured before the finally block runs.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
