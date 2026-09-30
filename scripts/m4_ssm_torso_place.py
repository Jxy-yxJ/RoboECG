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
    anatomical_axes,
    parse_vtk_polydata,
)

TORSO_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
NAMES = ("V1", "V2", "V3", "V4", "V5", "V6")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T_01", help="model id, e.g. T_01")
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

        points_mm, faces = parse_vtk_polydata(
            TORSO_DIR / f"{args.model}_torso_coarse_surface.vtk"
        )
        electrodes = np.loadtxt(
            TORSO_DIR / f"{args.model}_electrodes.csv",
            delimiter=",",
            skiprows=1,
        )
        points_m = points_mm / 1000.0
        v_m = electrodes[3:9] / 1000.0  # V1..V6

        a_m, b_m, c_m = anatomical_axes(points_m, v_m)
        a_w = np.asarray(landmarks.shoulder_left) - np.asarray(
            landmarks.shoulder_right
        )
        a_w = a_w / (np.linalg.norm(a_w) + 1e-12)  # patient left
        b_w = np.array([0.0, 0.0, 1.0])  # scene anterior hint (M4)
        c_w = np.cross(a_w, b_w)  # down the body (towards the feet)
        down_check = float(
            np.dot(
                np.asarray(landmarks.pelvis) - np.asarray(landmarks.neck_base),
                c_w,
            )
        )
        if down_check <= 0.0:
            raise RuntimeError(
                f"scene axes inconsistent: pelvis not below neck along c_w "
                f"({down_check:.4f})"
            )
        rotation = np.column_stack([a_w, b_w, c_w]) @ np.column_stack(
            [a_m, b_m, c_m]
        ).T
        det = float(np.linalg.det(rotation))
        if not np.isclose(det, 1.0, atol=1e-6):
            raise RuntimeError(f"placement rotation not proper (det={det:.6f})")

        world_points0 = points_m @ rotation.T
        electrodes0 = v_m @ rotation.T
        translation = np.zeros(3)
        # XY: align the electrode centroid with the biped's nominal centroid
        # (the cameras and the detector expect the chest there).
        translation[:2] = biped_center[:2] - electrodes0.mean(axis=0)[:2]
        # Z: prefer the 3D centroid match (keeps the camera-relative framing of
        # the training renders), but never sink the back below the table.
        z_align = float(biped_center[2] - electrodes0.mean(axis=0)[2])
        z_rest = float(
            scene_report["table_bounds"]["top_z"] + 0.002
        ) - float(world_points0[:, 2].min())
        translation[2] = max(z_align, z_rest)
        world_points = world_points0 + translation
        gt_world = electrodes0 + translation
        print(
            f"ssm_torso: z placement: align {z_align * 1000:+.0f} mm vs rest "
            f"{z_rest * 1000:+.0f} mm -> using {translation[2] * 1000:+.0f} mm",
            flush=True,
        )
        print(
            f"ssm_torso: {args.model} placed "
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

        detector = ChestLandmarkDetector()
        perceived = perceive_targets(
            stage, detector, rules, world=world, lateral_camera=LATERAL_CAMERA
        )
        fused = {t.name: t for t in perceived["fused"]}

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
            "model": args.model,
            "provenance": (
                f"I5-b: {args.model} (Bender et al., CC-BY-4.0) placed supine "
                "where the biped chest was; the biped is hidden; targets from "
                "the multiview perception chain; ground truth = the dataset's "
                "real electrode coordinates (non-circular)"
            ),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "biped_center_world": biped_center.tolist(),
            "table_top_z": scene_report["table_bounds"]["top_z"],
            "arm_proxies": bool(args.arm_proxies),
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
        report_path = RUNS_DIR / f"m4_report_ssm_torso_{args.model}.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"ssm_torso: report -> {report_path}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
