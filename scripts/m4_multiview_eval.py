#!/usr/bin/env python3
"""Multi-view POC (v3, I1): does a second depth view fix the V5/V6 lateral wall?

For a few candidate lateral-camera poses this script renders the overhead
perception depth plus the lateral depth of the same supine patient and
compares, at the V5/V6 mesh ground-truth contact points:

  * incidence angle from each camera (0 deg = head-on, 90 deg = grazing);
  * surface coverage: nearest distance from the GT point to the cloud and the
    number of cloud points within 1 cm, overhead cloud vs fused cloud;
  * local surface fit: PCA normal error and plane RMS within 2.5 cm, overhead
    vs fused, against a mesh-local reference normal.

Simulation caveat (documented): no registration step is needed in simulation
because both extrinsics are known; on a real robot the second view would be
calibrated, and the same comparison runs on the fused cloud.

Writes runs/m4/multiview_report.json + runs/m4/multiview.png (and the lateral
RGB of the winning pose).

Usage: ./scripts/run_headless.sh scripts/m4_multiview_eval.py
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

WIDTH, HEIGHT, FOV_DEG = 640, 360, 90.0
FIT_RADIUS_M = 0.025
NEAR_RADIUS_M = 0.010
CANDIDATE_NEIGHBORHOOD_M = 0.10

CANDIDATES = [
    ("side_mid", (-0.20, 0.85, 1.05), (-0.135, 0.135, 0.90)),
    ("side_high", (-0.13, 0.60, 1.35), (-0.133, 0.14, 0.89)),
    ("side_feet", (0.00, 0.75, 0.95), (-0.133, 0.14, 0.89)),
]


def local_pca_fit(cloud, point, normal_ref):
    """Normal + plane RMS of the cloud neighborhood around a point."""
    sizes = np.linalg.norm(cloud - point, axis=1)
    local = cloud[sizes <= FIT_RADIUS_M]
    result = {
        "points": int(local.shape[0]),
        "normal_error_deg": None,
        "plane_rms_mm": None,
    }
    if local.shape[0] >= 6:
        centered = local - local.mean(axis=0)
        covariance = centered.T @ centered / local.shape[0]
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, int(np.argmin(values))]
        if np.dot(normal, normal_ref) < 0:
            normal = -normal
        angle = float(
            np.degrees(np.arccos(np.clip(np.dot(normal, normal_ref), -1.0, 1.0)))
        )
        rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
        result["normal_error_deg"] = angle
        result["plane_rms_mm"] = rms * 1000.0
    return result


def cloud_metrics(cloud, point, normal_ref):
    sizes = np.linalg.norm(cloud - point, axis=1)
    nearest = float(sizes.min()) if sizes.size else float("nan")
    metrics = {
        "nearest_mm": nearest * 1000.0 if np.isfinite(nearest) else None,
        "coverage_points": int((sizes <= NEAR_RADIUS_M).sum()),
    }
    metrics.update(local_pca_fit(cloud, point, normal_ref))
    return metrics


def incidence_deg(point, normal, camera_position):
    ray = np.asarray(camera_position, dtype=float) - np.asarray(point, dtype=float)
    ray = ray / (np.linalg.norm(ray) + 1e-12)
    return float(
        np.degrees(np.arccos(np.clip(abs(np.dot(normal, ray)), 0.0, 1.0)))
    )


def main() -> None:
    app = boot(headless=True, width=WIDTH, height=HEIGHT)
    try:
        from isaacsim.core.api import World

        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
        )
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import (
            depth_to_world_points,
            estimate_surface_normal,
        )
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.rendering import capture_depth, capture_rgb

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        print("multiview: scene built", flush=True)

        rules = load_ecg_rules()
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        targets = {
            t.name: t
            for t in generate_v1_v6(landmarks, frame, mesh_points, rules).targets
        }
        intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_DEG)

        def camera_cloud(path, steps=3):
            for _ in range(steps):
                world.step(render=True)
            depth = capture_depth(path, WIDTH, HEIGHT)
            matrix = ecg_scene.world_matrix(stage, path)
            position = matrix[:3, 3]
            rotation = cv_rotation_from_usd(matrix[:3, :3])
            cloud = depth_to_world_points(depth, intrinsics, position, rotation)
            return cloud, position

        # mesh-local reference normals at V5/V6 (flipped toward the +Y flank)
        references = {}
        for name in ("V5", "V6"):
            point = np.asarray(targets[name].position, dtype=float)
            normal, inliers = estimate_surface_normal(
                mesh_points,
                point,
                radius=0.03,
                orient_toward=point + np.array([0.0, 0.05, 0.0]),
            )
            if normal is None:
                raise RuntimeError(f"no mesh reference normal at {name}")
            references[name] = {
                "point": point,
                "normal": normal,
                "mesh_inliers": int(inliers),
            }
            print(
                f"multiview: {name} GT {np.round(point, 3).tolist()} "
                f"n_ref {np.round(normal, 3).tolist()}",
                flush=True,
            )

        overhead_path = "/World/Cameras/PerceptionRGBD"
        overhead_cloud, overhead_position = camera_cloud(overhead_path)
        print(
            f"multiview: overhead cloud {overhead_cloud.shape[0]} points",
            flush=True,
        )

        candidate_rows = {}
        for name, position, look_at in CANDIDATES:
            path = f"/World/Cameras/MultiView_{name}"
            ecg_scene.add_camera(stage, path, position=position, look_at=look_at)
            cloud, cam_position = camera_cloud(path, steps=2)
            near = 0
            for reference in references.values():
                sizes = np.linalg.norm(
                    cloud - reference["point"], axis=1
                )
                near += int((sizes <= CANDIDATE_NEIGHBORHOOD_M).sum())
            incidences = [
                incidence_deg(reference["point"], reference["normal"], cam_position)
                for reference in references.values()
            ]
            candidate_rows[name] = {
                "position": [float(v) for v in cam_position],
                "look_at": [float(v) for v in look_at],
                "points": int(cloud.shape[0]),
                "points_near_v5v6_10cm": near,
                "incidence_deg": {
                    key: float(value)
                    for key, value in zip(references, incidences)
                },
            }
            print(
                f"multiview: candidate {name}: {cloud.shape[0]} pts, "
                f"{near} within 10 cm of V5/V6, incidence "
                + ", ".join(
                    f"{key}={value:.1f}"
                    for key, value in zip(references, incidences)
                ),
                flush=True,
            )

        best_name = max(
            candidate_rows,
            key=lambda key: (candidate_rows[key]["points_near_v5v6_10cm"],
                             -sum(candidate_rows[key]["incidence_deg"].values())),
        )
        best_position, best_look_at = CANDIDATES[
            [name for name, _, _ in CANDIDATES].index(best_name)
        ][1:]
        best_path = f"/World/Cameras/MultiView_{best_name}"
        lateral_cloud, lateral_position = camera_cloud(best_path)
        fused_cloud = np.concatenate([overhead_cloud, lateral_cloud], axis=0)
        rgb = capture_rgb(best_path, WIDTH, HEIGHT)
        from PIL import Image

        Image.fromarray(rgb.astype("uint8"), mode="RGB").save(
            RUNS_DIR / "multiview_lateral_rgb.png"
        )
        print(
            f"multiview: best candidate {best_name} "
            f"({lateral_cloud.shape[0]} lateral points, fused "
            f"{fused_cloud.shape[0]})",
            flush=True,
        )

        metrics = {}
        incidence = {}
        for name, reference in references.items():
            point = reference["point"]
            normal = reference["normal"]
            incidence[name] = {
                "overhead": incidence_deg(point, normal, overhead_position),
                f"lateral:{best_name}": incidence_deg(
                    point, normal, lateral_position
                ),
            }
            metrics[name] = {
                "overhead": cloud_metrics(overhead_cloud, point, normal),
                "fused": cloud_metrics(fused_cloud, point, normal),
            }
            print(
                f"multiview: {name}: incidence overhead "
                f"{incidence[name]['overhead']:.1f} deg vs lateral "
                f"{incidence[name][f'lateral:{best_name}']:.1f} deg | "
                f"nearest overhead {metrics[name]['overhead']['nearest_mm']:.1f} mm "
                f"vs fused {metrics[name]['fused']['nearest_mm']:.1f} mm | "
                f"normal err "
                f"{metrics[name]['overhead']['normal_error_deg']} -> "
                f"{metrics[name]['fused']['normal_error_deg']} deg",
                flush=True,
            )

        report = {
            "provenance": (
                "simulation POC: overhead (PerceptionRGBD) + lateral depth "
                "clouds in world coordinates; extrinsics known exactly, no "
                "registration step; metric = mesh ground-truth contact points"
            ),
            "settings": {
                "width": WIDTH,
                "height": HEIGHT,
                "fov_deg": FOV_DEG,
                "fit_radius_m": FIT_RADIUS_M,
                "near_radius_m": NEAR_RADIUS_M,
            },
            "targets": {
                name: {
                    "gt_world": [float(v) for v in reference["point"]],
                    "normal_ref": [float(v) for v in reference["normal"]],
                    "mesh_inliers": reference["mesh_inliers"],
                }
                for name, reference in references.items()
            },
            "candidates": candidate_rows,
            "best_candidate": best_name,
            "incidence_deg": incidence,
            "metrics": metrics,
        }
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / "multiview_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"multiview: wrote {out}", flush=True)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure = plt.figure(figsize=(13, 4.5))
            axis = figure.add_subplot(1, 3, 1, projection="3d")
            stride = max(1, overhead_cloud.shape[0] // 4000)
            axis.scatter(
                overhead_cloud[::stride, 0], overhead_cloud[::stride, 1],
                overhead_cloud[::stride, 2], s=1, color="0.6",
                label="overhead",
            )
            stride = max(1, lateral_cloud.shape[0] // 4000)
            axis.scatter(
                lateral_cloud[::stride, 0], lateral_cloud[::stride, 1],
                lateral_cloud[::stride, 2], s=1, color="tab:blue",
                label=f"lateral ({best_name})",
            )
            for name, reference in references.items():
                point = reference["point"]
                axis.scatter(*point, s=60, marker="*", color="crimson",
                             label=f"{name} GT")
            axis.set_title("clouds (world frame)")
            axis.legend(loc="upper right", fontsize=7)

            axis = figure.add_subplot(1, 3, 2)
            positions = np.arange(2)
            width = 0.35
            axis.bar(
                positions - width / 2,
                [metrics[n]["overhead"]["nearest_mm"] for n in references],
                width, label="overhead",
            )
            axis.bar(
                positions + width / 2,
                [metrics[n]["fused"]["nearest_mm"] for n in references],
                width, label="fused",
            )
            axis.set_xticks(positions, list(references))
            axis.set_ylabel("nearest GT distance [mm]")
            axis.legend(fontsize=8)
            axis.set_title("surface coverage")

            axis = figure.add_subplot(1, 3, 3)
            axis.bar(
                positions - width / 2,
                [
                    metrics[n]["overhead"]["normal_error_deg"] or 0.0
                    for n in references
                ],
                width, label="overhead",
            )
            axis.bar(
                positions + width / 2,
                [
                    metrics[n]["fused"]["normal_error_deg"] or 0.0
                    for n in references
                ],
                width, label="fused",
            )
            axis.set_xticks(positions, list(references))
            axis.set_ylabel("local normal error [deg]")
            axis.legend(fontsize=8)
            axis.set_title("local surface fit")
            figure.tight_layout()
            figure.savefig(RUNS_DIR / "multiview.png", dpi=140)
            print(f"multiview: wrote {RUNS_DIR / 'multiview.png'}", flush=True)
        except Exception as error:  # pragma: no cover - plotting optional
            print(f"multiview: plot skipped: {error}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_multiview_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
