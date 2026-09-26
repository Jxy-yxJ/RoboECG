"""M2 demo: RGB-D depth fusion of the V1-V6 targets with a local normal prior.

Pipeline:
    mesh targets (M1) + fitted chest-surface prior
      -> overhead RGB-D render
      -> per target: project, median-depth skin point, PCA normal
      -> accept the depth measurement only if it agrees with the local prior
      -> fused targets + evaluation against the mesh targets

The mesh targets are the simulation ground truth (M1 definition); the depth
path is the "real perception" surrogate, so the reported error is the error the
real path would introduce, not an improvement over the mesh (which by
construction sits on the ground truth).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboecg.coordinate_transform.camera import CameraIntrinsics
from roboecg.perception.chest_landmarks import read_chest_landmarks
from roboecg.perception.isaac_skeleton import read_joint_world_positions
from roboecg.perception.torso_prior import fit_chest_surface_prior
from roboecg.target_localization.chest_frame import build_chest_frame
from roboecg.target_localization.ecg import generate_v1_v6
from roboecg.target_localization.ecg_rules import load_ecg_rules
from roboecg.target_localization.fusion import evaluate_fusion, fuse_target
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.rendering import capture_depth, capture_rgb, save_png

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "m2"

CAMERA_FOV_DEG = 90.0


def camera_pose(stage, prim_path):
    from roboecg.coordinate_transform.camera import cv_rotation_from_usd
    from roboecg.task_manager.ecg_scene import world_matrix

    matrix = world_matrix(stage, prim_path)
    return matrix[:3, 3], cv_rotation_from_usd(matrix[:3, :3])


def add_fusion_visualization(stage, frame, nominal_targets, fused_targets):
    from roboecg.task_manager.ecg_scene import add_frame_axes, add_line, add_marker

    add_frame_axes(stage, "/World/Markers/ChestFrame", frame)
    for fused in fused_targets:
        color = (0.15, 0.85, 0.35) if fused.source == "depth" else (0.95, 0.35, 0.15)
        add_marker(
            stage, f"/World/Markers/M2_{fused.name}", fused.position, 0.011, color
        )
        add_line(
            stage,
            f"/World/Markers/M2_{fused.name}_normal",
            fused.position,
            np.asarray(fused.position) + np.asarray(fused.normal) * 0.04,
            color=color,
            width=0.003,
        )
    for target in nominal_targets:
        add_marker(
            stage,
            f"/World/Markers/M1_{target.name}",
            target.position,
            0.007,
            (0.25, 0.45, 0.95),
        )


def run_m2(app, gui: bool = False, width: int = 1280, height: int = 720) -> dict:
    from isaacsim.core.api import World

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0)
    stage, scene_report = ecg_scene.build_scene(world)
    print("M2: scene built", flush=True)

    joint_positions = read_joint_world_positions(stage)
    landmarks = read_chest_landmarks(joint_positions)
    frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
    rules = load_ecg_rules()

    points = ecg_scene.mesh_world_points(stage, "/World/Human")
    nominal = generate_v1_v6(landmarks, frame, points, rules)
    prior = fit_chest_surface_prior(points, frame)
    print(
        f"M2: prior fitted, rms={prior.fit_rms_m * 1000:.2f} mm, "
        f"samples={prior.samples}",
        flush=True,
    )

    camera_position, cv_rotation = camera_pose(stage, "/World/Cameras/PerceptionRGBD")
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, CAMERA_FOV_DEG)
    depth = capture_depth("/World/Cameras/PerceptionRGBD", width, height)
    print(
        f"M2: depth rendered, valid pixels="
        f"{int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))}",
        flush=True,
    )

    settings = rules["depth_fusion"]
    fused = [
        fuse_target(
            target,
            frame,
            prior,
            depth,
            intrinsics,
            camera_position,
            cv_rotation,
            settings,
        )
        for target in nominal.targets
    ]
    evaluation = evaluate_fusion(fused, nominal.targets, frame)
    print(
        f"M2: fusion done, hit rate={evaluation['depth_hit_rate']:.2f}, "
        f"position error mean={evaluation['position_error_m']['mean'] * 1000:.2f} mm",
        flush=True,
    )

    add_fusion_visualization(stage, frame, nominal.targets, fused)
    rgb = capture_rgb("/World/Cameras/ChestCloseup", width, height)
    save_png(RUNS_DIR / "fusion_closeup.png", rgb)
    depth_png = (np.clip(depth, 0.0, 2.0) / 2.0 * 255.0).astype("uint8")
    from PIL import Image

    Image.fromarray(depth_png, mode="L").save(RUNS_DIR / "depth.png")

    report = {
        # Acceptance: every target must produce a usable pose (depth or a
        # justified model fallback) and stay within the error bounds.  The
        # depth hit rate is reported but not required to be 1.0: at grazing
        # incidence (V6 midaxillary, ~70 deg) depth adds nothing and the
        # model target is used instead.
        "status": (
            "PASS"
            if evaluation["valid_rate"] >= 1.0
            and evaluation["position_error_m"]["max"] <= 0.02
            and evaluation["fused_normal_angle_vs_mesh_deg"]["max"] <= 25.0
            else "PARTIAL"
        ),
        "scene": scene_report,
        "prior": prior.describe(),
        "settings": settings,
        "targets": {
            fused_target.name: {
                "nominal_position_world": np.asarray(
                    next(
                        t.position
                        for t in nominal.targets
                        if t.name == fused_target.name
                    )
                ).tolist(),
                "fused_position_world": fused_target.position.tolist(),
                "fused_normal_world": fused_target.normal.tolist(),
                "source": fused_target.source,
                "info": fused_target.info,
            }
            for fused_target in fused
        },
        "evaluation": evaluation,
        "screenshots": {
            "closeup": str(RUNS_DIR / "fusion_closeup.png"),
            "depth": str(RUNS_DIR / "depth.png"),
        },
    }
    (RUNS_DIR / "m2_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report
