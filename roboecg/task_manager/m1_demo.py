"""M1 demo: measured V1-V6 targets -> UR3 planned and executed reach.

Pipeline (all quantities carry provenance, see configs/ecg_rules.yaml):
    rig landmarks -> ChestFrame -> measured chest surface
      -> V1-V6 targets (published SNND statistic + measured lateral lines)
      -> per-target collision-checked reach plan (plan_reach + table box)
      -> execution with per-frame clearance tracking + video evidence

The 4th->5th ICS drop uses the published regression on the measured torso
depth (independent GT: 25 torso models, DOI 10.5281/zenodo.20086105); the
report includes a +/-RMS sensitivity sweep around the prediction.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from roboecg.coordinate_transform.frames import (
    quat_wxyz_from_rotation,
    rotation_from_axes,
    rotation_from_quat_wxyz,
    slerp_wxyz,
)
from roboecg.perception.chest_landmarks import read_chest_landmarks
from roboecg.perception.isaac_skeleton import read_joint_world_positions
from roboecg.robot_controller.base_placement import (
    apply_base_placement,
    search_base_placement,
)
from roboecg.robot_controller.reach_plan import (
    link_world_positions,
    plan_reach,
    trajectory_clearance,
)
from roboecg.robot_controller.safety import CLEARANCE_MARGIN_M, body_capsules
from roboecg.target_localization.chest_frame import build_chest_frame
from roboecg.target_localization.ecg import generate_v1_v6
from roboecg.target_localization.ecg_rules import load_ecg_rules
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.overlay import overlay_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "m1"

# Depth render used to measure the chest surface for the targets
M1_WIDTH, M1_HEIGHT = 1280, 720

LAYOUT_M1 = {
    "motion_steps": 60,
    "pre_approach_distance_m": 0.06,
    "transition_steps": 30,
    "hold_frames": 15,
    "retreat_steps": 25,
    "video_resolution": [960, 540],
    "sensitivity_spacing_m": [0.079, 0.087, 0.095],
    # Pressing along the surface normal is preferred; when a target is not
    # reachable (UR3 reach vs. the midaxillary V6), the approach is tilted away
    # from the normal within a clinically acceptable bound (<= 30 deg) and the
    # contact point is kept fixed.  Ordered by increasing deviation.
    "approach_tilt_deg": [0.0, 10.0, 20.0, 30.0],
    "approach_azimuth_deg": [0.0, 90.0, 180.0, 270.0],
}

APPROACH_CANDIDATES = [
    (tilt, azimuth)
    for tilt in LAYOUT_M1["approach_tilt_deg"]
    for azimuth in (
        (0.0,)
        if tilt == 0.0
        else tuple(LAYOUT_M1["approach_azimuth_deg"])
    )
]


def table_box():
    layout = ecg_scene.LAYOUT
    center = (
        layout["table_center_xy"][0],
        layout["table_center_xy"][1],
        layout["table_top_z"] - layout["table_size_xyz"][2] / 2,
    )
    half = (
        layout["table_size_xyz"][0] / 2,
        layout["table_size_xyz"][1] / 2,
        layout["table_size_xyz"][2] / 2,
    )
    return np.asarray(center, dtype=float), np.asarray(half, dtype=float)


def tool0_pose_for_target(target, frame, electrode_offset_m, tilt_deg=0.0,
                          azimuth_deg=0.0):
    """Tool pose for pressing a target.

    `tilt_deg` rotates the approach direction away from the surface normal
    (up to a clinically acceptable bound) to keep the target reachable; the
    contact point is unchanged and only the wrist orientation is relaxed.
    """
    normal = np.asarray(target.normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    approach_normal = normal
    if tilt_deg:
        lateral_axis = np.asarray(frame.lateral, dtype=float)
        up_axis = np.cross(approach_normal, lateral_axis)
        up_axis /= np.linalg.norm(up_axis) + 1e-12
        # rotate the normal away from the surface in the (lateral, up) plane
        angle = np.radians(float(tilt_deg))
        azimuth = np.radians(float(azimuth_deg))
        axis = np.cos(azimuth) * lateral_axis + np.sin(azimuth) * up_axis
        approach_normal = (
            approach_normal * np.cos(angle)
            + np.cross(axis, approach_normal) * np.sin(angle)
            + axis * float(np.dot(axis, approach_normal)) * (1.0 - np.cos(angle))
        )
        approach_normal /= np.linalg.norm(approach_normal) + 1e-12
    approach = -approach_normal
    rotation = rotation_from_axes(frame.lateral, approach)
    tool0_world = np.asarray(target.position, dtype=float) + (
        approach_normal * electrode_offset_m
    )
    return tool0_world, rotation


def transit_pose(frame, height_m=0.30):
    """Safe transit tool pose: above the chest centre, pointing down."""
    position = np.asarray(frame.origin, dtype=float) + frame.anterior * height_m
    rotation = rotation_from_axes(frame.lateral, -frame.anterior)
    return position, rotation


def solve_tool_pose_warm(ik, base_matrix, position_world, rotation_world, warm_start):
    """IK for a world tool pose with fallback seeds; raises if unsolvable."""
    base_rotation = base_matrix[:3, :3]
    base_translation = base_matrix[:3, 3]
    local = base_rotation.T @ (np.asarray(position_world) - base_translation)
    quaternion = quat_wxyz_from_rotation(base_rotation.T @ np.asarray(rotation_world))
    seeds = [np.asarray(warm_start, dtype=float), np.zeros(6),
             np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0])]
    for seed in seeds:
        joints, ok = ik.solve_pose(local, quaternion, seed, orientation_tolerance=0.2)
        if ok and np.all(np.isfinite(joints)):
            return np.asarray(joints, dtype=float)
    raise RuntimeError(f"IK failed for tool pose {np.round(position_world, 3).tolist()}")


def cartesian_joint_path(ik, base_matrix, start_joints, start_pose, end_pose, steps):
    """Joint waypoints along a Cartesian line + orientation slerp (IK per step)."""
    start_position = np.asarray(start_pose[0], dtype=float)
    end_position = np.asarray(end_pose[0], dtype=float)
    start_quat = quat_wxyz_from_rotation(start_pose[1])
    end_quat = quat_wxyz_from_rotation(end_pose[1])
    path = []
    warm = np.asarray(start_joints, dtype=float)
    for t in np.linspace(0.0, 1.0, max(2, int(steps))):
        position = start_position + (end_position - start_position) * t
        rotation = rotation_from_quat_wxyz(slerp_wxyz(start_quat, end_quat, t))
        warm = solve_tool_pose_warm(ik, base_matrix, position, rotation, warm)
        path.append(warm)
    return path


def add_target_visualization(stage, frame, targets):
    from roboecg.task_manager.ecg_scene import add_frame_axes, add_line, add_marker

    add_frame_axes(stage, "/World/Markers/ChestFrame", frame)
    colors = {
        "V1": (0.95, 0.15, 0.15),
        "V2": (0.95, 0.45, 0.10),
        "V3": (0.95, 0.85, 0.10),
        "V4": (0.25, 0.85, 0.25),
        "V5": (0.15, 0.55, 0.95),
        "V6": (0.55, 0.25, 0.95),
    }
    for target in targets:
        color = colors.get(target.name, (0.9, 0.9, 0.9))
        add_marker(
            stage, f"/World/Markers/{target.name}", target.position, 0.011, color
        )
        add_line(
            stage,
            f"/World/Markers/{target.name}_normal",
            target.position,
            np.asarray(target.position) + np.asarray(target.normal) * 0.04,
            color=color,
            width=0.003,
        )


def make_render_product(camera_path, width, height, annotator_name="rgb"):
    import omni.replicator.core as rep

    product = rep.create.render_product(camera_path, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
    annotator.attach(product)
    rep.orchestrator.step(rt_subframes=1)
    rep.orchestrator.step(rt_subframes=1)
    return annotator


def annotator_frame(annotator, width, height):
    raw = np.asarray(annotator.get_data())
    if raw.ndim == 1 and raw.size == width * height * 4:
        raw = raw.reshape(height, width, 4)
    if raw.ndim != 3 or raw.shape[2] < 3:
        raise RuntimeError(f"invalid RGB frame shape {raw.shape}")
    return raw[:, :, :3].copy()


def save_png(path, array):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype("uint8"), mode="RGB").save(path)


def encode_video(frames_dir, video_path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
            "-i", str(frames_dir / "%06d.png"), "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
    )


def execute_sequence(
    app,
    world,
    stage,
    robot,
    ik,
    base_matrix,
    segments,
    video_annotator,
    closeup_annotator,
    width,
    height,
    capsules,
    gui=False,
    video=True,
    runs_dir=None,
    video_name="m1_reach.mp4",
    patient_motion=None,
    video_title="RoboECG - V1-V6 target reach",
):
    """Execute planned trajectories and interpolated transits, recording frames.

    Every motion sample (planned or interpolated) is checked against the patient
    capsules; the minimum clearance is reported even when negative, so a
    penetration can never be reported as success.

    `patient_motion(stage, frame_index, label) -> offset_m | None` is called
    once per motion frame (e.g. to breathe the patient during a press); the
    returned chest offset is logged per frame so the force model can account
    for it.
    """
    import omni.replicator.core as rep

    runs_dir = RUNS_DIR if runs_dir is None else Path(runs_dir)
    frames_dir = runs_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    base_rotation = base_matrix[:3, :3]
    base_translation = base_matrix[:3, 3]

    def tool0_world(joints):
        position, rotation = ik.forward(joints)
        return base_rotation @ position + base_translation, base_rotation @ rotation

    min_clearance = None
    min_clearance_info = None
    frame_index = 0
    per_segment = []
    target_metrics = []
    patient_motion_log = []
    force_track_log = []

    def move_to(q_target, label):
        nonlocal frame_index, min_clearance, min_clearance_info
        if patient_motion is not None:
            offset = patient_motion(stage, frame_index, label)
            if offset is not None:
                patient_motion_log.append(
                    {
                        "frame": frame_index,
                        "segment": label,
                        "offset_m": float(offset),
                    }
                )
        robot.set_dof_position_targets(q_target)
        world.step(render=True)
        q_actual = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
        position, rotation = tool0_world(q_actual)
        ecg_scene.set_tool_pose(stage, position, rotation)
        clearance = link_clearance_wrapper(ik, base_matrix, q_actual, capsules)
        if min_clearance is None or clearance[0] < min_clearance:
            min_clearance = float(clearance[0])
            min_clearance_info = {
                "clearance_m": float(clearance[0]),
                "link": clearance[1],
                "capsule": clearance[2],
                "frame": frame_index,
                "segment": label,
            }
        rgb = annotator_frame(video_annotator, width, height)
        frame_image = overlay_frame(
            rgb,
            title=video_title,
            stage=label,
            metrics=[
                f"min clearance so far: {min_clearance * 100:.1f} cm",
                f"frame {frame_index}",
            ],
        )
        if video:
            save_png(frames_dir / f"{frame_index:06d}.png", frame_image)
        rep.orchestrator.step(rt_subframes=1)
        if gui:
            app.update()
        frame_index += 1

    for segment in segments:
        label = segment["label"]
        if segment["kind"] == "stage":
            q_stage = np.asarray(segment["q"], dtype=float)
            robot.set_dof_positions(q_stage)
            robot.set_dof_position_targets(q_stage)
            for _ in range(3):
                world.step(render=True)
        elif segment["kind"] == "interp":
            start_q = (
                np.asarray(segment["start"], dtype=float)
                if segment.get("start") is not None
                else np.asarray(
                    robot.get_dof_positions(), dtype=float
                ).reshape(-1)
            )
            end_q = np.asarray(segment["end"], dtype=float)
            frames = max(2, int(segment["frames"]))
            for s_value in np.linspace(0.0, 1.0, frames):
                move_to(start_q + (end_q - start_q) * s_value, label)
            if segment.get("target_name"):
                q_hold = np.asarray(
                    robot.get_dof_positions(), dtype=float
                ).reshape(-1)
                hold_position, _ = tool0_world(q_hold)
                target_metrics.append(
                    {
                        "target": segment["target_name"],
                        "label": label,
                        "reach_error_m": float(
                            np.linalg.norm(
                                hold_position
                                - np.asarray(segment["requested_tool0_world"])
                            )
                        ),
                        "tracking_error_rad": float(
                            np.linalg.norm(q_hold - np.asarray(end_q))
                        ),
                    }
                )
        elif segment["kind"] == "force_track":
            from roboecg.robot_controller.compliant_press import (
                admittance_step,
            )
            from roboecg.robot_controller.press_plan import PressSettings
            from roboecg.robot_controller.reach_plan import solve_tool_pose

            controller = segment["controller"]
            contact = np.asarray(segment["contact_world"], dtype=float)
            normal = np.asarray(segment["normal_world"], dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            rotation = np.asarray(segment["rotation_world"], dtype=float)
            offset_electrode = float(segment["electrode_offset_m"])
            ctrl_settings = PressSettings(
                press_depth_m=float(segment.get("initial_depth_m", 0.004)),
                force_target_n=float(controller["force_target_n"]),
                max_press_depth_m=float(controller["depth_cap_m"]),
            )
            gain = 1.0 / (
                float(controller["k_est_n_m"]) * float(controller["tau_s"])
            )
            dt = 1.0 / float(controller.get("rate_hz", 60.0))
            k_plant = float(controller["plant_stiffness_n_m"])
            noise = float(controller.get("sensor_noise_n", 0.0))
            retract = float(controller.get("retract_limit_m", 0.010))
            max_step_value = controller.get("max_step_m")
            max_step = float(max_step_value) if max_step_value else None
            rng = np.random.default_rng(int(controller.get("seed", 0)))
            depth_cmd = float(
                segment.get("initial_depth_m", ctrl_settings.press_depth_m)
            )
            q_cmd = np.asarray(segment["q_seed"], dtype=float).reshape(-1)
            trace = []
            ik_failures = 0
            requested_flange = None
            for _ in range(int(segment["frames"])):
                offset = 0.0
                if patient_motion is not None:
                    value = patient_motion(stage, frame_index, label)
                    if value is not None:
                        offset = float(value)
                flange, _ = tool0_world(q_cmd)
                tip = flange - normal * offset_electrode
                surface = contact + np.array([0.0, 0.0, offset])
                indentation = max(0.0, float((surface - tip) @ normal))
                force_meas = k_plant * indentation
                if noise > 0.0:
                    force_meas += float(rng.normal(0.0, noise))
                depth_cmd = admittance_step(
                    depth_cmd,
                    force_meas,
                    ctrl_settings,
                    gain,
                    dt,
                    retract,
                    max_step,
                )
                # The command is relative to the *tracked* surface (the depth
                # camera keeps tracking the chest during the hold), so the
                # depth cap bounds the skin indentation, not the excursion
                # below the nominal plane.
                requested_flange = (
                    surface + normal * (offset_electrode - depth_cmd)
                )
                q_new, ok = solve_tool_pose(
                    ik,
                    base_matrix,
                    requested_flange,
                    rotation,
                    q_cmd,
                    position_tolerance=5e-4,
                    orientation_tolerance=0.05,
                )
                if ok:
                    q_cmd = q_new
                else:
                    ik_failures += 1
                move_to(q_cmd, label)
                flange_new, _ = tool0_world(q_cmd)
                tip_new = flange_new - normal * offset_electrode
                indent_after = max(0.0, float((surface - tip_new) @ normal))
                trace.append(
                    {
                        "step": len(trace),
                        "offset_m": offset,
                        "indentation_m": indentation,
                        "indentation_after_m": indent_after,
                        "flange_error_mm": float(
                            np.linalg.norm(flange_new - requested_flange) * 1000.0
                        ),
                        "force_n": force_meas,
                        "depth_cmd_m": depth_cmd,
                        "ik_ok": bool(ok),
                    }
                )
            indents = [entry["indentation_m"] for entry in trace]
            indents_after = [entry["indentation_after_m"] for entry in trace]
            forces = [entry["force_n"] for entry in trace]
            flange_errors = [entry["flange_error_mm"] for entry in trace]
            max_force = float(max(forces)) if forces else 0.0
            max_indent = (
                max(max(indents), max(indents_after)) if indents else 0.0
            )
            depth_ok = bool(
                max_indent <= ctrl_settings.max_press_depth_m + 1e-9
            )
            force_ok = bool(max_force <= float(controller["force_limit_n"]) + 1e-9)
            q_final = np.asarray(
                robot.get_dof_positions(), dtype=float
            ).reshape(-1)
            flange_final, _ = tool0_world(q_final)
            metrics = {
                "frames": len(trace),
                "force_target_n": ctrl_settings.force_target_n,
                "max_force_n": max_force,
                "max_indentation_m": max_indent,
                "depth_ok": depth_ok,
                "force_ok": force_ok,
                "violation": None
                if (depth_ok and force_ok)
                else ("depth" if not depth_ok else "force"),
                "contact_loss_fraction": float(
                    np.mean([d <= 1e-9 for d in indents])
                )
                if indents
                else 0.0,
                "ik_failures": int(ik_failures),
                "depth_cmd_final_m": float(depth_cmd),
                "max_flange_error_mm": float(max(flange_errors))
                if flange_errors
                else 0.0,
            }
            if segment.get("target_name"):
                target_metrics.append(
                    {
                        "target": segment["target_name"],
                        "label": label,
                        "mode": "force_tracking",
                        "reach_error_m": (
                            float(
                                np.linalg.norm(
                                    flange_final
                                    - np.asarray(requested_flange, dtype=float)
                                )
                            )
                            if requested_flange is not None
                            else None
                        ),
                        "tracking_error_rad": float(
                            np.linalg.norm(q_final - q_cmd)
                        ),
                    }
                )
            force_track_log.append(
                {
                    "target": segment.get("target_name"),
                    "label": label,
                    "metrics": metrics,
                    "controller": dict(controller),
                    "trace": trace,
                }
            )
        else:
            trajectory = segment["trajectory"]
            for q in trajectory:
                move_to(q, label)
            for _ in range(int(segment.get("hold", 0))):
                move_to(trajectory[-1], segment.get("hold_label", label))
            if segment.get("target_name"):
                q_hold = np.asarray(
                    robot.get_dof_positions(), dtype=float
                ).reshape(-1)
                hold_position, _ = tool0_world(q_hold)
                target_metrics.append(
                    {
                        "target": segment["target_name"],
                        "reach_error_m": float(
                            np.linalg.norm(
                                hold_position
                                - np.asarray(segment["requested_tool0_world"])
                            )
                        ),
                        "tracking_error_rad": float(
                            np.linalg.norm(q_hold - np.asarray(trajectory[-1]))
                        ),
                    }
                )
        per_segment.append(
            {"label": label, "kind": segment["kind"], "min_clearance_m": min_clearance}
        )
        clearance_text = "n/a" if min_clearance is None else f"{min_clearance:.4f} m"
        print(f"M1: segment done: {label} (min clearance {clearance_text})", flush=True)

    closeup = annotator_frame(closeup_annotator, width, height)
    save_png(runs_dir / "chest_closeup.png", closeup)

    video_path = None
    if video:
        video_path = runs_dir / video_name
        encode_video(frames_dir, video_path)

    return {
        "frames": frame_index,
        "min_clearance_m": min_clearance,
        "min_clearance_info": min_clearance_info,
        "segments": per_segment,
        "targets": target_metrics,
        "video": str(video_path) if video_path else None,
        "patient_motion": patient_motion_log,
        "force_track": force_track_log,
    }


def link_clearance_wrapper(ik, base_matrix, joints, capsules):
    from roboecg.robot_controller.safety import link_clearance

    return link_clearance(link_world_positions(ik, base_matrix, joints), capsules)


def _plan_target_with_approach_ladder(
    ik, base_matrix, target, frame, electrode_offset,
    q_seed, capsules, ground_box,
):
    """Plan one target, relaxing the approach direction if needed.

    Pressing along the surface normal is preferred; when a target is not
    reachable (UR3 reach vs. the midaxillary V6), the approach is tilted away
    from the normal within a clinically acceptable bound (<= 30 deg) while the
    contact point stays fixed.  Returns None if no candidate plans safely.
    """
    for tilt_deg, azimuth_deg in APPROACH_CANDIDATES:
        tool0_world, rotation_world = tool0_pose_for_target(
            target, frame, electrode_offset,
            tilt_deg=tilt_deg, azimuth_deg=azimuth_deg,
        )
        try:
            plan = plan_reach(
                ik,
                base_matrix,
                tool0_world,
                rotation_world,
                q_seed=q_seed,
                capsules=capsules,
                pre_distances=(LAYOUT_M1["pre_approach_distance_m"], 0.08, 0.10),
                steps=LAYOUT_M1["motion_steps"],
                ground_box=ground_box,
            )
        except RuntimeError:
            continue
        plan["approach_tilt_deg"] = float(tilt_deg)
        plan["approach_azimuth_deg"] = float(azimuth_deg)
        plan["requested_tool0_world"] = np.asarray(tool0_world, dtype=float)
        return plan
    return None


def run_m1(app, gui: bool = False, video: bool = True) -> dict:
    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import Articulation

    from roboecg.robot_controller.ur3_lula import UR3LulaIK, find_articulation_roots
    from roboecg.task_manager.ecg_scene import world_matrix

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0)
    stage, scene_report = ecg_scene.build_scene(world)
    print("M1: scene built")

    joint_positions = read_joint_world_positions(stage)
    landmarks = read_chest_landmarks(joint_positions)
    frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
    rules = load_ecg_rules()

    # Surface source: the overhead depth cloud, i.e. what the deployed
    # pipeline can actually measure.  The asset mesh is a simulation-only
    # reference and plays that role in the M2 fusion comparison instead.
    from roboecg.coordinate_transform.camera import (
        CameraIntrinsics,
        cv_rotation_from_usd,
    )
    from roboecg.perception.depth import depth_to_world_points
    from roboecg.task_manager.rendering import capture_depth

    camera_path = "/World/Cameras/PerceptionRGBD"
    camera_matrix = world_matrix(stage, camera_path)
    depth = capture_depth(camera_path, M1_WIDTH, M1_HEIGHT)
    points = depth_to_world_points(
        depth,
        CameraIntrinsics.from_horizontal_fov(M1_WIDTH, M1_HEIGHT, 90.0),
        camera_matrix[:3, 3],
        cv_rotation_from_usd(camera_matrix[:3, :3]),
        stride=1,
    )
    result = generate_v1_v6(landmarks, frame, points, rules)
    targets = {t.name: t for t in result.targets}
    print("M1: targets", {name: np.round(t.frame_coords, 3).tolist() for name, t in targets.items()})

    # sensitivity of the (assumed) intercostal spacing
    sensitivity = {}
    for spacing in LAYOUT_M1["sensitivity_spacing_m"]:
        print(f"M1: sweep {spacing:.3f} start", flush=True)
        try:
            variant = generate_v1_v6(landmarks, frame, points, rules, spacing_m=spacing)
            sensitivity[f"{spacing:.3f}"] = {
                t.name: t.position.tolist() for t in variant.targets
            }
        except Exception as error:  # noqa: BLE001 - diagnostics, reported below
            sensitivity[f"{spacing:.3f}"] = {"error": repr(error)}
        print(f"M1: sweep {spacing:.3f} done", flush=True)
    baseline = {t.name: t.position.tolist() for t in result.targets}
    max_shift = max(
        (
            float(np.linalg.norm(np.asarray(variant["V4"]) - np.asarray(baseline["V4"])))
            for variant in sensitivity.values()
            if "V4" in variant
        ),
        default=None,
    )

    print("M1: adding visualization", flush=True)
    add_target_visualization(stage, frame, result.targets)
    print("M1: visualization done", flush=True)

    print("M1: finding articulation roots", flush=True)
    roots = find_articulation_roots(stage)
    print(f"M1: roots={roots}", flush=True)
    if not roots:
        raise RuntimeError("UR3 articulation root not found")
    ur3_roots = [path for path in roots if "UR3" in path]
    if not ur3_roots:
        raise RuntimeError(f"no UR3 articulation root among {roots}")
    robot = Articulation(ur3_roots[0])
    print("M1: articulation ok", flush=True)
    q_start = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)

    ik = UR3LulaIK(frame="tool0")
    print("M1: IK ok", flush=True)
    capsules = body_capsules(joint_positions)
    box_center, box_half = table_box()
    electrode_offset = ecg_scene.LAYOUT["tool_electrode_offset_m"]
    print("M1: capsules ok", flush=True)

    target_order = ("V1", "V2", "V3", "V4", "V5", "V6")
    tool_poses = [
        tool0_pose_for_target(targets[name], frame, electrode_offset)
        for name in target_order
    ]

    # The nominal scene base cannot reach the contralateral targets; place the
    # robot at the best searched pose (same component as M0).
    print("M1: searching base placement", flush=True)
    placement = search_base_placement(
        ik, frame, tool_poses, capsules, (box_center, box_half)
    )
    if placement["best"] is None:
        raise RuntimeError("no base placement found")
    print(
        f"M1: base best={np.round(placement['best']['position'], 3).tolist()} "
        f"yaw={placement['best']['yaw_deg']} "
        f"ik={placement['best']['ik_success']}/{placement['targets']} "
        f"clearance={placement['best']['min_clearance_m']}",
        flush=True,
    )
    # IK feasibility alone is not enough: a base must also admit a
    # collision-free trajectory for every target.  Try the ranked candidates
    # (best first) until the whole plan succeeds.
    candidates = []
    seen = set()
    for item in [placement["best"], *placement["top10"], *placement["top30"][:30]]:
        if item is None:
            continue
        key = (tuple(np.round(item["position"], 4)), float(item["yaw_deg"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)

    plans = None
    chosen_placement = None
    failure_notes = []
    for candidate in candidates:
        apply_base_placement(
            stage, candidate["position"], candidate["yaw_deg"]
        )
        base_matrix = world_matrix(stage, "/World/UR3/base_link")
        plans = []
        for name in target_order:
            target = targets[name]
            plan = _plan_target_with_approach_ladder(
                ik, base_matrix, target, frame, electrode_offset,
                q_start, capsules, (box_center, box_half),
            )
            if plan is None:
                failure_notes.append(
                    f"base={np.round(candidate['position'], 3).tolist()}"
                    f"/yaw={candidate['yaw_deg']:.0f}: {name} unreachable"
                )
                plans = None
                break
            clearance = trajectory_clearance(
                ik,
                base_matrix,
                plan["trajectory"],
                joint_positions,
                ground_box=(box_center, box_half),
            )
            if not clearance["ok"]:
                failure_notes.append(
                    f"base={np.round(candidate['position'], 3).tolist()}"
                    f"/yaw={candidate['yaw_deg']:.0f}: {name} clearance"
                )
                plans = None
                break
            q_goal = np.asarray(plan["trajectory"][-1], dtype=float)
            fk_position, _ = ik.forward(q_goal)
            fk_world = base_matrix[:3, :3] @ fk_position + base_matrix[:3, 3]
            plan.update(
                {
                    "target_name": name,
                    "planned_clearance": clearance,
                    "ik_error_m": float(
                        np.linalg.norm(fk_world - np.asarray(plan["goal_tool0_world"]))
                    ),
                }
            )
            plans.append(plan)
        if plans is not None:
            chosen_placement = candidate
            break
    if plans is None:
        raise RuntimeError(
            "no base placement admits a full collision-free plan: "
            + "; ".join(failure_notes[:6])
        )
    print(
        f"M1: chosen base={np.round(chosen_placement['position'], 3).tolist()} "
        f"yaw={chosen_placement['yaw_deg']} (of {len(candidates)} candidates)",
        flush=True,
    )
    for plan in plans:
        print(
            f"M1: {plan['target_name']} planned, "
            f"clearance={plan['planned_clearance']['min_clearance_m']:.4f} m, "
            f"backoff={plan['goal_backoff_m'] * 1000:.1f} mm, "
            f"tilt={plan['approach_tilt_deg']:.0f} deg",
            flush=True,
        )
    print("M1: planning done", flush=True)
    width, height = LAYOUT_M1["video_resolution"]
    annotator = make_render_product("/World/Cameras/ThirdView", width, height)
    print("M1: render product 1 ok", flush=True)
    closeup_annotator = make_render_product("/World/Cameras/ChestCloseup", width, height)
    print("M1: render product 2 ok", flush=True)

    # M1 scope: per-target planning + execution of the collision-checked
    # approach from its pre-approach pose.  Free-space transit between targets
    # (the arm sweeping across the chest) is M4's multi-target sequencing work;
    # here the robot is staged at each pre-pose directly, and the staging is
    # reported as such.  A Cartesian transit prototype was measured to
    # penetrate the patient capsules (see ECG_M1_FINDINGS.md), which is exactly
    # why sequencing needs a real planner.
    segments = []
    staging = []
    for plan in plans:
        name = plan["target_name"]
        q_pre = np.asarray(plan["q_pre"], dtype=float)
        q_goal = np.asarray(plan["q_goal"], dtype=float)
        staging.append({"target": name, "staged_q_pre": q_pre.tolist()})
        segments.append({"kind": "stage", "q": q_pre.tolist(), "label": f"{name}: stage"})
        segments.append(
            {
                "kind": "trajectory",
                "trajectory": plan["trajectory"],
                "label": f"{name}: approach",
                "hold": LAYOUT_M1["hold_frames"],
                "hold_label": f"{name}: contact hold",
                "target_name": name,
                "requested_tool0_world": np.asarray(
                    plan["requested_tool0_world"]
                ).tolist(),
            }
        )
        segments.append(
            {
                "kind": "trajectory",
                "trajectory": list(reversed(plan["trajectory"])),
                "label": f"{name}: retreat",
            }
        )

    robot.set_dof_positions(np.asarray(plans[0]["q_pre"], dtype=float))
    robot.set_dof_position_targets(np.asarray(plans[0]["q_pre"], dtype=float))
    for _ in range(20):
        world.step(render=False)

    execution = execute_sequence(
        app,
        world,
        stage,
        robot,
        ik,
        base_matrix,
        segments,
        annotator,
        closeup_annotator,
        width,
        height,
        capsules,
        gui=gui,
        video=video,
    )

    scene_path = PROJECT_ROOT / "assets" / "local" / "m1_ecg_scene.usd"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(scene_path))

    report = {
        "status": (
            "PASS"
            if execution["min_clearance_m"] is not None
            and execution["min_clearance_m"] >= CLEARANCE_MARGIN_M
            and all(
                item["reach_error_m"] <= plan["goal_backoff_m"] + 0.005
                for item, plan in zip(execution["targets"], plans)
            )
            else "FAIL"
        ),
        "acceptance": {
            "min_clearance_m": execution["min_clearance_m"],
            "margin_m": CLEARANCE_MARGIN_M,
            "reach_error_rule": "reach_error <= planned backoff + 5 mm",
        },
        "scene": scene_report,
        "chest_frame": frame.describe(),
        "targets": {
            name: {
                "position_world": t.position.tolist(),
                "normal_world": t.normal.tolist(),
                "frame_coords_uvn": t.frame_coords.tolist(),
                "provenance": t.provenance,
            }
            for name, t in targets.items()
        },
        "vertical": result.vertical,
        "horizontal": result.horizontal,
        "spacing_sensitivity": {
            "spacings_m": LAYOUT_M1["sensitivity_spacing_m"],
            "max_v4_shift_m": max_shift,
            "positions": sensitivity,
        },
        "staging": staging,
        "staging_note": (
            "M1 executes each target from its pre-approach pose; free-space "
            "transit between targets is M4 (multi-target sequencing)."
        ),
        "plans": [
            {
                "target": plan["target_name"],
                "goal_backoff_m": plan["goal_backoff_m"],
                "goal_tool0_world": np.asarray(plan["goal_tool0_world"]).tolist(),
                "requested_tool0_world": np.asarray(
                    plan["requested_tool0_world"]
                ).tolist(),
                "ik_error_m": plan["ik_error_m"],
                "planned_clearance": plan["planned_clearance"],
                "branch_joint_distance_rad": plan["branch_joint_distance_rad"],
            }
            for plan in plans
        ],
        "base_placement": placement,
        "execution": execution,
        "safety": {
            "margin_m": CLEARANCE_MARGIN_M,
            "min_clearance_m": execution["min_clearance_m"],
            "min_clearance_info": execution["min_clearance_info"],
        },
        "scene_usd": str(scene_path),
    }
    (RUNS_DIR / "m1_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report
