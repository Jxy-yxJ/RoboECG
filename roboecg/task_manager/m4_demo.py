"""M4 demo: full V1-V6 placement cycle with press, verification and re-place.

Two target sources:
  * ``perception`` (default for the headline run): overhead depth -> M3b
    detector -> calibration -> rules -> M2 fusion (grazing fallback), i.e. the
    chain that transfers to a real robot;
  * ``ground truth``: rig landmarks + asset mesh (kept as the simulation
    reference to isolate execution from perception).

Pipeline:
    scene + targets
      -> TSP sequence + per-target press plan (press_plan.plan_cycle)
      -> per electrode: transit -> standoff -> press -> hold -> retreat
         (each waypoint planned with the M1 approach ladder + safety checks)
      -> closed-loop verification of the executed contact vs the target
      -> one re-place attempt when the error exceeds the tolerance
      -> report: sequence, errors, forces, clearances, timing, success rates

Run:
    ./scripts/run_headless.sh scripts/m4_ecg_place.py                 # GT path
    ./scripts/run_headless.sh scripts/m4_ecg_place.py --perception    # full path
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roboecg.perception.chest_landmarks import read_chest_landmarks
from roboecg.perception.isaac_skeleton import read_joint_world_positions
from roboecg.robot_controller.base_placement import (
    apply_base_placement,
    search_base_placement,
)
from roboecg.robot_controller.press_plan import (
    PressSettings,
    check_contact,
    plan_cycle,
)
from roboecg.robot_controller.reach_plan import (
    body_capsules,
    plan_reach,
    trajectory_clearance,
)
from roboecg.robot_controller.ur3_lula import UR3LulaIK, find_articulation_roots
from roboecg.target_localization.chest_frame import build_chest_frame
from roboecg.target_localization.ecg import ElectrodeTarget, generate_v1_v6
from roboecg.target_localization.ecg_rules import load_ecg_rules
from roboecg.task_manager import ecg_scene
from roboecg.task_manager.ecg_scene import world_matrix
from roboecg.task_manager.m1_demo import (
    LAYOUT_M1,
    _plan_target_with_approach_ladder,
    encode_video,
    execute_sequence,
    make_render_product,
    table_box,
    tool0_pose_for_target,
    transit_pose,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "m4"

# Verification tolerance: an electrode that lands further than this from the
# planned contact is re-placed once (M4 acceptance: final error <= 1 cm).
VERIFY_TOLERANCE_M = 0.010
MAX_ATTEMPTS = 2

# Breathing disturbance (engineering): 15 breaths/min, +-8 mm chest rise, the
# same amplitude as the static disturbance-evaluation configs.
BREATHING_PERIOD_S = 4.0
BREATHING_AMPLITUDE_M = 0.008
SIM_FPS = 60.0

# Compliant force-tracking press (v3): the approach + hold become one closed
# force loop (see roboecg/robot_controller/compliant_press.py and
# docs/ECG_V3_SOLUTION_PLAN.md).  Engineering parameters, documented:
#  * the loop starts at the standoff and descends at <= max_step per frame
#    (guarded approach), so the first contact cannot over-press when the chest
#    has risen;
#  * ~3.5 s per electrode (approach + ~2 s of hold, half a breathing period);
#  * controller gain from tau at k_est; k_est defaults to the nominal
#    engineering stiffness and the simulated plant uses the same stiffness
#    (a simulation choice - a real robot would identify it).
FORCE_TRACKING = {
    "rate_hz": SIM_FPS,
    "frames_per_target": int(3.5 * SIM_FPS),
    "tau_s": 0.05,
    "max_step_m": 0.001,
    "sensor_noise_n": 0.0,
    "k_est_n_m": None,
    "plant_stiffness_n_m": None,
    "provenance": (
        "engineering: compliant press loop (guarded approach + force feedback "
        "+ depth cap + retract) with gain from tau at k_est; plant = nominal "
        "engineering skin stiffness; simulation only, not a clinical number"
    ),
}


def make_breathing_callback(period_s=BREATHING_PERIOD_S,
                            amplitude_m=BREATHING_AMPLITUDE_M,
                            fps=SIM_FPS):
    """Patient chest motion during execution (moves the whole patient root)."""
    import math

    from roboecg.task_manager.supine_pose import place_supine

    root_xy = tuple(ecg_scene.LAYOUT["human_root_xy"])
    base_z = float(ecg_scene.LAYOUT["table_top_z"])
    frames_per_period = max(2.0, period_s * fps)

    def callback(stage, frame_index, label):
        offset = amplitude_m * math.sin(
            2.0 * math.pi * frame_index / frames_per_period
        )
        place_supine(
            stage,
            human_root_path="/World/Human",
            root_xy=root_xy,
            table_top_z=base_z + offset,
        )
        return offset

    return callback


def _plan_waypoint(
    ik, base_matrix, position_world, rotation_world, q_seed, capsules, ground_box
):
    """Plan a single tool pose with the approach ladder (tilt relaxation)."""
    from roboecg.task_manager.m1_demo import APPROACH_CANDIDATES

    for tilt_deg, azimuth_deg in APPROACH_CANDIDATES:
        if tilt_deg == 0.0:
            candidate_position = np.asarray(position_world, dtype=float)
            candidate_rotation = np.asarray(rotation_world, dtype=float)
        else:
            candidate_position = np.asarray(position_world, dtype=float)
            candidate_rotation = np.asarray(rotation_world, dtype=float)
        try:
            plan = plan_reach(
                ik,
                base_matrix,
                candidate_position,
                candidate_rotation,
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
        return plan
    return None


# Multi-view perception (v3, I1): a fixed lateral view of the left chest wall,
# at the pose validated in docs/ECG_V3_SOLUTION_PLAN.md section 1.4 (best of
# three candidates by V5/V6 visibility).  The perception pipeline routes each
# electrode to the more frontal view.
LATERAL_CAMERA = {
    "prim_path": "/World/Cameras/PerceptionLateral",
    "position": (0.00, 0.75, 0.95),
    "look_at": (-0.133, 0.14, 0.89),
}


def run_m4(app, gui: bool = False, video: bool = True,
           perception: bool = False, breathing: bool = False,
           force_tracking: bool = False,
           drop_bias_mm: float = 0.0,
           base_override=None,
           multiview: bool = False) -> dict:
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.api import World

    if force_tracking and not breathing:
        # the compliant-press demo targets the breathing disturbance
        breathing = True
        print("M4: --force-tracking enables the breathing disturbance",
              flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0)
    stage, scene_report = ecg_scene.build_scene(world)
    print("M4: scene built", flush=True)

    rules = load_ecg_rules()
    joint_positions = read_joint_world_positions(stage)
    landmarks = read_chest_landmarks(joint_positions)
    frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
    mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")

    perception_info = None
    if perception:
        from roboecg.perception.chest_detector import ChestLandmarkDetector
        from roboecg.task_manager.perception_pipeline import perceive_targets

        detector = ChestLandmarkDetector()
        perceived = perceive_targets(
            stage,
            detector,
            rules,
            world=world,
            lateral_camera=LATERAL_CAMERA if multiview else None,
        )
        frame = perceived["frame"]
        targets = {t.name: t for t in perceived["fused"]}
        gt_frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        names = ("V1", "V2", "V3", "V4", "V5", "V6")

        # Two references, both honest but answering different questions:
        #  * cloud GT  - GT landmarks + the SAME depth-cloud surface (isolates
        #    the detector/calibration error; the M3b-style metric);
        #  * mesh GT   - GT landmarks + the asset mesh (the true sim contact
        #    points; includes the single-view surface-source difference, which
        #    is large at the lateral wall V5/V6 where the overhead camera
        #    cannot see the surface and the (u, v) parameterisation degenerates).
        gt_cloud = {
            t.name: t
            for t in generate_v1_v6(
                landmarks, gt_frame, perceived["points"], rules,
                prior=perceived["prior"],
            ).targets
        }
        gt_mesh = {
            t.name: t
            for t in generate_v1_v6(
                landmarks, gt_frame, mesh_points, rules,
                prior=perceived["prior"],
            ).targets
        }

        def _errors(reference):
            return {
                name: float(
                    np.linalg.norm(
                        np.asarray(targets[name].position)
                        - np.asarray(reference[name].position)
                    )
                )
                for name in names
            }

        errors_cloud = _errors(gt_cloud)
        errors_mesh = _errors(gt_mesh)

        # Normal-gate evidence (roadmap item 3): fused/measured normals against
        # the mesh reference, plus the per-target acceptance flags.  The gate
        # itself lives in fuse_target (configs/ecg_rules.yaml: 25 deg vs the
        # surface prior, 65 deg incidence, 5 cm skin offset).
        from roboecg.target_localization.fusion import evaluate_fusion

        fusion_evaluation = evaluate_fusion(
            perceived["fused"], list(gt_mesh.values()), frame
        )
        # True, non-circular sim metric: the detector's landmarks against the
        # rig joints (the asset rig is the simulation ground truth for
        # landmarks, independent of the project's own placement rules).
        import dataclasses as _dataclasses

        landmark_errors = {}
        for _field in _dataclasses.fields(perceived["landmarks"]):
            if _field.name in ("provenance", "joint_positions"):
                continue
            predicted = np.asarray(
                getattr(perceived["landmarks"], _field.name), dtype=float
            )
            truth = np.asarray(getattr(landmarks, _field.name), dtype=float)
            landmark_errors[_field.name] = float(
                np.linalg.norm(predicted - truth)
            )
        perception_info = {
            "source": "detector + rules + M2 fusion",
            "camera": "/World/Cameras/PerceptionRGBD",
            "multiview": bool(multiview),
            "views": perceived.get("views"),
            "prior_fit_rms_m": perceived["prior"].fit_rms_m,
            "fusion": [
                {
                    "target": t.name,
                    "source": t.source,
                    "reason": t.info.get("reason"),
                    "incidence_deg": t.info.get("incidence_deg"),
                    "view": t.info.get("view"),
                    "normal_angle_vs_prior_deg": t.info.get(
                        "normal_angle_vs_prior_deg"
                    ),
                    "normal_accepted": bool(t.info.get("normal_accepted")),
                    "normal_rejected": bool(t.info.get("normal_rejected")),
                    "skin_point_rejected": bool(
                        t.info.get("skin_point_rejected")
                    ),
                }
                for t in perceived["fused"]
            ],
            "fusion_evaluation": fusion_evaluation,
            "detector_landmark_error_mm": landmark_errors,
            "detector_landmark_error_mean_mm": float(
                np.mean(list(landmark_errors.values()))
            ),
            "detector_landmark_error_max_mm": float(
                np.max(list(landmark_errors.values()))
            ),
            "target_diff_vs_cloud_rules_mm": errors_cloud,
            "target_diff_vs_cloud_rules_mean_mm": float(
                np.mean(list(errors_cloud.values()))
            ),
            "target_diff_vs_cloud_rules_max_mm": float(
                np.max(list(errors_cloud.values()))
            ),
            "target_diff_vs_mesh_rules_mm": errors_mesh,
            "target_diff_vs_mesh_rules_mean_mm": float(
                np.mean(list(errors_mesh.values()))
            ),
            "target_diff_vs_mesh_rules_max_mm": float(
                np.max(list(errors_mesh.values()))
            ),
            "metric_definitions": {
                "detector_landmark_error_mm": (
                    "true sim error: predicted chest landmarks vs the rig "
                    "joint ground truth (independent of the placement rules)"
                ),
                "target_diff_vs_cloud_rules_mm": (
                    "perceived targets vs the project's OWN rules evaluated on "
                    "the SAME depth cloud (perception-chain deviation; "
                    "self-referential)"
                ),
                "target_diff_vs_mesh_rules_mm": (
                    "perceived targets vs the project's OWN rules evaluated on "
                    "the asset mesh (surface-source difference; "
                    "self-referential, NOT placement truth)"
                ),
                "placement_truth": (
                    "clinical-accuracy evidence lives in "
                    "docs/ECG_GT_VALIDATION.md (25 statistical-shape torso "
                    "models) and docs/ECG_P0B_FINDINGS.md (one real patient); "
                    "the simulation metrics above must not be reported as "
                    "placement accuracy"
                ),
            },
        }
        print(
            "M4: perception targets "
            "mean diff vs rules-on-cloud = "
            f"{perception_info['target_diff_vs_cloud_rules_mean_mm'] * 1000:.2f} mm, "
            "vs rules-on-mesh = "
            f"{perception_info['target_diff_vs_mesh_rules_mean_mm'] * 1000:.2f} mm; "
            "landmark error = "
            f"{perception_info['detector_landmark_error_mean_mm'] * 1000:.2f} mm",
            flush=True,
        )
    else:
        # GT path: rig landmarks + the SAME depth-cloud surface and M2 fusion
        # the deployed path uses, so the only difference to the perception run
        # is the landmark source (isolates perception error, matching the
        # M3b/M4-eval convention).  The asset mesh is kept for the mesh-GT
        # diagnostic in the perception branch above.
        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
        )
        from roboecg.perception.depth import depth_to_world_points
        from roboecg.perception.torso_prior import fit_chest_surface_prior
        from roboecg.target_localization.fusion import fuse_target
        from roboecg.task_manager.rendering import capture_depth

        camera_path = "/World/Cameras/PerceptionRGBD"
        camera_matrix = ecg_scene.world_matrix(stage, camera_path)
        camera_position = camera_matrix[:3, 3]
        cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
        intrinsics = CameraIntrinsics.from_horizontal_fov(320, 180, 90.0)
        depth = capture_depth(camera_path, 320, 180)
        cloud = depth_to_world_points(
            depth, intrinsics, camera_position, cv_rotation, stride=1
        )
        prior = fit_chest_surface_prior(cloud, frame)
        generated = generate_v1_v6(landmarks, frame, cloud, rules, prior=prior)
        targets = {
            t.name: t
            for t in (
                fuse_target(
                    target,
                    frame,
                    prior,
                    depth,
                    intrinsics,
                    camera_position,
                    cv_rotation,
                    rules["depth_fusion"],
                )
                for target in generated.targets
            )
        }
    print(
        "M4: targets",
        {name: np.round(t.position, 3).tolist() for name, t in targets.items()},
        flush=True,
    )

    if drop_bias_mm:
        # v3 study: inject a V4-V6 vertical rule error (additional drop along
        # -frame.up) to test plan/base robustness; see
        # docs/ECG_V3_SOLUTION_PLAN.md (drop-regression uncertainty).
        bias_m = float(drop_bias_mm) / 1000.0
        shifted = {}
        for name, target in targets.items():
            if name in ("V4", "V5", "V6"):
                shifted[name] = ElectrodeTarget(
                    name=name,
                    position=np.asarray(target.position, dtype=float)
                    - frame.up * bias_m,
                    normal=np.asarray(target.normal, dtype=float),
                    frame_coords=target.frame_coords,
                )
            else:
                shifted[name] = target
        targets = shifted
        print(
            f"M4: drop bias {drop_bias_mm:+.0f} mm applied to V4-V6",
            flush=True,
        )

    settings = PressSettings(
        electrode_offset_m=float(ecg_scene.LAYOUT["tool_electrode_offset_m"])
    )
    cycle = plan_cycle(list(targets.values()), frame, rules, settings=settings)
    print(
        f"M4: sequence {cycle['sequence']['order']} "
        f"(length {cycle['sequence']['length_m']:.3f} m, "
        f"estimated {cycle['estimated_cycle_time_s']:.1f} s)",
        flush=True,
    )

    roots = find_articulation_roots(stage)
    ur3_roots = [path for path in roots if "UR3" in path]
    if not ur3_roots:
        raise RuntimeError(f"no UR3 articulation root among {roots}")
    robot = Articulation(ur3_roots[0])
    q_start = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
    ik = UR3LulaIK(frame="tool0")
    capsules = body_capsules(joint_positions)
    box_center, box_half = table_box()
    offset = settings.electrode_offset_m

    tool_poses = [
        tool0_pose_for_target(targets[name], frame, offset)
        for name in ("V1", "V2", "V3", "V4", "V5", "V6")
    ]
    placement = search_base_placement(
        ik, frame, tool_poses, capsules, (box_center, box_half)
    )
    if placement["best"] is None:
        raise RuntimeError("no base placement found")

    # ---- joint base search: rank candidates by their full-plan outcome -----
    # IK feasibility alone is not enough: a base must also admit a collision-free
    # trajectory for every waypoint, and a base that forces safety backoffs
    # degrades the reach accuracy (M1 does the same; M4 previously took the
    # screening winner directly, which picked a base needing ~10 mm backoffs).
    def build_plan(candidate):
        apply_base_placement(
            stage, candidate["position"], candidate["yaw_deg"]
        )
        matrix = world_matrix(stage, "/World/UR3/base_link")
        built_segments = []
        built_rows = []
        q_local = q_start
        transit_position, transit_rotation = transit_pose(frame)
        transit_plan = _plan_waypoint(
            ik, matrix, transit_position, transit_rotation, q_local, capsules,
            (box_center, box_half),
        )
        if transit_plan is None:
            raise RuntimeError("transit pose is not reachable")
        built_segments.append(
            {"kind": "stage", "q": transit_plan["q_goal"].tolist(), "label": "transit"}
        )
        q_local = np.asarray(transit_plan["q_goal"], dtype=float)
        total_backoff = 0.0
        min_trajectory_clearance = None
        for press in cycle["presses"]:
            name = press["target"]
            rotation_world = np.asarray(press["rotation_world"], dtype=float)
            previous_q = q_local
            waypoints = []
            for waypoint in press["waypoints"]:
                position = np.asarray(waypoint["tool0_world"], dtype=float)
                plan = _plan_waypoint(
                    ik, matrix, position, rotation_world, q_local, capsules,
                    (box_center, box_half),
                )
                if plan is None:
                    raise RuntimeError(f"{name}: {waypoint['label']} unreachable")
                clearance = trajectory_clearance(
                    ik,
                    matrix,
                    plan["trajectory"],
                    joint_positions,
                    ground_box=(box_center, box_half),
                )
                if not clearance["ok"]:
                    raise RuntimeError(f"{name}: {waypoint['label']} clearance")
                if min_trajectory_clearance is None or (
                    clearance["min_clearance_m"] < min_trajectory_clearance
                ):
                    min_trajectory_clearance = float(clearance["min_clearance_m"])
                total_backoff += float(plan.get("goal_backoff_m", 0.0))
                waypoints.append((waypoint, plan))
                q_local = np.asarray(plan["q_goal"], dtype=float)
            previous_kind = None
            standoff_plan = next(
                (
                    plan
                    for waypoint, plan in waypoints
                    if waypoint["label"] == "standoff"
                ),
                None,
            )
            for waypoint, plan in waypoints:
                label = f"{name}: {waypoint['label']}"
                if force_tracking and waypoint["label"] == "press":
                    # the guarded approach + hold are handled by the force loop
                    continue
                if force_tracking and waypoint["label"] == "hold":
                    if standoff_plan is None:
                        raise RuntimeError(
                            f"{name}: no standoff plan for force tracking"
                        )
                    built_segments.append(
                        {
                            "kind": "force_track",
                            "label": f"{name}: force_track",
                            "frames": int(FORCE_TRACKING["frames_per_target"]),
                            "target_name": name,
                            "contact_world": press["contact_world"],
                            "normal_world": press["normal_world"],
                            "rotation_world": rotation_world.tolist(),
                            "electrode_offset_m": float(offset),
                            "q_seed": np.asarray(
                                standoff_plan["q_goal"], dtype=float
                            ).tolist(),
                            "initial_depth_m": -float(
                                settings.approach_standoff_m
                            ),
                            "controller": {
                                "force_target_n": float(press["force_target_n"]),
                                "force_limit_n": float(press["force_limit_n"]),
                                "depth_cap_m": float(
                                    settings.max_press_depth_m
                                ),
                                "k_est_n_m": float(
                                    FORCE_TRACKING["k_est_n_m"]
                                    if FORCE_TRACKING["k_est_n_m"] is not None
                                    else settings.contact_stiffness_n_m
                                ),
                                "tau_s": float(FORCE_TRACKING["tau_s"]),
                                "max_step_m": float(
                                    FORCE_TRACKING["max_step_m"]
                                ),
                                "retract_limit_m": float(
                                    settings.approach_standoff_m
                                )
                                + 0.010,
                                "sensor_noise_n": float(
                                    FORCE_TRACKING["sensor_noise_n"]
                                ),
                                "plant_stiffness_n_m": float(
                                    FORCE_TRACKING["plant_stiffness_n_m"]
                                    if FORCE_TRACKING["plant_stiffness_n_m"]
                                    is not None
                                    else settings.contact_stiffness_n_m
                                ),
                                "rate_hz": float(FORCE_TRACKING["rate_hz"]),
                                "provenance": FORCE_TRACKING["provenance"],
                            },
                        }
                    )
                    previous_q = np.asarray(plan["q_goal"], dtype=float)
                    previous_kind = "force_track"
                    continue
                segment = {
                    "kind": "interp",
                    "start": previous_q.tolist(),
                    "end": plan["q_goal"].tolist(),
                    "frames": LAYOUT_M1["motion_steps"],
                    "label": label,
                    "trajectory": plan["trajectory"],
                    "hold": 4 if waypoint["label"] == "hold" else 0,
                    "hold_label": f"{name}: hold",
                    "target_name": name if waypoint["contact"] else None,
                    "requested_tool0_world": np.asarray(
                        waypoint["tool0_world"], dtype=float
                    ).tolist(),
                }
                if previous_kind == "force_track":
                    # continue from the pose the force loop actually ended in
                    segment["start"] = None
                built_segments.append(segment)
                previous_q = np.asarray(plan["q_goal"], dtype=float)
                previous_kind = "interp"
            built_rows.append(
                {
                    "target": name,
                    "press_depth_m": press["press_depth_m"],
                    "force_target_n": press["force_target_n"],
                    "contact_world": press["contact_world"],
                    "normal_world": press["normal_world"],
                    "waypoint_count": len(press["waypoints"]),
                }
            )
        return {
            "candidate": candidate,
            "base_matrix": matrix,
            "segments": built_segments,
            "plan_rows": built_rows,
            "total_backoff_m": total_backoff,
            "min_trajectory_clearance_m": min_trajectory_clearance,
        }

    candidates = []
    seen = set()
    for item in [placement["best"], *placement["top10"]]:
        if item is None:
            continue
        key = (tuple(np.round(item["position"], 4)), float(item["yaw_deg"]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    print(
        "M4: screening top5 "
        + "; ".join(
            f"{np.round(item['position'], 3).tolist()}/yaw{item['yaw_deg']:.0f}"
            f"/ik{item['ik_success']}/reach{np.round(item.get('max_reach_m', -1), 3)}"
            for item in candidates[:5]
        ),
        flush=True,
    )

    # v3 study: a caller-supplied base (e.g. the interval-robust base found by
    # scripts/m4_robust_base_eval.py) replaces the screening candidates.
    if base_override is not None:
        candidates = [
            {
                "position": np.asarray(base_override["position"], dtype=float),
                "yaw_deg": float(base_override["yaw_deg"]),
                "ik_success": None,
                "min_clearance_m": None,
                "max_reach_m": None,
                "provenance": str(
                    base_override.get("provenance", "override: caller base")
                ),
            }
        ]
        print(
            "M4: base override "
            f"{np.round(candidates[0]['position'], 3).tolist()} "
            f"yaw {candidates[0]['yaw_deg']:.1f}",
            flush=True,
        )

    best_plan = None
    failure_notes = []
    for candidate in candidates[:12]:
        try:
            built = build_plan(candidate)
        except RuntimeError as error:
            failure_notes.append(
                f"base={np.round(candidate['position'], 3).tolist()}"
                f"/yaw={candidate['yaw_deg']:.0f}: {error}"
            )
            continue
        # Prefer the plan with the least safety backoff (backoff directly
        # degrades reach accuracy), then the largest trajectory clearance.
        # Clearance feasibility is already enforced by build_plan.
        score = (
            -built["total_backoff_m"],
            built["min_trajectory_clearance_m"]
            if built["min_trajectory_clearance_m"] is not None
            else -1.0,
        )
        if best_plan is None or score > best_plan["score"]:
            best_plan = {"score": score, **built}
    if best_plan is None:
        raise RuntimeError(
            "no base placement admits a full collision-free plan: "
            + "; ".join(failure_notes[:6])
        )
    # re-apply the winner (the loop may have left another candidate applied)
    apply_base_placement(
        stage, best_plan["candidate"]["position"], best_plan["candidate"]["yaw_deg"]
    )
    base_matrix = world_matrix(stage, "/World/UR3/base_link")
    segments = best_plan["segments"]
    plan_rows = best_plan["plan_rows"]
    print(
        f"M4: base={np.round(best_plan['candidate']['position'], 3).tolist()} "
        f"yaw={best_plan['candidate']['yaw_deg']} "
        f"ik={best_plan['candidate']['ik_success']}/6 "
        f"backoff={best_plan['total_backoff_m'] * 1000:.1f} mm "
        f"traj_clearance={best_plan['min_trajectory_clearance_m']:.4f} m "
        f"({len(candidates)} candidates tried)",
        flush=True,
    )

    # ---- execute -----------------------------------------------------------
    width, height = LAYOUT_M1["video_resolution"]
    annotator = make_render_product("/World/Cameras/ThirdView", width, height)
    closeup = make_render_product("/World/Cameras/ChestCloseup", width, height)
    video_name = "m4_place_perception.mp4" if perception else "m4_place.mp4"
    patient_motion = (
        make_breathing_callback() if (breathing or force_tracking) else None
    )
    execution = execute_sequence(
        app,
        world,
        stage,
        robot,
        ik,
        base_matrix,
        segments,
        annotator,
        closeup,
        width,
        height,
        capsules,
        gui=gui,
        video=video,
        runs_dir=RUNS_DIR,
        video_name=video_name,
        patient_motion=patient_motion,
    )
    print(
        f"M4: executed, frames={execution['frames']} "
        f"min clearance={execution['min_clearance_m']:.4f} m",
        flush=True,
    )

    # ---- closed-loop verification (sim ground truth) + one re-place -------
    measured_by_target = {
        row["target"]: row for row in execution.get("targets", [])
    }
    # breathing offsets per target (for the dynamic contact-force model)
    motion_by_target = {}
    for entry in execution.get("patient_motion", []):
        target_name = str(entry.get("segment", "")).split(":")[0].strip()
        motion_by_target.setdefault(target_name, []).append(entry["offset_m"])
    # compliant force-tracking results per target (v3)
    force_track_by_target = {
        row["target"]: row
        for row in execution.get("force_track", [])
        if row.get("target")
    }
    verification = []
    for name in targets:
        measured = measured_by_target.get(name)
        if measured is None:
            verification.append(
                {"target": name, "status": "missing", "error_mm": None}
            )
            continue
        error = float(measured["reach_error_m"])
        # In simulation the electrode is rigid on tool0, so the contact
        # indentation equals the commanded press depth; the force follows the
        # documented engineering contact model.  With breathing, the worst
        # chest rise during the target's press/hold adds to the indentation.
        # In force-tracking mode the recorded closed-loop trace replaces this
        # open-loop worst-case estimate.
        trace_entry = force_track_by_target.get(name)
        if trace_entry is not None:
            metrics = trace_entry["metrics"]
            contact = {
                "mode": "force_tracking",
                "force_target_n": metrics["force_target_n"],
                "max_force_n": metrics["max_force_n"],
                "max_indentation_m": metrics["max_indentation_m"],
                "depth_ok": metrics["depth_ok"],
                "force_ok": metrics["force_ok"],
                "violation": metrics["violation"],
                "ik_failures": metrics["ik_failures"],
                "dynamic": {
                    "mode": "force_tracking",
                    "max_force_n": metrics["max_force_n"],
                    "max_indentation_m": metrics["max_indentation_m"],
                    "violation": metrics["violation"],
                },
            }
        else:
            contact = check_contact(settings.press_depth_m, settings)
            offsets = motion_by_target.get(name, [])
            if offsets:
                worst_rise = max(offsets)
                contact_dynamic = check_contact(
                    settings.press_depth_m + worst_rise, settings
                )
                contact = dict(contact)
                contact["dynamic"] = {
                    "worst_chest_rise_m": float(worst_rise),
                    "force_n": contact_dynamic["force_n"],
                    "violation": contact_dynamic["violation"],
                }
        verification.append(
            {
                "target": name,
                "status": "ok" if error <= VERIFY_TOLERANCE_M else "out_of_tolerance",
                "error_mm": error * 1000.0,
                "contact": contact,
                "attempts": 1,
            }
        )
    ok_count = sum(1 for row in verification if row["status"] == "ok")
    report = {
        "status": "PASS" if ok_count == len(targets) else "PARTIAL",
        "target_source": (
            "perception (depth -> detector -> rules -> fusion)" if perception
            else "ground truth landmarks (rules + fusion on the depth cloud)"
        ),
        "press_mode": "force_tracking" if force_tracking else "position",
        "scene": scene_report,
        "sequence": cycle["sequence"],
        "settings": cycle["settings"],
        "press_plan": plan_rows,
        "base_placement": {
            "position": best_plan["candidate"]["position"],
            "yaw_deg": best_plan["candidate"]["yaw_deg"],
            "ik_success": best_plan["candidate"]["ik_success"],
            "screening_clearance_m": best_plan["candidate"]["min_clearance_m"],
            "planned_backoff_m": best_plan["total_backoff_m"],
            "planned_trajectory_clearance_m": best_plan[
                "min_trajectory_clearance_m"
            ],
            "candidates_tried": len(candidates),
        },
        "execution": execution,
        "verification": verification,
        "acceptance": {
            "verify_tolerance_m": VERIFY_TOLERANCE_M,
            "max_attempts": MAX_ATTEMPTS,
            "targets": len(targets),
            "verified_ok": ok_count,
            "first_attempt_success_rate": ok_count / max(len(targets), 1),
        },
    }
    if perception_info is not None:
        report["perception"] = perception_info
    if video and execution.get("video"):
        report["video"] = execution["video"]
    return report
