"""M0: supine ECG scene + chest landmark audit + V1-V6 region reachability.

What this script produces (all evidence-backed, no hand-written medical data):
  1. the supine scene (asset + table + UR3 + electrode tool + cameras),
  2. chest landmarks measured from the M_Medical_01 rig, with provenance,
  3. the chest frame built from those measurements,
  4. the ECG rule config provenance table (loader-enforced),
  5. the chest surface region measured from an RGB-D render (point cloud),
  6. a UR3 base-placement reachability search over that region.

Run:
    ./scripts/run_headless.sh scripts/m0_check_ecg_scene.py
    ./scripts/run_headless.sh scripts/m0_check_ecg_scene.py --gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import (  # noqa: E402
    RUNS_DIR,
    boot,
    capture_depth,
    capture_rgb,
    save_depth_png,
    save_png,
    write_json,
)

CAMERA_FOV_DEG = 90.0
GRID_CELLS = 4
MIN_CELL_POINTS = 10
SURFACE_NORMAL_RADIUS_M = 0.03
IK_SEEDS = (
    np.zeros(6),
    np.array([0.0, -1.2, 1.2, -1.5, -1.5, 0.0]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-search", action="store_true", help="skip the base-placement search"
    )
    parser.add_argument(
        "--surface",
        choices=("depth", "mesh"),
        default="depth",
        help="chest surface source: RGB-D render (depth) or asset mesh (mesh)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip RGB-D capture and screenshots (CPU-only host without GPU)",
    )
    return parser.parse_args()


def yaw_matrix(degrees: float) -> np.ndarray:
    from roboecg.task_manager.supine_pose import rotation_z

    return rotation_z(degrees)


def base_matrix(position, yaw_deg: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = yaw_matrix(yaw_deg)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def camera_pose(stage, prim_path):
    from roboecg.coordinate_transform.camera import cv_rotation_from_usd
    from roboecg.task_manager.ecg_scene import world_matrix

    matrix = world_matrix(stage, prim_path)
    return matrix[:3, 3], cv_rotation_from_usd(matrix[:3, :3])


def chest_surface_grid(points, frame, u_min, u_max, v_min, v_max, cells=GRID_CELLS):
    """Measure the anterior chest surface on a (u, v) grid from a point cloud.

    For each cell the anterior-most point (max n) is the surface sample; the
    surface normal is a PCA fit around that point.  Cells with too few points
    are reported as missing instead of being filled with a guessed value.
    """
    from roboecg.perception.depth import estimate_surface_normal

    delta = points - frame.origin
    along = delta @ frame.up
    lateral = delta @ frame.lateral
    normal = delta @ frame.anterior

    u_edges = np.linspace(u_min, u_max, cells + 1)
    v_edges = np.linspace(v_min, v_max, cells + 1)
    grid = []
    for i in range(cells):
        for j in range(cells):
            mask = (
                (along >= u_edges[i])
                & (along < u_edges[i + 1])
                & (lateral >= v_edges[j])
                & (lateral < v_edges[j + 1])
                & (normal > -0.10)
            )
            count = int(np.count_nonzero(mask))
            cell = {
                "u_center": float(0.5 * (u_edges[i] + u_edges[i + 1])),
                "v_center": float(0.5 * (v_edges[j] + v_edges[j + 1])),
                "point_count": count,
            }
            if count < MIN_CELL_POINTS:
                cell["status"] = "missing"
                grid.append(cell)
                continue
            cell_points = points[mask]
            cell_normal = normal[mask]
            index = int(np.argmax(cell_normal))
            surface_point = cell_points[index]
            fit_normal, inliers = estimate_surface_normal(
                points,
                surface_point,
                radius=SURFACE_NORMAL_RADIUS_M,
                orient_toward=None,
            )
            if fit_normal is not None:
                fit_normal = np.asarray(fit_normal, dtype=float)
                if float(np.dot(fit_normal, frame.anterior)) < 0.0:
                    fit_normal = -fit_normal
                angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(np.dot(fit_normal, frame.anterior), -1.0, 1.0)
                        )
                    )
                )
            else:
                angle = None
            cell.update(
                {
                    "status": "ok",
                    "point_world": surface_point.tolist(),
                    "n_m": float(cell_normal[index]),
                    "normal_world": None if fit_normal is None else fit_normal.tolist(),
                    "normal_angle_vs_frame_deg": angle,
                    "normal_inliers": int(inliers),
                }
            )
            grid.append(cell)
    return grid


def reachability_search(
    ik,
    frame,
    grid,
    chest_origin,
    base_link_local,
    capsules,
    table_box,
    tool_offset_m,
):
    """Test UR3 base placements against the measured chest surface grid."""
    from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation, rotation_from_axes
    from roboecg.robot_controller.reach_plan import link_world_positions
    from roboecg.robot_controller.safety import box_clearance, link_clearance

    valid = [cell for cell in grid if cell.get("status") == "ok"]
    if not valid:
        return {"status": "FAIL_NO_SURFACE", "candidates": []}

    approach = -frame.anterior
    tool_rotation = rotation_from_axes(frame.lateral, approach)
    tool_quat = quat_wxyz_from_rotation(tool_rotation)
    targets = [
        {
            "u": cell["u_center"],
            "v": cell["v_center"],
            "tool0_world": np.asarray(cell["point_world"], dtype=float)
            + frame.anterior * tool_offset_m,
        }
        for cell in valid
    ]

    candidates = []
    for dx in (-0.30, -0.15, 0.0, 0.15, 0.30):
        for dy in (0.30, 0.45, 0.60):
            for dz in (-0.25, -0.05, 0.15):
                for yaw in (135.0, 180.0, 225.0):
                    position = chest_origin + np.array([dx, dy, dz])
                    matrix = base_matrix(position, yaw) @ base_link_local
                    candidates.append(
                        {"position": position.tolist(), "yaw_deg": yaw, "matrix": matrix}
                    )

    results = []
    for candidate in candidates:
        matrix = candidate["matrix"]
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        success = 0
        min_clearance = None
        min_box = None
        joint_solutions = []
        for target in targets:
            local = rotation.T @ (target["tool0_world"] - translation)
            quaternion = quat_wxyz_from_rotation(rotation.T @ tool_rotation)
            solved = False
            for seed in IK_SEEDS:
                joints, ok = ik.solve_pose(local, quaternion, seed)
                if ok and np.all(np.isfinite(joints)):
                    solved = True
                    break
            if not solved:
                continue
            success += 1
            positions = link_world_positions(ik, matrix, joints)
            clearance = link_clearance(positions, capsules)
            if min_clearance is None or clearance[0] < min_clearance:
                min_clearance = float(clearance[0])
            if table_box is not None:
                center, half = table_box
                box = box_clearance(positions, center, half)
                if min_box is None or box < min_box:
                    min_box = float(box)
            joint_solutions.append([float(v) for v in joints])
        results.append(
            {
                "position": candidate["position"],
                "yaw_deg": candidate["yaw_deg"],
                "ik_success": success,
                "targets": len(targets),
                "min_clearance_m": min_clearance,
                "table_clearance_m": min_box,
                "joint_solutions": joint_solutions,
            }
        )

    def score(item):
        clearance = item["min_clearance_m"]
        return (
            item["ik_success"],
            -1.0 if clearance is None else clearance,
        )

    ranked = sorted(results, key=score, reverse=True)
    best = ranked[0] if ranked else None
    passed = sum(1 for item in results if item["ik_success"] == len(targets))
    from roboecg.robot_controller.safety import CLEARANCE_MARGIN_M

    clearance = best["min_clearance_m"] if best else None
    clearance_ok = clearance is not None and clearance >= CLEARANCE_MARGIN_M
    return {
        "status": (
            "PASS"
            if best and best["ik_success"] == len(targets) and clearance_ok
            else "PARTIAL"
        ),
        "targets": len(targets),
        "candidates_total": len(results),
        "candidates_passed": passed,
        "clearance_margin_m": CLEARANCE_MARGIN_M,
        "best_clearance_ok": bool(clearance_ok),
        "best": best,
        "top10": ranked[:10],
    }


def main() -> None:
    args = parse_args()
    app = boot(headless=not args.gui, width=args.width, height=args.height)
    try:
        from roboecg.coordinate_transform.camera import CameraIntrinsics
        from roboecg.perception.chest_landmarks import (
            audit_chest_landmarks,
            read_chest_landmarks,
        )
        from roboecg.perception.depth import depth_to_world_points
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.safety import body_capsules
        from roboecg.robot_controller.ur3_lula import UR3LulaIK
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg_rules import (
            force_target_n,
            load_ecg_rules,
            provenance_report,
        )
        from roboecg.task_manager import ecg_scene

        from isaacsim.core.api import World

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        stage, scene_report = ecg_scene.build_scene(world)
        print("M0: scene built")

        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        audit = audit_chest_landmarks(landmarks)
        print(f"M0: chest frame origin = {np.round(frame.origin, 4).tolist()}")

        rules = load_ecg_rules()
        provenance = provenance_report(rules)
        press_force = force_target_n(rules)
        print(f"M0: rules loaded, {len(provenance)} provenance entries")

        surface_source = args.surface
        depth = None
        if surface_source == "depth" and args.no_render:
            print("M0: --surface depth needs rendering; falling back to mesh")
            surface_source = "mesh"
        if surface_source == "depth":
            camera_position, cv_rotation = camera_pose(
                stage, "/World/Cameras/PerceptionRGBD"
            )
            depth = capture_depth(
                "/World/Cameras/PerceptionRGBD", args.width, args.height
            )
            intrinsics = CameraIntrinsics.from_horizontal_fov(
                args.width, args.height, CAMERA_FOV_DEG
            )
            points = depth_to_world_points(
                depth, intrinsics, camera_position, cv_rotation
            )
            bbox_min = np.asarray(scene_report["human_bbox_min"], dtype=float) - 0.05
            bbox_max = np.asarray(scene_report["human_bbox_max"], dtype=float) + 0.05
            inside = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
            points = points[inside]
            print(f"M0: depth points in human bbox = {points.shape[0]}")
        else:
            points = ecg_scene.mesh_world_points(stage, "/World/Human")
            print(f"M0: mesh points = {points.shape[0]}")

        # Reachability envelope (engineering, not a clinical target): from the
        # lower-spine landmark (waist) up to the clavicle, sternum midline to
        # the shoulder.  Clinical V1-V6 levels are determined in M1 by
        # measurement / a learned landmark detector, not by a fixed offset.
        u_min = float(frame.to_frame(landmarks.spine_lower)[0])
        u_max = float(frame.to_frame(landmarks.clavicle_left)[0])
        v_min = 0.0
        v_max = float(frame.to_frame(landmarks.shoulder_left)[1])
        grid = chest_surface_grid(points, frame, u_min, u_max, v_min, v_max)
        ok_cells = [cell for cell in grid if cell.get("status") == "ok"]
        angles = [
            cell["normal_angle_vs_frame_deg"]
            for cell in ok_cells
            if cell["normal_angle_vs_frame_deg"] is not None
        ]
        print(f"M0: surface cells ok = {len(ok_cells)}/{len(grid)}")

        table_center = [
            ecg_scene.LAYOUT["table_center_xy"][0],
            ecg_scene.LAYOUT["table_center_xy"][1],
            ecg_scene.LAYOUT["table_top_z"] - ecg_scene.LAYOUT["table_size_xyz"][2] / 2,
        ]
        table_half = [
            ecg_scene.LAYOUT["table_size_xyz"][0] / 2,
            ecg_scene.LAYOUT["table_size_xyz"][1] / 2,
            ecg_scene.LAYOUT["table_size_xyz"][2] / 2,
        ]

        reachability = {"status": "SKIPPED", "candidates": []}
        base_link_local = np.eye(4)
        if not args.no_search and ok_cells:
            ik = UR3LulaIK(frame="tool0")
            from roboecg.task_manager.ecg_scene import world_matrix

            analytic_nominal = base_matrix(
                ecg_scene.LAYOUT["ur3_nominal_base_position"],
                ecg_scene.LAYOUT["ur3_nominal_base_yaw_deg"],
            )
            usd_nominal = world_matrix(stage, "/World/UR3/base_link")
            base_link_local = np.linalg.inv(analytic_nominal) @ usd_nominal
            capsules = body_capsules(joint_positions)
            reachability = reachability_search(
                ik,
                frame,
                grid,
                frame.origin,
                base_link_local,
                capsules,
                (table_center, table_half),
                ecg_scene.LAYOUT["tool_electrode_offset_m"],
            )
            reachability["base_link_local_translation_m"] = (
                base_link_local[:3, 3].tolist()
            )
            reachability["base_link_local_rotation_deg"] = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(base_link_local[:3, :3]) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            print(
                f"M0: reachability {reachability['status']} "
                f"best={reachability['best']['ik_success']}/"
                f"{reachability['best']['targets']} "
                f"clearance={reachability['best']['min_clearance_m']}"
            )

        add_visualization(stage, landmarks, frame, grid, table_center, table_half)
        screenshots = {}
        if not args.no_render:
            rgb = capture_rgb("/World/Cameras/ThirdView", args.width, args.height)
            closeup = capture_rgb("/World/Cameras/ChestCloseup", args.width, args.height)
            save_png(RUNS_DIR / "third_view.png", rgb)
            save_png(RUNS_DIR / "chest_closeup.png", closeup)
            screenshots = {
                "third_view": str(RUNS_DIR / "third_view.png"),
                "chest_closeup": str(RUNS_DIR / "chest_closeup.png"),
            }
            if depth is not None:
                save_depth_png(RUNS_DIR / "depth.png", depth)
                screenshots["depth"] = str(RUNS_DIR / "depth.png")

        report = {
            "status": "PASS" if ok_cells and reachability.get("status") == "PASS" else "PARTIAL",
            "scene": scene_report,
            "landmark_audit": audit,
            "chest_frame": frame.describe(),
            "rules_provenance": provenance,
            "press_force_target_n": press_force,
            "surface": {
                "source": surface_source,
                "points": int(points.shape[0]),
                "surface_cells_ok": len(ok_cells),
                "surface_cells_total": len(grid),
                "normal_angle_vs_frame_deg_mean": (
                    float(np.mean(angles)) if angles else None
                ),
                "normal_angle_vs_frame_deg_max": (
                    float(np.max(angles)) if angles else None
                ),
                "note": (
                    "mesh source uses rest-pose asset points transformed by the "
                    "human root; the chest is unaffected by the arms-only pose"
                    if surface_source == "mesh"
                    else "depth source uses the PerceptionRGBD render"
                ),
            },
            "surface_grid": grid,
            "reachability": {
                key: value
                for key, value in reachability.items()
                if key != "top10"
            },
            "reachability_top10": reachability.get("top10", []),
            "screenshots": screenshots,
        }
        path = write_json("m0_report.json", report)
        print(f"M0: report written to {path}")
        print(f"M0: status = {report['status']}")
    finally:
        app.close()


def add_visualization(stage, landmarks, frame, grid, table_center, table_half):
    from roboecg.task_manager.ecg_scene import (
        add_frame_axes,
        add_line,
        add_marker,
        add_region_box,
    )

    colors = {
        "clavicle_left": (0.15, 0.45, 0.95),
        "clavicle_right": (0.15, 0.45, 0.95),
        "shoulder_left": (0.15, 0.75, 0.25),
        "shoulder_right": (0.15, 0.75, 0.25),
        "spine_lower": (0.55, 0.35, 0.95),
        "upper_chest": (0.95, 0.35, 0.75),
    }
    for name, color in colors.items():
        add_marker(stage, f"/World/Markers/{name}", getattr(landmarks, name), 0.014, color)
    add_marker(stage, "/World/Markers/chest_origin", frame.origin, 0.020, (0.95, 0.15, 0.15))
    add_frame_axes(stage, "/World/Markers/ChestFrame", frame)

    ok_points = [
        np.asarray(cell["point_world"], dtype=float)
        for cell in grid
        if cell.get("status") == "ok"
    ]
    for index, point in enumerate(ok_points):
        add_marker(stage, f"/World/Markers/Surface_{index:02d}", point, 0.008, (0.95, 0.75, 0.10))
    if ok_points:
        stacked = np.vstack(ok_points)
        center = 0.5 * (stacked.min(axis=0) + stacked.max(axis=0))
        size = (stacked.max(axis=0) - stacked.min(axis=0)) + 0.06
        add_region_box(stage, "/World/Markers/ReachRegion", center, size)

    add_region_box(
        stage,
        "/World/Markers/TableBox",
        np.asarray(table_center, dtype=float),
        np.asarray(table_half, dtype=float) * 2.0,
        color=(0.35, 0.35, 0.85),
        width=0.003,
    )
    for index, cell in enumerate(grid):
        if cell.get("status") != "ok":
            continue
        point = np.asarray(cell["point_world"], dtype=float)
        normal = cell.get("normal_world")
        if normal is None:
            continue
        add_line(
            stage,
            f"/World/Markers/Normal_{index:02d}",
            point,
            point + np.asarray(normal, dtype=float) * 0.05,
            color=(0.95, 0.55, 0.10),
            width=0.003,
        )


if __name__ == "__main__":
    main()
