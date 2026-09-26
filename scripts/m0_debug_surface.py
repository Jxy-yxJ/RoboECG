"""Temporary diagnostic: compare mesh-based and depth-based chest surfaces."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot, capture_depth  # noqa: E402
from m0_check_ecg_scene import chest_surface_grid, mesh_world_points  # noqa: E402


def main() -> None:
    app = boot(headless=True, width=1280, height=720)
    try:
        from isaacsim.core.api import World
        from pxr import UsdGeom

        from roboecg.coordinate_transform.camera import CameraIntrinsics
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.depth import depth_to_world_points
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.supine_pose import world_bounds

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)

        root_matrix = ecg_scene.world_matrix(stage, "/World/Human")
        print("human root translation:", np.round(root_matrix[:3, 3], 4).tolist())
        print("human root rotation:\n", np.round(root_matrix[:3, :3], 4))

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if not path.startswith("/World/Human"):
                continue
            matrix = ecg_scene.world_matrix(stage, path)
            print(f"mesh {path}")
            print("  world translation:", np.round(matrix[:3, 3], 4).tolist())
            print("  local rel. to root:", np.round(
                np.linalg.inv(root_matrix) @ matrix, 4).tolist()[:3])

        min_xyz, max_xyz = world_bounds(stage, "/World/Human")
        points = mesh_world_points(stage, "/World/Human")
        print("bbox (scene):", np.round(min_xyz, 3).tolist(), np.round(max_xyz, 3).tolist())
        print("bbox (root-transformed mesh points):", np.round(points.min(axis=0), 3).tolist(), np.round(points.max(axis=0), 3).tolist())

        joints = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joints)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        breast_mid = 0.5 * (landmarks.breast_left + landmarks.breast_right)
        u_min = float(frame.to_frame(breast_mid)[0] - 0.10)
        u_max = float(frame.to_frame(landmarks.clavicle_left)[0])
        v_min = float(frame.to_frame(breast_mid)[1])
        v_max = float(frame.to_frame(landmarks.shoulder_left)[1])

        mesh_grid = chest_surface_grid(points, frame, u_min, u_max, v_min, v_max)

        camera_position, cv_rotation = ecg_scene.world_matrix(
            stage, "/World/Cameras/PerceptionRGBD"
        )[:3, 3], None
        from roboecg.coordinate_transform.camera import cv_rotation_from_usd

        cam_matrix = ecg_scene.world_matrix(stage, "/World/Cameras/PerceptionRGBD")
        camera_position = cam_matrix[:3, 3]
        cv_rotation = cv_rotation_from_usd(cam_matrix[:3, :3])
        print("camera position:", np.round(camera_position, 4).tolist())
        depth = capture_depth("/World/Cameras/PerceptionRGBD", 1280, 720)
        intrinsics = CameraIntrinsics.from_horizontal_fov(1280, 720, 90.0)
        depth_points = depth_to_world_points(depth, intrinsics, camera_position, cv_rotation)
        bbox_min = np.asarray(min_xyz, dtype=float) - 0.05
        bbox_max = np.asarray(max_xyz, dtype=float) + 0.05
        inside = np.all((depth_points >= bbox_min) & (depth_points <= bbox_max), axis=1)
        depth_points = depth_points[inside]
        depth_grid = chest_surface_grid(depth_points, frame, u_min, u_max, v_min, v_max)

        print("\ncell            mesh_n   depth_n   diff_cm   mesh_pt                       depth_pt")
        for m, d in zip(mesh_grid, depth_grid):
            if m["status"] != "ok" or d["status"] != "ok":
                continue
            mp = np.round(m["point_world"], 4).tolist()
            dp = np.round(d["point_world"], 4).tolist()
            diff = (d["n_m"] - m["n_m"]) * 100
            print(
                f"u={m['u_center']:+.3f} v={m['v_center']:+.3f} "
                f"{m['n_m']:+.4f} {d['n_m']:+.4f} {diff:+7.2f}   {mp} {dp}"
            )
    finally:
        app.close()


if __name__ == "__main__":
    main()
