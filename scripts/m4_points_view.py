"""Render labelled images of the final V1-V6 electrode positions.

The video makes the point distribution hard to read, so this script renders
high-resolution stills with the six electrodes marked and labelled:

  * overhead view (PerceptionRGBD): the point distribution on the chest
  * oblique view (ThirdView): the robot arm at the final (V6) electrode
  * chest closeup (ChestCloseup)

Labels are projected from 3D to pixels with the same intrinsics used by the
perception stack, so the overlay is geometrically exact.

Run:
    ./scripts/run_headless.sh scripts/m4_points_view.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "runs" / "m4" / "points"
WIDTH, HEIGHT = 1280, 720
FOV_DEG = 60.0


def main() -> None:
    app = boot(headless=True, width=WIDTH, height=HEIGHT)
    try:
        from isaacsim.core.api import World

        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
            project_world_to_pixel,
        )
        from roboecg.perception.chest_landmarks import read_chest_landmarks
        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.robot_controller.base_placement import (
            apply_base_placement,
            search_base_placement,
        )
        from roboecg.robot_controller.reach_plan import body_capsules
        from roboecg.robot_controller.ur3_lula import UR3LulaIK, find_articulation_roots
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.ecg_scene import world_matrix
        from roboecg.task_manager.m1_demo import (
            _plan_target_with_approach_ladder,
            add_target_visualization,
            annotator_frame,
            make_render_product,
            save_png,
            table_box,
            tool0_pose_for_target,
        )
        from roboecg.coordinate_transform.camera import (
            CameraIntrinsics,
            cv_rotation_from_usd,
        )
        from roboecg.perception.depth import depth_to_world_points
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.target_localization.ecg import generate_v1_v6
        from roboecg.target_localization.ecg_rules import load_ecg_rules
        from roboecg.task_manager.rendering import capture_depth

        from isaacsim.core.experimental.prims import Articulation

        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        joints = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joints)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        rules = load_ecg_rules()
        # surface measured by the deployed perception path (overhead depth)
        camera_path = "/World/Cameras/PerceptionRGBD"
        camera_matrix = ecg_scene.world_matrix(stage, camera_path)
        depth = capture_depth(camera_path, 1280, 720)
        points = depth_to_world_points(
            depth,
            CameraIntrinsics.from_horizontal_fov(1280, 720, 90.0),
            camera_matrix[:3, 3],
            cv_rotation_from_usd(camera_matrix[:3, :3]),
            stride=1,
        )
        result = generate_v1_v6(landmarks, frame, points, rules)
        targets = {t.name: t for t in result.targets}
        add_target_visualization(stage, frame, result.targets)
        print(
            "points view: targets",
            {n: np.round(t.position, 4).tolist() for n, t in targets.items()},
            flush=True,
        )

        # place the robot at the searched base and drive it to the LAST
        # electrode (V6) so the arm is visible at its final pose
        robot = Articulation(
            [p for p in find_articulation_roots(stage) if "UR3" in p][0]
        )
        ik = UR3LulaIK(frame="tool0")
        capsules = body_capsules(joints)
        box_center, box_half = table_box()
        offset = float(ecg_scene.LAYOUT["tool_electrode_offset_m"])
        tool_poses = [
            tool0_pose_for_target(targets[n], frame, offset)
            for n in ("V1", "V2", "V3", "V4", "V5", "V6")
        ]
        placement = search_base_placement(
            ik, frame, tool_poses, capsules, (box_center, box_half)
        )
        apply_base_placement(
            stage, placement["best"]["position"], placement["best"]["yaw_deg"]
        )
        base_matrix = world_matrix(stage, "/World/UR3/base_link")
        q = np.asarray(robot.get_dof_positions(), dtype=float).reshape(-1)
        v6 = targets["V6"]
        plan = _plan_target_with_approach_ladder(
            ik,
            base_matrix,
            v6,
            frame,
            offset,
            q,
            capsules,
            (box_center, box_half),
        )
        if plan is not None:
            q_goal = np.asarray(plan["trajectory"][-1], dtype=float)
            for s in np.linspace(0.0, 1.0, 90):
                robot.set_dof_position_targets(q + (q_goal - q) * s)
                world.step(render=True)
            print("points view: arm parked at V6", flush=True)
        else:
            print("points view: V6 pose unreachable, arm left at rest", flush=True)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        views = {
            "overhead": "/World/Cameras/PerceptionRGBD",
            "oblique": "/World/Cameras/ThirdView",
            "closeup": "/World/Cameras/ChestCloseup",
        }
        annotators = {
            name: make_render_product(path, WIDTH, HEIGHT)
            for name, path in views.items()
        }
        # warm up rendering
        for _ in range(5):
            world.step(render=True)
        for _ in range(3):
            world.step(render=True)

        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26
            )
            small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
            )
        except OSError:
            font = ImageFont.load_default()
            small = font

        colors = {
            "V1": (255, 40, 40),
            "V2": (255, 115, 25),
            "V3": (240, 215, 25),
            "V4": (65, 215, 65),
            "V5": (40, 140, 245),
            "V6": (140, 65, 245),
        }
        intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_DEG)

        # electrode spacing table (measured on the final targets)
        names = ("V1", "V2", "V3", "V4", "V5", "V6")
        spacing = {
            f"{a}-{b}": float(
                np.linalg.norm(
                    np.asarray(targets[a].position) - np.asarray(targets[b].position)
                )
                * 1000.0
            )
            for i, a in enumerate(names)
            for b in names[i + 1:]
        }

        for name, annotator in annotators.items():
            rgb = annotator_frame(annotator, WIDTH, HEIGHT)
            image = Image.fromarray(rgb).convert("RGB")
            if name == "overhead":
                image = image.rotate(-90, expand=False)
            draw = ImageDraw.Draw(image)
            camera_path = views[name]
            matrix = world_matrix(stage, camera_path)
            camera_position = matrix[:3, 3]
            cv_rotation = cv_rotation_from_usd(matrix[:3, :3])

            projected = {}
            for electrode in names:
                position = np.asarray(targets[electrode].position, dtype=float)
                pixel, _ = project_world_to_pixel(
                    position, camera_position, cv_rotation, intrinsics
                )
                if pixel is None:
                    continue
                u, v = float(pixel[0]), float(pixel[1])
                if name == "overhead":  # map through the 90 deg clockwise rotation
                    u, v = HEIGHT - 1 - v, u
                if 0 <= u < WIDTH and 0 <= v < HEIGHT:
                    projected[electrode] = (u, v, position)

            # markers: crosshair + short name only
            label_offsets = {
                "V1": (-64, -12),
                "V2": (24, -12),
                "V3": (24, 16),
                "V4": (-70, 18),
                "V5": (24, 40),
                "V6": (-72, 44),
            }
            for electrode, (u, v, position) in projected.items():
                color = colors[electrode]
                draw.ellipse([u - 12, v - 12, u + 12, v + 12], outline=color, width=4)
                draw.line([u - 20, v, u + 20, v], fill=color, width=2)
                draw.line([u, v - 20, u, v + 20], fill=color, width=2)
                dx, dy = label_offsets.get(electrode, (18, -30))
                draw.text((u + dx, v + dy), electrode, fill=color, font=font)

            # legend panel (right side): name, world position, spacing to V1
            panel_w = 430
            panel = Image.new("RGB", (panel_w, HEIGHT), (18, 18, 18))
            pdraw = ImageDraw.Draw(panel)
            pdraw.text((14, 12), "Electrode positions (m)", fill=(255, 255, 255), font=small)
            y = 46
            for electrode in names:
                if electrode not in projected:
                    continue
                _, _, position = projected[electrode]
                color = colors[electrode]
                pdraw.rectangle([14, y + 4, 34, y + 20], fill=color)
                pdraw.text((44, y), electrode, fill=color, font=font)
                pdraw.text(
                    (44, y + 30),
                    f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})",
                    fill=(225, 225, 225),
                    font=small,
                )
                y += 74
            pdraw.text((14, y + 8), "Spacing (mm)", fill=(255, 255, 255), font=small)
            y += 42
            for key in ("V1-V2", "V2-V4", "V4-V5", "V5-V6", "V4-V6"):
                if key in spacing:
                    pdraw.text(
                        (20, y), f"{key}: {spacing[key]:.1f}", fill=(210, 210, 210),
                        font=small,
                    )
                    y += 26
            combined_view = Image.new("RGB", (WIDTH + panel_w, HEIGHT), (18, 18, 18))
            combined_view.paste(image, (0, 0))
            combined_view.paste(panel, (WIDTH, 0))
            image = combined_view
            draw = ImageDraw.Draw(image)
            title = {
                "overhead": "Overhead view (head at top) - V1..V6 electrode positions",
                "oblique": "Oblique view - UR3 parked at V6 (midaxillary)",
                "closeup": "Chest closeup - V1..V6 distribution",
            }[name]
            draw.rectangle([0, 0, WIDTH + panel_w, 42], fill=(0, 0, 0))
            draw.text((12, 8), title, fill=(255, 255, 255), font=font)
            path = OUT_DIR / f"{name}.png"
            image.save(path)
            print(f"points view: wrote {path}", flush=True)

        # combined 2-panel image (overhead + oblique)
        top = Image.open(OUT_DIR / "overhead.png")
        bottom = Image.open(OUT_DIR / "oblique.png")
        combined = Image.new(
            "RGB", (max(top.width, bottom.width), top.height + bottom.height + 8),
            (20, 20, 20),
        )
        combined.paste(top, (0, 0))
        combined.paste(bottom, (0, top.height + 8))
        combined.save(OUT_DIR / "points_overview.png")
        print(f"points view: wrote {OUT_DIR / 'points_overview.png'}", flush=True)
    except BaseException:
        import traceback

        with open("/tmp/roboecg_points_view_traceback.txt", "w") as handle:
            traceback.print_exc(file=handle)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
