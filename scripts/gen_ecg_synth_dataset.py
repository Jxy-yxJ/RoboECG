"""M3b dataset generator: overhead depth renders with chest-landmark labels.

For each sample the patient's body scale and position are randomised, the scene
is re-placed and an overhead depth image is rendered.  Labels are the projected
pixel coordinates of the rig chest landmarks (clavicle joints, shoulders,
chest, neck base, pelvis), plus the camera pose and the 3D landmarks so the
evaluation can lift predictions back to 3D.

Run:
    ./scripts/run_headless.sh scripts/gen_ecg_synth_dataset.py --samples 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset"

LANDMARK_ORDER = (
    "clavicle_left",
    "clavicle_right",
    "shoulder_left",
    "shoulder_right",
    "chest",
    "neck_base",
    "pelvis",
)
WIDTH = 320
HEIGHT = 180
FOV_DEG = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scale-min", type=float, default=0.88)
    parser.add_argument("--scale-max", type=float, default=1.12)
    parser.add_argument("--camera-jitter", type=float, default=0.02)
    parser.add_argument(
        "--abduction-min", type=float, default=60.0,
        help="minimum left-arm abduction in degrees (M1 needs >=75 to reach V6)",
    )
    parser.add_argument(
        "--abduction-max", type=float, default=90.0,
        help="maximum left-arm abduction in degrees",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
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
        from roboecg.target_localization.chest_frame import build_chest_frame
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.rendering import capture_depth
        from roboecg.task_manager.supine_pose import (
            apply_supine_pose,
            place_supine,
            set_prim_scale,
            set_prim_translate,
        )

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.0)
        stage, _ = ecg_scene.build_scene(world)
        human_prim = stage.GetPrimAtPath("/World/Human")
        camera_prim = "/World/Cameras/PerceptionRGBD"
        intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_DEG)

        samples = []
        for index in range(args.samples):
            scale = float(rng.uniform(args.scale_min, args.scale_max))
            dx = float(rng.uniform(-0.04, 0.04))
            dy = float(rng.uniform(-0.03, 0.03))
            # Arm pose is randomised too: the robot needs the left arm abducted
            # to reach V6 (M1), while the patient may hold other positions, so
            # the detector must not depend on the arm.
            abduction = float(
                rng.uniform(args.abduction_min, args.abduction_max)
            )
            apply_supine_pose(
                stage,
                pose=(
                    (("L_UpArm",), (0.0, 0.0, 1.0), 90.0 - abduction),
                    (("R_UpArm",), (0.0, 0.0, 1.0), -90.0),
                ),
            )
            set_prim_scale(human_prim, scale)
            place_supine(
                stage,
                human_root_path="/World/Human",
                root_xy=(
                    ecg_scene.LAYOUT["human_root_xy"][0] + dx,
                    ecg_scene.LAYOUT["human_root_xy"][1] + dy,
                ),
                table_top_z=ecg_scene.LAYOUT["table_top_z"],
            )
            joints = read_joint_world_positions(stage)
            landmarks = read_chest_landmarks(joints)
            frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))

            # small camera mounting jitter (robustness to the real installation)
            base_camera = ecg_scene.CAMERAS["PerceptionRGBD"]["position"]
            jitter = args.camera_jitter
            set_prim_translate(
                stage.GetPrimAtPath(camera_prim),
                (
                    base_camera[0] + float(rng.uniform(-jitter, jitter)),
                    base_camera[1] + float(rng.uniform(-jitter, jitter)),
                    base_camera[2] + float(rng.uniform(-jitter, jitter)),
                ),
            )

            camera_matrix = ecg_scene.world_matrix(stage, camera_prim)
            camera_position = camera_matrix[:3, 3]
            cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
            depth = capture_depth(camera_prim, WIDTH, HEIGHT)

            pixels = []
            world_points = []
            for name in LANDMARK_ORDER:
                position = np.asarray(getattr(landmarks, name), dtype=float)
                pixel, _ = project_world_to_pixel(
                    position, camera_position, cv_rotation, intrinsics
                )
                if pixel is None:
                    raise RuntimeError(f"landmark {name} not visible in sample {index}")
                pixels.append(pixel)
                world_points.append(position.tolist())

            sample_path = DATASET_DIR / f"sample_{index:04d}.npz"
            np.savez_compressed(
                sample_path,
                depth=(depth * 1000.0).astype(np.uint16),
                pixels=np.asarray(pixels, dtype=np.float32),
                world_points=np.asarray(world_points, dtype=np.float32),
                camera_position=camera_position.astype(np.float32),
                cv_rotation=cv_rotation.astype(np.float32),
                frame_origin=frame.origin.astype(np.float32),
                frame_up=frame.up.astype(np.float32),
                frame_lateral=frame.lateral.astype(np.float32),
                frame_anterior=frame.anterior.astype(np.float32),
                scale=np.float32(scale),
            )
            samples.append(
                {
                    "index": index,
                    "scale": scale,
                    "dx": dx,
                    "dy": dy,
                    "arm_abduction_deg": abduction,
                }
            )
            if index % 20 == 0:
                print(f"M3b data: {index}/{args.samples}", flush=True)

        (DATASET_DIR / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": samples,
                    "landmark_order": list(LANDMARK_ORDER),
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fov_deg": FOV_DEG,
                    "seed": args.seed,
                    "scale_range": [args.scale_min, args.scale_max],
                    "arm_abduction_range_deg": [
                        args.abduction_min,
                        args.abduction_max,
                    ],
                    "provenance": (
                        "synthetic: Isaac Sim overhead depth renders with rig "
                        "joint landmark labels (simulation GT)"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"M3b data: wrote {args.samples} samples to {DATASET_DIR}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
