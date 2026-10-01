"""I5-c stage 2: fine-tune data for the chest-landmark detector from the SSM torsos.

Landmark labels are **electrode-free**: the biped's rig landmarks are expressed
as surface fractions (u along the torso span, v as a fraction of the local
half-width; `roboecg.perception.torso_mesh.surface_fraction_of`) and applied to
each model's own surface (`surface_fraction_landmarks`).  The models are placed
supine with a fixed model-to-world mapping plus per-sample scale and world-xy
jitter, the overhead depth is rendered with the same camera and intrinsics as
the perception pipeline, and the landmark pixels come from projecting the
labels.

Output: runs/m3b/dataset_ssm/{manifest.json, sample_XXXX.npz}

Run headless:
    ./scripts/run_headless.sh scripts/m4_ssm_detector_data.py
    ./scripts/run_headless.sh scripts/m4_ssm_detector_data.py --models 5
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
    surface_fraction_landmarks,
    surface_fraction_of,
)

TORSO_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
OUT_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset_ssm"
LANDMARK_ORDER = (
    "clavicle_left",
    "clavicle_right",
    "shoulder_left",
    "shoulder_right",
    "chest",
    "neck_base",
    "pelvis",
)
VARIATIONS = (
    (1.00, 0.000, 0.000),
    (0.93, 0.030, -0.020),
    (1.07, -0.020, 0.030),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=25, help="how many models")
    parser.add_argument("--subdiv", type=int, default=2)
    parser.add_argument("--smooth-iter", type=int, default=3)
    args = parser.parse_args()

    app = boot(headless=True, width=1280, height=720)
    try:
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom

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

        world = World(stage_units_in_meters=1.0)
        stage, scene_report = ecg_scene.build_scene(world)
        print("ssm_data: scene built", flush=True)

        # --- biped reference: landmark positions as surface fractions ---
        joint_positions = read_joint_world_positions(stage)
        landmarks = read_chest_landmarks(joint_positions)
        frame = build_chest_frame(landmarks, anterior_hint=(0.0, 0.0, 1.0))
        mesh_points = ecg_scene.mesh_world_points(stage, "/World/Human")
        # Torso-only reference: the biped mesh includes the head, legs and
        # (abducted) arms, so whole-mesh spans and half-widths are wrong --
        # that compressed the transferred labels ~4x in a first pass.  Restrict
        # to the pelvis..neck_top band and a torso |v| cap.
        mesh_u = mesh_points @ frame.up
        mesh_v = mesh_points @ frame.lateral
        u_top = float(np.asarray(landmarks.neck_top) @ frame.up) + 0.03
        u_bot = float(np.asarray(landmarks.pelvis) @ frame.up) - 0.03
        torso_mask = (
            (mesh_u >= u_bot) & (mesh_u <= u_top) & (np.abs(mesh_v) <= 0.25)
        )
        torso_points = mesh_points[torso_mask]
        print(
            f"ssm_data: torso reference points: {len(torso_points)} / "
            f"{len(mesh_points)}",
            flush=True,
        )
        fractions = {}
        for name in LANDMARK_ORDER:
            fractions[name] = surface_fraction_of(
                torso_points,
                frame.up,
                frame.lateral,
                frame.anterior,
                np.asarray(getattr(landmarks, name), dtype=float),
            )
        print(
            "ssm_data: biped fractions:",
            {k: (round(v[0], 3), round(v[1], 3)) for k, v in fractions.items()},
            flush=True,
        )
        chest_anchor = np.asarray(landmarks.chest, dtype=float)

        # --- fixed model -> world mapping (stage-1 axes): +X->+Y(left),
        # -Y->+Z(anterior), +Z->-X(head); right-handed by construction ---
        a_m = np.array([1.0, 0.0, 0.0])
        b_m = np.array([0.0, -1.0, 0.0])
        c_m = np.cross(a_m, b_m)
        a_w = np.array([0.0, 1.0, 0.0])
        b_w = np.array([0.0, 0.0, 1.0])
        c_w = np.cross(a_w, b_w)
        rotation = np.column_stack([a_w, b_w, c_w]) @ np.column_stack(
            [a_m, b_m, c_m]
        ).T
        up_w = -c_w  # towards the head is +Z_model -> -X_world
        lateral_w = a_w
        anterior_w = b_w

        # --- camera / intrinsics, same as the perception pipeline ---
        camera_path = "/World/Cameras/PerceptionRGBD"
        camera_matrix = ecg_scene.world_matrix(stage, camera_path)
        camera_position = camera_matrix[:3, 3]
        cv_rotation = cv_rotation_from_usd(camera_matrix[:3, :3])
        intrinsics = CameraIntrinsics.from_horizontal_fov(320, 180, 90.0)

        stage.GetPrimAtPath("/World/Human").SetActive(False)

        torso_prim = UsdGeom.Mesh.Define(stage, "/World/RealTorso")
        torso_prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        torso_prim.CreateDisplayColorAttr([Gf.Vec3f(0.87, 0.72, 0.63)])

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        samples = []
        index = 0
        skipped = 0
        for model_index in range(1, args.models + 1):
            model = f"T_{model_index:02d}"
            points_mm, faces = parse_vtk_polydata(
                TORSO_DIR / f"{model}_torso_coarse_surface.vtk"
            )
            points_m = points_mm / 1000.0
            if args.subdiv > 0 or args.smooth_iter > 0:
                points_m, faces = subdivide_and_smooth(
                    points_m,
                    faces,
                    subdivisions=args.subdiv,
                    smooth_iterations=args.smooth_iter,
                )
            centroid = points_m.mean(axis=0)
            for scale, dx, dy in VARIATIONS:
                scaled = (points_m - centroid) * float(scale)
                anchor = chest_anchor + np.array([float(dx), float(dy), 0.08])
                world_points = (scaled @ rotation.T) + anchor
                labels = surface_fraction_landmarks(
                    world_points, up_w, lateral_w, anterior_w, fractions
                )
                torso_prim.CreatePointsAttr(
                    [Gf.Vec3f(*[float(c) for c in p]) for p in world_points]
                )
                torso_prim.CreateFaceVertexCountsAttr([3] * len(faces))
                torso_prim.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
                world.step(render=True)
                depth = capture_depth(camera_path, 320, 180)
                pixels = []
                world_label_points = []
                for name in LANDMARK_ORDER:
                    position = labels[name]
                    pixel, _ = project_world_to_pixel(
                        position, camera_position, cv_rotation, intrinsics
                    )
                    if pixel is None:
                        pixels = None
                        break
                    pixels.append([float(pixel[0]), float(pixel[1])])
                    world_label_points.append([float(c) for c in position])
                if pixels is None or not (
                    all(0 <= p[0] < 320 and 0 <= p[1] < 180 for p in pixels)
                ):
                    skipped += 1
                    print(
                        f"ssm_data: {model} scale {scale}: labels leave the "
                        "image, skipped",
                        flush=True,
                    )
                    continue
                sample_path = OUT_DIR / f"sample_{index:04d}.npz"
                depth_mm = np.clip(
                    np.nan_to_num(np.asarray(depth, dtype=np.float32), posinf=0.0)
                    * 1000.0,
                    0.0,
                    65535.0,
                ).astype(np.uint16)
                np.savez_compressed(
                    sample_path,
                    depth=depth_mm,
                    pixels=np.asarray(pixels, dtype=np.float32),
                    world_points=np.asarray(world_label_points, dtype=np.float32),
                    camera_position=np.asarray(camera_position, dtype=np.float32),
                    cv_rotation=np.asarray(cv_rotation, dtype=np.float32),
                )
                samples.append(
                    {
                        "index": index,
                        "model": model,
                        "scale": float(scale),
                        "dx": float(dx),
                        "dy": float(dy),
                        "arm_abduction_deg": 0.0,
                    }
                )
                index += 1
                print(
                    f"ssm_data: sample {index - 1} ({model}, scale {scale}) "
                    f"valid pixels {int(np.count_nonzero(depth_mm))}",
                    flush=True,
                )

        manifest = {
            "samples": samples,
            "landmark_order": list(LANDMARK_ORDER),
            "width": 320,
            "height": 180,
            "fov_deg": 90.0,
            "provenance": (
                "I5-c stage 2: SSM torso surfaces (Bender et al., CC-BY-4.0) "
                "placed supine under the perception camera; labels are surface "
                "fractions transferred from the biped's rig landmarks -- "
                "electrode-free, so the real electrode coordinates never enter "
                "the training data"
            ),
        }
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"ssm_data: wrote {len(samples)} samples ({skipped} skipped) -> "
            f"{OUT_DIR}",
            flush=True,
        )
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
