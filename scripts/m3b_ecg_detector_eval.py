"""M3b evaluation: chest-landmark CNN -> frame -> V1-V6 vs ground truth.

Runs on the held-out synthetic samples: predicts the landmarks from the depth
image, lifts them to 3D with the same depth image, builds the chest frame,
generates the V1-V6 targets with the rules and compares everything against the
simulation ground truth.

Run:
    $ISAACSIM_ENV/bin/python scripts/m3b_ecg_detector_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.coordinate_transform.camera import (  # noqa: E402
    CameraIntrinsics,
    deproject_pixel,
)
from roboecg.coordinate_transform.frames import angle_between_deg  # noqa: E402
from roboecg.perception.chest_landmarks import ChestLandmarks  # noqa: E402
from roboecg.perception.depth import depth_to_world_points  # noqa: E402
from roboecg.perception.torso_prior import fit_chest_surface_prior  # noqa: E402
from roboecg.target_localization.chest_frame import build_chest_frame  # noqa: E402
from roboecg.target_localization.ecg import generate_v1_v6  # noqa: E402
from roboecg.target_localization.ecg_rules import load_ecg_rules  # noqa: E402
from roboecg.target_localization.fusion import fuse_target  # noqa: E402

from train_chest_landmark import DEPTH_MAX_M, DEPTH_MIN_M, ChestLandmarkNet  # noqa: E402
from train_chest_landmark_v2 import ChestLandmarkHeatmapNet, soft_argmax  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset"
MODEL_V1 = PROJECT_ROOT / "assets" / "models" / "chest_landmark_cnn.pt"
MODEL_V2 = PROJECT_ROOT / "assets" / "models" / "chest_landmark_heatmap.pt"
MODEL_PATH = MODEL_V2 if MODEL_V2.is_file() else MODEL_V1
REPORT_PATH = PROJECT_ROOT / "runs" / "m3b" / "eval_report.json"


def _axes_matrix(frame) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(frame.up, dtype=float),
            np.asarray(frame.lateral, dtype=float),
            np.asarray(frame.anterior, dtype=float),
        ]
    )


def _rotate_frame(frame, rotation: np.ndarray):
    """Rotate the frame axes by `rotation` expressed in its own basis."""
    from roboecg.target_localization.chest_frame import ChestFrame

    axes = _axes_matrix(frame) @ np.asarray(rotation, dtype=float).T
    up = axes[:, 0] / (np.linalg.norm(axes[:, 0]) + 1e-12)
    lateral = axes[:, 1] / (np.linalg.norm(axes[:, 1]) + 1e-12)
    anterior = axes[:, 2] / (np.linalg.norm(axes[:, 2]) + 1e-12)
    return ChestFrame(
        origin=np.asarray(frame.origin, dtype=float),
        up=up,
        lateral=lateral,
        anterior=anterior,
        provenance=dict(frame.provenance, rotation_calibration=True),
    )


def predict_points(model, model_type, depth_m, width, height, heatmap_stride,
                   intrinsics, camera_position, cv_rotation, landmark_order):
    """Model inference + per-pixel lifting (the real pipeline path)."""
    depth_normalised = np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M)
    depth_normalised = (depth_normalised - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    with torch.no_grad():
        output = model(torch.from_numpy(depth_normalised[None, None, :, :]).float())
        if model_type == "heatmap":
            predicted_pixels = soft_argmax(output)[0].numpy() * float(heatmap_stride)
        else:
            predicted_pixels = output[0].numpy() * np.array(
                [width, height], dtype=np.float32
            )
    points = {}
    for order_index, name in enumerate(landmark_order):
        u, v = predicted_pixels[order_index]
        u = float(np.clip(u, 0, width - 1))
        v = float(np.clip(v, 0, height - 1))
        z = float(depth_m[int(round(v)), int(round(u))])
        if z <= 0.0:
            z = float(np.median(depth_m[depth_m > 0]))
        points[name] = deproject_pixel(u, v, z, intrinsics, camera_position, cv_rotation)
    return predicted_pixels, points


def landmarks_from_points(points: dict, up_hint) -> ChestLandmarks:
    """Build the ChestLandmarks dataclass from the seven detected points."""
    clavicle_left = points["clavicle_left"]
    clavicle_right = points["clavicle_right"]
    neck_base = points["neck_base"]
    chest = points["chest"]
    up = np.asarray(neck_base, dtype=float) - np.asarray(chest, dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)
    return ChestLandmarks(
        spine_lower=np.asarray(points["pelvis"], dtype=float) - up * 0.05,
        spine_mid=np.asarray(points["pelvis"], dtype=float) + up * 0.10,
        chest=np.asarray(chest, dtype=float),
        upper_chest=np.asarray(chest, dtype=float) + up * 0.07,
        neck_base=np.asarray(neck_base, dtype=float),
        neck_top=np.asarray(neck_base, dtype=float) + up * 0.06,
        clavicle_left=np.asarray(clavicle_left, dtype=float),
        clavicle_right=np.asarray(clavicle_right, dtype=float),
        shoulder_left=np.asarray(points["shoulder_left"], dtype=float),
        shoulder_right=np.asarray(points["shoulder_right"], dtype=float),
        pelvis=np.asarray(points["pelvis"], dtype=float),
        provenance="learned_model: chest landmark CNN (M3b) + per-pixel lifting",
    )


def main() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    landmark_order = checkpoint["landmark_order"]
    width = checkpoint["width"]
    height = checkpoint["height"]
    model_type = checkpoint.get("model_type", "regressor")
    if model_type == "heatmap":
        model = ChestLandmarkHeatmapNet(len(landmark_order))
        heatmap_stride = int(checkpoint["heatmap_stride"])
    else:
        model = ChestLandmarkNet(len(landmark_order))
        heatmap_stride = 1
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"M3b eval: model={model_type} ({MODEL_PATH.name})", flush=True)

    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    report_name = (
        "train_report_v2.json" if model_type == "heatmap" else "train_report.json"
    )
    train_report = json.loads(
        (PROJECT_ROOT / "runs" / "m3b" / report_name).read_text()
    )
    val_indices = set(train_report["val_indices"])
    intrinsics = CameraIntrinsics.from_horizontal_fov(
        width, height, manifest["fov_deg"]
    )
    rules = load_ecg_rules()
    settings = rules["depth_fusion"]

    # Rotation calibration on the same training split: the detected landmarks
    # are lifted onto the skin while the GT landmarks are the joint centres
    # inside the body, which tilts the derived chest frame by a systematic
    # angle (measured: ~3.3 deg, std 1.7 deg).  Estimate the mean rotation
    # from predicted to GT axes and undo it at evaluation time.
    rotation_samples = []
    for sample in manifest["samples"]:
        if sample["index"] in val_indices:
            continue
        data = np.load(DATASET_DIR / f"sample_{sample['index']:04d}.npz")
        depth_train = data["depth"].astype(np.float32) / 1000.0
        _, points_train = predict_points(
            model,
            model_type,
            depth_train,
            width,
            height,
            heatmap_stride,
            intrinsics,
            data["camera_position"],
            data["cv_rotation"],
            landmark_order,
        )
        pred_frame_train = build_chest_frame(
            landmarks_from_points(points_train, up_hint=None),
            anterior_hint=(0.0, 0.0, 1.0),
        )
        gt_world_train = data["world_points"]
        gt_points_train = {
            name: gt_world_train[i] for i, name in enumerate(landmark_order)
        }
        gt_frame_train = build_chest_frame(
            landmarks_from_points(gt_points_train, up_hint=None),
            anterior_hint=(0.0, 0.0, 1.0),
        )
        rotation_samples.append(
            _axes_matrix(gt_frame_train).T @ _axes_matrix(pred_frame_train)
        )
    # Per-landmark offset calibration (train split): the detected landmarks are
    # lifted onto the skin while the GT landmarks are the joint centres, so
    # every landmark carries a systematic offset (dominated by the tissue depth
    # along n, but the lateral components matter too: v_shoulder/v_mcl feed the
    # V4-V6 rules and a 6-9% lateral bias shifted V4 by ~28 mm).
    landmark_offsets = []
    for sample in manifest["samples"]:
        if sample["index"] in val_indices:
            continue
        data = np.load(DATASET_DIR / f"sample_{sample['index']:04d}.npz")
        depth_train = data["depth"].astype(np.float32) / 1000.0
        _, points_train = predict_points(
            model,
            model_type,
            depth_train,
            width,
            height,
            heatmap_stride,
            intrinsics,
            data["camera_position"],
            data["cv_rotation"],
            landmark_order,
        )
        pred_frame_train = build_chest_frame(
            landmarks_from_points(points_train, up_hint=None),
            anterior_hint=(0.0, 0.0, 1.0),
        )
        gt_world_train = data["world_points"]
        row = []
        for i, name in enumerate(landmark_order):
            pred_uvn = np.asarray(
                pred_frame_train.to_frame(points_train[name]), dtype=float
            )
            gt_uvn = np.asarray(
                pred_frame_train.to_frame(gt_world_train[i]), dtype=float
            )
            row.extend((pred_uvn - gt_uvn).tolist())
        landmark_offsets.append(row)
    landmark_offset = np.mean(np.asarray(landmark_offsets), axis=0).reshape(
        len(landmark_order), 3
    )
    print(
        "M3b eval: landmark offsets (pred - gt, uvn mm): "
        + ", ".join(
            f"{name}={np.round(landmark_offset[i] * 1000.0, 1).tolist()}"
            for i, name in enumerate(landmark_order)
        ),
        flush=True,
    )

    rotation_mean = np.mean(np.asarray(rotation_samples), axis=0)
    u_rot, _, v_rot = np.linalg.svd(rotation_mean)
    rotation_calibration = u_rot @ v_rot
    calibration_path = MODEL_PATH.parent / "m3b_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "rotation_matrix": rotation_calibration.tolist(),
                "landmark_offset_uvn_m": {
                    name: [float(v) for v in landmark_offset[i]]
                    for i, name in enumerate(landmark_order)
                },
                "landmark_order": list(landmark_order),
                "provenance": (
                    "estimated on the training split of runs/m3b/dataset by "
                    "scripts/m3b_ecg_detector_eval.py (no validation leakage)"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    residual_deg = np.degrees(
        np.arccos(
            np.clip((np.trace(rotation_mean) - 1.0) / 2.0, -1.0, 1.0)
        )
    )
    print(
        f"M3b eval: rotation calibration (train split) = {residual_deg:.2f} deg",
        flush=True,
    )

    rows = []
    for sample in manifest["samples"]:
        index = sample["index"]
        if index not in val_indices:
            continue
        data = np.load(DATASET_DIR / f"sample_{index:04d}.npz")
        depth_m = data["depth"].astype(np.float32) / 1000.0
        gt_pixels = data["pixels"]
        gt_world = data["world_points"]
        camera_position = data["camera_position"]
        cv_rotation = data["cv_rotation"]

        predicted_pixels, predicted_points = predict_points(
            model,
            model_type,
            depth_m,
            width,
            height,
            heatmap_stride,
            intrinsics,
            camera_position,
            cv_rotation,
            landmark_order,
        )

        pixel_error = np.linalg.norm(predicted_pixels - gt_pixels, axis=1)
        per_landmark_pixel_error = {
            name: float(pixel_error[i]) for i, name in enumerate(landmark_order)
        }

        point_error_mm = [
            float(np.linalg.norm(predicted_points[name] - gt_world[order_index]) * 1000.0)
            for order_index, name in enumerate(landmark_order)
        ]

        gt_points = {name: gt_world[i] for i, name in enumerate(landmark_order)}
        gt_landmarks = landmarks_from_points(gt_points, up_hint=None)
        gt_frame = build_chest_frame(gt_landmarks, anterior_hint=(0.0, 0.0, 1.0))
        # Correct the detected landmarks for the systematic skin-vs-joint
        # offset measured on the training split before applying the rules.
        corrected_points = {}
        raw_frame = build_chest_frame(
            landmarks_from_points(predicted_points, up_hint=None),
            anterior_hint=(0.0, 0.0, 1.0),
        )
        for i, name in enumerate(landmark_order):
            offset = landmark_offset[i]
            corrected_points[name] = (
                np.asarray(predicted_points[name], dtype=float)
                - offset[0] * raw_frame.up
                - offset[1] * raw_frame.lateral
                - offset[2] * raw_frame.anterior
            )
        predicted_points = corrected_points
        pred_landmarks = landmarks_from_points(predicted_points, up_hint=None)
        pred_frame = build_chest_frame(pred_landmarks, anterior_hint=(0.0, 0.0, 1.0))
        pred_frame = _rotate_frame(pred_frame, rotation_calibration)

        delta = np.asarray(pred_frame.origin) - np.asarray(gt_frame.origin)
        up_pred = np.asarray(pred_frame.up, dtype=float)
        lat_pred = np.asarray(pred_frame.lateral, dtype=float)
        up_gt = np.asarray(gt_frame.up, dtype=float)
        lat_gt = np.asarray(gt_frame.lateral, dtype=float)
        ant_gt = np.asarray(gt_frame.anterior, dtype=float)

        def _signed(vec, axis_a, axis_b, base):
            return float(
                np.degrees(
                    np.arctan2(float(np.dot(vec, axis_a)), float(np.dot(vec, base)))
                )
            )

        frame_error = {
            "up_rot_anterior_deg": _signed(up_pred, ant_gt, lat_gt, up_gt),
            "up_rot_lateral_deg": _signed(up_pred, lat_gt, ant_gt, up_gt),
            "lateral_rot_up_deg": _signed(lat_pred, up_gt, ant_gt, lat_gt),
            "lateral_rot_anterior_deg": _signed(lat_pred, ant_gt, up_gt, lat_gt),
            "origin_mm": float(np.linalg.norm(delta) * 1000.0),
            "origin_du_mm": float(np.dot(delta, gt_frame.up) * 1000.0),
            "origin_dv_mm": float(np.dot(delta, gt_frame.lateral) * 1000.0),
            "origin_dn_mm": float(np.dot(delta, gt_frame.anterior) * 1000.0),
            "up_deg": angle_between_deg(pred_frame.up, gt_frame.up),
            "lateral_deg": angle_between_deg(pred_frame.lateral, gt_frame.lateral),
            "anterior_deg": angle_between_deg(pred_frame.anterior, gt_frame.anterior),
        }

        points = depth_to_world_points(
            depth_m, intrinsics, camera_position, cv_rotation, stride=1
        )
        try:
            gt_targets = generate_v1_v6(gt_landmarks, gt_frame, points, rules)
            pred_targets = generate_v1_v6(pred_landmarks, pred_frame, points, rules)
            target_errors = [
                float(
                    np.linalg.norm(
                        np.asarray(pred.position) - np.asarray(gt.position)
                    )
                    * 1000.0
                )
                for pred, gt in zip(pred_targets.targets, gt_targets.targets)
            ]
            prior = fit_chest_surface_prior(points, pred_frame)
            fused_errors = []
            for target, gt in zip(pred_targets.targets, gt_targets.targets):
                fused = fuse_target(
                    target,
                    pred_frame,
                    prior,
                    depth_m,
                    intrinsics,
                    camera_position,
                    cv_rotation,
                    settings,
                )
                fused_errors.append(
                    float(np.linalg.norm(np.asarray(fused.position) - np.asarray(gt.position)) * 1000.0)
                )
        except RuntimeError as error:
            target_errors = None
            fused_errors = None
            frame_error["target_error"] = repr(error)

        rows.append(
            {
                "index": index,
                "scale": sample["scale"],
                "pixel_error_mean": float(pixel_error.mean()),
                "pixel_error_max": float(pixel_error.max()),
                "per_landmark_pixel_error": per_landmark_pixel_error,
                "point_error_mm_mean": float(np.mean(point_error_mm)),
                "point_error_mm_max": float(np.max(point_error_mm)),
                "frame": frame_error,
                "target_errors_mm": target_errors,
                "fused_errors_mm": fused_errors,
            }
        )

    def summarise(key, nested=None):
        values = []
        for row in rows:
            value = row[key] if nested is None else row[key].get(nested)
            if value is None:
                continue
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
        if not values:
            return {"mean": None, "max": None, "count": 0}
        return {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    report = {
        "val_samples": len(rows),
        "landmark_order": landmark_order,
        "pixel_error": {
            "mean": float(np.mean([r["pixel_error_mean"] for r in rows])),
            "max": float(np.max([r["pixel_error_max"] for r in rows])),
        },
        "per_landmark_pixel_error": {
            name: {
                "mean": float(
                    np.mean([r["per_landmark_pixel_error"][name] for r in rows])
                ),
                "max": float(
                    np.max([r["per_landmark_pixel_error"][name] for r in rows])
                ),
            }
            for name in landmark_order
        },
        "point_error_mm": {
            "mean": float(np.mean([r["point_error_mm_mean"] for r in rows])),
            "max": float(np.max([r["point_error_mm_max"] for r in rows])),
        },
        "frame_origin_mm": summarise("frame", "origin_mm"),
        "frame_origin_du_mm": summarise("frame", "origin_du_mm"),
        "frame_origin_dv_mm": summarise("frame", "origin_dv_mm"),
        "frame_origin_dn_mm": summarise("frame", "origin_dn_mm"),
        "frame_up_deg": summarise("frame", "up_deg"),
        "frame_lateral_deg": summarise("frame", "lateral_deg"),
        "target_errors_mm": summarise("target_errors_mm"),
        "fused_errors_mm": summarise("fused_errors_mm"),
        "target_generation_failures": int(
            sum(1 for r in rows if r["target_errors_mm"] is None)
        ),
        "per_sample": rows,
        "model_type": model_type,
        "origin_calibration_uvn_m": None,
        "calibration_note": (
            "origin/rotation/landmark-offset calibration is estimated on the "
            "TRAINING split only; the per-landmark offsets subsume the old "
            "origin calibration (applying both double-corrected the origin)"
        ),
        "rotation_calibration_deg": float(residual_deg),
        "landmark_offset_uvn_m": {
            name: [float(v) for v in landmark_offset[i]]
            for i, name in enumerate(landmark_order)
        },
        "provenance": (
            "synthetic held-out samples; detector trained on Isaac overhead "
            "depth renders with rig joint labels (M3b)"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"M3b eval: {len(rows)} val samples")
    print(
        f"M3b eval: pixel error mean={report['pixel_error']['mean']:.2f} px "
        f"max={report['pixel_error']['max']:.2f} px"
    )
    print(
        f"M3b eval: landmark 3D error mean={report['point_error_mm']['mean']:.1f} mm "
        f"max={report['point_error_mm']['max']:.1f} mm "
        f"(joints are inside the body; the depth measures the skin)"
    )
    print(
        f"M3b eval: frame origin mean={report['frame_origin_mm']['mean']:.1f} mm "
        f"max={report['frame_origin_mm']['max']:.1f} mm"
    )
    print(
        f"M3b eval: target error mean={report['target_errors_mm']['mean']} mm "
        f"max={report['target_errors_mm']['max']} mm"
    )
    print(
        f"M3b eval: fused error mean={report['fused_errors_mm']['mean']} mm "
        f"max={report['fused_errors_mm']['max']} mm"
    )
    print(f"M3b eval: target generation failures={report['target_generation_failures']}")


if __name__ == "__main__":
    main()
