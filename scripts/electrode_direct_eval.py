"""P0 evaluation: direct electrode regression vs the rule pipeline.

Both paths are evaluated on the same held-out samples:
  rules  : landmark detector -> chest frame -> clinical rules -> depth fusion
           (numbers from runs/m3b/eval_report.json)
  direct : depth image -> 6 electrode pixels -> per-pixel depth lifting

Run:
    $ISAACSIM_ENV/bin/python scripts/electrode_direct_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecg_dataset_common import (  # noqa: E402
    ELECTRODE_ORDER,
    load_manifest,
    load_sample,
)
from train_chest_landmark import DEPTH_MAX_M, DEPTH_MIN_M  # noqa: E402
from train_chest_landmark_v2 import ChestLandmarkHeatmapNet, soft_argmax  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "electrode_direct_heatmap.pt"
TRAIN_REPORT = PROJECT_ROOT / "runs" / "p0" / "train_report.json"
LABEL_CACHE = PROJECT_ROOT / "runs" / "p0" / "electrode_labels.npz"
REPORT_PATH = PROJECT_ROOT / "runs" / "p0" / "eval_report.json"
RULES_REPORT = PROJECT_ROOT / "runs" / "m3b" / "eval_report.json"


def main() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    width = checkpoint["width"]
    height = checkpoint["height"]
    stride = int(checkpoint["heatmap_stride"])
    model = ChestLandmarkHeatmapNet(len(ELECTRODE_ORDER))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    manifest = load_manifest()
    train_report = json.loads(TRAIN_REPORT.read_text())
    val_indices = set(train_report["val_indices"])
    labels = np.load(LABEL_CACHE)
    gt_pixels_all, gt_world_all = labels["pixels"], labels["world"]
    index_of = {
        sample["index"]: position
        for position, sample in enumerate(manifest["samples"])
    }

    from roboecg.coordinate_transform.camera import (
        CameraIntrinsics,
        deproject_pixel,
    )

    rows = []
    for sample in manifest["samples"]:
        index = sample["index"]
        if index not in val_indices:
            continue
        position = index_of[index]
        data = load_sample(index)
        depth_m = data["depth"].astype(np.float32) / 1000.0
        depth_normalised = np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M)
        depth_normalised = (depth_normalised - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        with torch.no_grad():
            heatmaps = model(
                torch.from_numpy(depth_normalised[None, None, :, :]).float()
            )
            predicted_pixels = soft_argmax(heatmaps)[0].numpy() * float(stride)

        intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, 90.0)
        camera_position = data["camera_position"]
        cv_rotation = data["cv_rotation"]
        gt_world = gt_world_all[position]
        gt_pixels = gt_pixels_all[position]

        predicted_world = []
        for point in range(len(ELECTRODE_ORDER)):
            u, v = predicted_pixels[point]
            u = float(np.clip(u, 0, width - 1))
            v = float(np.clip(v, 0, height - 1))
            z = float(depth_m[int(round(v)), int(round(u))])
            if z <= 0.0:
                z = float(np.median(depth_m[depth_m > 0]))
            predicted_world.append(
                deproject_pixel(u, v, z, intrinsics, camera_position, cv_rotation)
            )
        predicted_world = np.asarray(predicted_world)

        errors = np.linalg.norm(predicted_world - gt_world, axis=1) * 1000.0
        pixel_errors = np.linalg.norm(predicted_pixels - gt_pixels, axis=1)
        rows.append(
            {
                "index": index,
                "errors_mm": errors.tolist(),
                "pixel_errors": pixel_errors.tolist(),
            }
        )

    errors = np.array([row["errors_mm"] for row in rows])
    pixel_errors = np.array([row["pixel_errors"] for row in rows])

    rules_report = json.loads(RULES_REPORT.read_text()) if RULES_REPORT.is_file() else None
    report = {
        "val_samples": len(rows),
        "electrode_order": list(ELECTRODE_ORDER),
        "direct": {
            "pixel_error_mean": float(pixel_errors.mean()),
            "pixel_error_max": float(pixel_errors.max()),
            "error_mm_mean": float(errors.mean()),
            "error_mm_max": float(errors.max()),
            "per_electrode_mm": {
                name: {
                    "mean": float(errors[:, i].mean()),
                    "max": float(errors[:, i].max()),
                }
                for i, name in enumerate(ELECTRODE_ORDER)
            },
        },
        "rules": (
            None
            if rules_report is None
            else {
                "error_mm_mean": rules_report["target_errors_mm"]["mean"],
                "error_mm_max": rules_report["target_errors_mm"]["max"],
                "pixel_error_mean": rules_report["pixel_error"]["mean"],
                "val_samples": rules_report["val_samples"],
                "same_val_split": set(rules_report["per_sample"][i]["index"] for i in range(len(rules_report["per_sample"])))
                == val_indices,
            }
        ),
        "provenance": (
            "synthetic held-out samples; direct = depth->electrode pixels->lifting, "
            "rules = landmark->frame->clinical rules (P0 comparison)"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    print(f"P0 eval: {len(rows)} val samples (same split as rules: "
          f"{report['rules']['same_val_split'] if report['rules'] else 'n/a'})")
    print(
        f"P0 direct: pixel {report['direct']['pixel_error_mean']:.2f} px | "
        f"error {report['direct']['error_mm_mean']:.1f} mm mean / "
        f"{report['direct']['error_mm_max']:.1f} mm max"
    )
    if report["rules"]:
        print(
            f"P0 rules : pixel {report['rules']['pixel_error_mean']:.2f} px | "
            f"error {report['rules']['error_mm_mean']:.1f} mm mean / "
            f"{report['rules']['error_mm_max']:.1f} mm max"
        )
    print("per electrode (direct):")
    for name, stats in report["direct"]["per_electrode_mm"].items():
        print(f"  {name}: {stats['mean']:.1f} mm (max {stats['max']:.1f})")


if __name__ == "__main__":
    main()
