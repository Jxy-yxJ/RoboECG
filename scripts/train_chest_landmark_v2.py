"""M3b v2 trainer: heatmap U-Net for chest landmarks with soft-argmax decoding.

Improvements over v1 (direct coordinate regression, 1.38 px / 21.8 mm):
  * heatmap supervision + soft-argmax for sub-pixel coordinates
  * a small U-Net with skip connections at 1/2 resolution
  * on-the-fly augmentation (depth noise, gain jitter, translation, occlusion)

Run:
    /home/jxy/isaacsim-compat/env/bin/python scripts/train_chest_landmark_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset"
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "chest_landmark_heatmap.pt"
REPORT_PATH = PROJECT_ROOT / "runs" / "m3b" / "train_report_v2.json"

# Model architecture + decoding live in the shared detector module so that
# training and deployment (M4 demo / disturbance eval) use one definition.
from roboecg.perception.chest_detector import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    HEATMAP_STRIDE,
    ChestLandmarkHeatmapNet,
    soft_argmax,
)

GAUSSIAN_SIGMA_PX = 2.0


def load_dataset(extra_dirs=None):
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    dirs = [DATASET_DIR] + [Path(d) for d in (extra_dirs or [])]
    depth_list, pixel_list = [], []
    for dataset_dir in dirs:
        dir_manifest = json.loads((dataset_dir / "manifest.json").read_text())
        for sample in dir_manifest["samples"]:
            data = np.load(dataset_dir / f"sample_{sample['index']:04d}.npz")
            depth = data["depth"].astype(np.float32) / 1000.0
            depth = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
            depth = (depth - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
            depth_list.append(depth[None, :, :])
            pixel_list.append(data["pixels"])
    depths = np.stack(depth_list).astype(np.float32)
    pixels = np.stack(pixel_list).astype(np.float32)
    return depths, pixels, manifest


def gaussian_targets(pixels, height, width, stride, sigma):
    batch, points, _ = pixels.shape
    device = pixels.device
    ys = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, width)
    centres = pixels / stride
    cx = centres[:, :, 0].view(batch, points, 1, 1)
    cy = centres[:, :, 1].view(batch, points, 1, 1)
    return torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))


def augment(depth, pixels, rng, width, height):
    """On-the-fly augmentation: noise, gain jitter, translation, occlusion."""
    out = depth.clone()
    batch = out.shape[0]
    for i in range(batch):
        if rng.random() < 0.8:
            sigma = float(rng.uniform(0.002, 0.008))
            out[i] = torch.clamp(out[i] + torch.randn_like(out[i]) * sigma, 0.0, 1.0)
        if rng.random() < 0.5:
            out[i] = torch.clamp(out[i] * float(rng.uniform(0.99, 1.01)), 0.0, 1.0)
        if rng.random() < 0.5:
            shift_x = int(rng.integers(-4, 5))
            shift_y = int(rng.integers(-4, 5))
            out[i] = torch.roll(out[i], shifts=(shift_y, shift_x), dims=(1, 2))
            pixels[i, :, 0] += shift_x
            pixels[i, :, 1] += shift_y
        if rng.random() < 0.3:
            patch_w = int(rng.integers(10, 40))
            patch_h = int(rng.integers(10, 30))
            x0 = int(rng.integers(0, max(1, width - patch_w)))
            y0 = int(rng.integers(0, max(1, height - patch_h)))
            out[i, :, y0 : y0 + patch_h, x0 : x0 + patch_w] = 0.0
    return out, pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--model-out",
        default=None,
        help="checkpoint output path (default: the deployed MODEL_PATH)",
    )
    parser.add_argument(
        "--extra-dataset",
        default=None,
        help="append samples from another dataset dir (I5-c fine-tune)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "seed for the train/val split (default: --seed); fix it while "
            "varying --seed to measure training variance on one split"
        ),
    )
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    depths, pixels, manifest = load_dataset(
        [args.extra_dataset] if getattr(args, "extra_dataset", None) else None
    )
    n = depths.shape[0]
    width, height = manifest["width"], manifest["height"]
    heatmap_h = height // HEATMAP_STRIDE
    heatmap_w = width // HEATMAP_STRIDE
    split_rng = np.random.default_rng(
        args.seed if args.split_seed is None else args.split_seed
    )
    permutation = split_rng.permutation(n)
    val_idx = permutation[: args.val_samples]
    train_idx = permutation[args.val_samples:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestLandmarkHeatmapNet(len(manifest["landmark_order"])).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )
    mse = nn.MSELoss()

    x_train = torch.from_numpy(depths[train_idx]).to(device)
    y_train = torch.from_numpy(pixels[train_idx]).to(device)
    x_val = torch.from_numpy(depths[val_idx]).to(device)
    y_val = torch.from_numpy(pixels[val_idx]).to(device)

    best_val = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(x_train.shape[0])
        total = 0.0
        for start in range(0, x_train.shape[0], args.batch_size):
            batch = order[start : start + args.batch_size]
            depth_batch = x_train[batch].clone()
            pixel_batch = y_train[batch].clone()
            if not args.no_augment:
                depth_batch, pixel_batch = augment(
                    depth_batch, pixel_batch, rng, width, height
                )
            targets = gaussian_targets(
                pixel_batch, heatmap_h, heatmap_w, HEATMAP_STRIDE, GAUSSIAN_SIGMA_PX
            )
            optimiser.zero_grad()
            heatmaps = model(depth_batch)
            loss_heat = mse(heatmaps, targets)
            coords = soft_argmax(heatmaps) * HEATMAP_STRIDE
            loss_coord = F.smooth_l1_loss(coords, pixel_batch)
            loss = loss_heat + 0.01 * loss_coord
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * batch.shape[0]
        train_loss = total / x_train.shape[0]

        model.eval()
        with torch.no_grad():
            heatmaps = model(x_val)
            coords = soft_argmax(heatmaps) * HEATMAP_STRIDE
            error_px = (coords - y_val).norm(dim=2)
            mean_px = float(error_px.mean())
            max_px = float(error_px.max())
            val_loss = float(mse(heatmaps, gaussian_targets(
                y_val, heatmap_h, heatmap_w, HEATMAP_STRIDE, GAUSSIAN_SIGMA_PX
            )))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "val_mean_px": mean_px, "val_max_px": max_px}
        )
        scheduler.step()
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            out_path = Path(args.model_out) if args.model_out else MODEL_PATH
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": "heatmap",
                    "state_dict": model.state_dict(),
                    "landmark_order": manifest["landmark_order"],
                    "width": width,
                    "height": height,
                    "heatmap_stride": HEATMAP_STRIDE,
                    "depth_min_m": DEPTH_MIN_M,
                    "depth_max_m": DEPTH_MAX_M,
                },
                out_path,
            )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:3d} train={train_loss:.5f} val={val_loss:.5f} "
                f"mean_px={mean_px:.2f} max_px={max_px:.2f}",
                flush=True,
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "model": "heatmap_unet",
                "epochs": args.epochs,
                "samples": int(n),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "val_indices": val_idx.tolist(),
                "augment": not args.no_augment,
                "best_val_loss": best_val,
                "final_val_mean_px": history[-1]["val_mean_px"],
                "final_val_max_px": history[-1]["val_max_px"],
                "model_path": str(out_path),
                "history": history,
                "provenance": (
                    "synthetic dataset (Isaac overhead depth + rig joint labels); "
                    "M3b v2 heatmap detector"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"M3b v2 train: model saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
