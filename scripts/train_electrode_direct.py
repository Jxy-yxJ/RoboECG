"""P0 experiment: direct electrode-position regression vs the rule pipeline.

Trains the same heatmap U-Net but with 6 output channels predicting the V1-V6
*pixel* positions directly from the depth image (Li-style learned electrode
localisation), then evaluates both paths on the same held-out samples:

  rules  : detector -> landmarks -> chest frame -> clinical rules -> depth fusion
  direct : depth image -> 6 electrode pixels -> per-pixel depth lifting

The ground truth for both is the rule-generated electrode position (the
simulation's electrode definition), so the comparison measures how much error
the landmark -> frame -> rules chain adds on top of perception.

Run:
    $ISAACSIM_ENV/bin/python scripts/train_electrode_direct.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecg_dataset_common import (  # noqa: E402
    DATASET_DIR,
    ELECTRODE_ORDER,
    gt_electrodes,
    load_manifest,
    load_sample,
)
from train_chest_landmark import DEPTH_MAX_M, DEPTH_MIN_M  # noqa: E402
from train_chest_landmark_v2 import (  # noqa: E402
    ChestLandmarkHeatmapNet,
    augment,
    gaussian_targets,
    soft_argmax,
)

MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "electrode_direct_heatmap.pt"
REPORT_PATH = PROJECT_ROOT / "runs" / "p0" / "train_report.json"
LABEL_CACHE = PROJECT_ROOT / "runs" / "p0" / "electrode_labels.npz"


def prepare_labels(manifest: dict, rules: dict):
    if LABEL_CACHE.is_file():
        cached = np.load(LABEL_CACHE)
        return cached["pixels"], cached["world"]
    pixels, world = [], []
    for position, sample in enumerate(manifest["samples"]):
        data = load_sample(sample["index"])
        gt = gt_electrodes(data, rules)
        pixels.append(gt["pixels"])
        world.append(gt["world"])
        if position % 50 == 0:
            print(f"labels: {position}/{len(manifest['samples'])}", flush=True)
    pixels = np.stack(pixels).astype(np.float32)
    world = np.stack(world).astype(np.float32)
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LABEL_CACHE, pixels=pixels, world=world)
    return pixels, world


def load_depths(manifest: dict):
    depths = []
    for sample in manifest["samples"]:
        data = load_sample(sample["index"])
        depth = data["depth"].astype(np.float32) / 1000.0
        depth = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
        depth = (depth - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        depths.append(depth[None, :, :])
    return np.stack(depths).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--val-indices-from",
        type=str,
        default=None,
        help="JSON report whose val_indices to reuse (same split comparison)",
    )
    args = parser.parse_args()

    from roboecg.target_localization.ecg_rules import load_ecg_rules

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # same split as the M3b v2 landmark detector
    rng = np.random.default_rng(args.seed)
    rules = load_ecg_rules()
    manifest = load_manifest()
    pixels, world = prepare_labels(manifest, rules)
    depths = load_depths(manifest)
    width, height = manifest["width"], manifest["height"]
    stride = 2
    heatmap_h, heatmap_w = height // stride, width // stride

    n = depths.shape[0]
    if args.val_indices_from:
        reference = json.loads(Path(args.val_indices_from).read_text())
        val_idx = np.asarray(reference["val_indices"], dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = np.where(mask)[0]
    else:
        permutation = np.random.permutation(n)
        val_idx = permutation[: args.val_samples]
        train_idx = permutation[args.val_samples:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestLandmarkHeatmapNet(len(ELECTRODE_ORDER)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )
    mse = torch.nn.MSELoss()

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
            depth_batch, pixel_batch = augment(
                depth_batch, pixel_batch, rng, width, height
            )
            targets = gaussian_targets(
                pixel_batch, heatmap_h, heatmap_w, stride, 2.0
            )
            optimiser.zero_grad()
            heatmaps = model(depth_batch)
            loss_heat = mse(heatmaps, targets)
            coords = soft_argmax(heatmaps) * stride
            loss_coord = F.smooth_l1_loss(coords, pixel_batch)
            loss = loss_heat + 0.01 * loss_coord
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * batch.shape[0]
        train_loss = total / x_train.shape[0]

        model.eval()
        with torch.no_grad():
            heatmaps = model(x_val)
            coords = soft_argmax(heatmaps) * stride
            error_px = (coords - y_val).norm(dim=2)
            mean_px = float(error_px.mean())
            max_px = float(error_px.max())
        history.append(
            {"epoch": epoch, "train_loss": train_loss,
             "val_mean_px": mean_px, "val_max_px": max_px}
        )
        scheduler.step()
        if best_val is None or mean_px < best_val:
            best_val = mean_px
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": "heatmap",
                    "state_dict": model.state_dict(),
                    "electrode_order": list(ELECTRODE_ORDER),
                    "width": width,
                    "height": height,
                    "heatmap_stride": stride,
                    "depth_min_m": DEPTH_MIN_M,
                    "depth_max_m": DEPTH_MAX_M,
                },
                MODEL_PATH,
            )
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:3d} train={train_loss:.5f} "
                f"mean_px={mean_px:.2f} max_px={max_px:.2f}",
                flush=True,
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "model": "electrode_direct_heatmap",
                "electrode_order": list(ELECTRODE_ORDER),
                "epochs": args.epochs,
                "samples": int(n),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "val_indices": val_idx.tolist(),
                "best_val_mean_px": best_val,
                "final_val_mean_px": history[-1]["val_mean_px"],
                "final_val_max_px": history[-1]["val_max_px"],
                "model_path": str(MODEL_PATH),
                "history": history,
                "provenance": (
                    "synthetic dataset (Isaac overhead depth + rule-generated "
                    "electrode labels); direct electrode regression (P0)"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"P0 direct train: model saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
