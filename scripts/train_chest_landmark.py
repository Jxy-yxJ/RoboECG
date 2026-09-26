"""M3b trainer: chest-landmark CNN on synthetic overhead depth images.

Input : normalised overhead depth image (90 x 160)
Output: 7 chest landmark pixel coordinates (normalised), see
        gen_ecg_synth_dataset.LANDMARK_ORDER

Run (Isaac env python has torch+CUDA):
    $ISAACSIM_ENV/bin/python scripts/train_chest_landmark.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "runs" / "m3b" / "dataset"
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "chest_landmark_cnn.pt"
REPORT_PATH = PROJECT_ROOT / "runs" / "m3b" / "train_report.json"

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 3.0


def load_dataset():
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
    depth_list = []
    pixel_list = []
    world_list = []
    for sample in manifest["samples"]:
        path = DATASET_DIR / f"sample_{sample['index']:04d}.npz"
        data = np.load(path)
        depth = data["depth"].astype(np.float32) / 1000.0
        depth = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
        depth = (depth - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        depth_list.append(depth[None, :, :])
        pixel_list.append(data["pixels"])
        world_list.append(data["world_points"])
    width = manifest["width"]
    height = manifest["height"]
    depths = np.stack(depth_list).astype(np.float32)
    pixels = np.stack(pixel_list).astype(np.float32)
    normalised = pixels.copy()
    normalised[:, :, 0] /= width
    normalised[:, :, 1] /= height
    return depths, normalised, pixels, np.stack(world_list), manifest


class ChestLandmarkNet(nn.Module):
    def __init__(self, n_points: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_points * 2)

    def forward(self, x):
        features = self.features(x).flatten(1)
        return self.head(features).view(-1, self.head.out_features // 2, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    depths, normalised, pixels, world, manifest = load_dataset()
    n = depths.shape[0]
    width, height = manifest["width"], manifest["height"]
    permutation = np.random.permutation(n)
    val_idx = permutation[: args.val_samples]
    train_idx = permutation[args.val_samples:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestLandmarkNet(len(manifest["landmark_order"])).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, args.epochs), eta_min=args.lr * 0.02
    )
    loss_fn = nn.SmoothL1Loss()

    x_train = torch.from_numpy(depths[train_idx]).to(device)
    y_train = torch.from_numpy(normalised[train_idx]).to(device)
    x_val = torch.from_numpy(depths[val_idx]).to(device)
    y_val = torch.from_numpy(normalised[val_idx]).to(device)

    best_val = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation_epoch = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, x_train.shape[0], args.batch_size):
            batch = permutation_epoch[start : start + args.batch_size]
            optimiser.zero_grad()
            prediction = model(x_train[batch])
            loss = loss_fn(prediction, y_train[batch])
            loss.backward()
            optimiser.step()
            total_loss += float(loss) * batch.shape[0]
        train_loss = total_loss / x_train.shape[0]

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val)
            val_loss = float(loss_fn(val_prediction, y_val))
            error_px = (
                (val_prediction - y_val)
                * torch.tensor([width, height], device=device, dtype=torch.float32)
            ).norm(dim=2)
            mean_px = float(error_px.mean())
            max_px = float(error_px.max())
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "val_mean_px": mean_px, "val_max_px": max_px}
        )
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "landmark_order": manifest["landmark_order"],
                    "width": width,
                    "height": height,
                    "depth_min_m": DEPTH_MIN_M,
                    "depth_max_m": DEPTH_MAX_M,
                },
                MODEL_PATH,
            )
        scheduler.step()
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
                "epochs": args.epochs,
                "samples": int(n),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "val_indices": val_idx.tolist(),
                "best_val_loss": best_val,
                "final_val_mean_px": history[-1]["val_mean_px"],
                "final_val_max_px": history[-1]["val_max_px"],
                "model_path": str(MODEL_PATH),
                "history": history,
                "provenance": (
                    "synthetic dataset (Isaac overhead depth + rig joint labels); "
                    "trained for the ECG chest-landmark detector (M3b)"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"M3b train: model saved to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
