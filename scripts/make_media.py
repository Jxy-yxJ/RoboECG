"""Compose the README teaser and copy the result figures into media/.

Run after m4_points_view.py / m4_points_map.py / m2_ecg_depth.py have produced
fresh figures:

    python scripts/make_media.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIA = PROJECT_ROOT / "media"
RUNS = PROJECT_ROOT / "runs"

COPIES = {
    "runs/m4/points/chest_map.png": "chest_map.png",
    "runs/m4/points/overhead.png": "overhead.png",
    "runs/m4/points/oblique.png": "oblique.png",
    "runs/m4/points/closeup.png": "closeup.png",
    "runs/m2/depth.png": "depth.png",
    "runs/m2/fusion_closeup.png": "fusion_closeup.png",
}
GAP = 16
BACKGROUND = (255, 255, 255)


def main() -> None:
    MEDIA.mkdir(exist_ok=True)
    for source, name in COPIES.items():
        path = PROJECT_ROOT / source
        if path.is_file():
            shutil.copy2(path, MEDIA / name)
            print("copied", name)

    # teaser: oblique view (left) + 2D electrode map (right), same height
    left = Image.open(MEDIA / "oblique.png")
    right = Image.open(MEDIA / "chest_map.png")
    height = 720
    left = left.resize((int(left.width * height / left.height), height), Image.LANCZOS)
    right = right.resize(
        (int(right.width * height / right.height), height), Image.LANCZOS
    )
    width = left.width + GAP + right.width
    teaser = Image.new("RGB", (width, height), BACKGROUND)
    teaser.paste(left, (0, 0))
    teaser.paste(right, (left.width + GAP, 0))
    teaser.save(MEDIA / "teaser.png", optimize=True)
    print("teaser:", teaser.size)


if __name__ == "__main__":
    main()
