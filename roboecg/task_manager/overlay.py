# Copied from farus_thyroid_isaac/farus/task_manager/overlay.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Video frame overlays: stage banners, metric lines and picture-in-picture."""
from __future__ import annotations

import numpy as np

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_target_crosshair(image, pixel, color=(255, 60, 60), size=20, width=6):
    """Draw a crosshair at a pixel on an RGB image (for the perception inset)."""
    from PIL import Image, ImageDraw

    result = Image.fromarray(np.asarray(image, dtype="uint8"), mode="RGB")
    if pixel is None:
        return np.asarray(result)
    draw = ImageDraw.Draw(result)
    u, v = float(pixel[0]), float(pixel[1])
    draw.line([u - size, v, u + size, v], fill=color, width=width)
    draw.line([u, v - size, u, v + size], fill=color, width=width)
    draw.ellipse(
        [u - size // 3, v - size // 3, u + size // 3, v + size // 3],
        outline=color,
        width=width,
    )
    return np.asarray(result)


def overlay_frame(
    frame,
    title: str | None = None,
    stage: str | None = None,
    metrics=None,
    inset=None,
    inset_label: str | None = None,
):
    """Draw title/stage/metrics text and an inset image onto one RGB frame."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype="uint8"), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    title_font = _load_font(20)
    stage_font = _load_font(30)
    small_font = _load_font(18)

    if title:
        draw.rectangle([0, 0, width, 34], fill=(0, 0, 0, 155))
        draw.text((12, 6), title, font=title_font, fill=(255, 255, 255))
    if stage:
        draw.rectangle([0, 38, width, 80], fill=(0, 0, 0, 120))
        draw.text((12, 42), stage, font=stage_font, fill=(255, 215, 0))
    if metrics:
        y = height - 24 * len(metrics) - 10
        for line in metrics:
            draw.rectangle(
                [0, y - 3, min(width, 14 + 11 * len(line)), y + 21],
                fill=(0, 0, 0, 140),
            )
            draw.text((10, y), line, font=small_font, fill=(255, 255, 255))
            y += 24
    if inset is not None:
        inset_image = Image.fromarray(np.asarray(inset, dtype="uint8"), mode="RGB")
        target_width = width // 3
        ratio = target_width / inset_image.width
        inset_image = inset_image.resize(
            (target_width, max(1, int(inset_image.height * ratio)))
        )
        x = width - inset_image.width - 14
        y = 92
        image.paste(inset_image, (x, y))
        draw.rectangle(
            [x - 2, y - 2, x + inset_image.width + 1, y + inset_image.height + 1],
            outline=(255, 215, 0),
            width=2,
        )
        if inset_label:
            draw.rectangle(
                [
                    x - 2,
                    y + inset_image.height + 2,
                    x + inset_image.width + 1,
                    y + inset_image.height + 30,
                ],
                fill=(0, 0, 0, 170),
            )
            draw.text(
                (x + 5, y + inset_image.height + 5),
                inset_label,
                font=small_font,
                fill=(255, 255, 255),
            )
    return np.asarray(image)
