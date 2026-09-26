"""Render-product capture helpers for the ECG task modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_render_product(camera_path, width, height, annotator_name="rgb"):
    import omni.replicator.core as rep

    product = rep.create.render_product(camera_path, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
    annotator.attach(product)
    rep.orchestrator.step(rt_subframes=1)
    rep.orchestrator.step(rt_subframes=1)
    return annotator


def capture_rgb(camera_path, width, height):
    raw = np.asarray(make_render_product(camera_path, width, height, "rgb").get_data())
    if raw.ndim == 1 and raw.size == width * height * 4:
        raw = raw.reshape(height, width, 4)
    if raw.ndim != 3 or raw.shape[2] < 3:
        raise RuntimeError(f"camera {camera_path} returned invalid RGB shape {raw.shape}")
    return raw[:, :, :3].copy()


def capture_depth(camera_path, width, height):
    annotator = make_render_product(
        camera_path, width, height, "distance_to_image_plane"
    )
    raw = np.asarray(annotator.get_data())
    if raw.ndim == 1 and raw.size == width * height:
        raw = raw.reshape(height, width)
    if raw.ndim != 2:
        raise RuntimeError(f"camera {camera_path} returned invalid depth shape {raw.shape}")
    return raw.copy()


def save_png(path, array) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype="uint8"), mode="RGB").save(path)
