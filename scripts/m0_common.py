"""Shared helpers for the ECG M0 verification script.

Adapted from `farus_thyroid_isaac/scripts/m0_common.py` (2026-09-18); only the
project root and the fallback cache name changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs" / "m0"
LOCAL_ASSETS_DIR = PROJECT_ROOT / "assets" / "local"


def boot(headless: bool = True, width: int = 1280, height: int = 720):
    """Start SimulationApp with the settings validated on this host."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": headless,
            "width": width,
            "height": height,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
        }
    )


def ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def write_json(name: str, payload: dict) -> Path:
    path = ensure_runs_dir() / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


def _json_default(value):
    import numpy as np

    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _render_annotator(camera_path: str, width: int, height: int, annotator_name: str):
    import numpy as np
    import omni.replicator.core as rep

    product = rep.create.render_product(camera_path, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
    annotator.attach(product)
    rep.orchestrator.step(rt_subframes=1)
    rep.orchestrator.step(rt_subframes=1)
    return np.asarray(annotator.get_data()).copy()


def capture_rgb(camera_path: str, width: int, height: int):
    raw = _render_annotator(camera_path, width, height, "rgb")
    if raw.ndim == 1 and raw.size == width * height * 4:
        raw = raw.reshape(height, width, 4)
    if raw.ndim != 3 or raw.shape[2] < 3:
        raise RuntimeError(f"camera {camera_path} returned invalid RGB shape {raw.shape}")
    return raw[:, :, :3]


def capture_depth(camera_path: str, width: int, height: int):
    raw = _render_annotator(camera_path, width, height, "distance_to_image_plane")
    if raw.ndim == 1 and raw.size == width * height:
        raw = raw.reshape(height, width)
    if raw.ndim != 2:
        raise RuntimeError(f"camera {camera_path} returned invalid depth shape {raw.shape}")
    return raw


def save_png(path: Path, array) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype("uint8"), mode="RGB").save(path)


def save_depth_png(path: Path, depth) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise RuntimeError("depth image has no finite values")
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, 1e-6)
    normalized = np.clip((depth - lo) / span, 0.0, 1.0) * 255.0
    Image.fromarray(normalized.astype("uint8"), mode="L").save(path)
