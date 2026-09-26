# Copied from farus_thyroid_isaac/farus/perception/body_tracking.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""Run the external MediaPipe pose estimator (simulation body tracking)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENV = Path("/home/jxy/.venvs/farus_pose")
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task"
ESTIMATOR_SCRIPT = PROJECT_ROOT / "scripts" / "pose_estimator_mediapipe.py"


def pose_venv_python() -> Path:
    venv = Path(os.environ.get("FARUS_POSE_VENV", str(DEFAULT_VENV)))
    python = venv / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"pose venv python not found at {python}; create it with:\n"
            "  python -m venv ~/.venvs/farus_pose && "
            "~/.venvs/farus_pose/bin/pip install mediapipe"
        )
    return python


def run_pose_estimator(
    image_path: Path,
    out_json: Path,
    overlay_path: Path | None = None,
    model_path: Path | None = None,
    timeout: int = 120,
) -> dict:
    """Call the MediaPipe estimator in its own venv and return its JSON report."""
    python = pose_venv_python()
    model = Path(model_path) if model_path else DEFAULT_MODEL
    if not model.is_file():
        raise FileNotFoundError(f"pose model not found: {model}")

    command = [
        str(python),
        str(ESTIMATOR_SCRIPT),
        "--image",
        str(image_path),
        "--model",
        str(model),
        "--out",
        str(out_json),
    ]
    if overlay_path is not None:
        command += ["--overlay", str(overlay_path)]

    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pose estimator failed ({result.returncode}):\n{result.stderr[-2000:]}"
        )
    if not out_json.is_file():
        raise RuntimeError("pose estimator did not write its output JSON")
    return json.loads(out_json.read_text())
