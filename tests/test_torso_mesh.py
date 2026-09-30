"""Tests for the SSM torso mesh I/O (I5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roboecg.perception.torso_mesh import parse_vtk_polydata, write_obj

DATA_DIR = Path(__file__).resolve().parents[1] / "assets" / "external" / "torso_models"

SYNTHETIC = """# vtk DataFile Version 4.2
vtk output
ASCII
DATASET POLYDATA
POINTS 4 float
0 0 0 1 0 0 2 0 0 0 1 0
POLYGONS 2 8
3 0 1 3
3 1 2 3
"""


def test_parse_synthetic_polydata(tmp_path):
    path = tmp_path / "tiny.vtk"
    path.write_text(SYNTHETIC, encoding="utf-8")
    points, faces = parse_vtk_polydata(path)
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[2], [2.0, 0.0, 0.0])
    assert faces.shape == (2, 3)
    np.testing.assert_array_equal(faces[0], [0, 1, 3])


def test_parse_real_model_counts():
    points, faces = parse_vtk_polydata(
        DATA_DIR / "T_01_torso_coarse_surface.vtk"
    )
    assert points.shape == (2986, 3)
    assert faces.shape == (5968, 3)
    # mm-space torso: extents of a few hundred mm per axis
    extent = points.max(axis=0) - points.min(axis=0)
    assert 200.0 < extent[0] < 700.0
    assert 100.0 < extent[1] < 500.0
    assert 300.0 < extent[2] < 800.0


def test_write_obj_roundtrip(tmp_path):
    path = tmp_path / "tiny.vtk"
    path.write_text(SYNTHETIC, encoding="utf-8")
    points_mm, faces = parse_vtk_polydata(path)
    obj = tmp_path / "tiny.obj"
    write_obj(points_mm / 1000.0, faces, obj)
    lines = obj.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4 + 2
    assert lines[0].startswith("v 0 0 0")
    assert lines[4] == "f 1 2 4"


def test_parse_rejects_missing_blocks(tmp_path):
    path = tmp_path / "bad.vtk"
    path.write_text("DATASET POLYDATA\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_vtk_polydata(path)
