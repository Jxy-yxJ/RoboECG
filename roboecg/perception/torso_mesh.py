"""Mesh I/O for the SSM torso surfaces (I5).

The GT torso files (assets/external/torso_models, Bender et al., Zenodo
10.5281/zenodo.20086105, CC-BY-4.0) are ASCII POLYDATA: a POINTS block (mm) and
a POLYGONS block of triangles.  This module parses them without any VTK
dependency and writes Wavefront OBJ, so the meshes can be inspected offline or
converted to USD inside the Isaac environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_vtk_polydata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (points Nx3, triangles Mx3) from an ASCII POLYDATA file."""
    tokens = Path(path).read_text(encoding="utf-8").split()
    try:
        points_at = tokens.index("POINTS")
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: no POINTS block") from exc
    n_points = int(tokens[points_at + 1])
    start = points_at + 3  # skip "POINTS N float"
    values = np.asarray(
        [float(t) for t in tokens[start : start + 3 * n_points]], dtype=float
    )
    if values.size != 3 * n_points:
        raise ValueError(f"{Path(path).name}: truncated POINTS block")
    points = values.reshape(n_points, 3)

    try:
        polys_at = tokens.index("POLYGONS")
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: no POLYGONS block") from exc
    n_polys = int(tokens[polys_at + 1])
    cursor = polys_at + 3  # skip "POLYGONS M K"
    faces = np.empty((n_polys, 3), dtype=np.int64)
    for index in range(n_polys):
        n_verts = int(tokens[cursor])
        if n_verts != 3:
            raise ValueError(
                f"{Path(path).name}: polygon {index} has {n_verts} "
                "vertices (expected 3)"
            )
        faces[index] = [int(t) for t in tokens[cursor + 1 : cursor + 4]]
        cursor += 4
    return points, faces


def write_obj(points_m: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Write a vertices/faces mesh as Wavefront OBJ (points in meters)."""
    lines = [f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in points_m]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
