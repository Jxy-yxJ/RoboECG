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


def anatomical_axes(points_m: np.ndarray, electrodes_m: np.ndarray):
    """Anatomical axes of an SSM torso: (left, anterior, down), right-handed.

    left: V1->V2 (the parasternal pair is unambiguous); anterior: the mean
    outward surface normal at V1/V2 (frontal electrodes) orthogonalised
    against left -- an electrode-cloud centroid offset would be dominated by
    the lateral component and pick the wrong axis; down: left x anterior (the
    anatomical identity, no extra sign needed).  Raises when the mesh does not
    support the construction or the axes are inconsistent.
    """
    from roboecg.perception.depth import estimate_surface_normal

    a = electrodes_m[1] - electrodes_m[0]
    a = a / (np.linalg.norm(a) + 1e-12)
    centroid = points_m.mean(axis=0)
    normals = []
    for index in (0, 1):
        position = electrodes_m[index]
        outward = position + (position - centroid)
        normal, _ = estimate_surface_normal(
            points_m, position, radius=0.030, orient_toward=outward
        )
        if normal is None:
            raise RuntimeError(f"no surface patch near V{index + 1}")
        normals.append(np.asarray(normal, dtype=float))
    b = np.mean(normals, axis=0)
    b = b - float(np.dot(b, a)) * a
    b = b / (np.linalg.norm(b) + 1e-12)
    c = np.cross(a, b)
    down_check = float(
        np.dot(c, electrodes_m[3:6].mean(axis=0) - electrodes_m[0:2].mean(axis=0))
    )
    if down_check <= 0.0:
        raise ValueError(
            f"model axes inconsistent: left x anterior = {down_check:.4f} "
            "(expected > 0, pointing towards V4-V6)"
        )
    return a, b, c


def subdivide_and_smooth(
    points_m: np.ndarray, faces: np.ndarray,
    subdivisions: int = 2, smooth_iterations: int = 3, lam: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint-subdivide then Laplacian-smooth a closed triangle mesh.

    The SSM surfaces are deliberately coarse (~3-4 cm triangles), far below a
    real depth camera's chest sampling; the faceting, not the anatomy,
    dominates the detector's input statistics (I5-c stage 2 isolates this
    factor).  Subdividing and smoothing approximates smooth skin while moving
    the surface only a few millimetres.
    """
    points = np.asarray(points_m, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    for _ in range(int(subdivisions)):
        edge_midpoint: dict = {}
        vertices = [row for row in points]

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            index = edge_midpoint.get(key)
            if index is None:
                index = len(vertices)
                vertices.append(0.5 * (points[a] + points[b]))
                edge_midpoint[key] = index
            return index

        new_faces = []
        for a, b, c in faces:
            ab = midpoint(int(a), int(b))
            bc = midpoint(int(b), int(c))
            ca = midpoint(int(c), int(a))
            new_faces.append((a, ab, ca))
            new_faces.append((ab, b, bc))
            new_faces.append((ca, bc, c))
            new_faces.append((ab, bc, ca))
        points = np.asarray(vertices, dtype=float)
        faces = np.asarray(new_faces, dtype=np.int64)

    for _ in range(int(smooth_iterations)):
        sums = np.zeros_like(points)
        counts = np.zeros(len(points))
        np.add.at(sums, faces[:, 0], points[faces[:, 1]])
        np.add.at(sums, faces[:, 0], points[faces[:, 2]])
        np.add.at(sums, faces[:, 1], points[faces[:, 0]])
        np.add.at(sums, faces[:, 1], points[faces[:, 2]])
        np.add.at(sums, faces[:, 2], points[faces[:, 0]])
        np.add.at(sums, faces[:, 2], points[faces[:, 1]])
        np.add.at(counts, faces[:, 0], 2.0)
        np.add.at(counts, faces[:, 1], 2.0)
        np.add.at(counts, faces[:, 2], 2.0)
        neighbours = sums / np.maximum(counts, 1.0)[:, None]
        points = points + float(lam) * (neighbours - points)
    return points, faces


def surface_fraction_of(points, up, lateral, anterior, position) -> tuple:
    """(u_frac, v_frac) of a point on a torso surface.

    u_frac runs 0 at the head end of the surface span to 1 at the hip end
    (both measured along ``up``); v_frac is the lateral coordinate as a
    fraction of the local surface half-width (98th percentile), measured from
    the bilateral midpoint.  Electrode-free by construction: it only uses the
    surface and a few landmark-independent directions.
    """
    points = np.asarray(points, dtype=float)
    up = np.asarray(up, dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)
    lateral = np.asarray(lateral, dtype=float)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-12)
    position = np.asarray(position, dtype=float)
    u_all = points @ up
    v_all = points @ lateral
    u_head, u_hips = float(u_all.max()), float(u_all.min())
    v_mid = 0.5 * (float(v_all.max()) + float(v_all.min()))
    u = float(position @ up)
    v = float(position @ lateral)
    band = np.abs(u_all - u) <= 0.02
    if np.count_nonzero(band) < 10:
        band = np.abs(u_all - u) <= 0.05
    half_width = float(np.percentile(np.abs(v_all[band] - v_mid), 98.0))
    u_frac = (u_head - u) / max(u_head - u_hips, 1e-9)
    v_frac = (v - v_mid) / max(half_width, 1e-9)
    return float(u_frac), float(v_frac)


def surface_fraction_landmarks(points, up, lateral, anterior, fractions) -> dict:
    """Place landmarks on a torso surface from (u_frac, v_frac) pairs.

    The detector's fine-tune (I5-c stage 2) needs landmark labels on the SSM
    torsos.  Deriving them from the real electrode coordinates would leak the
    evaluation target into the training data; instead the biped's rig
    landmarks are expressed as surface fractions (see `surface_fraction_of`)
    and applied here to the model surface.  Positions sit on the anterior
    surface at the requested (u, v).
    """
    points = np.asarray(points, dtype=float)
    up = np.asarray(up, dtype=float)
    up = up / (np.linalg.norm(up) + 1e-12)
    lateral = np.asarray(lateral, dtype=float)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-12)
    anterior = np.asarray(anterior, dtype=float)
    anterior = anterior / (np.linalg.norm(anterior) + 1e-12)
    u_all = points @ up
    v_all = points @ lateral
    n_all = points @ anterior
    u_head, u_hips = float(u_all.max()), float(u_all.min())
    v_mid = 0.5 * (float(v_all.max()) + float(v_all.min()))
    landmarks = {}
    for name, (u_frac, v_frac) in fractions.items():
        u = u_head - float(u_frac) * (u_head - u_hips)
        band = np.abs(u_all - u) <= 0.02
        if np.count_nonzero(band) < 10:
            band = np.abs(u_all - u) <= 0.05
        half_width = float(np.percentile(np.abs(v_all[band] - v_mid), 98.0))
        v = v_mid + float(v_frac) * half_width
        near = band & (np.abs(v_all - v) <= 0.03)
        if np.count_nonzero(near) >= 5:
            n = float(np.percentile(n_all[near], 95.0))
        else:
            n = float(np.percentile(n_all[band], 95.0))
        landmarks[name] = u * up + v * lateral + n * anterior
    return landmarks
