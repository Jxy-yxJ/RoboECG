"""Import the 25 SSM torso surfaces into mesh formats Isaac can consume (I5).

Source: assets/external/torso_models/T_*_torso_coarse_surface.vtk (Bender et
al., Zenodo 10.5281/zenodo.20086105, CC-BY-4.0).  Parsing lives in
roboecg.perception.torso_mesh; this script writes Wavefront OBJ (always, in
meters) plus a USD mesh when pxr is importable (i.e. inside the Isaac
environment: `--usd`).

Usage:
    python3 scripts/import_torso_ssm.py                # all 25 -> OBJ
    python3 scripts/import_torso_ssm.py --usd          # + USD (Isaac env)
    python3 scripts/import_torso_ssm.py --model T_01   # a single model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboecg.perception.torso_mesh import (  # noqa: E402
    parse_vtk_polydata,
    write_obj,
)

DATA_DIR = PROJECT_ROOT / "assets" / "external" / "torso_models"
OUT_DIR = DATA_DIR / "converted"


def write_usd(points_m, faces, path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Torso")
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(c) for c in p]) for p in points_m])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="model id(s) like T_01 (default: all T_01..T_25)",
    )
    parser.add_argument("--usd", action="store_true", help="also write USD meshes")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    ids = args.model or [f"T_{i:02d}" for i in range(1, 26)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in ids:
        vtk = DATA_DIR / f"{model_id}_torso_coarse_surface.vtk"
        points_mm, faces = parse_vtk_polydata(vtk)
        points_m = points_mm / 1000.0
        obj_path = args.out_dir / f"{model_id}.obj"
        write_obj(points_m, faces, obj_path)
        note = f"{len(points_mm)} verts / {len(faces)} tris"
        if args.usd:
            usd_path = args.out_dir / f"{model_id}.usd"
            write_usd(points_m, faces, usd_path)
            note += f" -> {obj_path.name} + {usd_path.name}"
        else:
            note += f" -> {obj_path.name}"
        extent = (points_mm.max(axis=0) - points_mm.min(axis=0)) / 1000.0
        print(
            f"import_torso_ssm: {model_id}: {note} "
            f"(extent {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m)",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"import_torso_ssm: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
