"""Temporary diagnostic: inspect a human USD asset (meshes, skeleton, bbox).

Usage:
    ./scripts/run_headless.sh scripts/m0_debug_assets.py <usd_url> [--hide PATTERN ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot, save_png  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_url")
    parser.add_argument("--hide", action="append", default=[])
    parser.add_argument("--render", default=None, help="save an RGB screenshot to this path")
    args = parser.parse_args()

    app = boot(headless=True, width=960, height=720)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Gf, UsdGeom, UsdLux, UsdSkel

        from roboecg.task_manager.ecg_scene import add_camera, world_matrix

        world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()
        add_reference_to_stage(args.usd_url, "/World/Human")
        world.reset()
        stage.Load("/World/Human")

        hidden = []
        for pattern in args.hide:
            for prim in stage.Traverse():
                if pattern.lower() in prim.GetName().lower() and prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(str(prim.GetPath()))
        print("hidden:", hidden)

        meshes = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            imageable = UsdGeom.Imageable(prim)
            visible = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            meshes.append((str(prim.GetPath()), len(points) if points else 0, visible))
        print(f"meshes: {len(meshes)}")
        for path, count, visible in meshes:
            print(f"  {'VIS' if visible else 'HID'} {count:7d}  {path}")

        for prim in stage.Traverse():
            if prim.IsA(UsdSkel.Skeleton):
                joints = list(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or [])
                print(f"skeleton: {prim.GetPath()} joints={len(joints)}")
                print("  ", [j.rsplit('/', 1)[-1] for j in joints][:60])

        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bound = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Human")).ComputeAlignedRange()
        print("bbox:", np.round(bound.GetMin(), 3).tolist(), np.round(bound.GetMax(), 3).tolist())

        if args.render:
            add_camera(
                stage,
                "/World/Cameras/AssetView",
                [0.0, -1.6, 1.2],
                [0.0, 0.0, 0.9],
            )
            light = UsdLux.DomeLight.Define(stage, "/World/Dome")
            light.CreateIntensityAttr(600.0)
            from m0_common import capture_rgb

            rgb = capture_rgb("/World/Cameras/AssetView", 960, 720)
            save_png(Path(args.render), rgb)
            print("render saved:", args.render)
    finally:
        app.close()


if __name__ == "__main__":
    main()
