"""Diagnostic: inspect the M_Medical_01 prim xform ops and supine placement.

Kept for the rig-orientation finding (the standing rig faces -Y); the patient
asset is now the bare-body biped_demo, whose orientation follows the same
convention.  See docs/ECG_M0_FINDINGS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402


def main() -> None:
    app = boot(headless=True, width=640, height=480)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import UsdGeom

        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.task_manager import ecg_scene
        from roboecg.task_manager.supine_pose import place_supine, world_bounds

        world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()
        world.scene.add_default_ground_plane()
        add_reference_to_stage(ecg_scene.HUMAN_USD, "/World/Human")
        world.reset()
        stage.Load("/World/Human")

        prim = stage.GetPrimAtPath("/World/Human")
        print("== ops before ==")
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            print("  ", op.GetOpType(), op.GetPrecision(), op.Get())
        min_xyz, max_xyz = world_bounds(stage, "/World/Human")
        print("bbox before:", np.round(min_xyz, 3).tolist(), np.round(max_xyz, 3).tolist())

        joints = read_joint_world_positions(stage)
        by_leaf = {k.rsplit("/", 1)[-1]: np.asarray(v) for k, v in joints.items()}
        for leaf in ("NeckTwist01", "L_Clavicle", "L_Breast", "Pelvis", "L_ToeBase", "L_Hand", "R_Hand", "Spine02"):
            if leaf in by_leaf:
                print(f"  {leaf}: {np.round(by_leaf[leaf], 3).tolist()}")

        report = place_supine(
            stage,
            human_root_path="/World/Human",
            root_xy=ecg_scene.LAYOUT["human_root_xy"],
            table_top_z=ecg_scene.LAYOUT["table_top_z"],
        )
        print("== supine report ==", report)
        print("== ops after ==")
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            print("  ", op.GetOpType(), op.GetPrecision(), op.Get())

        joints = read_joint_world_positions(stage)
        by_leaf = {k.rsplit("/", 1)[-1]: np.asarray(v) for k, v in joints.items()}
        for leaf in ("NeckTwist01", "L_Clavicle", "L_Breast", "Pelvis", "L_ToeBase", "L_Hand", "R_Hand", "Spine02"):
            if leaf in by_leaf:
                print(f"  {leaf}: {np.round(by_leaf[leaf], 3).tolist()}")

        from pxr import UsdGeom

        all_points = []
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith("/World/Human"):
                pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if pts:
                    all_points.append(np.array([[v[0], v[1], v[2]] for v in pts]))
        if all_points:
            pts = np.vstack(all_points)
            print("mesh points:", pts.shape)
            for label, z_lo, z_hi in (
                ("head", 1.50, 1.85),
                ("chest", 1.15, 1.50),
                ("pelvis", 0.80, 1.15),
                ("legs", 0.00, 0.80),
            ):
                band = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] < z_hi)]
                if band.shape[0]:
                    print(
                        f"  band {label}: y_max={band[:,1].max():.3f} "
                        f"y_min={band[:,1].min():.3f} n={band.shape[0]}"
                    )
            print(f"  global y_max={pts[:,1].max():.3f}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
