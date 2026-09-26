"""Supine ECG scene: table + patient + UR3 + electrode tool + cameras.

All layout numbers are engineering choices (table height, pedestal, camera
positions); the patient geometry comes from the M_Medical_01 asset and the
chest landmarks are measured at runtime.  No clinical constants live here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roboecg.task_manager.supine_pose import (
    apply_supine_pose,
    place_supine,
    set_prim_rotate_xyz,
    set_prim_translate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "m0"
LOCAL_ASSETS_DIR = PROJECT_ROOT / "assets" / "local"

ASSETS_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
)
UR3_USD = f"{ASSETS_ROOT}/Isaac/Robots/UniversalRobots/ur3/ur3.usd"
# Bare-body patient: ECG electrodes require bare skin, and M_Medical_01 has no
# torso surface under its lab coat / scrub shirt.  biped_demo is the official
# bare-body mesh (single 16.8k-point mesh, 81-joint skeleton).
HUMAN_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
    "/Isaac/People/Characters/biped_demo/biped_demo_meters.usd"
)

LAYOUT = {
    # Engineering scene layout (not clinical data).
    "table_center_xy": [0.125, 0.0],
    "table_size_xyz": [1.95, 0.80, 0.08],
    "table_top_z": 0.70,
    "human_root_xy": [1.05, 0.0],
    "ur3_nominal_base_position": [0.10, 0.70, 0.50],
    "ur3_nominal_base_yaw_deg": 180.0,
    "tool_electrode_radius_m": 0.010,
    "tool_electrode_offset_m": 0.088,
    "video_resolution": [960, 540],
    "provenance": (
        "engineering: table height/size, pedestal and camera placement; "
        "patient body dimensions come from the asset bounding box"
    ),
}

CAMERAS = {
    # ThirdView frames patient + robot (robot is on the patient's left, +Y).
    "ThirdView": {"position": [-1.45, -1.35, 2.00], "look_at": [-0.10, 0.25, 0.80]},
    "ChestCloseup": {"position": [-0.10, -0.95, 1.30], "look_at": [0.0, 0.0, 0.92]},
    # Overhead perception camera: a supine chest faces +Z, so an oblique
    # single view cannot see the lateral chest (measured in M0: 9/16 cells).
    "PerceptionRGBD": {"position": [-0.35, 0.00, 2.10], "look_at": [-0.35, 0.05, 1.00]},
}


def add_camera(stage, prim_path, position, look_at, aspect=16.0 / 9.0):
    """Create a pinhole camera with square pixels.

    USD defaults the vertical aperture to 15.2908 mm independently of the
    horizontal one; at 16:9 that gives a vertical FOV of ~72 deg while the
    pinhole intrinsics model (fx = fy) assumes ~59 deg, which stretches the
    deprojected point cloud.  Setting vertical_aperture = horizontal / aspect
    makes fx = fy hold for the rendered image.
    """
    from pxr import Gf, UsdGeom

    from roboecg.coordinate_transform.frames import look_at_rotation, quat_wxyz_from_rotation

    horizontal_aperture = 20.955
    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.CreateFocalLengthAttr(10.4775)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(horizontal_aperture / float(aspect))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    quat = quat_wxyz_from_rotation(look_at_rotation(position, look_at))
    camera.AddOrientOp().Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))
    )
    return camera


def add_table(stage, layout=None):
    from pxr import Gf, UsdGeom, UsdPhysics

    layout = layout or LAYOUT
    center = layout["table_center_xy"]
    size = layout["table_size_xyz"]
    top_z = layout["table_top_z"]
    prim = UsdGeom.Cube.Define(stage, "/World/Table")
    prim.CreateSizeAttr(1.0)
    prim.AddTranslateOp().Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(top_z - size[2] / 2))
    )
    prim.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    prim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.72, 0.74, 0.78)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

    leg_radius = 0.03
    leg_height = float(top_z - size[2])
    for index, (sx, sy) in enumerate(
        [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ):
        leg = UsdGeom.Cylinder.Define(stage, f"/World/Table/Leg_{index}")
        leg.CreateRadiusAttr(leg_radius)
        leg.CreateHeightAttr(leg_height)
        leg.AddTranslateOp().Set(
            Gf.Vec3d(
                float(center[0] + sx * (size[0] / 2 - 0.12)),
                float(center[1] + sy * (size[1] / 2 - 0.08)),
                leg_height / 2,
            )
        )
        leg.CreateDisplayColorAttr().Set([Gf.Vec3f(0.45, 0.47, 0.50)])
    return prim


def add_electrode_tool(stage, prefix="/World/ToolAssembly"):
    """Dry-electrode end effector along tool +Z: flange, FT sensor, holder, disc.

    The disc radius sets the contact area used by the press-force target
    (configs/ecg_rules.yaml -> contact.contact_area_m2).
    """
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, prefix)
    xform = UsdGeom.Xformable(root.GetPrim())
    xform.AddTranslateOp()
    xform.AddOrientOp()

    def cylinder(name, radius, height, z0, color):
        prim = UsdGeom.Cylinder.Define(stage, f"{prefix}/{name}")
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0 + height / 2.0))
        prim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return prim

    radius = LAYOUT["tool_electrode_radius_m"]
    offset = LAYOUT["tool_electrode_offset_m"]
    cylinder("FlangeAdapter", 0.030, 0.018, 0.000, (0.35, 0.35, 0.35))
    cylinder("ForceTorqueSensor", 0.036, 0.014, 0.018, (0.55, 0.55, 0.60))
    cylinder("ElectrodeHolder", 0.016, offset - 0.032, 0.032, (0.12, 0.12, 0.14))
    cylinder("Electrode", radius, 0.004, offset, (0.85, 0.55, 0.15))
    return root


def set_tool_pose(stage, position, rotation, prefix="/World/ToolAssembly"):
    from pxr import Gf, UsdGeom

    from roboecg.coordinate_transform.frames import quat_wxyz_from_rotation

    prim = stage.GetPrimAtPath(prefix)
    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpType(): op for op in xformable.GetOrderedXformOps()}
    ops[UsdGeom.XformOp.TypeTranslate].Set(Gf.Vec3d(*[float(v) for v in position]))
    quat = quat_wxyz_from_rotation(rotation)
    ops[UsdGeom.XformOp.TypeOrient].Set(
        Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])))
    )


def add_marker(stage, prim_path, position, radius, color):
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color])])
    return sphere


def add_line(stage, prim_path, start, end, color=(0.95, 0.6, 0.1), width=0.004):
    from pxr import Gf, UsdGeom

    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr([2])
    curves.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in start]), Gf.Vec3f(*[float(v) for v in end])])
    curves.CreateWidthsAttr([float(width)])
    curves.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color])])
    return curves


def add_frame_axes(stage, prefix, frame, length=0.12):
    """Visualize a ChestFrame: up=blue, lateral=green, anterior=red."""
    origin = np.asarray(frame.origin, dtype=float)
    add_line(stage, f"{prefix}/Up", origin, origin + frame.up * length, (0.15, 0.35, 0.95))
    add_line(
        stage, f"{prefix}/Lateral", origin, origin + frame.lateral * length, (0.15, 0.75, 0.25)
    )
    add_line(
        stage,
        f"{prefix}/Anterior",
        origin,
        origin + frame.anterior * length,
        (0.95, 0.20, 0.15),
    )


def add_region_box(stage, prefix, center, size, color=(0.95, 0.75, 0.10), width=0.004):
    """Wireframe box for the reachability region visualization."""
    center = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    corners = [
        center + np.array([sx * half[0], sy * half[1], sz * half[2]])
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    for index, (a, b) in enumerate(edges):
        add_line(stage, f"{prefix}/Edge_{index:02d}", corners[a], corners[b], color, width)


def build_scene(world, layout=None):
    """Assemble the M0 scene; returns (stage, scene_report)."""
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Gf, UsdLux

    layout = layout or LAYOUT
    stage = get_current_stage()
    world.scene.add_default_ground_plane()

    add_table(stage, layout)

    add_reference_to_stage(UR3_USD, "/World/UR3")
    set_prim_translate(stage.GetPrimAtPath("/World/UR3"), layout["ur3_nominal_base_position"])
    set_prim_rotate_xyz(
        stage.GetPrimAtPath("/World/UR3"),
        (0.0, 0.0, layout["ur3_nominal_base_yaw_deg"]),
    )

    add_reference_to_stage(HUMAN_USD, "/World/Human")
    world.reset()
    stage.Load("/World/UR3")
    stage.Load("/World/Human")

    apply_supine_pose(stage, human_root_path="/World/Human")
    supine_report = place_supine(
        stage,
        human_root_path="/World/Human",
        root_xy=layout["human_root_xy"],
        table_top_z=layout["table_top_z"],
    )

    add_electrode_tool(stage)
    add_camera(stage, "/World/Cameras/ThirdView", **CAMERAS["ThirdView"])
    add_camera(stage, "/World/Cameras/ChestCloseup", **CAMERAS["ChestCloseup"])
    add_camera(stage, "/World/Cameras/PerceptionRGBD", **CAMERAS["PerceptionRGBD"])

    fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(450.0)
    fill.AddRotateXYZOp().Set(Gf.Vec3f(50.0, 0.0, -30.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(600.0)

    report = {
        "layout": layout,
        "supine": supine_report,
        "human_bbox_min": supine_report["bbox_min_world"],
        "human_bbox_max": supine_report["bbox_max_world"],
        "table_bounds": {
            "x": [
                layout["table_center_xy"][0] - layout["table_size_xyz"][0] / 2,
                layout["table_center_xy"][0] + layout["table_size_xyz"][0] / 2,
            ],
            "y": [
                layout["table_center_xy"][1] - layout["table_size_xyz"][1] / 2,
                layout["table_center_xy"][1] + layout["table_size_xyz"][1] / 2,
            ],
            "top_z": layout["table_top_z"],
        },
        "cameras": CAMERAS,
    }
    return stage, report


def mesh_world_points(stage, human_root_path="/World/Human"):
    """World-space asset mesh points (rest pose, invisible prims excluded).

    The chest is rigidly transformed by the human root, so an arms-only pose
    does not affect it.  Used for the CPU-only surface path and for target
    snapping.
    """
    from pxr import UsdGeom

    matrix = world_matrix(stage, human_root_path)
    chunks = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not str(prim.GetPath()).startswith(human_root_path):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points:
            chunks.append(np.array([[p[0], p[1], p[2]] for p in points], dtype=float))
    if not chunks:
        raise RuntimeError(f"no mesh points under {human_root_path}")
    points = np.vstack(chunks)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def world_matrix(stage, prim_path):
    """Prim world transform as a column-vector 4x4 numpy matrix."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    rotation_row = transform.ExtractRotationMatrix()
    rotation = np.array(
        [[rotation_row[i][j] for j in range(3)] for i in range(3)], dtype=float
    ).T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [translation[0], translation[1], translation[2]]
    return matrix
