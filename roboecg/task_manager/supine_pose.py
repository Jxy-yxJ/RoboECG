"""Supine placement for the official M_Medical_01 rig.

Measured rig orientation (M0 debug, 2026-09-18): standing T-pose with up +Z,
patient left +X and the body facing -Y (the breast joints sit at y < 0).

Supine therefore is ``Rz(90) @ Rx(-90)``: the body's -Y (anterior) maps to +Z
(chest up), its +Z (up) maps to -X (head toward -X) and its +X (left) stays
+Y (patient left toward the robot side).  The root Z is solved from the world
bounding box so the most posterior point rests on the table top.

The rotation is written into the asset's existing RotateXYZ op (order matters:
appending an Orient op would place it after the Scale op).  No clinical data is
involved: the table height is an engineering choice in the scene config.
"""
from __future__ import annotations

import numpy as np

SUPINE_ROTATE_XYZ_DEG = (-90.0, 0.0, 90.0)


def rotation_x(degrees: float) -> np.ndarray:
    theta = np.radians(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(degrees: float) -> np.ndarray:
    theta = np.radians(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(degrees: float) -> np.ndarray:
    theta = np.radians(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _get_or_add_op(xformable, op_type):
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xformable.AddXformOp(op_type, UsdGeom.XformOp.PrecisionDouble)


# Supine arm pose.  Joint names and rotation axes are rig-specific and were
# determined empirically per rig (scripts/m0_debug_biped_pose.py):
#   M_Medical_01: L/R_Upperarm local Z ∓75 lowers the arm
#   biped_demo:   L/R_UpArm    local Z ±90 lowers the arm
#                 (measured hand travel 0.72 m -> 0.19 m from the body axis)
#
# The LEFT arm (the side the robot works on) is abducted from the alongside
# position: with the arm alongside, it occupies the space lateral to the chest
# and blocks V6 at the midaxillary line (measured at the V6 goal, 2026-09-19:
# 30 deg -> forearm_link vs upper_arm_l -75 mm; 45 deg -> -67 mm; 60 deg ->
# -21 mm; 75 deg -> +23 mm).  Clinical practice likewise abducts the arm to
# reach the axillary line.  The right arm stays alongside (outside the working
# area).
LEFT_ARM_ABDUCTION_DEG = 75.0

SUPINE_POSE_MEDICAL = (
    (("L_Upperarm",), (0.0, 0.0, 1.0), -75.0 + LEFT_ARM_ABDUCTION_DEG),
    (("R_Upperarm",), (0.0, 0.0, 1.0), 75.0),
    (("L_Forearm",), (1.0, 0.0, 0.0), 15.0),
    (("R_Forearm",), (1.0, 0.0, 0.0), 15.0),
)

SUPINE_POSE_BIPED = (
    (("L_UpArm",), (0.0, 0.0, 1.0), 90.0 - LEFT_ARM_ABDUCTION_DEG),
    (("R_UpArm",), (0.0, 0.0, 1.0), -90.0),
)


def default_pose_for(index) -> tuple:
    """Pick the supine pose whose joints exist in the rig."""
    if "L_Upperarm" in index:
        return SUPINE_POSE_MEDICAL
    if "L_UpArm" in index:
        return SUPINE_POSE_BIPED
    raise KeyError("no known supine pose for this rig")


def find_rig(stage, human_root_path="/World/Human"):
    """Return (skeleton_prim, binding_target).

    binding_target is the UsdSkel.Root ancestor when present, otherwise the
    human root prim (the bare-body biped asset has no SkelRoot; binding the
    animation to the asset root still drives the manual-FK reader).
    """
    from pxr import UsdSkel

    skeleton = None
    skel_root = None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(human_root_path):
            continue
        if prim.IsA(UsdSkel.Skeleton) and skeleton is None:
            skeleton = prim
        if prim.IsA(UsdSkel.Root) and skel_root is None:
            skel_root = prim
    if skeleton is None:
        raise RuntimeError(f"skeleton not found under {human_root_path}")
    if skel_root is None:
        skel_root = stage.GetPrimAtPath(human_root_path)
        if not skel_root.IsValid():
            raise RuntimeError(f"human root not found: {human_root_path}")
    return skeleton, skel_root


def apply_supine_pose(stage, human_root_path="/World/Human", pose=None):
    """Author and bind the arms-alongside-torso pose for the supine patient.

    `pose` overrides the default SUPINE_POSE (used by the axis-search
    diagnostic to determine the rig-specific rotation axes).
    """
    from pxr import Gf, UsdSkel, Vt

    skeleton, skel_root = find_rig(stage, human_root_path)
    joints = list(UsdSkel.Skeleton(skeleton).GetJointsAttr().Get())
    rest = list(UsdSkel.Skeleton(skeleton).GetRestTransformsAttr().Get())
    index = {name.rsplit("/", 1)[-1]: i for i, name in enumerate(joints)}
    pose = pose if pose is not None else default_pose_for(index)

    rotations = []
    translations = []
    scales = []
    for matrix in rest:
        transform = Gf.Transform(Gf.Matrix4d(matrix))
        rotations.append(Gf.Quatf(transform.GetRotation().GetQuat()))
        translations.append(Gf.Vec3f(transform.GetTranslation()))
        scales.append(Gf.Vec3h(transform.GetScale()))

    applied = []
    for names, axis, angle_deg in pose:
        leaf = next((name for name in names if name in index), None)
        if leaf is None:
            continue
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half = np.radians(angle_deg) / 2.0
        extra = Gf.Quatf(
            float(np.cos(half)), Gf.Vec3f(*(np.sin(half) * axis).tolist())
        )
        rotations[index[leaf]] = Gf.Quatf(rotations[index[leaf]]) * extra
        applied.append(leaf)
    if not applied:
        raise KeyError(f"no supine pose joints found in rig: {pose}")

    anim = UsdSkel.Animation.Define(stage, f"{human_root_path}/SupinePose")
    anim.CreateJointsAttr(joints)
    anim.CreateRotationsAttr(Vt.QuatfArray(rotations))
    anim.CreateTranslationsAttr(Vt.Vec3fArray(translations))
    anim.CreateScalesAttr(Vt.Vec3hArray(scales))
    # Author the skel:animationSource relationship directly: constructing the
    # multi-apply BindingAPI object crashes this Isaac build on prims where it
    # was never applied.
    relationship = skel_root.CreateRelationship("skel:animationSource", custom=False)
    relationship.SetTargets([anim.GetPath()])
    return anim


def set_prim_translate(prim, position) -> None:
    from pxr import Gf, UsdGeom

    op = _get_or_add_op(UsdGeom.Xformable(prim), UsdGeom.XformOp.TypeTranslate)
    op.Set(Gf.Vec3d(*[float(v) for v in position]))


def set_prim_scale(prim, scale: float) -> None:
    """Set the prim's uniform scale through its existing Scale op."""
    from pxr import Gf, UsdGeom

    op = _get_or_add_op(UsdGeom.Xformable(prim), UsdGeom.XformOp.TypeScale)
    value = float(scale)
    op.Set(Gf.Vec3d(value, value, value))


def set_prim_rotate_xyz(prim, degrees_xyz) -> None:
    """Set the prim rotation through its RotateXYZ op (USD order: Rz @ Ry @ Rx).

    Writing the existing RotateXYZ op avoids appending a new op after the
    asset's Scale op, which would rotate the scaled geometry.
    """
    from pxr import Gf, UsdGeom

    op = _get_or_add_op(UsdGeom.Xformable(prim), UsdGeom.XformOp.TypeRotateXYZ)
    op.Set(Gf.Vec3d(*[float(v) for v in degrees_xyz]))


def world_bounds(stage, prim_path: str):
    """(min_xyz, max_xyz) of the prim's world bounding box."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.asarray(bound.GetMin(), dtype=float), np.asarray(bound.GetMax(), dtype=float)


def place_supine(
    stage,
    human_root_path="/World/Human",
    root_xy=(0.0, 0.0),
    table_top_z=0.70,
    yaw_deg=0.0,
    settle_offset_m=0.0,
) -> dict:
    """Lay the rig supine and drop it onto the table top.

    Returns a report with the final root transform, the world bounding box and
    the rotation/provenance so the scene can be audited.
    """
    rotation = rotation_z(yaw_deg) @ rotation_z(90.0) @ rotation_x(-90.0)
    prim = stage.GetPrimAtPath(human_root_path)
    if not prim.IsValid():
        raise RuntimeError(f"human prim not found: {human_root_path}")

    set_prim_translate(prim, [root_xy[0], root_xy[1], 0.0])
    set_prim_rotate_xyz(prim, (-90.0, 0.0, 90.0 + float(yaw_deg)))

    min_xyz, max_xyz = world_bounds(stage, human_root_path)
    dz = float(table_top_z) - float(min_xyz[2]) + float(settle_offset_m)
    set_prim_translate(prim, [root_xy[0], root_xy[1], dz])
    min_xyz, max_xyz = world_bounds(stage, human_root_path)

    return {
        "human_root_path": human_root_path,
        "root_translate": [float(root_xy[0]), float(root_xy[1]), float(dz)],
        "rotation_matrix": rotation.tolist(),
        "rotation": (
            "Rz(yaw) @ Rz(90) @ Rx(-90): supine, head toward -X, chest toward "
            "+Z, patient left toward +Y (rig faces -Y when standing)"
        ),
        "bbox_min_world": min_xyz.tolist(),
        "bbox_max_world": max_xyz.tolist(),
        "table_top_z": float(table_top_z),
        "provenance": (
            "engineering: supine root rotation + table height; body dimensions "
            "come from the M_Medical_01 asset bounding box"
        ),
    }
