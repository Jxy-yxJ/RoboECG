# Copied from farus_thyroid_isaac/farus/perception/isaac_skeleton.py on 2026-09-18.
# Local change: import namespace farus -> roboecg, plus a manual-FK fallback
# for rigs without a UsdSkel.Root ancestor (the bare-body biped_demo asset).
"""Read skeleton joint world transforms from an Isaac Sim USD stage.

Two backends:
  * ``UsdSkel.Cache`` when the skeleton has a ``UsdSkel.Root`` ancestor
    (M_Medical_01 and most rigs);
  * manual forward kinematics when it does not (the official ``biped_demo``
    bare-body asset has a ``UsdSkel.Skeleton`` without a SkelRoot, and feeding
    an invalid root to ``UsdSkel.Cache.Populate`` crashes the process).

Manual FK composes each joint's local transform (bound animation if present,
otherwise the skeleton rest transform) down the parent chain, then applies the
skeleton prim's world transform.  USD matrices are row-vector convention, so
``child_world = child_local * parent_world``.
"""
from __future__ import annotations


def find_skeleton_prim(stage):
    from pxr import UsdSkel

    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            return prim
    return None


def find_prim_path_by_name(stage, name: str) -> str | None:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def find_root_ancestor(skeleton_prim):
    from pxr import UsdSkel

    prim = skeleton_prim
    while prim and prim.IsValid():
        if prim.IsA(UsdSkel.Root):
            return prim
        prim = prim.GetParent()
    return None


def read_joint_world_positions(stage, skeleton_prim=None) -> dict:
    """Return {joint_token: [x, y, z]} in world coordinates."""
    skeleton_prim = skeleton_prim or find_skeleton_prim(stage)
    if skeleton_prim is None:
        raise RuntimeError("no UsdSkel.Skeleton prim found in stage")

    root = find_root_ancestor(skeleton_prim)
    if root is not None:
        return _read_with_skel_cache(stage, skeleton_prim, root)
    return _read_with_manual_fk(stage, skeleton_prim)


def _read_with_skel_cache(stage, skeleton_prim, root) -> dict:
    from pxr import Usd, UsdGeom, UsdSkel

    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root(root), Usd.PrimAllPrimsPredicate)
    query = cache.GetSkelQuery(UsdSkel.Skeleton(skeleton_prim))
    transforms = query.ComputeJointWorldTransforms(UsdGeom.XformCache())
    joints = list(UsdSkel.Skeleton(skeleton_prim).GetJointsAttr().Get() or [])
    return {
        name: [float(v) for v in matrix.ExtractTranslation()]
        for name, matrix in zip(joints, transforms)
    }


def find_bound_animation(stage, skeleton_prim):
    """Locate the animation source bound to the skeleton or its neighbours.

    Reads the raw ``skel:animationSource`` relationship instead of the
    BindingAPI wrapper: constructing the multi-apply API object on prims that
    do not have it crashes this build.
    """
    from pxr import UsdSkel

    base = str(skeleton_prim.GetPath()).rsplit("/", 1)[0]
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(base):
            continue
        relationship = prim.GetRelationship("skel:animationSource")
        if not relationship or not relationship.IsValid():
            continue
        targets = relationship.GetTargets()
        if targets:
            animation = UsdSkel.Animation(stage.GetPrimAtPath(targets[0]))
            if animation:
                return animation
    return None


def _read_with_manual_fk(stage, skeleton_prim) -> dict:
    from pxr import Gf, UsdGeom, UsdSkel

    skeleton = UsdSkel.Skeleton(skeleton_prim)
    joints = list(skeleton.GetJointsAttr().Get() or [])
    rest = list(skeleton.GetRestTransformsAttr().Get() or [])

    # Parents are derived from the joint path names: both supported rigs name
    # joints as paths (e.g. "Root/Pelvis/L_UpLeg").  Reading the optional
    # `jointParents` attribute is avoided because calling Get() on an
    # unauthored UsdAttribute crashes the process in this build.
    index = {name: i for i, name in enumerate(joints)}
    parents = []
    for name in joints:
        parent = name.rsplit("/", 1)[0] if "/" in name else None
        parents.append(index.get(parent, -1))

    animation = find_bound_animation(stage, skeleton_prim)
    rotations = translations = scales = None
    if animation:
        rotations = list(animation.GetRotationsAttr().Get() or [])
        translations = list(animation.GetTranslationsAttr().Get() or [])
        scales = list(animation.GetScalesAttr().Get() or [])
        if not (len(rotations) == len(translations) == len(scales) == len(joints)):
            rotations = translations = scales = None

    local = []
    for index in range(len(joints)):
        if rotations:
            quaternion = rotations[index]
            transform = Gf.Transform()
            transform.SetRotation(
                Gf.Rotation(
                    Gf.Quatd(
                        float(quaternion.GetReal()),
                        Gf.Vec3d(quaternion.GetImaginary()),
                    )
                )
            )
            transform.SetTranslation(Gf.Vec3d(translations[index]))
            transform.SetScale(Gf.Vec3d(scales[index]))
            matrix = transform.GetMatrix()
        else:
            matrix = Gf.Matrix4d(rest[index])
        local.append(matrix)

    world = [None] * len(joints)

    def joint_world(index: int):
        if world[index] is not None:
            return world[index]
        if parents[index] < 0:
            world[index] = local[index]
        else:
            world[index] = local[index] * joint_world(parents[index])
        return world[index]

    skel_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(skeleton_prim)
    return {
        name: [
            float(v)
            for v in (joint_world(index) * skel_to_world).ExtractTranslation()
        ]
        for index, name in enumerate(joints)
    }
