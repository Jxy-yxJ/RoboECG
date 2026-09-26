# Copied from farus_thyroid_isaac/farus/robot_controller/ur3_lula.py on 2026-09-18.
# Upstream: FARUS thyroid scanning reproduction (frozen deliverable).
# Local change: import namespace farus -> roboecg.
"""UR3 inverse kinematics through Lula (official Isaac Sim UR3 configs)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

EXTENSION_NAME = "isaacsim.robot_motion.motion_generation"
# Only used when the extension registry lookup fails; override the Isaac Sim
# installation root with ISAACSIM_ENV if the default does not exist.
FALLBACK_CONFIG_DIR = (
    Path(os.environ.get("ISAACSIM_ENV", "/home/jxy/isaacsim-compat/env"))
    / "lib/python3.12/site-packages/isaacsim/extsDeprecated"
    / "isaacsim.robot_motion.motion_generation/motion_policy_configs"
    / "universal_robots/ur3"
)


def find_config_dir() -> Path:
    candidates = []
    try:
        from omni.kit.app import get_app

        ext_path = get_app().get_extension_manager().get_extension_path(EXTENSION_NAME)
        if ext_path:
            candidates.append(
                Path(ext_path) / "motion_policy_configs/universal_robots/ur3"
            )
    except Exception:
        pass
    candidates.append(FALLBACK_CONFIG_DIR)
    for candidate in candidates:
        if (candidate / "ur3.urdf").is_file():
            return candidate
    raise FileNotFoundError(f"UR3 Lula config not found in: {candidates}")


class UR3LulaIK:
    def __init__(self, frame: str = "tool0", config_dir: Path | None = None) -> None:
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

        config = Path(config_dir) if config_dir else find_config_dir()
        self.frame = frame
        self.solver = LulaKinematicsSolver(
            str(config / "rmpflow/ur3_robot_description.yaml"),
            str(config / "ur3.urdf"),
        )

    @property
    def joint_names(self) -> list[str]:
        return list(self.solver.get_joint_names())

    def forward(self, joints, frame: str | None = None):
        position, rotation = self.solver.compute_forward_kinematics(
            frame or self.frame, np.asarray(joints, dtype=float)
        )
        return np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)

    def solve_pose(
        self,
        position,
        quaternion_wxyz,
        warm_start,
        frame: str | None = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.2,
    ):
        joints, success = self.solver.compute_inverse_kinematics(
            frame or self.frame,
            np.asarray(position, dtype=float),
            target_orientation=np.asarray(quaternion_wxyz, dtype=float),
            warm_start=np.asarray(warm_start, dtype=float),
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        return np.asarray(joints, dtype=float), bool(success)

    def solve_position(
        self,
        position,
        warm_start,
        frame: str | None = None,
        position_tolerance: float = 0.005,
    ):
        joints, success = self.solver.compute_inverse_kinematics(
            frame or self.frame,
            np.asarray(position, dtype=float),
            warm_start=np.asarray(warm_start, dtype=float),
            position_tolerance=position_tolerance,
        )
        return np.asarray(joints, dtype=float), bool(success)


def find_articulation_roots(stage) -> list[str]:
    from pxr import UsdPhysics

    return [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
