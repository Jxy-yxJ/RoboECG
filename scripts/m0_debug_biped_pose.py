"""Diagnostic: find the arm-lowering rotation axes for the biped_demo rig.

The supine pose rotates the arms alongside the torso.  Rotation axes are
rig-specific, so this script applies single-joint candidate rotations and
reports the resulting hand position in the standing frame.  The arm is
"lowered" when the hand moves from the T-pose (x ~ 0.72, z ~ 1.47) toward the
body axis (small x) and downward (smaller z).

Run:
    ./scripts/run_headless.sh scripts/m0_debug_biped_pose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_common import boot  # noqa: E402

BIPED_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.0/Isaac/People/Characters/biped_demo/biped_demo_meters.usd"
)

AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def main() -> None:
    app = boot(headless=True, width=640, height=480)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

        from roboecg.perception.isaac_skeleton import read_joint_world_positions
        from roboecg.task_manager.supine_pose import apply_supine_pose

        world = World(stage_units_in_meters=1.0)
        stage = get_current_stage()
        add_reference_to_stage(BIPED_USD, "/World/Human")
        world.reset()
        stage.Load("/World/Human")

        def hand():
            joints = read_joint_world_positions(stage)
            leaf = {k.rsplit("/", 1)[-1]: np.asarray(v) for k, v in joints.items()}
            return leaf

        base = hand()
        print(
            "T-pose: hand",
            np.round(base["L_MiddleFinger1"], 3).tolist(),
            "elbow",
            np.round(base["L_LoArm"], 3).tolist(),
        )

        results = []
        for leaf in ("L_UpArm", "L_LoArm"):
            for axis_name, axis in AXES.items():
                for sign in (1.0, -1.0):
                    for angle in (60.0, 90.0, 120.0):
                        prim = stage.GetPrimAtPath("/World/Human/SupinePose")
                        if prim and prim.IsValid():
                            stage.RemovePrim("/World/Human/SupinePose")
                        pose = (((leaf,), axis, sign * angle),)
                        apply_supine_pose(stage, "/World/Human", pose=pose)
                        joints = hand()
                        hand_pos = joints["L_MiddleFinger1"]
                        elbow = joints["L_LoArm"]
                        results.append(
                            {
                                "joint": leaf,
                                "axis": axis_name,
                                "angle": sign * angle,
                                "hand": hand_pos,
                                "elbow": elbow,
                            }
                        )
        for item in sorted(results, key=lambda r: r["hand"][0] + abs(r["hand"][1])):
            print(
                f"{item['joint']:9s} {item['axis']} {item['angle']:+6.0f}  "
                f"hand={np.round(item['hand'], 3).tolist()} "
                f"elbow={np.round(item['elbow'], 3).tolist()}"
            )
    finally:
        app.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
