#!/usr/bin/env python3
"""Move G1 smoothly from the initial MuJoCo state to a hands-beside-head pose.

Run from your unitree_mujoco/scripts directory, or put this file there:
    python3 g1_move_to_symmetric_pose.py

This is a kinematic pose animation: it interpolates qpos directly and calls
mj_forward(), so the robot will not fall or fight gravity during the preview.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "unitree_robots" / "g1" / "scene_29dof.xml"

DURATION = 3.0      # seconds from initial pose to target pose
HOLD_SECONDS = 2.0  # after reaching target, keep holding it; set None to hold forever
FPS = 60.0

# Your final target pose, in radians.
TARGET_POSE = {
    "left_shoulder_pitch_joint": 0.338,
    "left_shoulder_roll_joint": 0.504,
    "left_shoulder_yaw_joint": -0.419,
    "left_elbow_joint": -0.529,
    "left_wrist_roll_joint": -0.394,
    "left_wrist_pitch_joint": -0.339,
    "left_wrist_yaw_joint": -0.0969,

    "right_shoulder_pitch_joint": 0.309,
    "right_shoulder_roll_joint": -0.466,
    "right_shoulder_yaw_joint": 0.393,
    "right_elbow_joint": -0.529,
    "right_wrist_roll_joint": 0.177,
    "right_wrist_pitch_joint": -0.258,
    "right_wrist_yaw_joint": -0.113,
}


def smoothstep(x: float) -> float:
    """0->1 smooth interpolation curve with zero speed at both ends."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def joint_qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    """Return qpos index for a 1-DoF joint."""
    joint = model.joint(joint_name)
    return int(joint.qposadr[0])


def set_arm_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    for joint_name, angle in pose.items():
        data.qpos[joint_qpos_index(model, joint_name)] = angle


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    # Use the XML/default state as the initial state.
    mujoco.mj_forward(model, data)
    start_qpos = data.qpos.copy()

    # Precompute target qpos by modifying only the target arm joints.
    target_qpos = start_qpos.copy()
    for joint_name, angle in TARGET_POSE.items():
        target_qpos[joint_qpos_index(model, joint_name)] = angle

    print("G1 will move from the initial pose to the target hands-beside-head pose.")
    print("Close the MuJoCo window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.75)
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -8.0

        start_time = time.time()

        while viewer.is_running():
            elapsed = time.time() - start_time
            alpha = smoothstep(elapsed / DURATION)

            # Interpolate all qpos values, but only arm joints differ from start.
            data.qpos[:] = (1.0 - alpha) * start_qpos + alpha * target_qpos
            mujoco.mj_forward(model, data)
            viewer.sync()

            if HOLD_SECONDS is not None and elapsed > DURATION + HOLD_SECONDS:
                # Continue holding the final pose instead of closing automatically.
                data.qpos[:] = target_qpos
                mujoco.mj_forward(model, data)
                viewer.sync()

            time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
