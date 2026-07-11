#!/usr/bin/env python3
"""Display a symmetric G1 hands-beside-head pose in MuJoCo."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "unitree_robots" / "g1" / "scene_29dof.xml"

# Joint angles in radians.  The right arm is the exact sagittal-plane mirror
# of the left arm: roll/yaw signs change, while pitch/elbow signs stay equal.
POSE = {
    # Match the reference above the wrist: upper arm extends sideways and the
    # forearm rises toward the ear.  The wrist then turns the hand upward and
    # inward, producing a cupped "listening carefully" gesture.
    "left_shoulder_pitch_joint": -0.814,
    "left_shoulder_roll_joint": 1.69,
    "left_shoulder_yaw_joint": 0.131,
    "left_elbow_joint": -0.796,
    "left_wrist_roll_joint": 0.177,
    "left_wrist_pitch_joint": -0.0646,
    "left_wrist_yaw_joint": 1.28,
    # Exact sagittal-plane mirror of the left arm.
    "right_shoulder_pitch_joint": -0.353,
    "right_shoulder_roll_joint": -0.37,
    "right_shoulder_yaw_joint": -0.445,
    "right_elbow_joint": 1.21,
    "right_wrist_roll_joint": -0.592,
    "right_wrist_pitch_joint": -0.194,
    "right_wrist_yaw_joint": 0.0807,
}


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Write the pose by joint name and update forward kinematics."""
    for joint_name, angle in POSE.items():
        joint = model.joint(joint_name)
        data.qpos[joint.qposadr[0]] = angle
    mujoco.mj_forward(model, data)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    apply_pose(model, data)

    print("已加载 G1 对称动作。关闭 MuJoCo 窗口即可退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Front view, framed around the full robot.
        viewer.cam.lookat[:] = (0.0, 0.0, 0.75)
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -8.0
        while viewer.is_running():
            # No physics stepping: keep the requested pose exactly fixed.
            apply_pose(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
