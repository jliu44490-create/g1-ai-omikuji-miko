#!/usr/bin/env python3
"""
MuJoCo preview version of g1_arm7_sdk_dds.py.

This script does NOT connect to a real G1 and does NOT use DDS.
It only loads a G1 MuJoCo model and visualizes the same 14-arm-joint target
trajectory that you used in the safer DDS version.

Example:
    python3 g1_arm7_mujoco_viewer.py --model /path/to/unitree_robots/g1/scene.xml

If the motion looks too small:
    python3 g1_arm7_mujoco_viewer.py --model /path/to/scene.xml --scale 2.0
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer


# Same safe target from your DDS test:
# 7 left-arm joints + 7 right-arm joints.
TARGET_POS = np.array([
    0.0, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0,
    0.0, -0.3, 0.0, 0.5, 0.0, 0.0, 0.0,
], dtype=float)


# Logical arm joint order matches TARGET_POS.
# Each entry has possible MuJoCo joint names because different Unitree model
# versions sometimes use slightly different names.
ARM_JOINT_NAME_CANDIDATES: List[Tuple[str, List[str]]] = [
    ("LeftShoulderPitch", ["left_shoulder_pitch_joint", "left_shoulder_pitch"]),
    ("LeftShoulderRoll",  ["left_shoulder_roll_joint",  "left_shoulder_roll"]),
    ("LeftShoulderYaw",   ["left_shoulder_yaw_joint",   "left_shoulder_yaw"]),
    ("LeftElbow",         ["left_elbow_joint",          "left_elbow"]),
    ("LeftWristRoll",     ["left_wrist_roll_joint",     "left_wrist_roll"]),
    ("LeftWristPitch",    ["left_wrist_pitch_joint",    "left_wrist_pitch"]),
    ("LeftWristYaw",      ["left_wrist_yaw_joint",      "left_wrist_yaw"]),

    ("RightShoulderPitch", ["right_shoulder_pitch_joint", "right_shoulder_pitch"]),
    ("RightShoulderRoll",  ["right_shoulder_roll_joint",  "right_shoulder_roll"]),
    ("RightShoulderYaw",   ["right_shoulder_yaw_joint",   "right_shoulder_yaw"]),
    ("RightElbow",         ["right_elbow_joint",          "right_elbow"]),
    ("RightWristRoll",     ["right_wrist_roll_joint",     "right_wrist_roll"]),
    ("RightWristPitch",    ["right_wrist_pitch_joint",    "right_wrist_pitch"]),
    ("RightWristYaw",      ["right_wrist_yaw_joint",      "right_wrist_yaw"]),
]


DEFAULT_MODEL_CANDIDATES = [
    "unitree_robots/g1/scene.xml",
    "unitree_robots/g1/scene_29dof.xml",
    "unitree_robots/g1/g1_29dof.xml",
    "unitree_robots/g1/g1.xml",
    "g1/scene.xml",
    "g1/scene_29dof.xml",
    "scene.xml",
    # Common paths on your machine style. They are safe to leave here;
    # the script only uses them if they exist.
    "~/Desktop/unitree_mujoco/unitree_robots/g1/scene.xml",
    "~/Desktop/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
    "~/Desktop/unitree_mujoco (第一版)/unitree_robots/g1/scene.xml",
    "~/Desktop/unitree_mujoco (第一版)/unitree_robots/g1/scene_29dof.xml",
]


def smoothstep(x: float) -> float:
    """0 -> 1 smooth interpolation."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def resolve_model_path(model_arg: Optional[str]) -> Path:
    if model_arg:
        p = Path(model_arg).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Model file does not exist: {p}")
        return p

    for cand in DEFAULT_MODEL_CANDIDATES:
        p = Path(cand).expanduser()
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find a G1 MuJoCo XML automatically.\n"
        "Please pass it explicitly, for example:\n"
        '  python3 g1_arm7_mujoco_viewer.py --model "/home/jimi/Desktop/unitree_mujoco/unitree_robots/g1/scene.xml"\n'
        'or:\n'
        '  python3 g1_arm7_mujoco_viewer.py --model "/home/jimi/Desktop/unitree_mujoco (第一版)/unitree_robots/g1/scene.xml"\n'
    )


def set_keyframe_if_requested(model: mujoco.MjModel, data: mujoco.MjData, keyframe: Optional[str]) -> None:
    """Optionally initialize qpos/qvel from a MuJoCo keyframe."""
    if keyframe is None:
        return

    if keyframe == "auto":
        if model.nkey > 0:
            data.qpos[:] = model.key_qpos[0]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print("[INFO] Loaded keyframe 0 as initial posture.")
        return

    # Try integer keyframe index first.
    try:
        idx = int(keyframe)
        if idx < 0 or idx >= model.nkey:
            raise ValueError
        data.qpos[:] = model.key_qpos[idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        print(f"[INFO] Loaded keyframe index {idx} as initial posture.")
        return
    except ValueError:
        pass

    # Then try keyframe name.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            for i in range(model.nkey)
        ]
        raise ValueError(f"Unknown keyframe '{keyframe}'. Available keyframes: {names}")

    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[INFO] Loaded keyframe '{keyframe}' as initial posture.")


def find_joint_qpos_addresses(model: mujoco.MjModel) -> Dict[str, int]:
    """Map logical arm joint names to MuJoCo qpos addresses."""
    mapping: Dict[str, int] = {}
    missing_required = []

    for logical_name, candidates in ARM_JOINT_NAME_CANDIDATES:
        found_id = -1
        found_name = None

        for name in candidates:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                found_id = jid
                found_name = name
                break

        target_index = [x[0] for x in ARM_JOINT_NAME_CANDIDATES].index(logical_name)
        target_value = TARGET_POS[target_index]

        if found_id < 0:
            # Missing zero-target wrist joints are okay for 23-DoF models.
            if abs(target_value) < 1e-9 and "Wrist" in logical_name:
                print(f"[WARN] Optional zero-target joint not found, skipping: {logical_name}")
                continue
            missing_required.append((logical_name, candidates))
            continue

        qpos_adr = int(model.jnt_qposadr[found_id])
        mapping[logical_name] = qpos_adr
        print(f"[INFO] {logical_name:20s} -> {found_name:30s} qpos[{qpos_adr}]")

    if missing_required:
        all_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ]
        raise RuntimeError(
            "Some required arm joints were not found in this MuJoCo model:\n"
            + "\n".join(f"  {logical}: tried {cands}" for logical, cands in missing_required)
            + "\n\nAvailable joint names in this model:\n"
            + "\n".join(f"  {name}" for name in all_joint_names if name is not None)
        )

    return mapping


def desired_arm_q(t: float, duration: float, target: np.ndarray) -> np.ndarray:
    """
    Original-like preview trajectory:
      Stage 1: hold neutral for duration
      Stage 2: move to target for 2 * duration
      Stage 3: return to neutral for 3 * duration
      Stage 4: hold neutral for duration
    """
    if t < duration:
        return np.zeros_like(target)

    if t < duration * 3.0:
        r = smoothstep((t - duration) / (duration * 2.0))
        return r * target

    if t < duration * 6.0:
        r = smoothstep((t - duration * 3.0) / (duration * 3.0))
        return (1.0 - r) * target

    return np.zeros_like(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to G1 MuJoCo XML, for example unitree_robots/g1/scene.xml",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Base duration in seconds. Same meaning as your DDS script. Default: 3.0",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply target arm angles for visualization. Try 2.0 if motion is too small.",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        default="auto",
        help="Initial keyframe: 'auto', a keyframe name, an index, or 'none'. Default: auto",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play once and then hold final neutral pose. Default is loop.",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    model_path = resolve_model_path(args.model)
    print(f"[INFO] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.keyframe != "none":
        set_keyframe_if_requested(model, data, args.keyframe)

    joint_qpos_addr = find_joint_qpos_addresses(model)
    logical_order = [name for name, _ in ARM_JOINT_NAME_CANDIDATES]
    target = TARGET_POS * args.scale
    total_time = args.duration * 7.0

    # Force initial arm pose to neutral.
    for logical_name, qpos_adr in joint_qpos_addr.items():
        data.qpos[qpos_adr] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print("\n[INFO] MuJoCo preview started.")
    print("[INFO] This script only changes arm qpos for visualization; it does not control DDS or the real robot.")
    print("[INFO] Close the viewer window to quit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = max(viewer.cam.distance, 3.0)
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 140

        start_time = time.time()
        while viewer.is_running():
            elapsed = time.time() - start_time
            if args.once:
                t = min(elapsed, total_time)
            else:
                t = elapsed % total_time

            q_arm = desired_arm_q(t, args.duration, target)

            for i, logical_name in enumerate(logical_order):
                if logical_name not in joint_qpos_addr:
                    continue
                data.qpos[joint_qpos_addr[logical_name]] = q_arm[i]

            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
