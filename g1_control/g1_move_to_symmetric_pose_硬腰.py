#!/usr/bin/env python3
"""Move a real 29-DOF Unitree G1 to the configured arm pose with a hard waist hold.

This version:
- controls arms through rt/arm_sdk, same as the original script;
- reads the current waist pose at startup;
- holds waist yaw/roll/pitch at that startup pose with higher stiffness;
- ramps the waist stiffness in/out together with the Arm SDK weight.

WARNING:
Use this only with the robot suspended or physically protected, with E-stop ready.
Hard waist holding can fight the lower-body balance controller if the stiffness is too high.
"""

import argparse
import signal
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


CONTROL_DT = 0.02
ENABLE_SECONDS = 2.0
MOVE_SECONDS = 12.0
HOLD_SECONDS = 2.0
RETURN_SECONDS = 12.0
RELEASE_SECONDS = 2.0

# Arm stiffness.
ARM_KP = 40.0
ARM_KD = 2.0

# Hard waist lock stiffness.
# Compared with the previous soft-lock version, this is intentionally stronger.
# If the lower body shakes or fights balance, reduce to 25~35 / 1.5~2.5 first.
WAIST_KP_HARD = 45.0
WAIST_KD_HARD = 3.0

ARM_SDK_ENABLE_INDEX = 29

# G1 29-DOF DDS indices:
# waist: 12..14, left arm: 15..21, right arm: 22..28.
# Please verify these indices against your own Unitree G1 SDK joint map before running.
WAIST_JOINTS = [12, 13, 14]  # waist_yaw, waist_roll, waist_pitch
ARM_JOINTS = list(range(15, 29))

TARGET = np.array(
    [
        -0.814,
        1.69,
        0.131,
        -0.796,
        0.177,
        -0.0646,
        1.28,
        0.165,
        -0.178,
        -0.262,
        1.17,
        0.0986,
        0.0807,
        0.0323,
    ],
    dtype=float,
)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class RealG1ArmController:
    def __init__(self, waist_kp: float, waist_kd: float) -> None:
        self.low_state = None
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.stop_requested = False

        self.waist_kp = float(waist_kp)
        self.waist_kd = float(waist_kd)
        self.waist_lock_position = None

        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._state_handler, 10)

    def _state_handler(self, msg: LowState_) -> None:
        self.low_state = msg

    def request_stop(self, _signum=None, _frame=None) -> None:
        self.stop_requested = True

    def wait_for_state(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while self.low_state is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("10秒内未收到 rt/lowstate，请检查网卡和 G1 连接")
            time.sleep(0.05)

    def current_arm_position(self) -> np.ndarray:
        return np.array(
            [self.low_state.motor_state[joint].q for joint in ARM_JOINTS],
            dtype=float,
        )

    def current_waist_position(self) -> np.ndarray:
        return np.array(
            [self.low_state.motor_state[joint].q for joint in WAIST_JOINTS],
            dtype=float,
        )

    def set_waist_lock_position(self, waist_positions: np.ndarray) -> None:
        self.waist_lock_position = np.array(waist_positions, dtype=float)

    def publish(self, arm_positions: np.ndarray, weight: float) -> None:
        """Publish arm targets and hard-hold waist at the startup pose.

        weight controls both:
        - Arm SDK enable index;
        - waist stiffness ramp, so the waist lock does not suddenly snap on/off.
        """
        weight = float(np.clip(weight, 0.0, 1.0))
        self.cmd.motor_cmd[ARM_SDK_ENABLE_INDEX].q = weight

        # Arm control.
        for joint, position in zip(ARM_JOINTS, arm_positions):
            motor = self.cmd.motor_cmd[joint]
            motor.q = float(position)
            motor.dq = 0.0
            motor.kp = ARM_KP
            motor.kd = ARM_KD
            motor.tau = 0.0

        # Hard waist lock at the measured startup waist pose.
        # The q target stays fixed; kp/kd ramp with weight.
        if self.waist_lock_position is not None:
            for joint, position in zip(WAIST_JOINTS, self.waist_lock_position):
                motor = self.cmd.motor_cmd[joint]
                motor.q = float(position)
                motor.dq = 0.0
                motor.kp = self.waist_kp * weight
                motor.kd = self.waist_kd * weight
                motor.tau = 0.0

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)

    def interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        start_weight: float = 1.0,
        end_weight: float = 1.0,
    ) -> None:
        begin = time.monotonic()
        while True:
            elapsed = time.monotonic() - begin
            phase = smoothstep(elapsed / duration)
            position = (1.0 - phase) * start + phase * end
            weight = (1.0 - phase) * start_weight + phase * end_weight
            self.publish(position, weight)
            if elapsed >= duration or self.stop_requested:
                return
            time.sleep(max(0.0, CONTROL_DT - ((time.monotonic() - begin) % CONTROL_DT)))

    def hold(self, positions: np.ndarray, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and not self.stop_requested:
            self.publish(positions, 1.0)
            time.sleep(CONTROL_DT)

    def run(self) -> None:
        self.wait_for_state()
        initial_arm = self.current_arm_position()
        initial_waist = self.current_waist_position()
        self.set_waist_lock_position(initial_waist)

        print("已收到 G1 状态。")
        print("initial_arm   =", np.round(initial_arm, 4))
        print("target_arm    =", np.round(TARGET, 4))
        print("arm_delta     =", np.round(TARGET - initial_arm, 4))
        print("max arm delta =", float(np.max(np.abs(TARGET - initial_arm))))
        print("initial_waist =", np.round(initial_waist, 4))
        print(f"硬锁腰部参数: WAIST_KP={self.waist_kp}, WAIST_KD={self.waist_kd}")

        print("正在平滑接管手臂，并逐渐加硬腰部锁定…")
        self.interpolate(initial_arm, initial_arm, ENABLE_SECONDS, 0.0, 1.0)

        if not self.stop_requested:
            print("正在移动到目标姿态，腰部保持启动时角度…")
            self.interpolate(initial_arm, TARGET, MOVE_SECONDS)

        if not self.stop_requested:
            self.hold(TARGET, HOLD_SECONDS)

        # A stop request still returns to the measured starting pose before release.
        return_start = self.current_arm_position()
        print("正在返回初始手臂姿态，腰部仍保持硬锁…")
        self.stop_requested = False
        self.interpolate(return_start, initial_arm, RETURN_SECONDS)

        print("正在释放 Arm SDK 控制，并逐渐释放腰部硬锁…")
        self.interpolate(initial_arm, initial_arm, RELEASE_SECONDS, 1.0, 0.0)
        self.publish(initial_arm, 0.0)
        print("完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move a real 29-DOF G1 arm pose with hard waist lock")
    parser.add_argument("network_interface", help="G1 有线网卡，例如 enp2s0")
    parser.add_argument(
        "--waist-kp",
        type=float,
        default=WAIST_KP_HARD,
        help="腰部硬锁 kp，默认 45.0；如果抖动建议降到 25~35",
    )
    parser.add_argument(
        "--waist-kd",
        type=float,
        default=WAIST_KD_HARD,
        help="腰部硬锁 kd，默认 3.0；如果抖动建议降到 1.5~2.5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("WARNING: 仅适用于 29DOF G1。请先吊装机器人，清空手臂/腰部周围，并准备急停。")
    print("WARNING: 当前版本会硬锁腰部启动瞬间姿态；如果下肢明显对抗或抖动，请立刻急停并降低 --waist-kp/--waist-kd。")
    answer = input("输入 YES 后回车开始: ").strip()
    if answer != "YES":
        print("已取消。")
        return

    ChannelFactoryInitialize(0, args.network_interface)
    controller = RealG1ArmController(waist_kp=args.waist_kp, waist_kd=args.waist_kd)
    signal.signal(signal.SIGINT, controller.request_stop)
    signal.signal(signal.SIGTERM, controller.request_stop)
    controller.run()


if __name__ == "__main__":
    try:
        main()
    except (TimeoutError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
