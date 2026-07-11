#!/usr/bin/env python3
"""Move a real 29-DOF Unitree G1 to the configured arm pose via Arm SDK."""

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

KP = 40.0
KD = 2.0
ARM_SDK_ENABLE_INDEX = 29

# G1 29-DOF DDS indices: left arm 15..21, right arm 22..28.
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
        -0.353,
        -0.37,
        -0.445,
        1.21,
        -0.592,
        -0.194,
        0.0807,
    ],
    dtype=float,
)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class RealG1ArmController:
    def __init__(self) -> None:
        self.low_state = None
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.stop_requested = False

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

    def publish(self, positions: np.ndarray, weight: float) -> None:
        self.cmd.motor_cmd[ARM_SDK_ENABLE_INDEX].q = float(np.clip(weight, 0.0, 1.0))
        for joint, position in zip(ARM_JOINTS, positions):
            motor = self.cmd.motor_cmd[joint]
            motor.q = float(position)
            motor.dq = 0.0
            motor.kp = KP
            motor.kd = KD
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
        initial = self.current_arm_position()

        print("已收到 G1 状态，正在平滑接管手臂…")
        self.interpolate(initial, initial, ENABLE_SECONDS, 0.0, 1.0)

        if not self.stop_requested:
            print("正在移动到目标姿态…")
            self.interpolate(initial, TARGET, MOVE_SECONDS)
        if not self.stop_requested:
            self.hold(TARGET, HOLD_SECONDS)

        # A stop request still returns to the measured starting pose before release.
        return_start = self.current_arm_position()
        print("正在返回初始姿态…")
        self.stop_requested = False
        self.interpolate(return_start, initial, RETURN_SECONDS)
        print("正在释放 Arm SDK 控制…")
        self.interpolate(initial, initial, RELEASE_SECONDS, 1.0, 0.0)
        self.publish(initial, 0.0)
        print("完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move a real 29-DOF G1 arm pose")
    parser.add_argument("network_interface", help="G1 有线网卡，例如 enp2s0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("WARNING: 仅适用于 29DOF G1。请先吊装机器人，清空手臂周围，并准备急停。")
    answer = input("输入 YES 后回车开始: ").strip()
    if answer != "YES":
        print("已取消。")
        return

    ChannelFactoryInitialize(0, args.network_interface)
    controller = RealG1ArmController()
    signal.signal(signal.SIGINT, controller.request_stop)
    signal.signal(signal.SIGTERM, controller.request_stop)
    controller.run()


if __name__ == "__main__":
    try:
        main()
    except (TimeoutError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
