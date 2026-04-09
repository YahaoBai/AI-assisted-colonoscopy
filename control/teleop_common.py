from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from control.sim2real_bridge import BridgeCommand


def log_event(event: str, **fields: Any) -> None:
    """
    输出结构化关键日志，避免高频刷屏。

    Inputs:
        event: 事件名（建议全大写，便于检索）
        fields: 结构化字段
    """
    parts = [f"[{event}]"]
    for key in sorted(fields.keys()):
        parts.append(f"{key}={fields[key]}")
    print(" ".join(parts))


def apply_deadzone(value: float, deadzone: float) -> float:
    """
    对摇杆轴值做 deadzone + 线性重标定。

    Input:
        value: 原始轴值，期望范围 [-1, 1]
        deadzone: 死区阈值，范围 [0, 1)
    Output:
        处理后的轴值，范围仍为 [-1, 1]
    """
    v = float(np.clip(float(value), -1.0, 1.0))
    dz = float(deadzone)
    if dz <= 0.0:
        return v
    if dz >= 1.0:
        return 0.0
    if abs(v) <= dz:
        return 0.0
    scaled = (abs(v) - dz) / (1.0 - dz)
    return float(np.sign(v) * scaled)


def axis_to_rate(
    raw_axis: float,
    deadzone: float,
    max_rate_rad_s: float,
    invert: bool = False,
) -> float:
    """
    将轴值映射为角速度（rad/s）。
    """
    cleaned = apply_deadzone(raw_axis, deadzone)
    if invert:
        cleaned = -cleaned
    return float(cleaned * max(0.0, float(max_rate_rad_s)))


def should_trigger_disconnect_estop(
    now_sec: float,
    last_input_ok_sec: float,
    timeout_sec: float,
    controller_connected: bool,
) -> bool:
    """
    判断是否应触发断连/超时急停。
    """
    if not controller_connected:
        return True
    return (float(now_sec) - float(last_input_ok_sec)) > float(timeout_sec)


def should_send_follow_command(
    command: BridgeCommand,
    should_send_flag: bool,
    estop_latched: bool,
) -> bool:
    """
    锁存后禁止普通跟随帧，这是安全门（safety gate）。
    """
    if estop_latched:
        return False
    if command != BridgeCommand.CMD:
        return False
    return bool(should_send_flag)


@dataclass
class HoldLatch:
    """
    按钮长按触发器（一次按压仅触发一次）。
    """

    pressed_since_sec: Optional[float] = None
    fired: bool = False

    def update(self, active: bool, now_sec: float, hold_sec: float) -> bool:
        now_v = float(now_sec)
        hold_v = max(0.0, float(hold_sec))

        if not active:
            self.pressed_since_sec = None
            self.fired = False
            return False

        if self.pressed_since_sec is None:
            self.pressed_since_sec = now_v
            self.fired = False
            return False

        if self.fired:
            return False

        if now_v - self.pressed_since_sec >= hold_v:
            self.fired = True
            return True

        return False
