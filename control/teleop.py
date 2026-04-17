"""
手柄遥操作入口（Hardware Direct Teleop）

控制链路总览（展示版）:
1) Gamepad Input (pygame controller)
   -> 2) axis/button parsing
   -> 3) yaw/pitch direct mm mapping + feed Y/B continuous trigger
   -> 4) Sim2RealBridge (angle/motor limit + estop latch)
   -> 5) MotorMapper + ActuatorTx + FeedTx (F3 follow / auto-estop / boot reset)

设计目标:
- 与 il.py 自动驾驶入口解耦，专注手动上机联调。
- 安全优先：保留自动急停（断连/监控/发送失败等）与启动自动复位。
- 代码可讲解：模块分层清晰、关键逻辑有注释、结构化关键日志。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from control.teleop_common import (
    apply_deadzone,
    axis_to_rate,
    log_event,
    should_send_follow_command,
    should_trigger_disconnect_estop,
)
from control.teleop_config import TeleopConfig, load_teleop_config
from control.teleop_input import GamepadSample, PygameGamepadInput
from control.teleop_runtime import (
    log_input_snapshot,
    on_forward_state_change,
    perform_reset_sequence,
    run_teleop,
    send_follow_target_mm,
)

DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("sim2real_config.yaml"))


# Backward compatibility: keep legacy imports available from control.teleop.
__all__ = [
    "TeleopConfig",
    "GamepadSample",
    "PygameGamepadInput",
    "log_event",
    "apply_deadzone",
    "axis_to_rate",
    "should_trigger_disconnect_estop",
    "should_send_follow_command",
    "send_follow_target_mm",
    "perform_reset_sequence",
    "on_forward_state_change",
    "log_input_snapshot",
    "load_teleop_config",
    "run_teleop",
    "build_arg_parser",
    "main",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gamepad teleoperation for Sim2Real colonoscope actuator control."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run input/control pipeline without serial output.",
    )
    parser.add_argument(
        "--debug-input",
        action="store_true",
        help="Print low-frequency input snapshots (state + m1~m4 target counts).",
    )
    parser.add_argument(
        "--list-controllers",
        action="store_true",
        help="List detected controllers and exit.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run_teleop(
            config_path=DEFAULT_CONFIG_PATH,
            dry_run=bool(args.dry_run),
            debug_input=bool(args.debug_input),
            list_controllers_only=bool(args.list_controllers),
        )
    except Exception as exc:
        log_event("FATAL", reason=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
