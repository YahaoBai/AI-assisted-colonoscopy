"""
手柄遥操作入口（Hardware Direct Teleop, Toggle Recording）

与 control/teleop.py 的主要区别：
- 启动时不自动创建 teleop_capture 记录目录。
- 在终端里按数字键 `6` 时开始记录 `controls.csv`。
- 再按一次 `6` 停止记录；再次按下会新开一个记录 session。

其余控制、安全、进给、限幅、急停与 teleop 保持一致。
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import termios
    import tty
except Exception:  # pragma: no cover - platform dependent import guard
    termios = None
    tty = None

import numpy as np

from control.sim2real_bridge import (
    MotorMapper,
    Sim2RealBridge,
    load_dagger_sim2real_runtime_config,
)
from control.teleop_config import load_teleop_config
from control.teleop_input import PygameGamepadInput
from control.teleop_runtime import (
    TELEOP_CAPTURE_ROOT,
    TeleopRecorder,
    TeleopRuntimeContext,
    _cleanup_runtime,
    _compute_direct_motor_target_mm,
    _compute_loop_dt,
    _create_loop_state,
    _disabled_feed_state,
    _handle_disconnect_estop,
    _handle_monitor_fault,
    _list_controllers_and_exit,
    _maybe_log_debug_snapshot,
    _run_boot_reset,
    _send_direct_target_and_check_limits,
    _set_estop_log_freeze,
    _setup_actuator_io,
    _setup_feed_runtime,
    _sleep_for_rate,
    _sync_monitor_active_state,
    _update_forward_state_and_feed,
    log_event,
)

DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("sim2real_config.yaml"))
RECORD_TOGGLE_KEY = "6"
RECORD_TOGGLE_DEBOUNCE_SEC = 0.25


class TerminalRecordingTogglePoller:
    """
    终端键盘轮询器：在当前控制台中监听单键 `6`，用于开关录制。

    实现选择标准库 raw/cbreak 方案，避免引入 GUI/桌面依赖。
    """

    def __init__(
        self,
        toggle_key: str = RECORD_TOGGLE_KEY,
        debounce_sec: float = RECORD_TOGGLE_DEBOUNCE_SEC,
    ) -> None:
        if len(str(toggle_key)) != 1:
            raise ValueError("toggle_key must be a single character")

        self.toggle_key = str(toggle_key)
        self.debounce_sec = max(0.0, float(debounce_sec))
        self._enabled = False
        self._fd: Optional[int] = None
        self._stream = sys.stdin
        self._saved_termios = None
        self._last_toggle_sec = -1e9
        self.reason = ""

    def open(self) -> bool:
        if termios is None or tty is None:
            self.reason = "TERMINAL_RAW_MODE_UNAVAILABLE"
            return False
        if not hasattr(self._stream, "isatty") or (not self._stream.isatty()):
            self.reason = "STDIN_NOT_TTY"
            return False

        try:
            fd = int(self._stream.fileno())
            self._saved_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception as exc:
            self.reason = f"RAW_MODE_SETUP_FAIL:{exc}"
            self._saved_termios = None
            return False

        self._fd = fd
        self._enabled = True
        self.reason = "OK"
        return True

    def poll_toggle(self) -> bool:
        if not self._enabled:
            return False

        toggled = False
        while True:
            try:
                readable, _, _ = select.select([self._stream], [], [], 0.0)
            except Exception:
                return toggled
            if not readable:
                break

            try:
                char = self._stream.read(1)
            except Exception:
                break
            if not char:
                break

            if char != self.toggle_key:
                continue

            now_sec = time.monotonic()
            if (now_sec - self._last_toggle_sec) < self.debounce_sec:
                continue
            self._last_toggle_sec = now_sec
            toggled = True

        return toggled

    def close(self) -> None:
        if self._fd is None or self._saved_termios is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_termios)
        except Exception:
            pass
        finally:
            self._fd = None
            self._saved_termios = None
            self._enabled = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gamepad teleoperation for Sim2Real colonoscope actuator control with toggle recording."
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


def _resolve_unique_capture_dir(now_sec: Optional[float] = None) -> Path:
    timestamp = time.strftime(
        "%Y%m%d_%H%M%S",
        time.localtime(time.time() if now_sec is None else float(now_sec)),
    )
    base_dir = Path(TELEOP_CAPTURE_ROOT) / timestamp
    if not base_dir.exists():
        return base_dir

    suffix = 1
    while True:
        candidate = Path(TELEOP_CAPTURE_ROOT) / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _create_toggle_recorder(now_sec: Optional[float] = None) -> TeleopRecorder:
    return TeleopRecorder.create(output_dir=_resolve_unique_capture_dir(now_sec=now_sec))


def _stop_recording_session(recorder: Optional[TeleopRecorder]) -> None:
    if recorder is None:
        return
    recorder.close()


def _toggle_recording_session(
    recorder: Optional[TeleopRecorder],
    *,
    now_sec: Optional[float] = None,
) -> Optional[TeleopRecorder]:
    if recorder is None:
        new_recorder = _create_toggle_recorder(now_sec=now_sec)
        log_event("RECORDING_START", key=RECORD_TOGGLE_KEY, path=str(new_recorder.output_dir))
        return new_recorder

    output_dir = recorder.output_dir
    _stop_recording_session(recorder)
    log_event("RECORDING_STOP", key=RECORD_TOGGLE_KEY, path=str(output_dir))
    return None


def run_teleop2(
    config_path: str,
    dry_run: bool,
    debug_input: bool = False,
    list_controllers_only: bool = False,
    toggle_poller_factory: Callable[[], TerminalRecordingTogglePoller] = TerminalRecordingTogglePoller,
) -> int:
    _set_estop_log_freeze(False)
    runtime_cfg = load_dagger_sim2real_runtime_config(config_path)
    teleop_cfg = load_teleop_config(config_path)
    feed_only_mode = bool(teleop_cfg.allow_feed_without_actuator)
    for warning in teleop_cfg.deprecation_warnings:
        log_event("DEPRECATED_CONFIG", message=warning)

    sim2real_cfg = runtime_cfg.bridge
    actuator_cfg = runtime_cfg.actuator
    monitor_cfg = runtime_cfg.monitor

    gamepad = PygameGamepadInput(teleop_cfg)
    if list_controllers_only:
        return _list_controllers_and_exit(gamepad)
    gamepad.open()
    log_event(
        "CONTROLLER_READY",
        index=gamepad.controller_index,
        name=gamepad.controller_name,
    )

    toggle_poller = toggle_poller_factory()
    toggle_ready = bool(toggle_poller.open())
    if toggle_ready:
        log_event("RECORD_TOGGLE_READY", key=RECORD_TOGGLE_KEY)
    else:
        log_event("RECORD_TOGGLE_DISABLED", key=RECORD_TOGGLE_KEY, reason=toggle_poller.reason or "UNKNOWN")

    sim2real_bridge = Sim2RealBridge(sim2real_cfg)
    motor_mapper = MotorMapper(actuator_cfg, sim2real_cfg.motor_order)
    actuator_ids = actuator_cfg.ids_for_motor_order(sim2real_cfg.motor_order)
    actuator_tx = None
    actuator_monitor = None
    serial_link = None
    feed_state = _disabled_feed_state()
    recorder: Optional[TeleopRecorder] = None

    try:
        if not feed_only_mode:
            actuator_tx, actuator_monitor, serial_link = _setup_actuator_io(
                runtime_cfg=runtime_cfg,
                sim2real_cfg=sim2real_cfg,
                monitor_cfg=monitor_cfg,
                actuator_ids=actuator_ids,
                dry_run=dry_run,
            )
        else:
            log_event("ACTUATOR_BYPASS", mode="FEED_ONLY")

        feed_state = _setup_feed_runtime(
            config_path=config_path,
            dry_run=dry_run,
            actuator_serial_port=sim2real_cfg.serial_port,
            check_port_conflict=(not feed_only_mode),
        )
        ctx = TeleopRuntimeContext(
            sim2real_bridge=sim2real_bridge,
            motor_mapper=motor_mapper,
            actuator_ids=actuator_ids,
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            monitor_settle_sec=monitor_cfg.response_timeout_sec,
            feed_state=feed_state,
            dry_run=dry_run,
            allow_feed_without_actuator=feed_only_mode,
        )
        loop_state = _create_loop_state(control_hz=sim2real_cfg.control_hz)
        if feed_only_mode:
            log_event("BOOT_READY", mode="FEED_ONLY")
        else:
            _run_boot_reset(ctx)

        log_event("RECORDING_STANDBY", key=RECORD_TOGGLE_KEY, note="press 6 to start/stop controls.csv capture")

        while True:
            loop_start = time.monotonic()
            if toggle_poller.poll_toggle():
                recorder = _toggle_recording_session(recorder, now_sec=time.time())

            wall_time_sec = time.time()
            _ = _compute_loop_dt(loop_state, loop_start)

            ctx.feed_state.last_delta_pulses = 0
            sample = gamepad.poll()
            _update_forward_state_and_feed(sample, loop_state, ctx, loop_start)
            _handle_disconnect_estop(sample, loop_start, teleop_cfg, loop_state, ctx)
            _handle_monitor_fault(ctx)
            _sync_monitor_active_state(ctx)

            if feed_only_mode:
                yaw_cmd_mm = 0.0
                pitch_cmd_mm = 0.0
                motor_target_mm = np.zeros(4, dtype=np.float64)
                motor_limit_active = (False, False, False, False)
            else:
                yaw_cmd_mm, pitch_cmd_mm, motor_target_mm, motor_limit_active = _compute_direct_motor_target_mm(
                    sample=sample,
                    teleop_cfg=teleop_cfg,
                    motor_limit_mm=ctx.sim2real_bridge.motor_limit_mm,
                )

            _maybe_log_debug_snapshot(
                enabled=debug_input,
                loop_start=loop_start,
                sample=sample,
                yaw_rate=yaw_cmd_mm,
                pitch_rate=pitch_cmd_mm,
                delta_yaw=0.0,
                delta_pitch=0.0,
                estop_latched=ctx.sim2real_bridge.estop_latched,
                motor_target_mm=motor_target_mm,
                loop_state=loop_state,
                ctx=ctx,
            )

            if not feed_only_mode:
                _send_direct_target_and_check_limits(
                    motor_target_mm=motor_target_mm,
                    motor_limit_active=motor_limit_active,
                    loop_state=loop_state,
                    ctx=ctx,
                )

            if recorder is not None:
                recorder.record(
                    wall_time_sec=wall_time_sec,
                    motor_target_mm=np.asarray(ctx.sim2real_bridge.motor_target_mm, dtype=np.float64),
                    feed_delta_pulses=int(ctx.feed_state.last_delta_pulses),
                    estop_latched=ctx.sim2real_bridge.estop_latched,
                )

            _sleep_for_rate(loop_start, loop_state.period_sec)
    except KeyboardInterrupt:
        log_event("STOP", reason="KEYBOARD_INTERRUPT")
    finally:
        if recorder is not None:
            _stop_recording_session(recorder)
        toggle_poller.close()
        _cleanup_runtime(
            gamepad=gamepad,
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            actuator_ids=actuator_ids,
            serial_link=serial_link,
            feed_state=feed_state,
        )

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run_teleop2(
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
