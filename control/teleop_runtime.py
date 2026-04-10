from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover - import guard only
    serial = None

from control.sim2real_bridge import (
    ActuatorMonitor,
    ActuatorTx,
    BridgeCommand,
    FaultClearVerification,
    MotorMapper,
    Sim2RealBridge,
    load_dagger_sim2real_runtime_config,
)
from control.feed import (
    FeedConfig,
    SerialLink,
    build_en_control_frame,
    build_pos_control_frame,
    compute_next_forward_target,
    resolve_feed_config,
    resolve_forward_dir,
)
from control.teleop_common import (
    HoldLatch,
    axis_to_rate,
    log_event,
    should_send_follow_command,
    should_trigger_disconnect_estop,
)
from control.teleop_config import load_teleop_config
from control.teleop_input import GamepadSample, PygameGamepadInput

DEBUG_INPUT_HZ_DEFAULT = 5.0
FEED_RESET_DISABLE_SETTLE_SEC = 0.15
FEED_RESET_ENABLE_SETTLE_SEC = 0.15
FEED_POST_RESET_PREMOVE_SETTLE_SEC = 0.10


@dataclass
class FeedRuntimeState:
    """
    Teleop 内的滑台进给运行时状态。
    """

    enabled: bool
    cfg: Optional[FeedConfig]
    serial_link: Optional[SerialLink]
    current_pulses: int
    locked: bool
    reenable_pending: bool = False
    next_forward_send_ts: float = 0.0
    forward_limit_latched: bool = False


@dataclass
class TeleopRuntimeContext:
    """
    运行时硬件与桥接上下文。

    该结构把主循环依赖集中，避免在多处传递大量参数。
    """

    sim2real_bridge: Sim2RealBridge
    motor_mapper: MotorMapper
    actuator_ids: Sequence[int]
    actuator_tx: Optional[ActuatorTx]
    actuator_monitor: Optional[ActuatorMonitor]
    monitor_settle_sec: float
    feed_state: FeedRuntimeState
    dry_run: bool
    allow_feed_without_actuator: bool


@dataclass
class TeleopLoopState:
    """
    主循环可变状态。

    包括长按判定状态、前进占位边沿状态、循环节拍与调试打印节拍。
    """

    estop_hold: HoldLatch
    reset_hold: HoldLatch
    forward_pressed_prev: bool
    period_sec: float
    loop_ts_prev: float
    last_input_ok_sec: float
    debug_interval_sec: float
    next_debug_ts: float
    has_debug_snapshot: bool
    last_debug_sample: Optional[GamepadSample]
    last_debug_yaw_rate: float
    last_debug_pitch_rate: float
    last_debug_estop_latched: bool
    last_debug_motor_counts: Optional[Tuple[int, int, int, int]]


def send_follow_target_mm(
    actuator_tx: ActuatorTx,
    motor_mapper: MotorMapper,
    actuator_ids: Sequence[int],
    motor_target_mm: np.ndarray,
    frame_name: str = "F3_FOLLOW",
    critical: bool = False,
) -> bool:
    """
    将语义电机 mm 目标映射为 count 并发送广播跟随帧。
    """
    try:
        target_counts = motor_mapper.mm_targets_to_counts(motor_target_mm)
    except Exception as exc:
        log_event("MM_TO_COUNT_FAIL", reason=str(exc))
        return False

    return actuator_tx.send_follow_broadcast(
        actuator_ids,
        target_counts.tolist(),
        critical=critical,
        frame_name=frame_name,
    )


def perform_reset_sequence(
    actuator_tx: Optional[ActuatorTx],
    actuator_monitor: Optional[ActuatorMonitor],
    motor_mapper: MotorMapper,
    actuator_ids: Sequence[int],
    sim2real_bridge: Sim2RealBridge,
    monitor_settle_sec: float,
    dry_run: bool,
    feed_state: Optional[FeedRuntimeState] = None,
    allow_feed_without_actuator: bool = False,
) -> Tuple[bool, str]:
    """
    执行复位序列:
    fault_clear -> work_start -> follow_zero -> verify -> bridge.reset

    返回:
        (ok, reason)
    """
    if dry_run:
        sim2real_bridge.reset()
        if actuator_monitor is not None:
            actuator_monitor.clear_fault()
        if feed_state is not None and feed_state.enabled:
            feed_state.locked = False
        return True, "DRY_RUN"

    if actuator_tx is None:
        if not allow_feed_without_actuator:
            return False, "NO_ACTUATOR_TX"
        if feed_state is not None and feed_state.enabled:
            feed_ok, feed_reason = _feed_reset_disable_enable(feed_state)
            if not feed_ok:
                return False, feed_reason
        sim2real_bridge.reset()
        if actuator_monitor is not None:
            actuator_monitor.clear_fault()
        return True, "OK_FEED_ONLY"

    fault_clear_ok = actuator_tx.send_fault_clear_all(actuator_ids, critical=True)
    if not fault_clear_ok:
        return False, "FAULT_CLEAR_FAIL"

    work_start_ok = actuator_tx.send_work_start_all(actuator_ids, critical=True)
    if not work_start_ok:
        return False, "WORK_START_FAIL"

    zero_ok = send_follow_target_mm(
        actuator_tx=actuator_tx,
        motor_mapper=motor_mapper,
        actuator_ids=actuator_ids,
        motor_target_mm=np.zeros(4, dtype=np.float64),
        frame_name="F3_FOLLOW_ZERO",
        critical=True,
    )
    if not zero_ok:
        return False, "FOLLOW_ZERO_FAIL"

    if actuator_monitor is not None and actuator_monitor.monitor_cfg.enabled:
        verify_result: FaultClearVerification = actuator_monitor.verify_fault_clear(
            actuator_ids=actuator_ids,
            attempts=2,
            settle_time_sec=max(0.01, float(monitor_settle_sec)),
        )
        if not verify_result.ok:
            details = []
            if verify_result.failures_by_id:
                fail_text = ",".join(
                    f"id={aid}:{reason}"
                    for aid, reason in sorted(verify_result.failures_by_id.items())
                )
                details.append(f"QUERY_FAIL[{fail_text}]")
            if verify_result.uncleared_error_bits_by_id:
                bit_text = ",".join(
                    f"id={aid}:0x{bits:02X}"
                    for aid, bits in sorted(verify_result.uncleared_error_bits_by_id.items())
                )
                details.append(f"ERROR_BITS[{bit_text}]")
            return False, ";".join(details) if details else "VERIFY_FAIL"

    if feed_state is not None and feed_state.enabled:
        feed_ok, feed_reason = _feed_reset_disable_enable(feed_state)
        if not feed_ok:
            return False, feed_reason

    sim2real_bridge.reset()
    if actuator_monitor is not None:
        actuator_monitor.clear_fault()
    return True, "OK"


def on_forward_state_change(is_pressed: bool, ts_sec: float) -> None:
    """
    前进状态回调（仅记录边沿状态，真实推进由上层触发）。
    """
    log_event(
        "FORWARD_STATE_CHANGE",
        pressed=int(bool(is_pressed)),
        ts=f"{float(ts_sec):.3f}",
    )


def log_input_snapshot(
    sample: GamepadSample,
    yaw_rate: float,
    pitch_rate: float,
    delta_yaw: float,
    delta_pitch: float,
    estop_latched: bool,
    motor_counts: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """
    打印低频输入快照（精简字段版）。

    说明：
    - 展示和联调优先，仅保留操作状态与安全状态字段，避免刷屏。
    - 速率和增量仍在上层参与“变化检测”，但不再直接打印。
    """
    _ = yaw_rate
    _ = pitch_rate
    _ = delta_yaw
    _ = delta_pitch
    reset_active = sample.reset_primary_pressed and sample.reset_secondary_pressed
    payload: dict[str, Any] = {
        "connected": int(sample.connected),
        "forward": int(sample.forward_pressed),
        "estop": int(sample.estop_pressed),
        "reset": int(reset_active),
        "estop_latched": int(estop_latched),
    }
    if motor_counts is not None:
        payload["m1"] = int(motor_counts[0])
        payload["m2"] = int(motor_counts[1])
        payload["m3"] = int(motor_counts[2])
        payload["m4"] = int(motor_counts[3])
    log_event("INPUT", **payload)


def _disabled_feed_state() -> FeedRuntimeState:
    return FeedRuntimeState(
        enabled=False,
        cfg=None,
        serial_link=None,
        current_pulses=0,
        locked=False,
        next_forward_send_ts=0.0,
        forward_limit_latched=False,
    )


def _validate_feed_port_conflict(
    dry_run: bool,
    actuator_serial_port: str,
    feed_cfg: FeedConfig,
    check_conflict: bool = True,
) -> None:
    """
    实机模式下，电缸串口与滑台串口必须分离。
    """
    if not check_conflict:
        return
    if dry_run:
        return
    if not feed_cfg.teleop_enabled:
        return
    if str(actuator_serial_port).strip() == str(feed_cfg.port).strip():
        raise ValueError(
            f"teleop feed port conflict: actuator_port={actuator_serial_port} "
            f"must be different from feed.port={feed_cfg.port}"
        )


def _feed_send_frame(feed_state: FeedRuntimeState, frame: bytes, event: str) -> bool:
    if not feed_state.enabled or feed_state.cfg is None:
        return False

    cfg = feed_state.cfg
    if cfg.dry_run:
        log_event(event, frame=frame.hex(" ").upper())
        return True

    if feed_state.serial_link is None:
        log_event("FEED_TX_FAIL", reason="SERIAL_NOT_READY")
        return False

    try:
        wrote = feed_state.serial_link.write(frame)
        feed_state.serial_link.flush()
    except Exception as exc:
        log_event("FEED_TX_FAIL", reason=str(exc))
        return False

    if wrote != len(frame):
        log_event("FEED_TX_FAIL", reason=f"SHORT_WRITE wrote={wrote} expected={len(frame)}")
        return False

    log_event(event, frame=frame.hex(" ").upper())
    return True


def _feed_send_enable(feed_state: FeedRuntimeState, state: bool) -> bool:
    if not feed_state.enabled or feed_state.cfg is None:
        return True
    frame = build_en_control_frame(
        addr=feed_state.cfg.addr,
        state=state,
        sync_start=False,
    )
    return _feed_send_frame(
        feed_state=feed_state,
        frame=frame,
        event="FEED_ENABLE" if state else "FEED_DISABLE",
    )


def _feed_send_forward_once(feed_state: FeedRuntimeState, step_pulses: int) -> bool:
    if not feed_state.enabled or feed_state.cfg is None:
        return True
    cfg = feed_state.cfg
    frame = build_pos_control_frame(
        addr=cfg.addr,
        dir_flag=resolve_forward_dir(cfg.invert_dir),
        vel=cfg.default_vel,
        acc=cfg.default_acc,
        clk=int(step_pulses),
        relative_mode=True,
        sync_start=False,
    )
    return _feed_send_frame(feed_state=feed_state, frame=frame, event="FEED_FORWARD_TX")


def _setup_feed_runtime(
    config_path: str,
    dry_run: bool,
    actuator_serial_port: str,
    check_port_conflict: bool = True,
) -> FeedRuntimeState:
    """
    初始化 teleop 的 feed 链路。
    """
    feed_cfg = resolve_feed_config(dry_run=dry_run, config_path=config_path)
    if not feed_cfg.teleop_enabled:
        return _disabled_feed_state()

    _validate_feed_port_conflict(
        dry_run=dry_run,
        actuator_serial_port=actuator_serial_port,
        feed_cfg=feed_cfg,
        check_conflict=check_port_conflict,
    )

    feed_serial_link: Optional[SerialLink] = None
    if not dry_run:
        if serial is None:
            raise RuntimeError("pyserial is required. Install with: pip install pyserial")
        feed_serial_link = serial.Serial(
            feed_cfg.port,
            baudrate=feed_cfg.baudrate,
            timeout=feed_cfg.timeout,
            write_timeout=0.2,
        )

    current_pulses = max(feed_cfg.min_pulses, min(0, feed_cfg.max_pulses))
    feed_state = FeedRuntimeState(
        enabled=True,
        cfg=feed_cfg,
        serial_link=feed_serial_link,
        current_pulses=current_pulses,
        locked=False,
        next_forward_send_ts=0.0,
        forward_limit_latched=False,
    )
    log_event(
        "FEED_READY",
        mode="DRY_RUN" if dry_run else "HARDWARE",
        port=feed_cfg.port,
        addr=feed_cfg.addr,
        step=feed_cfg.step_pulses,
        repeat_hz=feed_cfg.repeat_hz,
        current=current_pulses,
        min=feed_cfg.min_pulses,
        max=feed_cfg.max_pulses,
    )

    if feed_cfg.enable_on_start and (not _feed_send_enable(feed_state, state=True)):
        raise RuntimeError("feed enable_on_start failed")

    return feed_state


def _handle_feed_forward_rising_edge(ctx: TeleopRuntimeContext) -> str:
    """
    执行一次“前进一步进”尝试。

    返回:
        "sent"   : 本次成功发送
        "limit"  : 命中限位（本次按住期间锁存）
        "blocked": 锁存/禁用等状态不允许发送
        "failed" : 发送失败并触发急停
    """
    feed_state = ctx.feed_state
    if not feed_state.enabled or feed_state.cfg is None:
        return "blocked"
    if feed_state.locked or ctx.sim2real_bridge.estop_latched:
        log_event("FEED_FORWARD_BLOCKED", reason="LOCKED")
        return "blocked"
    if feed_state.forward_limit_latched:
        return "blocked"

    if feed_state.reenable_pending:
        if not _feed_send_enable(feed_state, state=True):
            _latch_estop(ctx, reason="FEED_REENABLE_FAIL")
            return "failed"
        time.sleep(FEED_POST_RESET_PREMOVE_SETTLE_SEC)
        feed_state.reenable_pending = False
        log_event("FEED_REENABLE_OK")

    cfg = feed_state.cfg
    ok, next_pulses = compute_next_forward_target(
        current_pulses=feed_state.current_pulses,
        step_pulses=cfg.step_pulses,
        min_pulses=cfg.min_pulses,
        max_pulses=cfg.max_pulses,
    )
    if not ok:
        log_event(
            "LIMIT_HIT",
            subsystem="FEED",
            current=feed_state.current_pulses,
            step=cfg.step_pulses,
            min=cfg.min_pulses,
            max=cfg.max_pulses,
        )
        feed_state.forward_limit_latched = True
        return "limit"

    if not _feed_send_forward_once(feed_state=feed_state, step_pulses=cfg.step_pulses):
        _latch_estop(ctx, reason="FEED_TX_FAIL")
        return "failed"

    feed_state.current_pulses = int(next_pulses)
    log_event("FEED_FORWARD_OK", current=feed_state.current_pulses, step=cfg.step_pulses)
    return "sent"


def _feed_reset_disable_enable(feed_state: FeedRuntimeState) -> Tuple[bool, str]:
    """
    feed 复位定义：失能 -> 使能（不移动）。
    """
    if not feed_state.enabled:
        return True, "SKIP"
    if not _feed_send_enable(feed_state, state=False):
        return False, "FEED_DISABLE_FAIL"
    time.sleep(FEED_RESET_DISABLE_SETTLE_SEC)
    if not _feed_send_enable(feed_state, state=True):
        return False, "FEED_ENABLE_FAIL"
    time.sleep(FEED_RESET_ENABLE_SETTLE_SEC)
    feed_state.locked = False
    feed_state.reenable_pending = True
    feed_state.forward_limit_latched = False
    feed_state.next_forward_send_ts = 0.0
    return True, "OK"


def _setup_actuator_io(
    runtime_cfg: Any,
    sim2real_cfg: Any,
    monitor_cfg: Any,
    actuator_ids: Sequence[int],
    dry_run: bool,
) -> Tuple[Optional[ActuatorTx], Optional[ActuatorMonitor], Any]:
    """
    初始化串口、发送器与监控线程。

    返回:
        (actuator_tx, actuator_monitor, serial_link)
    """
    if dry_run:
        return None, None, None

    if serial is None:
        raise RuntimeError("pyserial is required. Install with: pip install pyserial")

    serial_link = serial.Serial(
        sim2real_cfg.serial_port,
        baudrate=sim2real_cfg.serial_baudrate,
        timeout=sim2real_cfg.serial_timeout,
        write_timeout=sim2real_cfg.serial_write_timeout,
    )
    actuator_tx = ActuatorTx(
        serial_link,
        critical_retry_count=sim2real_cfg.serial_critical_retry_count,
        critical_retry_interval_sec=sim2real_cfg.serial_critical_retry_interval_sec,
        print_tx_frame=runtime_cfg.output.print_tx_frame,
    )
    actuator_monitor = ActuatorMonitor(
        serial_link=serial_link,
        actuator_ids=actuator_ids,
        monitor_cfg=monitor_cfg,
        serial_lock=actuator_tx.serial_lock,
    )
    actuator_monitor.start()
    actuator_monitor.set_active(True)
    return actuator_tx, actuator_monitor, serial_link


def _latch_estop(
    ctx: TeleopRuntimeContext,
    reason: str,
    actuator_id: Optional[int] = None,
    error_bits: Optional[int] = None,
) -> bool:
    """
    执行 ESTOP 锁存并可选下发急停帧。

    返回:
        True: 本次新触发锁存
        False: 已处于锁存状态
    """
    if ctx.sim2real_bridge.estop_latched:
        return False

    ctx.sim2real_bridge.estop_latched = True
    payload: dict[str, Any] = {"reason": reason}
    if actuator_id is not None:
        payload["actuator_id"] = actuator_id
    if error_bits is not None:
        payload["error_bits"] = f"0x{int(error_bits):02X}"
    log_event("ESTOP_LATCHED", **payload)

    if ctx.actuator_tx is not None:
        ctx.actuator_tx.send_estop_all(ctx.actuator_ids, critical=True)

    if ctx.feed_state.enabled:
        ctx.feed_state.locked = True
        if not _feed_send_enable(ctx.feed_state, state=False):
            log_event("FEED_DISABLE_FAIL", reason="ESTOP_LATCH")
    return True


def _run_boot_reset(ctx: TeleopRuntimeContext) -> None:
    """
    启动阶段执行一次复位流程。
    """
    ok, reason = perform_reset_sequence(
        actuator_tx=ctx.actuator_tx,
        actuator_monitor=ctx.actuator_monitor,
        motor_mapper=ctx.motor_mapper,
        actuator_ids=ctx.actuator_ids,
        sim2real_bridge=ctx.sim2real_bridge,
        monitor_settle_sec=ctx.monitor_settle_sec,
        dry_run=ctx.dry_run,
        feed_state=ctx.feed_state,
        allow_feed_without_actuator=ctx.allow_feed_without_actuator,
    )
    if not ok:
        log_event("BOOT_RESET_FAIL", reason=reason)
        _latch_estop(ctx, reason="BOOT_RESET_FAIL")
    else:
        log_event("BOOT_READY", mode="DRY_RUN" if ctx.dry_run else "HARDWARE")


def _create_loop_state(control_hz: float) -> TeleopLoopState:
    """
    初始化主循环状态。
    """
    now = time.monotonic()
    return TeleopLoopState(
        estop_hold=HoldLatch(),
        reset_hold=HoldLatch(),
        forward_pressed_prev=False,
        period_sec=1.0 / float(control_hz),
        loop_ts_prev=now,
        last_input_ok_sec=now,
        debug_interval_sec=1.0 / max(0.1, float(DEBUG_INPUT_HZ_DEFAULT)),
        next_debug_ts=now,
        has_debug_snapshot=False,
        last_debug_sample=None,
        last_debug_yaw_rate=0.0,
        last_debug_pitch_rate=0.0,
        last_debug_estop_latched=False,
        last_debug_motor_counts=None,
    )


def _compute_loop_dt(loop_state: TeleopLoopState, loop_start: float) -> float:
    """
    计算本轮 dt，并限制最大步长以抑制暂停恢复后的突变。
    """
    dt = loop_start - loop_state.loop_ts_prev
    loop_state.loop_ts_prev = loop_start
    if dt <= 0.0:
        dt = loop_state.period_sec
    return min(dt, 0.2)


def _feed_repeat_period_sec(feed_state: FeedRuntimeState) -> float:
    if feed_state.cfg is None:
        return 0.1
    return 1.0 / max(0.1, float(feed_state.cfg.repeat_hz))


def _update_forward_state_and_feed(
    sample: GamepadSample,
    loop_state: TeleopLoopState,
    ctx: TeleopRuntimeContext,
    loop_start: float,
) -> None:
    """
    前进键处理：边沿日志 + 按住连续进给（定时发送）。
    """
    feed_state = ctx.feed_state
    prev_pressed = bool(loop_state.forward_pressed_prev)
    now_pressed = bool(sample.forward_pressed)
    repeat_period = _feed_repeat_period_sec(feed_state)

    if now_pressed != prev_pressed:
        loop_state.forward_pressed_prev = now_pressed
        on_forward_state_change(now_pressed, time.time())
        if not now_pressed:
            feed_state.forward_limit_latched = False
            feed_state.next_forward_send_ts = loop_start
            return

        feed_state.forward_limit_latched = False
        result = _handle_feed_forward_rising_edge(ctx)
        feed_state.next_forward_send_ts = loop_start + repeat_period
        if result == "limit":
            feed_state.forward_limit_latched = True
        return

    if not now_pressed:
        return
    if feed_state.forward_limit_latched:
        return
    if loop_start < feed_state.next_forward_send_ts:
        return

    result = _handle_feed_forward_rising_edge(ctx)
    feed_state.next_forward_send_ts = loop_start + repeat_period
    if result == "limit":
        feed_state.forward_limit_latched = True


def _handle_disconnect_estop(
    sample: GamepadSample,
    loop_start: float,
    teleop_cfg: Any,
    loop_state: TeleopLoopState,
    ctx: TeleopRuntimeContext,
) -> None:
    """
    断连/超时安全判定。
    """
    if sample.connected:
        loop_state.last_input_ok_sec = loop_start

    disconnected = should_trigger_disconnect_estop(
        now_sec=loop_start,
        last_input_ok_sec=loop_state.last_input_ok_sec,
        timeout_sec=teleop_cfg.disconnect_timeout_sec,
        controller_connected=sample.connected,
    )
    if disconnected:
        _latch_estop(ctx, reason="GAMEPAD_DISCONNECT_OR_TIMEOUT")


def _handle_manual_estop_and_reset(
    sample: GamepadSample,
    loop_start: float,
    teleop_cfg: Any,
    loop_state: TeleopLoopState,
    ctx: TeleopRuntimeContext,
) -> None:
    """
    手动急停与复位组合键处理。
    """
    reset_combo_active = sample.reset_primary_pressed and sample.reset_secondary_pressed
    estop_hold_active = sample.estop_pressed and (not reset_combo_active)

    if loop_state.estop_hold.update(
        active=estop_hold_active,
        now_sec=loop_start,
        hold_sec=teleop_cfg.estop_hold_sec,
    ):
        _latch_estop(ctx, reason="MANUAL_HOLD")

    if not loop_state.reset_hold.update(
        active=reset_combo_active,
        now_sec=loop_start,
        hold_sec=teleop_cfg.reset_hold_sec,
    ):
        return

    if not ctx.sim2real_bridge.estop_latched:
        log_event("RESET_IGNORED", reason="NOT_LATCHED")
        return

    if ctx.actuator_monitor is not None:
        ctx.actuator_monitor.set_active(False)
    ok, reason = perform_reset_sequence(
        actuator_tx=ctx.actuator_tx,
        actuator_monitor=ctx.actuator_monitor,
        motor_mapper=ctx.motor_mapper,
        actuator_ids=ctx.actuator_ids,
        sim2real_bridge=ctx.sim2real_bridge,
        monitor_settle_sec=ctx.monitor_settle_sec,
        dry_run=ctx.dry_run,
        feed_state=ctx.feed_state,
        allow_feed_without_actuator=ctx.allow_feed_without_actuator,
    )
    if ok:
        log_event("RESET_SUCCESS")
    else:
        ctx.sim2real_bridge.estop_latched = True
        log_event("RESET_FAIL", reason=reason)


def _handle_monitor_fault(ctx: TeleopRuntimeContext) -> None:
    """
    监控线程故障上报处理。
    """
    monitor_fault = (
        ctx.actuator_monitor.get_fault()
        if ctx.actuator_monitor is not None and ctx.actuator_monitor.has_fault()
        else None
    )
    if monitor_fault is None:
        return

    _latch_estop(
        ctx,
        reason=f"MONITOR_{monitor_fault.reason}",
        actuator_id=monitor_fault.actuator_id,
        error_bits=monitor_fault.error_bits,
    )


def _sync_monitor_active_state(ctx: TeleopRuntimeContext) -> None:
    """
    根据锁存状态控制监控线程是否处于 active。
    """
    if ctx.actuator_monitor is not None:
        ctx.actuator_monitor.set_active(not ctx.sim2real_bridge.estop_latched)


def _compute_control_delta(
    sample: GamepadSample,
    teleop_cfg: Any,
    dt: float,
) -> Tuple[float, float, float, float]:
    """
    从手柄轴值计算速率与角度增量。

    返回:
        (yaw_rate, pitch_rate, delta_yaw, delta_pitch)
    """
    yaw_rate = axis_to_rate(
        raw_axis=sample.axis_x,
        deadzone=teleop_cfg.deadzone,
        max_rate_rad_s=teleop_cfg.max_yaw_rate_rad_s,
        invert=teleop_cfg.invert_yaw,
    )
    pitch_rate = axis_to_rate(
        raw_axis=sample.axis_y,
        deadzone=teleop_cfg.deadzone,
        max_rate_rad_s=teleop_cfg.max_pitch_rate_rad_s,
        invert=teleop_cfg.invert_pitch,
    )
    delta_yaw = yaw_rate * dt
    delta_pitch = pitch_rate * dt
    return yaw_rate, pitch_rate, delta_yaw, delta_pitch


def _maybe_log_debug_snapshot(
    enabled: bool,
    loop_start: float,
    sample: GamepadSample,
    yaw_rate: float,
    pitch_rate: float,
    delta_yaw: float,
    delta_pitch: float,
    estop_latched: bool,
    loop_state: TeleopLoopState,
    ctx: TeleopRuntimeContext,
) -> None:
    """
    输入变化触发的调试日志（含最小间隔限频）。

    设计目的:
    - 仅在输入/状态变化时打印，减少静止时刷屏。
    - 仍保留最小间隔限制，避免高频抖动产生过多日志。
    """
    if not enabled:
        return

    motor_counts: Optional[Tuple[int, int, int, int]] = None
    try:
        motor_counts_arr = ctx.motor_mapper.mm_targets_to_counts(ctx.sim2real_bridge.motor_target_mm)
        motor_counts_vec = np.asarray(motor_counts_arr, dtype=np.int32).reshape(-1)
        if motor_counts_vec.shape == (4,):
            motor_counts = tuple(int(v) for v in motor_counts_vec.tolist())
    except Exception:
        # 调试日志不应影响主控制流：映射失败时跳过计数字段。
        motor_counts = None

    changed = False
    axis_eps = 0.03
    rate_eps = 0.03
    prev_sample = loop_state.last_debug_sample
    if not loop_state.has_debug_snapshot or prev_sample is None:
        changed = True
    else:
        changed = any(
            (
                sample.connected != prev_sample.connected,
                abs(sample.axis_x - prev_sample.axis_x) >= axis_eps,
                abs(sample.axis_y - prev_sample.axis_y) >= axis_eps,
                sample.forward_pressed != prev_sample.forward_pressed,
                sample.estop_pressed != prev_sample.estop_pressed,
                sample.reset_primary_pressed != prev_sample.reset_primary_pressed,
                sample.reset_secondary_pressed != prev_sample.reset_secondary_pressed,
                abs(yaw_rate - loop_state.last_debug_yaw_rate) >= rate_eps,
                abs(pitch_rate - loop_state.last_debug_pitch_rate) >= rate_eps,
                estop_latched != loop_state.last_debug_estop_latched,
                motor_counts != loop_state.last_debug_motor_counts,
            )
        )

    if not changed or loop_start < loop_state.next_debug_ts:
        return

    log_input_snapshot(
        sample=sample,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        delta_yaw=delta_yaw,
        delta_pitch=delta_pitch,
        estop_latched=estop_latched,
        motor_counts=motor_counts,
    )
    loop_state.next_debug_ts = loop_start + loop_state.debug_interval_sec
    loop_state.has_debug_snapshot = True
    loop_state.last_debug_sample = sample
    loop_state.last_debug_yaw_rate = yaw_rate
    loop_state.last_debug_pitch_rate = pitch_rate
    loop_state.last_debug_estop_latched = estop_latched
    loop_state.last_debug_motor_counts = motor_counts


def _step_bridge_and_send(
    delta_yaw: float,
    delta_pitch: float,
    dt: float,
    ctx: TeleopRuntimeContext,
) -> None:
    """
    执行桥接一步并在允许时发送 F3 跟随帧。
    """
    bridge_result = ctx.sim2real_bridge.step(delta_yaw, delta_pitch, dt)
    if bridge_result.command == BridgeCommand.ESTOP:
        _latch_estop(ctx, reason=bridge_result.reason or "BRIDGE_ESTOP")

    if not should_send_follow_command(
        command=bridge_result.command,
        should_send_flag=bridge_result.should_send,
        estop_latched=ctx.sim2real_bridge.estop_latched,
    ):
        return

    if ctx.dry_run:
        return

    assert ctx.actuator_tx is not None
    send_ok = send_follow_target_mm(
        actuator_tx=ctx.actuator_tx,
        motor_mapper=ctx.motor_mapper,
        actuator_ids=ctx.actuator_ids,
        motor_target_mm=bridge_result.motor_target_mm,
        frame_name="F3_FOLLOW",
        critical=False,
    )
    if not send_ok:
        _latch_estop(ctx, reason="SERIAL_TX_FAIL")


def _sleep_for_rate(loop_start: float, period_sec: float) -> None:
    """
    固定频率节拍控制。
    """
    elapsed = time.monotonic() - loop_start
    sleep_sec = period_sec - elapsed
    if sleep_sec > 0.0:
        time.sleep(sleep_sec)


def _cleanup_runtime(
    gamepad: PygameGamepadInput,
    actuator_tx: Optional[ActuatorTx],
    actuator_monitor: Optional[ActuatorMonitor],
    actuator_ids: Sequence[int],
    serial_link: Any,
    feed_state: FeedRuntimeState,
) -> None:
    """
    退出阶段清理资源。
    """
    if actuator_monitor is not None:
        actuator_monitor.set_active(False)
        actuator_monitor.stop(join_timeout_sec=1.0)

    if actuator_tx is not None:
        try:
            actuator_tx.send_estop_all(actuator_ids, critical=True)
        except Exception:
            pass

    if serial_link is not None:
        try:
            serial_link.close()
        except Exception:
            pass

    if feed_state.enabled and feed_state.cfg is not None:
        try:
            if feed_state.cfg.disable_on_exit:
                _feed_send_enable(feed_state, state=False)
        except Exception:
            pass
        if feed_state.serial_link is not None:
            try:
                feed_state.serial_link.close()
            except Exception:
                pass

    gamepad.close()


def _list_controllers_and_exit(gamepad: PygameGamepadInput) -> int:
    """
    列出当前可见手柄并退出。
    """
    items = gamepad.list_controllers()
    if len(items) == 0:
        print("No controller device detected.")
    else:
        print("Detected controllers:")
        for idx, name, supported in items:
            print(
                f"  - index={idx} name={name} "
                f"sdl2_gamecontroller={'yes' if supported else 'no'}"
            )
    gamepad.close()
    return 0


def run_teleop(
    config_path: str,
    dry_run: bool,
    debug_input: bool = False,
    list_controllers_only: bool = False,
) -> int:
    """
    teleop 主流程。
    """
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

    sim2real_bridge = Sim2RealBridge(sim2real_cfg)
    motor_mapper = MotorMapper(actuator_cfg, sim2real_cfg.motor_order)
    actuator_ids = actuator_cfg.ids_for_motor_order(sim2real_cfg.motor_order)
    actuator_tx: Optional[ActuatorTx] = None
    actuator_monitor: Optional[ActuatorMonitor] = None
    serial_link = None
    feed_state = _disabled_feed_state()

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

        while True:
            loop_start = time.monotonic()
            dt = _compute_loop_dt(loop_state, loop_start)

            sample = gamepad.poll()
            _update_forward_state_and_feed(sample, loop_state, ctx, loop_start)
            _handle_disconnect_estop(sample, loop_start, teleop_cfg, loop_state, ctx)
            _handle_manual_estop_and_reset(sample, loop_start, teleop_cfg, loop_state, ctx)
            _handle_monitor_fault(ctx)
            _sync_monitor_active_state(ctx)

            if feed_only_mode:
                yaw_rate, pitch_rate, delta_yaw, delta_pitch = 0.0, 0.0, 0.0, 0.0
            else:
                yaw_rate, pitch_rate, delta_yaw, delta_pitch = _compute_control_delta(
                    sample=sample,
                    teleop_cfg=teleop_cfg,
                    dt=dt,
                )
            _maybe_log_debug_snapshot(
                enabled=debug_input,
                loop_start=loop_start,
                sample=sample,
                yaw_rate=yaw_rate,
                pitch_rate=pitch_rate,
                delta_yaw=delta_yaw,
                delta_pitch=delta_pitch,
                estop_latched=ctx.sim2real_bridge.estop_latched,
                loop_state=loop_state,
                ctx=ctx,
            )
            if not feed_only_mode:
                _step_bridge_and_send(
                    delta_yaw=delta_yaw,
                    delta_pitch=delta_pitch,
                    dt=dt,
                    ctx=ctx,
                )
            _sleep_for_rate(loop_start, loop_state.period_sec)
    except KeyboardInterrupt:
        log_event("STOP", reason="KEYBOARD_INTERRUPT")
    finally:
        _cleanup_runtime(
            gamepad=gamepad,
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            actuator_ids=actuator_ids,
            serial_link=serial_link,
            feed_state=feed_state,
        )

    return 0
