from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_SERIAL_TIMEOUT_UNSET = object()

DEFAULT_J_4X2_MM_PER_RAD = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=np.float64,
)

MOTOR_NAME_SET = ("m1", "m2", "m3", "m4")


class BridgeCommand(str, Enum):
    CMD = "CMD"
    ESTOP = "ESTOP"


@dataclass
class Sim2RealConfig:
    J_4x2_mm_per_rad: np.ndarray
    yaw_limit_deg: float = 120.0
    pitch_limit_deg: float = 120.0
    motor_limit_mm: float = 15.0
    control_hz: float = 30.0
    motor_order: Tuple[str, str, str, str] = ("m1", "m2", "m3", "m4")
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 115200
    serial_timeout: float = 0.0
    serial_write_timeout: float = 0.2
    serial_critical_retry_count: int = 3
    serial_critical_retry_interval_sec: float = 0.02

    def __post_init__(self) -> None:
        self.J_4x2_mm_per_rad = np.asarray(self.J_4x2_mm_per_rad, dtype=np.float64)
        if self.J_4x2_mm_per_rad.shape != (4, 2):
            raise ValueError(
                f"J_4x2_mm_per_rad must have shape (4, 2), got {self.J_4x2_mm_per_rad.shape}"
            )
        motor_order_tuple = tuple(str(x).strip() for x in self.motor_order)
        if len(motor_order_tuple) != 4:
            raise ValueError(f"motor_order must contain 4 entries, got {len(motor_order_tuple)}")
        if set(motor_order_tuple) != set(MOTOR_NAME_SET):
            raise ValueError("motor_order must be a permutation of ('m1','m2','m3','m4').")
        self.motor_order = motor_order_tuple
        if float(self.motor_limit_mm) <= 0.0:
            raise ValueError("motor_limit_mm must be > 0")
        if float(self.control_hz) <= 0.0:
            raise ValueError("control_hz must be > 0")
        self.serial_write_timeout = float(self.serial_write_timeout)
        if self.serial_write_timeout < 0.0:
            raise ValueError("serial_write_timeout must be >= 0")
        self.serial_critical_retry_count = max(1, int(self.serial_critical_retry_count))
        self.serial_critical_retry_interval_sec = float(self.serial_critical_retry_interval_sec)
        if self.serial_critical_retry_interval_sec < 0.0:
            raise ValueError("serial_critical_retry_interval_sec must be >= 0")


@dataclass
class EstopAlarmConfig:
    enabled: bool = True
    repeat: int = 3
    terminal_bell: bool = True
    banner_width: int = 70

    def __post_init__(self) -> None:
        self.repeat = max(1, int(self.repeat))
        self.banner_width = max(40, int(self.banner_width))


@dataclass
class MotorOutputConfig:
    print_tx_frame: bool = False
    print_every_n: int = 100
    save_csv: bool = True
    csv_path: str = "./sim2real_motor_log.csv"
    save_plot: bool = True
    plot_path: str = "./sim2real_motor_plot.png"
    plot_dpi: int = 120
    max_plot_points: int = 4000

    def __post_init__(self) -> None:
        self.print_every_n = max(0, int(self.print_every_n))
        self.plot_dpi = max(60, int(self.plot_dpi))
        self.max_plot_points = max(200, int(self.max_plot_points))


@dataclass
class MonitorConfig:
    enabled: bool = True
    query_hz: float = 20.0
    response_timeout_sec: float = 0.02
    failure_threshold: int = 3
    error_mask: int = 0x0F
    log_every_n: int = 5

    def __post_init__(self) -> None:
        self.query_hz = float(self.query_hz)
        self.response_timeout_sec = float(self.response_timeout_sec)
        self.failure_threshold = max(1, int(self.failure_threshold))
        self.error_mask = int(self.error_mask) & 0xFF
        self.log_every_n = max(0, int(self.log_every_n))

        if self.query_hz <= 0.0:
            raise ValueError("monitor.query_hz must be > 0")
        if self.response_timeout_sec <= 0.0:
            raise ValueError("monitor.response_timeout_sec must be > 0")


@dataclass
class MotorAxisMapConfig:
    zero_count: int = 1000
    count_per_mm: float = 20.0
    sign: int = 1
    soft_min_count: int = 300
    soft_max_count: int = 1700

    def __post_init__(self) -> None:
        self.zero_count = int(self.zero_count)
        self.count_per_mm = float(self.count_per_mm)
        self.sign = -1 if int(self.sign) < 0 else 1
        self.soft_min_count = int(self.soft_min_count)
        self.soft_max_count = int(self.soft_max_count)

        if self.count_per_mm <= 0.0:
            raise ValueError("count_per_mm must be > 0")
        if self.soft_min_count > self.soft_max_count:
            raise ValueError("soft_min_count must be <= soft_max_count")


@dataclass
class ActuatorConfig:
    mode: str = "broadcast_follow_no_feedback"
    # ids 语义固定为 [m1,m2,m3,m4]，不随 motor_order 改变
    ids: Tuple[int, int, int, int] = (1, 2, 3, 4)
    # 显式语义映射，避免 ids 与 motor_order 关系仅靠人工约定
    id_by_motor: Mapping[str, int] = field(
        default_factory=lambda: {"m1": 1, "m2": 2, "m3": 3, "m4": 4}
    )
    position_index: int = 0x37
    count_min: int = 0
    count_max: int = 2000
    m1: MotorAxisMapConfig = field(default_factory=MotorAxisMapConfig)
    m2: MotorAxisMapConfig = field(default_factory=MotorAxisMapConfig)
    m3: MotorAxisMapConfig = field(default_factory=MotorAxisMapConfig)
    m4: MotorAxisMapConfig = field(default_factory=MotorAxisMapConfig)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"broadcast_follow_no_feedback"}:
            raise ValueError(
                "actuator.mode must be 'broadcast_follow_no_feedback' in this stage."
            )
        self.mode = mode

        ids_tuple = tuple(int(x) for x in self.ids)
        if len(ids_tuple) != 4:
            raise ValueError(f"actuator.ids must contain 4 ids, got {len(ids_tuple)}")
        if len(set(ids_tuple)) != 4:
            raise ValueError("actuator.ids must be unique.")
        for aid in ids_tuple:
            if not (1 <= aid <= 254):
                raise ValueError(f"actuator id out of range [1,254]: {aid}")

        raw_id_by_motor = self.id_by_motor
        if not isinstance(raw_id_by_motor, Mapping):
            raise ValueError("actuator.id_by_motor must be a mapping/object.")

        id_map: Dict[str, int] = {}
        for name in MOTOR_NAME_SET:
            if name not in raw_id_by_motor:
                raise ValueError(f"`sim2real.actuator.id_by_motor.{name}` is required.")
            aid = int(raw_id_by_motor[name])
            if not (1 <= aid <= 254):
                raise ValueError(f"actuator id_by_motor.{name} out of range [1,254]: {aid}")
            id_map[name] = aid
        if len(set(id_map.values())) != 4:
            raise ValueError("actuator.id_by_motor ids must be unique.")

        expected_ids = tuple(id_map[name] for name in MOTOR_NAME_SET)
        if ids_tuple != expected_ids:
            raise ValueError(
                "actuator.ids must follow semantic order [m1,m2,m3,m4] and match actuator.id_by_motor."
            )

        self.ids = ids_tuple
        self.id_by_motor = id_map

        self.position_index = _coerce_int(self.position_index)
        if not (0 <= self.position_index <= 255):
            raise ValueError("actuator.position_index must be in [0,255]")

        self.count_min = int(self.count_min)
        self.count_max = int(self.count_max)
        if self.count_min >= self.count_max:
            raise ValueError("actuator.count_min must be < actuator.count_max")

        for name in MOTOR_NAME_SET:
            cfg = getattr(self, name)
            if not isinstance(cfg, MotorAxisMapConfig):
                raise ValueError(f"actuator.per_motor.{name} config is invalid")
            if cfg.soft_min_count < self.count_min or cfg.soft_max_count > self.count_max:
                raise ValueError(
                    f"actuator.per_motor.{name}.soft_* must be within [{self.count_min},{self.count_max}]"
                )

    def axis_cfg(self, motor_name: str) -> MotorAxisMapConfig:
        if motor_name not in MOTOR_NAME_SET:
            raise KeyError(f"unknown motor name: {motor_name}")
        return getattr(self, motor_name)

    def ids_for_motor_order(self, motor_order: Sequence[str]) -> Tuple[int, int, int, int]:
        order = tuple(str(x).strip() for x in motor_order)
        if len(order) != 4 or set(order) != set(MOTOR_NAME_SET):
            raise ValueError("motor_order must be a permutation of ('m1','m2','m3','m4').")
        return tuple(int(self.id_by_motor[name]) for name in order)


@dataclass
class DaggerSim2RealRuntimeConfig:
    bridge: Sim2RealConfig
    actuator: ActuatorConfig
    alarm: EstopAlarmConfig
    monitor: MonitorConfig
    output: MotorOutputConfig


@dataclass
class BridgeResult:
    command: BridgeCommand
    should_send: bool = False
    serial_frame: str = ""
    motor_delta_mm: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    motor_target_mm: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    yaw_accum_rad: float = 0.0
    pitch_accum_rad: float = 0.0
    reason: str = ""


@dataclass
class ActuatorStatus:
    actuator_id: int
    target_count: int
    current_count: int
    temperature_c: int
    error_bits: int
    raw_frame: bytes
    timestamp_sec: float


@dataclass
class MonitorFault:
    reason: str
    actuator_id: int
    consecutive_failures: int
    raw: bytes = b""
    error_bits: int = 0
    timestamp_sec: float = 0.0


@dataclass
class FaultClearVerification:
    ok: bool
    statuses_by_id: Dict[int, ActuatorStatus] = field(default_factory=dict)
    failures_by_id: Dict[int, str] = field(default_factory=dict)
    uncleared_error_bits_by_id: Dict[int, int] = field(default_factory=dict)


class Sim2RealBridge:
    def __init__(self, config: Sim2RealConfig):
        self.config = config
        self.yaw_limit_rad = np.deg2rad(float(config.yaw_limit_deg))
        self.pitch_limit_rad = np.deg2rad(float(config.pitch_limit_deg))
        self.motor_limit_mm = float(config.motor_limit_mm)
        self.control_period_sec = 1.0 / float(config.control_hz)

        self.yaw_accum_rad = 0.0
        self.pitch_accum_rad = 0.0
        self.motor_target_mm = np.zeros(4, dtype=np.float64)
        self.control_dt_accum = 0.0
        self.estop_latched = False

    def _make_estop(self, reason: str) -> BridgeResult:
        return BridgeResult(
            command=BridgeCommand.ESTOP,
            should_send=True,
            motor_target_mm=self.motor_target_mm.copy(),
            yaw_accum_rad=self.yaw_accum_rad,
            pitch_accum_rad=self.pitch_accum_rad,
            reason=reason,
        )

    def step(self, delta_yaw_rad: float, delta_pitch_rad: float, dt: float) -> BridgeResult:
        delta_yaw = float(delta_yaw_rad)
        delta_pitch = float(delta_pitch_rad)
        dt_val = float(dt)

        if not np.isfinite(delta_yaw) or not np.isfinite(delta_pitch) or not np.isfinite(dt_val) or dt_val < 0.0:
            self.estop_latched = True
            return self._make_estop("INVALID_ACTION")

        if self.estop_latched:
            return self._make_estop("ANGLE_LIMIT")

        yaw_next = self.yaw_accum_rad + delta_yaw
        pitch_next = self.pitch_accum_rad + delta_pitch
        over_limit = abs(yaw_next) > self.yaw_limit_rad or abs(pitch_next) > self.pitch_limit_rad

        if over_limit:
            self.estop_latched = True
            return self._make_estop("ANGLE_LIMIT")

        self.yaw_accum_rad = yaw_next
        self.pitch_accum_rad = pitch_next

        action = np.array([delta_yaw, delta_pitch], dtype=np.float64)
        motor_delta_mm = self.config.J_4x2_mm_per_rad @ action

        self.motor_target_mm += motor_delta_mm
        self.motor_target_mm = np.clip(
            self.motor_target_mm,
            -self.motor_limit_mm,
            self.motor_limit_mm,
        )

        should_send = False
        if dt_val > 0.0:
            self.control_dt_accum += dt_val
            if self.control_dt_accum + 1e-12 >= self.control_period_sec:
                should_send = True
                self.control_dt_accum = math.fmod(self.control_dt_accum, self.control_period_sec)

        return BridgeResult(
            command=BridgeCommand.CMD,
            should_send=should_send,
            motor_delta_mm=motor_delta_mm,
            motor_target_mm=self.motor_target_mm.copy(),
            yaw_accum_rad=self.yaw_accum_rad,
            pitch_accum_rad=self.pitch_accum_rad,
        )

    def reset(self) -> None:
        self.yaw_accum_rad = 0.0
        self.pitch_accum_rad = 0.0
        self.motor_target_mm = np.zeros(4, dtype=np.float64)
        self.control_dt_accum = 0.0
        self.estop_latched = False


class LAFrameBuilder:
    @staticmethod
    def checksum(frame_body: Sequence[int]) -> int:
        return sum(int(x) for x in frame_body) & 0xFF

    @staticmethod
    def build_single_control_frame(actuator_id: int, cmd_value: int) -> bytes:
        aid = int(actuator_id)
        cmd = int(cmd_value)
        if not (1 <= aid <= 254):
            raise ValueError(f"actuator_id out of range [1,254]: {aid}")
        if not (0 <= cmd <= 255):
            raise ValueError(f"cmd_value out of range [0,255]: {cmd}")

        body = [0x03, aid, 0x04, 0x00, cmd]
        return bytes([0x55, 0xAA] + body + [LAFrameBuilder.checksum(body)])

    @staticmethod
    def build_status_query_frame(actuator_id: int) -> bytes:
        aid = int(actuator_id)
        if not (1 <= aid <= 254):
            raise ValueError(f"actuator_id out of range [1,254]: {aid}")

        body = [0x03, aid, 0x04, 0x00, 0x22]
        return bytes([0x55, 0xAA] + body + [LAFrameBuilder.checksum(body)])

    @staticmethod
    def build_follow_broadcast_frame(actuator_ids: Sequence[int], target_counts: Sequence[int]) -> bytes:
        if len(actuator_ids) != len(target_counts):
            raise ValueError("actuator_ids and target_counts length mismatch")

        n = len(actuator_ids)
        if n < 1 or n > 15:
            raise ValueError(f"broadcast follow supports 1~15 actuators, got {n}")

        body = [1 + 3 * n, 0xFF, 0xF3]
        for aid_raw, count_raw in zip(actuator_ids, target_counts):
            aid = int(aid_raw)
            count = int(count_raw)
            if not (1 <= aid <= 254):
                raise ValueError(f"actuator_id out of range [1,254]: {aid}")
            if not (0 <= count <= 0xFFFF):
                raise ValueError(f"target count out of range [0,65535]: {count}")
            body.extend([aid, count & 0xFF, (count >> 8) & 0xFF])

        return bytes([0x55, 0xAA] + body + [LAFrameBuilder.checksum(body)])


class MotorMapper:
    def __init__(self, actuator_cfg: ActuatorConfig, motor_order: Sequence[str]):
        self.actuator_cfg = actuator_cfg
        motor_order_tuple = tuple(str(x).strip() for x in motor_order)
        if len(motor_order_tuple) != 4 or set(motor_order_tuple) != set(MOTOR_NAME_SET):
            raise ValueError("motor_order must be a permutation of ('m1','m2','m3','m4').")
        self.motor_order = motor_order_tuple
        self.semantic_index = {name: idx for idx, name in enumerate(MOTOR_NAME_SET)}

    def mm_targets_to_counts(self, mm_targets: np.ndarray) -> np.ndarray:
        arr = np.asarray(mm_targets, dtype=np.float64).reshape(-1)
        if arr.shape != (4,):
            raise ValueError(f"mm_targets must have shape (4,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("mm_targets contain non-finite values")

        out = []
        for motor_name in self.motor_order:
            mm_val = float(arr[self.semantic_index[motor_name]])
            axis_cfg = self.actuator_cfg.axis_cfg(motor_name)
            raw = axis_cfg.zero_count + axis_cfg.sign * mm_val * axis_cfg.count_per_mm
            clipped = np.clip(raw, axis_cfg.soft_min_count, axis_cfg.soft_max_count)
            clipped = np.clip(clipped, self.actuator_cfg.count_min, self.actuator_cfg.count_max)
            out.append(int(round(float(clipped))))
        return np.asarray(out, dtype=np.int32)

    def zero_counts(self) -> np.ndarray:
        return self.mm_targets_to_counts(np.zeros(4, dtype=np.float64))


class ActuatorTx:
    def __init__(
        self,
        serial_link: Any,
        critical_retry_count: int = 3,
        critical_retry_interval_sec: float = 0.02,
        print_tx_frame: bool = False,
        serial_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.serial_link = serial_link
        self.critical_retry_count = max(1, int(critical_retry_count))
        self.critical_retry_interval_sec = max(0.0, float(critical_retry_interval_sec))
        self.print_tx_frame = bool(print_tx_frame)
        self.serial_lock = serial_lock if serial_lock is not None else threading.Lock()

    def _send_frame_locked(self, frame: bytes, frame_name: str = "FRAME") -> bool:
        if self.serial_link is None:
            print(f"❌ [ActuatorTx] 串口链路未就绪，拒绝发送 {frame_name}。")
            return False

        payload = bytes(frame)
        try:
            bytes_written = self.serial_link.write(payload)
            self.serial_link.flush()
        except Exception as exc:
            print(f"⚠️ [ActuatorTx] 串口发送失败: {exc} | {frame_name}")
            return False

        if bytes_written != len(payload):
            print(
                f"⚠️ [ActuatorTx] 串口短写: wrote={bytes_written} expected={len(payload)} | {frame_name}"
            )
            return False

        if self.print_tx_frame:
            print(f"[ActuatorTx TX] {frame_name}: {payload.hex(' ').upper()}")
        return True

    def send_frame(self, frame: bytes, frame_name: str = "FRAME") -> bool:
        with self.serial_lock:
            return self._send_frame_locked(frame, frame_name)

    def send_critical(self, frames: Sequence[bytes], frame_name: str) -> bool:
        frame_seq = [bytes(x) for x in frames]
        if len(frame_seq) == 0:
            return True

        for attempt in range(1, self.critical_retry_count + 1):
            with self.serial_lock:
                ok = True
                for idx, frame in enumerate(frame_seq):
                    if not self._send_frame_locked(frame, f"{frame_name}[{idx}]"):
                        ok = False
                        break
            if ok:
                if attempt > 1:
                    print(f"✅ [ActuatorTx] 关键帧 {frame_name} 在第 {attempt} 次重发后成功。")
                return True

            print(
                f"⚠️ [ActuatorTx] 关键帧 {frame_name} 发送失败（{attempt}/{self.critical_retry_count}）。"
            )
            if attempt < self.critical_retry_count and self.critical_retry_interval_sec > 0.0:
                time.sleep(self.critical_retry_interval_sec)

        print(f"❌ [ActuatorTx] 关键帧 {frame_name} 连续 {self.critical_retry_count} 次发送失败。")
        return False

    def send_follow_broadcast(
        self,
        actuator_ids: Sequence[int],
        target_counts: Sequence[int],
        critical: bool = False,
        frame_name: str = "F3_FOLLOW",
    ) -> bool:
        frame = LAFrameBuilder.build_follow_broadcast_frame(actuator_ids, target_counts)
        if critical:
            return self.send_critical([frame], frame_name)
        return self.send_frame(frame, frame_name)

    def send_estop_all(self, actuator_ids: Sequence[int], critical: bool = True) -> bool:
        frames = [LAFrameBuilder.build_single_control_frame(aid, 0x23) for aid in actuator_ids]
        if critical:
            return self.send_critical(frames, "ESTOP_ALL")
        ok = True
        for idx, frame in enumerate(frames):
            ok = self.send_frame(frame, f"ESTOP[{idx}]") and ok
        return ok

    def send_work_start_all(self, actuator_ids: Sequence[int], critical: bool = True) -> bool:
        frames = [LAFrameBuilder.build_single_control_frame(aid, 0x04) for aid in actuator_ids]
        if critical:
            return self.send_critical(frames, "WORK_START_ALL")
        ok = True
        for idx, frame in enumerate(frames):
            ok = self.send_frame(frame, f"WORK_START[{idx}]") and ok
        return ok

    def send_fault_clear_all(self, actuator_ids: Sequence[int], critical: bool = True) -> bool:
        frames = [LAFrameBuilder.build_single_control_frame(aid, 0x1E) for aid in actuator_ids]
        if critical:
            return self.send_critical(frames, "FAULT_CLEAR_ALL")
        ok = True
        for idx, frame in enumerate(frames):
            ok = self.send_frame(frame, f"FAULT_CLEAR[{idx}]") and ok
        return ok


class LAFrameParser:
    RESPONSE_HEADER = b"\xAA\x55"

    @staticmethod
    def extract_first_response_frame(payload: bytes) -> Optional[bytes]:
        frame, _ = LAFrameParser.extract_first_response_frame_with_consumed(payload)
        return frame

    @staticmethod
    def extract_first_response_frame_with_consumed(payload: bytes) -> Tuple[Optional[bytes], int]:
        data = bytes(payload)
        data_len = len(data)
        if data_len < 2:
            return None, 0

        idx = 0
        while idx + 1 < data_len:
            if data[idx] != 0xAA or data[idx + 1] != 0x55:
                idx += 1
                continue

            if idx + 3 >= data_len:
                return None, idx

            frame_len = int(data[idx + 2])
            total_len = frame_len + 5
            if total_len < 7:
                idx += 1
                continue
            if idx + total_len <= data_len:
                end = idx + total_len
                return data[idx:end], end
            return None, idx

        return None, max(0, data_len - 1)

    @staticmethod
    def parse_status_response(frame: bytes, expected_id: Optional[int] = None) -> ActuatorStatus:
        payload = bytes(frame)
        if len(payload) < 7:
            raise ValueError("response frame too short")
        if payload[0:2] != LAFrameParser.RESPONSE_HEADER:
            raise ValueError("invalid response header")

        frame_len = int(payload[2])
        expected_total = frame_len + 5
        if len(payload) != expected_total:
            raise ValueError(f"response length mismatch: got {len(payload)} expected {expected_total}")

        body = payload[2:-1]
        checksum = int(payload[-1])
        checksum_calc = LAFrameBuilder.checksum(body)
        if checksum != checksum_calc:
            raise ValueError(f"invalid checksum: got {checksum} expected {checksum_calc}")

        actuator_id = int(payload[3])
        if expected_id is not None and actuator_id != int(expected_id):
            raise ValueError(f"id mismatch: got {actuator_id} expected {int(expected_id)}")

        cmd = int(payload[4])
        reserved = int(payload[5])
        index = int(payload[6])
        if cmd != 0x04:
            raise ValueError(f"unexpected response cmd: {cmd}")
        if reserved != 0x00:
            raise ValueError(f"unexpected response reserved byte: {reserved}")
        if index != 0x22:
            raise ValueError(f"unexpected response index: {index}")

        # For 0x22 replies, the manual defines:
        # B7~B8 target, B9~B10 current, B11 temp, B12~B13 current,
        # B14 force low, B15 error, B16 force high, B17~B20 internal data.
        data = payload[7:-1]
        if len(data) < 14:
            raise ValueError("status response data too short")

        target_count = int(data[0]) | (int(data[1]) << 8)
        current_count = int.from_bytes(data[2:4], byteorder="little", signed=True)
        temperature_c = int.from_bytes(bytes([int(data[4])]), byteorder="little", signed=True)
        error_bits = int(data[8]) & 0xFF

        return ActuatorStatus(
            actuator_id=actuator_id,
            target_count=target_count,
            current_count=current_count,
            temperature_c=temperature_c,
            error_bits=error_bits,
            raw_frame=payload,
            timestamp_sec=time.time(),
        )


class ActuatorMonitor:
    def __init__(
        self,
        serial_link: Any,
        actuator_ids: Sequence[int],
        monitor_cfg: MonitorConfig,
        serial_lock: Optional[threading.Lock] = None,
    ) -> None:
        ids = tuple(int(x) for x in actuator_ids)
        if len(ids) == 0:
            raise ValueError("actuator_ids must not be empty")
        for aid in ids:
            if not (1 <= aid <= 254):
                raise ValueError(f"actuator id out of range [1,254]: {aid}")

        self.serial_link = serial_link
        self.actuator_ids = ids
        self.monitor_cfg = monitor_cfg
        self.serial_lock = serial_lock if serial_lock is not None else threading.Lock()

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_event = threading.Event()
        self._fault_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._rr_index = 0
        self._poll_count = 0
        self._consecutive_failures_by_id: Dict[int, int] = {
            aid: 0 for aid in self.actuator_ids
        }
        self._last_status: Dict[int, ActuatorStatus] = {}
        self._last_fault: Optional[MonitorFault] = None

    def start(self) -> None:
        if not self.monitor_cfg.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sim2real-actuator-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_sec: float = 1.0) -> None:
        self._stop_event.set()
        self._active_event.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout_sec)))

    def set_active(self, active: bool) -> None:
        if not self.monitor_cfg.enabled:
            return
        if bool(active):
            self._active_event.set()
        else:
            was_active = self._active_event.is_set()
            self._active_event.clear()
            if was_active:
                with self._state_lock:
                    self._reset_failure_streaks_locked()

    def _reset_failure_streaks_locked(self) -> None:
        for aid in self._consecutive_failures_by_id:
            self._consecutive_failures_by_id[aid] = 0

    def clear_fault(self) -> None:
        with self._state_lock:
            self._fault_event.clear()
            self._last_fault = None
            self._reset_failure_streaks_locked()

    def verify_fault_clear(
        self,
        actuator_ids: Optional[Sequence[int]] = None,
        attempts: int = 2,
        settle_time_sec: float = 0.01,
        error_mask: Optional[int] = None,
    ) -> FaultClearVerification:
        if not self.monitor_cfg.enabled:
            return FaultClearVerification(ok=True)

        ids = tuple(self.actuator_ids if actuator_ids is None else actuator_ids)
        if len(ids) == 0:
            return FaultClearVerification(ok=True)

        max_attempts = max(1, int(attempts))
        settle_sec = max(0.0, float(settle_time_sec))
        mask = (
            self.monitor_cfg.error_mask if error_mask is None else (int(error_mask) & 0xFF)
        )

        last_statuses: Dict[int, ActuatorStatus] = {}
        last_failures: Dict[int, str] = {}
        last_uncleared: Dict[int, int] = {}

        for attempt_idx in range(max_attempts):
            last_statuses = {}
            last_failures = {}
            last_uncleared = {}

            for actuator_id in ids:
                ok, status, reason, _ = self._query_once(int(actuator_id))
                if not ok or status is None:
                    last_failures[int(actuator_id)] = str(reason)
                    continue

                last_statuses[status.actuator_id] = status
                with self._state_lock:
                    self._last_status[status.actuator_id] = status

                if (status.error_bits & mask) != 0:
                    last_uncleared[status.actuator_id] = status.error_bits

            if not last_failures and not last_uncleared:
                return FaultClearVerification(
                    ok=True,
                    statuses_by_id=last_statuses,
                )

            if attempt_idx + 1 < max_attempts and settle_sec > 0.0:
                time.sleep(settle_sec)

        return FaultClearVerification(
            ok=False,
            statuses_by_id=last_statuses,
            failures_by_id=last_failures,
            uncleared_error_bits_by_id=last_uncleared,
        )

    def has_fault(self) -> bool:
        return self._fault_event.is_set()

    def get_fault(self) -> Optional[MonitorFault]:
        with self._state_lock:
            return self._last_fault

    def snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "poll_count": int(self._poll_count),
                "consecutive_failures": max(self._consecutive_failures_by_id.values(), default=0),
                "consecutive_failures_by_id": dict(self._consecutive_failures_by_id),
                "fault": self._last_fault,
                "last_status": {k: v for k, v in self._last_status.items()},
            }

    def _make_round_summary_line_locked(self) -> Optional[str]:
        num_axes = len(self.actuator_ids)
        if num_axes <= 0:
            return None

        log_every_n = self.monitor_cfg.log_every_n
        if log_every_n <= 0:
            return None
        if self._poll_count <= 0 or (self._poll_count % num_axes) != 0:
            return None

        round_count = self._poll_count // num_axes
        if round_count <= 0 or (round_count % log_every_n) != 0:
            return None

        parts = []
        for actuator_id in self.actuator_ids:
            failures = self._consecutive_failures_by_id.get(actuator_id, 0)
            status = self._last_status.get(actuator_id)
            if status is None:
                parts.append(f"id={actuator_id} NO_DATA fail={failures}")
                continue

            if failures > 0:
                parts.append(
                    f"id={actuator_id} fail={failures} "
                    f"last_tgt={status.target_count} last_cur={status.current_count} "
                    f"temp={status.temperature_c}C err=0x{status.error_bits:02X}"
                )
            else:
                parts.append(
                    f"id={actuator_id} tgt={status.target_count} cur={status.current_count} "
                    f"temp={status.temperature_c}C err=0x{status.error_bits:02X}"
                )

        return f"[Monitor] round={round_count} poll={self._poll_count} | " + " | ".join(parts)

    def _run_loop(self) -> None:
        period_sec = 1.0 / self.monitor_cfg.query_hz
        idle_sleep_sec = min(0.02, period_sec)

        while not self._stop_event.is_set():
            if not self._active_event.is_set():
                time.sleep(idle_sleep_sec)
                continue

            tick_start = time.monotonic()
            actuator_id = self.actuator_ids[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.actuator_ids)

            ok, status, reason, raw = self._query_once(actuator_id)

            if not self._active_event.is_set():
                with self._state_lock:
                    self._reset_failure_streaks_locked()
                continue

            if ok and status is not None:
                self._record_success(status)
                if (status.error_bits & self.monitor_cfg.error_mask) != 0:
                    self._raise_fault(
                        reason="ACTUATOR_ERROR_BITS",
                        actuator_id=status.actuator_id,
                        consecutive_failures=self._consecutive_failures_by_id.get(
                            status.actuator_id,
                            0,
                        ),
                        raw=status.raw_frame,
                        error_bits=status.error_bits,
                    )
            else:
                self._record_failure(reason=reason, actuator_id=actuator_id, raw=raw)

            elapsed = time.monotonic() - tick_start
            remain = period_sec - elapsed
            if remain > 0.0:
                time.sleep(remain)

    def _record_success(self, status: ActuatorStatus) -> None:
        summary_line = None
        with self._state_lock:
            self._poll_count += 1
            self._last_status[status.actuator_id] = status
            self._consecutive_failures_by_id[status.actuator_id] = 0
            summary_line = self._make_round_summary_line_locked()

        if summary_line is not None:
            print(summary_line)

    def _record_failure(self, reason: str, actuator_id: int, raw: bytes) -> None:
        summary_line = None
        with self._state_lock:
            self._poll_count += 1
            self._consecutive_failures_by_id[actuator_id] += 1
            failures = self._consecutive_failures_by_id[actuator_id]
            summary_line = self._make_round_summary_line_locked()

        if summary_line is not None:
            print(summary_line)

        if failures >= self.monitor_cfg.failure_threshold:
            self._raise_fault(
                reason=reason,
                actuator_id=actuator_id,
                consecutive_failures=failures,
                raw=raw,
            )

    def _raise_fault(
        self,
        reason: str,
        actuator_id: int,
        consecutive_failures: int,
        raw: bytes = b"",
        error_bits: int = 0,
    ) -> None:
        with self._state_lock:
            if self._fault_event.is_set():
                return
            self._last_fault = MonitorFault(
                reason=str(reason),
                actuator_id=int(actuator_id),
                consecutive_failures=int(consecutive_failures),
                raw=bytes(raw),
                error_bits=int(error_bits) & 0xFF,
                timestamp_sec=time.time(),
            )
            self._fault_event.set()

    def _query_once(self, actuator_id: int) -> Tuple[bool, Optional[ActuatorStatus], str, bytes]:
        if self.serial_link is None or not getattr(self.serial_link, "is_open", True):
            return False, None, "SERIAL_NOT_READY", b""

        query_frame = LAFrameBuilder.build_status_query_frame(actuator_id)
        original_timeout: Any = _SERIAL_TIMEOUT_UNSET
        timeout_overridden = False

        try:
            with self.serial_lock:
                original_timeout = getattr(self.serial_link, "timeout", _SERIAL_TIMEOUT_UNSET)
                if original_timeout is not _SERIAL_TIMEOUT_UNSET and original_timeout != 0.0:
                    self.serial_link.timeout = 0.0
                    timeout_overridden = True
                if hasattr(self.serial_link, "reset_input_buffer"):
                    self.serial_link.reset_input_buffer()
                try:
                    bytes_written = self.serial_link.write(query_frame)
                    if hasattr(self.serial_link, "flush"):
                        self.serial_link.flush()
                    if bytes_written != len(query_frame):
                        return False, None, "SHORT_WRITE", query_frame

                    status, response_frame, reason = self._read_status_response_locked(
                        self.monitor_cfg.response_timeout_sec,
                        expected_id=actuator_id,
                    )
                finally:
                    if timeout_overridden:
                        self.serial_link.timeout = original_timeout
        except Exception as exc:
            return False, None, f"SERIAL_EXCEPTION:{exc}", b""

        if status is None:
            return False, None, reason, response_frame

        return True, status, "OK", response_frame

    def _read_status_response_locked(
        self,
        timeout_sec: float,
        expected_id: Optional[int] = None,
    ) -> Tuple[Optional[ActuatorStatus], bytes, str]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        recv = bytearray()
        last_parse_error_reason: Optional[str] = None
        last_parse_error_frame = b""

        while time.monotonic() < deadline:
            read_size = 1
            try:
                in_waiting = int(getattr(self.serial_link, "in_waiting", 0))
            except Exception:
                in_waiting = 0
            if in_waiting > 0:
                read_size = min(256, in_waiting)

            chunk = self.serial_link.read(read_size)
            if chunk:
                recv.extend(chunk)
                while True:
                    frame, consumed = LAFrameParser.extract_first_response_frame_with_consumed(
                        bytes(recv)
                    )
                    if consumed > 0:
                        del recv[:consumed]
                    if frame is None:
                        break

                    try:
                        status = LAFrameParser.parse_status_response(frame)
                    except Exception as exc:
                        last_parse_error_reason = f"PARSE_ERROR:{exc}"
                        last_parse_error_frame = frame
                        continue

                    if expected_id is not None and status.actuator_id != int(expected_id):
                        continue
                    return status, frame, "OK"
            else:
                time.sleep(0.001)
        if last_parse_error_reason is not None:
            return None, last_parse_error_frame, last_parse_error_reason
        return None, b"", "READ_TIMEOUT"


def default_dagger_sim2real_runtime_config() -> DaggerSim2RealRuntimeConfig:
    bridge_cfg = Sim2RealConfig(
        J_4x2_mm_per_rad=DEFAULT_J_4X2_MM_PER_RAD.copy(),
        yaw_limit_deg=120.0,
        pitch_limit_deg=120.0,
        motor_limit_mm=15.0,
        control_hz=30.0,
        motor_order=("m1", "m2", "m3", "m4"),
        serial_port="/dev/ttyUSB0",
        serial_baudrate=115200,
        serial_timeout=0.0,
        serial_write_timeout=0.2,
        serial_critical_retry_count=3,
        serial_critical_retry_interval_sec=0.02,
    )
    actuator_cfg = ActuatorConfig(
        mode="broadcast_follow_no_feedback",
        ids=(1, 2, 3, 4),
        id_by_motor={"m1": 1, "m2": 2, "m3": 3, "m4": 4},
        position_index=0x37,
        count_min=0,
        count_max=2000,
        m1=MotorAxisMapConfig(zero_count=1000, count_per_mm=20.0, sign=1, soft_min_count=300, soft_max_count=1700),
        m2=MotorAxisMapConfig(zero_count=1000, count_per_mm=20.0, sign=1, soft_min_count=300, soft_max_count=1700),
        m3=MotorAxisMapConfig(zero_count=1000, count_per_mm=20.0, sign=1, soft_min_count=300, soft_max_count=1700),
        m4=MotorAxisMapConfig(zero_count=1000, count_per_mm=20.0, sign=1, soft_min_count=300, soft_max_count=1700),
    )
    alarm_cfg = EstopAlarmConfig(
        enabled=True,
        repeat=3,
        terminal_bell=True,
        banner_width=70,
    )
    monitor_cfg = MonitorConfig(
        enabled=True,
        query_hz=20.0,
        response_timeout_sec=0.02,
        failure_threshold=3,
        error_mask=0x0F,
        log_every_n=5,
    )
    output_cfg = MotorOutputConfig(
        print_tx_frame=False,
        print_every_n=100,
        save_csv=True,
        csv_path="./sim2real_motor_log.csv",
        save_plot=True,
        plot_path="./sim2real_motor_plot.png",
        plot_dpi=120,
        max_plot_points=4000,
    )
    return DaggerSim2RealRuntimeConfig(
        bridge=bridge_cfg,
        actuator=actuator_cfg,
        alarm=alarm_cfg,
        monitor=monitor_cfg,
        output=output_cfg,
    )


def load_dagger_sim2real_runtime_config(config_path: str | Path) -> DaggerSim2RealRuntimeConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load sim2real YAML config. Install with: pip install pyyaml"
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sim2Real config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, Mapping):
        raise ValueError("Top-level YAML content must be a mapping/object.")

    defaults = default_dagger_sim2real_runtime_config()
    sim2real_raw = raw.get("sim2real", {})
    if not isinstance(sim2real_raw, Mapping):
        raise ValueError("`sim2real` must be a mapping/object.")

    serial_raw = sim2real_raw.get("serial", {})
    if serial_raw is None:
        serial_raw = {}
    if not isinstance(serial_raw, Mapping):
        raise ValueError("`sim2real.serial` must be a mapping/object.")

    alarm_raw = sim2real_raw.get("alarm", {})
    if alarm_raw is None:
        alarm_raw = {}
    if not isinstance(alarm_raw, Mapping):
        raise ValueError("`sim2real.alarm` must be a mapping/object.")

    output_raw = sim2real_raw.get("output", {})
    if output_raw is None:
        output_raw = {}
    if not isinstance(output_raw, Mapping):
        raise ValueError("`sim2real.output` must be a mapping/object.")

    monitor_raw = sim2real_raw.get("monitor", {})
    if monitor_raw is None:
        monitor_raw = {}
    if not isinstance(monitor_raw, Mapping):
        raise ValueError("`sim2real.monitor` must be a mapping/object.")

    actuator_raw = sim2real_raw.get("actuator", None)
    if actuator_raw is None:
        raise ValueError("`sim2real.actuator` is required for direct LA actuator control.")
    if not isinstance(actuator_raw, Mapping):
        raise ValueError("`sim2real.actuator` must be a mapping/object.")

    motor_order_raw = sim2real_raw.get("motor_order", defaults.bridge.motor_order)
    if isinstance(motor_order_raw, str):
        raise ValueError("`sim2real.motor_order` must be a list containing m1,m2,m3,m4 in some order.")
    motor_order = tuple(str(x).strip() for x in motor_order_raw)
    if len(motor_order) != 4:
        raise ValueError(f"`sim2real.motor_order` must have 4 entries, got {len(motor_order)}")
    if set(motor_order) != set(MOTOR_NAME_SET):
        raise ValueError("`sim2real.motor_order` must be a permutation of ['m1','m2','m3','m4']")

    bridge_cfg = Sim2RealConfig(
        J_4x2_mm_per_rad=_coerce_matrix(
            sim2real_raw.get("J_4x2_mm_per_rad", defaults.bridge.J_4x2_mm_per_rad)
        ),
        yaw_limit_deg=float(sim2real_raw.get("yaw_limit_deg", defaults.bridge.yaw_limit_deg)),
        pitch_limit_deg=float(sim2real_raw.get("pitch_limit_deg", defaults.bridge.pitch_limit_deg)),
        motor_limit_mm=float(sim2real_raw.get("motor_limit_mm", defaults.bridge.motor_limit_mm)),
        control_hz=float(sim2real_raw.get("control_hz", defaults.bridge.control_hz)),
        motor_order=motor_order,
        serial_port=str(serial_raw.get("port", defaults.bridge.serial_port)),
        serial_baudrate=int(serial_raw.get("baudrate", defaults.bridge.serial_baudrate)),
        serial_timeout=float(serial_raw.get("timeout", defaults.bridge.serial_timeout)),
        serial_write_timeout=float(
            serial_raw.get("write_timeout", defaults.bridge.serial_write_timeout)
        ),
        serial_critical_retry_count=int(
            serial_raw.get("critical_retry_count", defaults.bridge.serial_critical_retry_count)
        ),
        serial_critical_retry_interval_sec=float(
            serial_raw.get(
                "critical_retry_interval_sec",
                defaults.bridge.serial_critical_retry_interval_sec,
            )
        ),
    )

    per_motor_raw = actuator_raw.get("per_motor", None)
    if per_motor_raw is None or not isinstance(per_motor_raw, Mapping):
        raise ValueError("`sim2real.actuator.per_motor` must be a mapping and include m1~m4.")

    id_by_motor_raw = actuator_raw.get("id_by_motor", None)
    if id_by_motor_raw is None or not isinstance(id_by_motor_raw, Mapping):
        raise ValueError("`sim2real.actuator.id_by_motor` must be a mapping and include m1~m4.")

    actuator_cfg = ActuatorConfig(
        mode=str(actuator_raw.get("mode", defaults.actuator.mode)),
        ids=tuple(int(x) for x in actuator_raw.get("ids", defaults.actuator.ids)),
        id_by_motor=_load_id_by_motor_cfg(id_by_motor_raw),
        position_index=_coerce_int(actuator_raw.get("position_index", defaults.actuator.position_index)),
        count_min=int(actuator_raw.get("count_min", defaults.actuator.count_min)),
        count_max=int(actuator_raw.get("count_max", defaults.actuator.count_max)),
        m1=_load_axis_map_cfg(per_motor_raw, "m1", defaults.actuator.m1),
        m2=_load_axis_map_cfg(per_motor_raw, "m2", defaults.actuator.m2),
        m3=_load_axis_map_cfg(per_motor_raw, "m3", defaults.actuator.m3),
        m4=_load_axis_map_cfg(per_motor_raw, "m4", defaults.actuator.m4),
    )

    alarm_cfg = EstopAlarmConfig(
        enabled=bool(alarm_raw.get("enabled", defaults.alarm.enabled)),
        repeat=int(alarm_raw.get("repeat", defaults.alarm.repeat)),
        terminal_bell=bool(alarm_raw.get("terminal_bell", defaults.alarm.terminal_bell)),
        banner_width=int(alarm_raw.get("banner_width", defaults.alarm.banner_width)),
    )

    monitor_cfg = MonitorConfig(
        enabled=bool(monitor_raw.get("enabled", defaults.monitor.enabled)),
        query_hz=float(monitor_raw.get("query_hz", defaults.monitor.query_hz)),
        response_timeout_sec=float(
            monitor_raw.get("response_timeout_sec", defaults.monitor.response_timeout_sec)
        ),
        failure_threshold=int(
            monitor_raw.get("failure_threshold", defaults.monitor.failure_threshold)
        ),
        error_mask=_coerce_int(monitor_raw.get("error_mask", defaults.monitor.error_mask)),
        log_every_n=int(monitor_raw.get("log_every_n", defaults.monitor.log_every_n)),
    )

    output_cfg = MotorOutputConfig(
        print_tx_frame=bool(output_raw.get("print_tx_frame", defaults.output.print_tx_frame)),
        print_every_n=int(output_raw.get("print_every_n", defaults.output.print_every_n)),
        save_csv=bool(output_raw.get("save_csv", defaults.output.save_csv)),
        csv_path=str(output_raw.get("csv_path", defaults.output.csv_path)),
        save_plot=bool(output_raw.get("save_plot", defaults.output.save_plot)),
        plot_path=str(output_raw.get("plot_path", defaults.output.plot_path)),
        plot_dpi=int(output_raw.get("plot_dpi", defaults.output.plot_dpi)),
        max_plot_points=int(output_raw.get("max_plot_points", defaults.output.max_plot_points)),
    )

    return DaggerSim2RealRuntimeConfig(
        bridge=bridge_cfg,
        actuator=actuator_cfg,
        alarm=alarm_cfg,
        monitor=monitor_cfg,
        output=output_cfg,
    )


def _coerce_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 2):
        raise ValueError(f"`sim2real.J_4x2_mm_per_rad` must have shape (4,2), got {arr.shape}")
    return arr


def _coerce_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _load_id_by_motor_cfg(raw: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in MOTOR_NAME_SET:
        if name not in raw:
            raise ValueError(f"`sim2real.actuator.id_by_motor.{name}` is required.")
        out[name] = _coerce_int(raw[name])
    return out


def _load_axis_map_cfg(
    per_motor_raw: Mapping[str, Any],
    axis_name: str,
    defaults: MotorAxisMapConfig,
) -> MotorAxisMapConfig:
    axis_raw = per_motor_raw.get(axis_name, None)
    if axis_raw is None:
        raise ValueError(f"`sim2real.actuator.per_motor.{axis_name}` is required.")
    if not isinstance(axis_raw, Mapping):
        raise ValueError(f"`sim2real.actuator.per_motor.{axis_name}` must be a mapping/object.")

    return MotorAxisMapConfig(
        zero_count=int(axis_raw.get("zero_count", defaults.zero_count)),
        count_per_mm=float(axis_raw.get("count_per_mm", defaults.count_per_mm)),
        sign=int(axis_raw.get("sign", defaults.sign)),
        soft_min_count=int(axis_raw.get("soft_min_count", defaults.soft_min_count)),
        soft_max_count=int(axis_raw.get("soft_max_count", defaults.soft_max_count)),
    )
