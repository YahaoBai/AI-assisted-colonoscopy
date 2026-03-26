from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

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
    ) -> None:
        self.serial_link = serial_link
        self.critical_retry_count = max(1, int(critical_retry_count))
        self.critical_retry_interval_sec = max(0.0, float(critical_retry_interval_sec))
        self.print_tx_frame = bool(print_tx_frame)

    def send_frame(self, frame: bytes, frame_name: str = "FRAME") -> bool:
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

    def send_critical(self, frames: Sequence[bytes], frame_name: str) -> bool:
        frame_seq = [bytes(x) for x in frames]
        if len(frame_seq) == 0:
            return True

        for attempt in range(1, self.critical_retry_count + 1):
            ok = True
            for idx, frame in enumerate(frame_seq):
                if not self.send_frame(frame, f"{frame_name}[{idx}]"):
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
