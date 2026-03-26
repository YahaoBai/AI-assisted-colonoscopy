from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

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
    serial_port: str = "/dev/pts/3"
    serial_baudrate: int = 115200
    serial_timeout: float = 0.0
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
class DaggerSim2RealRuntimeConfig:
    bridge: Sim2RealConfig
    alarm: EstopAlarmConfig
    output: MotorOutputConfig


@dataclass
class BridgeResult:
    command: BridgeCommand
    serial_frame: str
    should_send: bool = False
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
        self.cmd_send_indices = tuple(MOTOR_NAME_SET.index(name) for name in self.config.motor_order)

    def _motor_target_for_cmd_order(self, motor_target_mm: np.ndarray) -> np.ndarray:
        return np.asarray(motor_target_mm, dtype=np.float64)[list(self.cmd_send_indices)]

    def _make_estop(self, reason: str) -> BridgeResult:
        return BridgeResult(
            command=BridgeCommand.ESTOP,
            serial_frame=self.build_estop_frame(reason),
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

        cmd_frame = (
            self.build_cmd_frame(self._motor_target_for_cmd_order(self.motor_target_mm).tolist())
            if should_send
            else ""
        )

        return BridgeResult(
            command=BridgeCommand.CMD,
            serial_frame=cmd_frame,
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

    def get_reset_frames(self) -> Tuple[str, str]:
        zero_target_cmd_order = self._motor_target_for_cmd_order(np.zeros(4, dtype=np.float64))
        return (
            self.build_reset_frame(),
            self.build_cmd_frame(zero_target_cmd_order.tolist()),
        )

    @staticmethod
    def build_cmd_frame(motor_target_mm: Sequence[float]) -> str:
        if len(motor_target_mm) != 4:
            raise ValueError(f"motor_target_mm must contain 4 entries, got {len(motor_target_mm)}")
        return "CMD,{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
            float(motor_target_mm[0]),
            float(motor_target_mm[1]),
            float(motor_target_mm[2]),
            float(motor_target_mm[3]),
        )

    @staticmethod
    def build_estop_frame(reason: str = "ANGLE_LIMIT") -> str:
        return f"ESTOP,{reason}\n"

    @staticmethod
    def build_reset_frame() -> str:
        return "RESET\n"


def default_dagger_sim2real_runtime_config() -> DaggerSim2RealRuntimeConfig:
    bridge_cfg = Sim2RealConfig(
        J_4x2_mm_per_rad=DEFAULT_J_4X2_MM_PER_RAD.copy(),
        yaw_limit_deg=120.0,
        pitch_limit_deg=120.0,
        motor_limit_mm=15.0,
        control_hz=30.0,
        motor_order=("m1", "m2", "m3", "m4"),
        serial_port="/dev/pts/3",
        serial_baudrate=115200,
        serial_timeout=0.0,
        serial_critical_retry_count=3,
        serial_critical_retry_interval_sec=0.02,
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
    return DaggerSim2RealRuntimeConfig(bridge=bridge_cfg, alarm=alarm_cfg, output=output_cfg)


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

    return DaggerSim2RealRuntimeConfig(bridge=bridge_cfg, alarm=alarm_cfg, output=output_cfg)


def _coerce_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 2):
        raise ValueError(f"`sim2real.J_4x2_mm_per_rad` must have shape (4,2), got {arr.shape}")
    return arr
