from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TeleopConfig:
    """
    teleop 运行时配置模型。

    公开（推荐）参数面：
    - controller_index
    - invert_yaw
    - invert_pitch
    - disconnect_timeout_sec
    - allow_feed_without_actuator

    其余字段用于内部默认或兼容旧配置，不建议新配置直接使用。
    """

    # ---- Public ----
    controller_index: int = 0
    invert_yaw: bool = False
    invert_pitch: bool = True
    disconnect_timeout_sec: float = 0.5
    # 显式开启“无电缸仅滑台”模式（默认关闭，避免影响原流程）
    allow_feed_without_actuator: bool = False

    # ---- Internal defaults / backward compatibility ----
    backend: str = "pygame"
    controller_name_contains: str = "Wireless Controller"
    left_stick_x_axis: int = 0
    left_stick_y_axis: int = 1
    forward_button: str = "north"
    deprecation_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.backend = str(self.backend).strip().lower()
        if self.backend != "pygame":
            raise ValueError("teleop.backend currently supports only 'pygame'.")

        self.controller_name_contains = str(self.controller_name_contains).strip()
        self.controller_index = int(self.controller_index)
        self.left_stick_x_axis = int(self.left_stick_x_axis)
        self.left_stick_y_axis = int(self.left_stick_y_axis)
        self.invert_yaw = bool(self.invert_yaw)
        self.invert_pitch = bool(self.invert_pitch)
        self.allow_feed_without_actuator = bool(self.allow_feed_without_actuator)
        self.forward_button = str(self.forward_button).strip().lower()
        self.disconnect_timeout_sec = float(self.disconnect_timeout_sec)
        self.deprecation_warnings = tuple(str(x) for x in self.deprecation_warnings)

        if self.disconnect_timeout_sec <= 0.0:
            raise ValueError("teleop.disconnect_timeout_sec must be > 0.")


def _load_raw_teleop_mapping(config_path: str | Path) -> Mapping[str, Any]:
    """
    读取 YAML 并提取 sim2real.teleop 节点。
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, Mapping):
        raise ValueError("Top-level YAML content must be a mapping/object.")

    sim2real_raw = raw.get("sim2real", {})
    if sim2real_raw is None:
        sim2real_raw = {}
    if not isinstance(sim2real_raw, Mapping):
        raise ValueError("`sim2real` must be a mapping/object.")

    teleop_raw = sim2real_raw.get("teleop", {})
    if teleop_raw is None:
        teleop_raw = {}
    if not isinstance(teleop_raw, Mapping):
        raise ValueError("`sim2real.teleop` must be a mapping/object.")

    return teleop_raw


def _warn_deprecated_config_field(
    warnings: list[str],
    field_name: str,
    replacement_hint: str,
) -> None:
    warnings.append(
        f"`sim2real.teleop.{field_name}` is deprecated, use `{replacement_hint}`."
    )


def _build_effective_teleop_config(teleop_raw: Mapping[str, Any]) -> TeleopConfig:
    """
    将公共参数与 legacy 字段合并为运行时配置（集中兼容映射入口）。
    """
    defaults = TeleopConfig()
    warnings: list[str] = []

    removed_fields = (
        "deadzone",
        "max_yaw_rate_rad_s",
        "max_pitch_rate_rad_s",
        "hold_sec",
        "estop_button",
        "reset_combo",
        "estop_hold_sec",
        "reset_hold_sec",
    )
    removed_hits = [name for name in removed_fields if name in teleop_raw]
    if removed_hits:
        text = ", ".join(removed_hits)
        raise ValueError(
            "Removed teleop fields are not supported: "
            f"{text}. Teleop now uses direct mm mapping and no in-app manual estop/reset fields."
        )

    # ---- Public ----
    controller_index = int(teleop_raw.get("controller_index", defaults.controller_index))
    invert_yaw = bool(teleop_raw.get("invert_yaw", defaults.invert_yaw))
    invert_pitch = bool(teleop_raw.get("invert_pitch", defaults.invert_pitch))
    disconnect_timeout_sec = float(
        teleop_raw.get("disconnect_timeout_sec", defaults.disconnect_timeout_sec)
    )
    allow_feed_without_actuator = bool(
        teleop_raw.get(
            "allow_feed_without_actuator",
            defaults.allow_feed_without_actuator,
        )
    )

    # ---- Legacy compatibility ----
    backend = defaults.backend
    if "backend" in teleop_raw:
        backend = str(teleop_raw.get("backend", defaults.backend))
        _warn_deprecated_config_field(warnings, "backend", "fixed internal default")

    controller_name_contains = defaults.controller_name_contains
    if "controller_name_contains" in teleop_raw:
        controller_name_contains = str(
            teleop_raw.get("controller_name_contains", defaults.controller_name_contains)
        )
        _warn_deprecated_config_field(
            warnings,
            "controller_name_contains",
            "fixed internal default",
        )

    left_stick_x_axis = defaults.left_stick_x_axis
    if "left_stick_x_axis" in teleop_raw:
        left_stick_x_axis = int(teleop_raw.get("left_stick_x_axis", defaults.left_stick_x_axis))
        _warn_deprecated_config_field(warnings, "left_stick_x_axis", "fixed internal default")

    left_stick_y_axis = defaults.left_stick_y_axis
    if "left_stick_y_axis" in teleop_raw:
        left_stick_y_axis = int(teleop_raw.get("left_stick_y_axis", defaults.left_stick_y_axis))
        _warn_deprecated_config_field(warnings, "left_stick_y_axis", "fixed internal default")

    forward_button = defaults.forward_button
    if "forward_button" in teleop_raw:
        forward_button = str(teleop_raw.get("forward_button", defaults.forward_button))
        _warn_deprecated_config_field(warnings, "forward_button", "fixed internal default")

    return TeleopConfig(
        controller_index=controller_index,
        invert_yaw=invert_yaw,
        invert_pitch=invert_pitch,
        disconnect_timeout_sec=disconnect_timeout_sec,
        allow_feed_without_actuator=allow_feed_without_actuator,
        backend=backend,
        controller_name_contains=controller_name_contains,
        left_stick_x_axis=left_stick_x_axis,
        left_stick_y_axis=left_stick_y_axis,
        forward_button=forward_button,
        deprecation_warnings=tuple(warnings),
    )


def load_teleop_config(config_path: str | Path) -> TeleopConfig:
    """
    从 YAML 加载 teleop 配置（公共参数 + 部分旧字段兼容）。
    """
    teleop_raw = _load_raw_teleop_mapping(config_path)
    return _build_effective_teleop_config(teleop_raw)
