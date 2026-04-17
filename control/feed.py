"""
单滑台进给控制脚本（Emm_V5 协议，最小交互版）。

职责边界:
- 本脚本只负责“滑台进给”。
- 不复用电缸 yaw/pitch 控制协议。

交互命令:
- f            前进一次（+step_pulses，相对模式）
- b            后退一次（-backward_step_pulses，相对模式）
- h            打印帮助

CLI 参数:
- --dry-run    不下发串口，仅打印帧
- --list-ports 扫描串口并退出

退出策略:
- 仅支持 Ctrl+C 退出。
- 退出时仅释放串口，不主动下发失能。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

try:
    import serial
except ImportError:  # pragma: no cover - import guard only
    serial = None


DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("sim2real_config.yaml"))
CHK = 0x6B


class SerialLink(Protocol):
    def write(self, data: bytes) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def log_event(event: str, **fields: Any) -> None:
    parts = [f"[FEED][{event}]"]
    for key in sorted(fields.keys()):
        parts.append(f"{key}={fields[key]}")
    print(" ".join(parts))


@dataclass
class FeedConfig:
    config_path: str = DEFAULT_CONFIG_PATH
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.05
    addr: int = 1

    microstep: int = 16
    steps_per_rev: int = 200

    step_pulses: int = 200
    backward_step_pulses: Optional[int] = None
    repeat_hz: float = 10.0
    default_vel: int = 100
    default_acc: int = 0

    enable_on_start: bool = True
    teleop_enabled: bool = False

    dry_run: bool = False

    def validate(self) -> None:
        self.port = str(self.port)
        self.baudrate = int(self.baudrate)
        self.timeout = float(self.timeout)
        self.addr = int(self.addr)
        self.microstep = int(self.microstep)
        self.steps_per_rev = int(self.steps_per_rev)
        self.step_pulses = int(self.step_pulses)
        if self.backward_step_pulses is None:
            self.backward_step_pulses = self.step_pulses
        else:
            self.backward_step_pulses = int(self.backward_step_pulses)
        self.repeat_hz = float(self.repeat_hz)
        self.default_vel = int(self.default_vel)
        self.default_acc = int(self.default_acc)
        self.enable_on_start = bool(self.enable_on_start)
        self.teleop_enabled = bool(self.teleop_enabled)
        self.dry_run = bool(self.dry_run)

        if not (1 <= self.addr <= 255):
            raise ValueError("feed.addr must be in [1, 255]")
        if self.baudrate <= 0:
            raise ValueError("feed.baudrate must be > 0")
        if self.timeout < 0.0:
            raise ValueError("feed.timeout must be >= 0")
        if self.microstep <= 0:
            raise ValueError("feed.microstep must be > 0")
        if self.steps_per_rev <= 0:
            raise ValueError("feed.steps_per_rev must be > 0")
        if self.step_pulses <= 0:
            raise ValueError("feed.step_pulses must be > 0")
        if int(self.backward_step_pulses) <= 0:
            raise ValueError("feed.backward_step_pulses must be > 0")
        if self.repeat_hz <= 0.0 or self.repeat_hz > 100.0:
            raise ValueError("feed.repeat_hz must be in (0, 100]")
        if not (0 <= self.default_vel <= 0xFFFF):
            raise ValueError("feed.default_vel must be in [0, 65535]")
        if not (0 <= self.default_acc <= 0xFF):
            raise ValueError("feed.default_acc must be in [0, 255]")


# -------------------------
# Emm_V5 frame builders
# -------------------------


def build_en_control_frame(addr: int, state: bool, sync_start: bool = False) -> bytes:
    """
    使能/失能控制帧。

    格式（来自参考代码）:
    [addr, 0xF3, 0xAB, state, snF, 0x6B]
    """
    return bytes(
        [
            int(addr) & 0xFF,
            0xF3,
            0xAB,
            0x01 if bool(state) else 0x00,
            0x01 if bool(sync_start) else 0x00,
            CHK,
        ]
    )


def build_pos_control_frame(
    addr: int,
    dir_flag: int,
    vel: int,
    acc: int,
    clk: int,
    relative_mode: bool = True,
    sync_start: bool = False,
) -> bytes:
    """
    位置控制帧（13 字节）。

    格式（来自参考代码）:
    [addr, 0xFD, dir, velH, velL, acc, clk3, clk2, clk1, clk0, raF, snF, 0x6B]

    其中:
    - dir: 0=正向(CW), 1=反向(CCW)
    - raF: 0=相对模式, 1=绝对模式
    """
    addr_v = int(addr)
    dir_v = int(dir_flag)
    vel_v = int(vel)
    acc_v = int(acc)
    clk_v = int(clk)

    if not (1 <= addr_v <= 255):
        raise ValueError("addr must be in [1, 255]")
    if dir_v not in (0, 1):
        raise ValueError("dir_flag must be 0 or 1")
    if not (0 <= vel_v <= 0xFFFF):
        raise ValueError("vel must be in [0, 65535]")
    if not (0 <= acc_v <= 0xFF):
        raise ValueError("acc must be in [0, 255]")
    if not (0 <= clk_v <= 0xFFFFFFFF):
        raise ValueError("clk must be in [0, 4294967295]")

    raF = 0x00 if bool(relative_mode) else 0x01
    snF = 0x01 if bool(sync_start) else 0x00

    return bytes(
        [
            addr_v & 0xFF,
            0xFD,
            dir_v & 0xFF,
            (vel_v >> 8) & 0xFF,
            vel_v & 0xFF,
            acc_v & 0xFF,
            (clk_v >> 24) & 0xFF,
            (clk_v >> 16) & 0xFF,
            (clk_v >> 8) & 0xFF,
            clk_v & 0xFF,
            raF,
            snF,
            CHK,
        ]
    )


def resolve_direction_flag(forward: bool) -> int:
    """
    固定方向位:
    - forward=True  -> dir=0（前进）
    - forward=False -> dir=1（后退）
    """
    return 0 if bool(forward) else 1


def resolve_step_pulses(cfg: FeedConfig, forward: bool) -> int:
    """
    根据方向解析本次应发送的步长。
    """
    return int(cfg.step_pulses) if bool(forward) else int(cfg.backward_step_pulses)


# -------------------------
# Config loading
# -------------------------


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _load_feed_raw(config_path: str) -> Mapping[str, Any]:
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

    feed_raw = sim2real_raw.get("feed", {})
    if feed_raw is None:
        feed_raw = {}
    if not isinstance(feed_raw, Mapping):
        raise ValueError("`sim2real.feed` must be a mapping/object.")

    return feed_raw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-slider feed control (Emm_V5 protocol, minimal interactive mode)."
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Scan available serial ports and exit",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print frames without serial writes")
    return parser


def resolve_feed_config(dry_run: bool, config_path: str = DEFAULT_CONFIG_PATH) -> FeedConfig:
    feed_raw = _load_feed_raw(str(config_path))
    removed_fields = ("invert_dir", "min_pulses", "max_pulses")
    removed_hits = [name for name in removed_fields if name in feed_raw]
    if removed_hits:
        text = ", ".join(removed_hits)
        raise ValueError(
            "Removed feed fields are not supported: "
            f"{text}. Feed now uses fixed direction (Y/B) and no software limit."
        )

    cfg = FeedConfig(
        config_path=str(config_path),
        port=str(feed_raw.get("port", "/dev/ttyUSB0")),
        baudrate=int(feed_raw.get("baudrate", 115200)),
        timeout=float(feed_raw.get("timeout", 0.05)),
        addr=int(feed_raw.get("addr", 1)),
        microstep=int(feed_raw.get("microstep", 16)),
        steps_per_rev=int(feed_raw.get("steps_per_rev", 200)),
        step_pulses=int(feed_raw.get("step_pulses", 200)),
        backward_step_pulses=feed_raw.get("backward_step_pulses"),
        repeat_hz=float(feed_raw.get("repeat_hz", 10.0)),
        default_vel=int(feed_raw.get("default_vel", 100)),
        default_acc=int(feed_raw.get("default_acc", 0)),
        enable_on_start=_coerce_bool(feed_raw.get("enable_on_start", True), True),
        teleop_enabled=_coerce_bool(feed_raw.get("teleop_enabled", False), False),
        dry_run=bool(dry_run),
    )

    cfg.validate()
    return cfg


# -------------------------
# Runtime
# -------------------------


def list_serial_ports() -> int:
    if serial is None:
        log_event("FATAL", reason="pyserial is required. Install with: pip install pyserial")
        return 1

    try:
        from serial.tools import list_ports
    except Exception as exc:
        log_event("FATAL", reason=f"list_ports_unavailable: {exc}")
        return 1

    ports = sorted(list_ports.comports(), key=lambda item: item.device or "")
    if not ports:
        print("Detected serial ports: (none)")
        return 0

    print("Detected serial ports:")
    for idx, port in enumerate(ports):
        print(
            f"  - index={idx} device={port.device} description={port.description} hwid={port.hwid}"
        )
    return 0


def _send_frame(
    serial_link: Optional[SerialLink],
    frame: bytes,
    dry_run: bool,
    event: str,
) -> bool:
    if dry_run:
        log_event(event, frame=frame.hex(" ").upper())
        return True

    if serial_link is None:
        log_event("TX_FAIL", reason="SERIAL_NOT_READY")
        return False

    try:
        wrote = serial_link.write(frame)
        serial_link.flush()
    except Exception as exc:
        log_event("TX_FAIL", reason=str(exc))
        return False

    if wrote != len(frame):
        log_event("TX_FAIL", reason=f"SHORT_WRITE wrote={wrote} expected={len(frame)}")
        return False

    log_event(event, frame=frame.hex(" ").upper())
    return True


def _send_enable(serial_link: Optional[SerialLink], cfg: FeedConfig, state: bool) -> bool:
    frame = build_en_control_frame(addr=cfg.addr, state=state, sync_start=False)
    return _send_frame(
        serial_link=serial_link,
        frame=frame,
        dry_run=cfg.dry_run,
        event="ENABLE" if state else "DISABLE",
    )


def _send_step_once(
    serial_link: Optional[SerialLink],
    cfg: FeedConfig,
    step_pulses: int,
    forward: bool,
) -> bool:
    dir_flag = resolve_direction_flag(forward=forward)
    frame = build_pos_control_frame(
        addr=cfg.addr,
        dir_flag=dir_flag,
        vel=cfg.default_vel,
        acc=cfg.default_acc,
        clk=step_pulses,
        relative_mode=True,
        sync_start=False,
    )
    event = "FORWARD_TX" if forward else "BACKWARD_TX"
    return _send_frame(serial_link=serial_link, frame=frame, dry_run=cfg.dry_run, event=event)


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  f            forward one step (step_pulses)\n"
        "  b            backward one step (backward_step_pulses)\n"
        "  h            show help\n"
        "  Ctrl+C       exit\n"
    )


def run_feed(cfg: FeedConfig) -> int:
    serial_link = None
    if not cfg.dry_run:
        if serial is None:
            raise RuntimeError("pyserial is required. Install with: pip install pyserial")
        serial_link = serial.Serial(
            cfg.port,
            baudrate=cfg.baudrate,
            timeout=cfg.timeout,
            write_timeout=0.2,
        )

    # 软件侧“相对位移累计计数”（仅日志显示，不做限位拦截）。
    current_pulses = 0
    forward_step_pulses = int(cfg.step_pulses)
    backward_step_pulses = int(cfg.backward_step_pulses)

    log_event(
        "READY",
        mode="DRY_RUN" if cfg.dry_run else "HARDWARE",
        port=cfg.port,
        addr=cfg.addr,
        forward_step=forward_step_pulses,
        backward_step=backward_step_pulses,
        current=current_pulses,
    )

    if cfg.enable_on_start:
        if not _send_enable(serial_link, cfg, state=True):
            log_event("ABORT", reason="ENABLE_ON_START_FAIL")
            return 2

    _print_help()

    try:
        while True:
            cmd = input("feed> ").strip().lower()
            if not cmd:
                continue

            if cmd in {"h", "help", "?"}:
                _print_help()
                continue

            if cmd == "f":
                step_pulses = resolve_step_pulses(cfg, forward=True)
                if not _send_step_once(
                    serial_link,
                    cfg,
                    step_pulses=step_pulses,
                    forward=True,
                ):
                    log_event("ABORT", reason="FORWARD_TX_FAIL")
                    return 3

                current_pulses += step_pulses
                log_event("FORWARD_OK", current=current_pulses, step=step_pulses)
                continue

            if cmd == "b":
                step_pulses = resolve_step_pulses(cfg, forward=False)
                if not _send_step_once(
                    serial_link,
                    cfg,
                    step_pulses=step_pulses,
                    forward=False,
                ):
                    log_event("ABORT", reason="BACKWARD_TX_FAIL")
                    return 3

                current_pulses -= step_pulses
                log_event("BACKWARD_OK", current=current_pulses, step=step_pulses)
                continue

            log_event("BAD_CMD", reason="UNKNOWN", value=cmd)

    except EOFError:
        log_event("STOP", reason="STDIN_EOF")
        return 0
    except KeyboardInterrupt:
        log_event("STOP", reason="KEYBOARD_INTERRUPT")
        return 130
    finally:
        if serial_link is not None:
            try:
                serial_link.close()
            except Exception:
                pass


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        if bool(args.list_ports):
            return list_serial_ports()
        cfg = resolve_feed_config(dry_run=bool(args.dry_run))
        return run_feed(cfg)
    except Exception as exc:
        log_event("FATAL", reason=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
