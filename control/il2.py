#####################

#这个脚本用来实机验证

#####################



from __future__ import annotations

import csv
import math
import subprocess
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime environment dependent
    cv2 = None
    _CV2_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from perception.api import predict
except Exception as exc:  # pragma: no cover - runtime environment dependent
    predict = None
    _POLICY_IMPORT_ERROR: Optional[Exception] = exc
else:
    _POLICY_IMPORT_ERROR = None

try:
    from perception.lumen_center_api import get_lumen_center, init as init_lumen_model
except Exception as exc:  # pragma: no cover - runtime environment dependent
    get_lumen_center = None
    init_lumen_model = None
    _LUMEN_IMPORT_ERROR: Optional[Exception] = exc
else:
    _LUMEN_IMPORT_ERROR = None

from control.sim2real_bridge import (
    ActuatorMonitor,
    ActuatorTx,
    BridgeCommand,
    BridgeResult,
    DaggerSim2RealRuntimeConfig,
    MotorMapper,
    Sim2RealBridge,
    load_dagger_sim2real_runtime_config,
)
from control.teleop_config import load_teleop_config
from control.teleop_input import GamepadSample, PygameGamepadInput
from control.teleop_runtime import (
    FeedRuntimeState,
    TeleopRuntimeContext,
    _cleanup_runtime,
    _compute_direct_motor_target_mm,
    _feed_send_enable,
    _handle_feed_step_once,
    _setup_actuator_io,
    _setup_feed_runtime,
    perform_reset_sequence,
)


SIM2REAL_CONFIG_PATH = Path(__file__).with_name("sim2real_config.yaml")
CAMERA_DEVICE_INDEX = 2
WINDOW_NAME = "IL2 Camera Stream"
DEFAULT_OUTPUT_ROOT = Path("camera_capture")
OUTPUT_DIR = ""
SAVE_SESSION_ARTIFACTS = True
POLICY_IMAGE_SIZE = 256
MAX_CAMERA_READ_FAILURES = 5
FOLLOW_TX_LOG_EVERY_N = 30
EXIT_DIAGNOSTICS_FILENAME = "il2_exit_diagnostics.txt"

CSV_FIELDNAMES = [
    "frame_idx",
    "timestamp_sec",
    "image_width",
    "image_height",
    "scope_center_x",
    "scope_center_y",
    "lumen_center_x",
    "lumen_center_y",
    "error_x_px",
    "error_y_px",
    "error_norm_px",
    "inference_ms",
    "status",
    "control_mode",
    "autopilot_on",
    "policy_step_yaw",
    "policy_step_pitch",
    "bridge_command",
    "estop_latched",
    "forward_pressed",
    "backward_pressed",
    "hold_pressed",
    "manual_hold_active",
    "feed_delta_pulses",
    "m1_target_mm",
    "m2_target_mm",
    "m3_target_mm",
    "m4_target_mm",
]


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _classify_exception_reason(prefix: str, exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return f"{prefix}_RAM_OOM"

    message = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in message or "cannot allocate memory" in message or "std::bad_alloc" in message:
        if "cuda" in message or "cudnn" in message:
            return f"{prefix}_CUDA_OOM"
        return f"{prefix}_RAM_OOM"
    return prefix


def _read_proc_status_snapshot() -> dict[str, int]:
    snapshot: dict[str, int] = {}
    path = Path("/proc/self/status")
    if not path.exists():
        return snapshot

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if len(fields) == 0:
                continue
            try:
                raw = int(fields[0])
            except ValueError:
                continue
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                snapshot[key] = raw * 1024
            elif key == "Threads":
                snapshot[key] = raw
    except Exception:
        return {}

    return snapshot


def _read_proc_meminfo_snapshot() -> dict[str, int]:
    snapshot: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return snapshot

    wanted = {"MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key not in wanted:
                continue
            fields = value.strip().split()
            if len(fields) == 0:
                continue
            try:
                snapshot[key] = int(fields[0]) * 1024
            except ValueError:
                continue
    except Exception:
        return {}

    return snapshot


def _collect_torch_cuda_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}

    try:
        import torch
    except Exception as exc:
        snapshot["backend_error"] = f"torch unavailable: {exc}"
        return snapshot

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        snapshot["backend_error"] = f"torch.cuda probe failed: {exc}"
        return snapshot

    snapshot["cuda_available"] = cuda_available
    if not cuda_available:
        return snapshot

    try:
        device_idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_idx)
        snapshot["device_idx"] = device_idx
        snapshot["device_name"] = str(props.name)
        snapshot["allocated_bytes"] = int(torch.cuda.memory_allocated(device_idx))
        snapshot["reserved_bytes"] = int(torch.cuda.memory_reserved(device_idx))
        snapshot["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_idx))
        snapshot["total_bytes"] = int(props.total_memory)
        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            snapshot["free_bytes"] = int(free_bytes)
            snapshot["total_bytes"] = int(total_bytes)
    except Exception as exc:
        snapshot["query_error"] = str(exc)
    return snapshot


def _collect_nvidia_smi_snapshot() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception as exc:
        return {"query_error": str(exc)}

    if result.returncode != 0:
        err_text = (result.stderr or result.stdout).strip()
        return {"query_error": err_text or f"nvidia-smi exited with code {result.returncode}"}

    line = next((x.strip() for x in result.stdout.splitlines() if x.strip()), "")
    if not line:
        return {"query_error": "nvidia-smi returned no GPU rows"}

    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 4:
        return {"query_error": f"unexpected nvidia-smi output: {line}"}

    return {
        "name": parts[0],
        "total_mib": parts[1],
        "used_mib": parts[2],
        "free_mib": parts[3],
    }


def _build_exit_diagnostic_report(
    reason: str,
    *,
    exc: Optional[BaseException] = None,
    traceback_text: str = "",
    loop_state: Optional[dict[str, Any]] = None,
    proc_status: Optional[dict[str, int]] = None,
    meminfo: Optional[dict[str, int]] = None,
    torch_cuda: Optional[dict[str, Any]] = None,
    nvidia_smi: Optional[dict[str, str]] = None,
) -> str:
    loop_state = loop_state or {}
    proc_status = _read_proc_status_snapshot() if proc_status is None else proc_status
    meminfo = _read_proc_meminfo_snapshot() if meminfo is None else meminfo
    torch_cuda = _collect_torch_cuda_snapshot() if torch_cuda is None else torch_cuda
    nvidia_smi = _collect_nvidia_smi_snapshot() if nvidia_smi is None else nvidia_smi

    lines = [
        ">>> [IL2] Exit diagnostics",
        f"reason: {reason}",
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if loop_state:
        lines.append(
            "last_state: "
            f"frame_idx={loop_state.get('frame_idx', 'n/a')} "
            f"autopilot_on={loop_state.get('autopilot_on', 'n/a')} "
            f"status={loop_state.get('status', 'n/a')} "
            f"bridge_command={loop_state.get('bridge_command', 'n/a')}"
        )

    lines.append(
        "ram: "
        f"proc_rss={_format_bytes(proc_status.get('VmRSS'))} "
        f"proc_peak={_format_bytes(proc_status.get('VmHWM'))} "
        f"proc_vmsize={_format_bytes(proc_status.get('VmSize'))} "
        f"threads={proc_status.get('Threads', 'unknown')}"
    )
    lines.append(
        "system_mem: "
        f"available={_format_bytes(meminfo.get('MemAvailable'))} "
        f"free={_format_bytes(meminfo.get('MemFree'))} "
        f"total={_format_bytes(meminfo.get('MemTotal'))} "
        f"swap_free={_format_bytes(meminfo.get('SwapFree'))} "
        f"swap_total={_format_bytes(meminfo.get('SwapTotal'))}"
    )

    if torch_cuda.get("cuda_available"):
        lines.append(
            "vram(torch): "
            f"device={torch_cuda.get('device_name', 'unknown')} "
            f"allocated={_format_bytes(torch_cuda.get('allocated_bytes'))} "
            f"reserved={_format_bytes(torch_cuda.get('reserved_bytes'))} "
            f"free={_format_bytes(torch_cuda.get('free_bytes'))} "
            f"total={_format_bytes(torch_cuda.get('total_bytes'))} "
            f"max_allocated={_format_bytes(torch_cuda.get('max_allocated_bytes'))}"
        )
        if "query_error" in torch_cuda:
            lines.append(f"vram(torch)_query_error: {torch_cuda['query_error']}")
    else:
        lines.append(
            "vram(torch): "
            + (torch_cuda.get("backend_error") or "cuda unavailable or not initialized")
        )

    if nvidia_smi and "name" in nvidia_smi:
        lines.append(
            "vram(nvidia-smi): "
            f"device={nvidia_smi['name']} "
            f"used={nvidia_smi['used_mib']} MiB "
            f"free={nvidia_smi['free_mib']} MiB "
            f"total={nvidia_smi['total_mib']} MiB"
        )
    elif nvidia_smi and "query_error" in nvidia_smi:
        lines.append(f"vram(nvidia-smi): {nvidia_smi['query_error']}")

    if exc is not None:
        lines.append(f"exception: {type(exc).__name__}: {exc}")
    if traceback_text.strip():
        lines.append("traceback:")
        lines.append(traceback_text.rstrip())

    lines.append("note: OS-level hard OOM kills may terminate Python before this report can run.")
    return "\n".join(lines)


def _write_exit_diagnostic_report(report: str, output_root: Optional[Path] = None) -> Path:
    target_dir = output_root if output_root is not None else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / EXIT_DIAGNOSTICS_FILENAME
    report_path.write_text(report + "\n", encoding="utf-8")
    return report_path


def _emit_exit_diagnostics(
    reason: str,
    *,
    exc: Optional[BaseException] = None,
    traceback_text: str = "",
    output_root: Optional[Path] = None,
    loop_state: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    report = _build_exit_diagnostic_report(
        reason,
        exc=exc,
        traceback_text=traceback_text,
        loop_state=loop_state,
    )
    print(report)
    try:
        report_path = _write_exit_diagnostic_report(report, output_root=output_root)
    except Exception as write_exc:
        print(f"⚠️ [IL2] 写入退出诊断失败: {write_exc}")
        return None

    print(f">>> [IL2] 退出诊断已保存: {report_path}")
    return report_path


@dataclass
class IL2YamlBoundControl:
    runtime_cfg: DaggerSim2RealRuntimeConfig
    sim2real_bridge: Sim2RealBridge
    motor_mapper: MotorMapper
    actuator_ids: Tuple[int, int, int, int]


@dataclass
class FeedButtonLoopState:
    forward_pressed_prev: bool = False
    backward_pressed_prev: bool = False
    next_send_ts: float = 0.0


@dataclass
class ManualPoseLoopState:
    hold_active: bool = False
    hold_pressed_prev: bool = False
    hold_target_mm: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    last_motor_limit_active: Tuple[bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
    )


@dataclass
class ManualPoseResult:
    command: str
    motor_target_mm: np.ndarray
    motor_limit_active: Tuple[bool, bool, bool, bool]
    hold_active: bool


@dataclass
class SessionRecorder:
    output_root: Path
    raw_dir: Path
    mask_dir: Path
    overlay_dir: Path
    csv_path: Path
    csv_file: Any
    writer: csv.DictWriter

    @classmethod
    def create(cls, output_dir: str) -> "SessionRecorder":
        output_root = resolve_output_root(output_dir)
        raw_dir = output_root / "raw"
        mask_dir = output_root / "mask"
        overlay_dir = output_root / "overlay"
        csv_path = output_root / "frame_metrics.csv"

        raw_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        csv_file = csv_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        csv_file.flush()
        return cls(
            output_root=output_root,
            raw_dir=raw_dir,
            mask_dir=mask_dir,
            overlay_dir=overlay_dir,
            csv_path=csv_path,
            csv_file=csv_file,
            writer=writer,
        )

    def close(self) -> None:
        self.csv_file.close()


@dataclass
class MotorTargetRecorder:
    frames: list[int]
    targets: list[np.ndarray]

    def record(self, frame_idx: int, motor_target_mm: np.ndarray) -> None:
        self.frames.append(int(frame_idx))
        self.targets.append(np.asarray(motor_target_mm, dtype=np.float64).copy())

    def save(self, output_cfg: Any) -> None:
        if len(self.targets) == 0:
            print(">>> [IL2] 本次无可导出的电机目标位移记录。")
            return

        frames = np.asarray(self.frames, dtype=np.int64)
        motors = np.asarray(self.targets, dtype=np.float64)

        if output_cfg.save_csv:
            csv_path = Path(output_cfg.csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_data = np.column_stack([frames, motors])
            np.savetxt(
                csv_path,
                csv_data,
                delimiter=",",
                header="frame,m1_target_mm,m2_target_mm,m3_target_mm,m4_target_mm",
                comments="",
                fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"],
            )
            print(f">>> [IL2] 电机目标位移 CSV 已保存: {csv_path}")

        if output_cfg.save_plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("⚠️ [IL2] 未安装 matplotlib，跳过电机目标位移曲线导出。")
                return

            plot_path = Path(output_cfg.plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frames, motors[:, 0], label="m1")
            ax.plot(frames, motors[:, 1], label="m2")
            ax.plot(frames, motors[:, 2], label="m3")
            ax.plot(frames, motors[:, 3], label="m4")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Motor Absolute Target (mm)")
            ax.set_title("Motor Absolute Target Trajectory")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=output_cfg.plot_dpi)
            plt.close(fig)
            print(f">>> [IL2] 电机目标位移曲线图已保存: {plot_path}")


def resolve_output_root(output_dir: str) -> Path:
    if output_dir.strip():
        return Path(output_dir).expanduser()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def maybe_start_session_recorder(
    recorder: Optional[SessionRecorder],
    *,
    output_dir: str,
    save_session_artifacts: bool,
) -> tuple[Optional[SessionRecorder], bool]:
    if recorder is not None:
        return recorder, False
    if not save_session_artifacts:
        return None, False
    return SessionRecorder.create(output_dir), True


def should_record_frame(autopilot_on: bool, recorder: Optional[SessionRecorder]) -> bool:
    return bool(autopilot_on and recorder is not None)


def load_yaml_bound_control(
    config_path: str | Path = SIM2REAL_CONFIG_PATH,
) -> IL2YamlBoundControl:
    runtime_cfg = load_dagger_sim2real_runtime_config(config_path)
    sim2real_bridge = Sim2RealBridge(runtime_cfg.bridge)
    motor_mapper = MotorMapper(runtime_cfg.actuator, runtime_cfg.bridge.motor_order)
    actuator_ids = runtime_cfg.actuator.ids_for_motor_order(runtime_cfg.bridge.motor_order)
    return IL2YamlBoundControl(
        runtime_cfg=runtime_cfg,
        sim2real_bridge=sim2real_bridge,
        motor_mapper=motor_mapper,
        actuator_ids=actuator_ids,
    )


def apply_policy_action(
    control: IL2YamlBoundControl,
    action: Sequence[float],
    dt: float,
) -> BridgeResult:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {arr.shape}")
    return control.sim2real_bridge.step(float(arr[0]), float(arr[1]), float(dt))


def map_motor_target_to_counts(
    control: IL2YamlBoundControl,
    motor_target_mm: np.ndarray,
) -> np.ndarray:
    return control.motor_mapper.mm_targets_to_counts(motor_target_mm)


def sample_hold_pressed(sample: GamepadSample) -> bool:
    return bool(getattr(sample, "hold_pressed", False))


def control_mode_text(autopilot_on: bool, manual_pose_state: ManualPoseLoopState) -> str:
    if autopilot_on:
        return "AUTO"
    if manual_pose_state.hold_active:
        return "MANUAL_HOLD"
    return "MANUAL"


def clear_manual_pose_hold(
    manual_pose_state: ManualPoseLoopState,
    *,
    current_hold_pressed: bool = False,
) -> None:
    manual_pose_state.hold_active = False
    manual_pose_state.hold_pressed_prev = bool(current_hold_pressed)
    manual_pose_state.hold_target_mm = np.zeros(4, dtype=np.float64)
    manual_pose_state.last_motor_limit_active = (False, False, False, False)


def compute_manual_pose_result(
    *,
    sample: GamepadSample,
    teleop_cfg: Any,
    motor_limit_mm: float,
    manual_pose_state: ManualPoseLoopState,
    current_motor_target_mm: np.ndarray,
) -> ManualPoseResult:
    hold_now = sample_hold_pressed(sample)
    hold_edge = hold_now and (not manual_pose_state.hold_pressed_prev)
    manual_pose_state.hold_pressed_prev = hold_now

    current_target = np.asarray(current_motor_target_mm, dtype=np.float64).reshape(4)
    if hold_edge:
        if manual_pose_state.hold_active:
            manual_pose_state.hold_active = False
            manual_pose_state.hold_target_mm = np.zeros(4, dtype=np.float64)
        else:
            manual_pose_state.hold_active = True
            manual_pose_state.hold_target_mm = current_target.copy()

    if manual_pose_state.hold_active:
        limit_mm = max(0.0, float(motor_limit_mm))
        target = np.clip(
            np.asarray(manual_pose_state.hold_target_mm, dtype=np.float64).reshape(4),
            -limit_mm,
            limit_mm,
        )
        active = tuple(
            bool(abs(float(v)) >= (limit_mm - 1e-9)) for v in target.tolist()
        )
        manual_pose_state.hold_target_mm = target.copy()
        return ManualPoseResult(
            command="MANUAL_HOLD",
            motor_target_mm=target,
            motor_limit_active=active,
            hold_active=True,
        )

    _, _, motor_target_mm, motor_limit_active = _compute_direct_motor_target_mm(
        sample=sample,
        teleop_cfg=teleop_cfg,
        motor_limit_mm=float(motor_limit_mm),
    )
    return ManualPoseResult(
        command="MANUAL_CMD",
        motor_target_mm=np.asarray(motor_target_mm, dtype=np.float64),
        motor_limit_active=tuple(bool(x) for x in motor_limit_active),
        hold_active=False,
    )


def send_manual_pose_target_mm(
    *,
    control: IL2YamlBoundControl,
    ctx: TeleopRuntimeContext,
    manual_pose_state: ManualPoseLoopState,
    motor_target_mm: np.ndarray,
    motor_limit_active: Tuple[bool, bool, bool, bool],
    actuator_tx: Optional[ActuatorTx],
) -> bool:
    if ctx.sim2real_bridge.estop_latched:
        return True

    target = np.asarray(motor_target_mm, dtype=np.float64).reshape(4)
    limit_mm = float(ctx.sim2real_bridge.motor_limit_mm)
    target = np.clip(target, -limit_mm, limit_mm)

    prev_active = np.asarray(manual_pose_state.last_motor_limit_active, dtype=bool)
    current_active = np.asarray(motor_limit_active, dtype=bool)
    new_hits = current_active & (~prev_active)
    if np.any(new_hits):
        motor_names = ("m1", "m2", "m3", "m4")
        hit_text = ",".join(
            name for name, is_hit in zip(motor_names, new_hits.tolist()) if is_hit
        )
        print(f">>> [IL2] MANUAL_MOTOR_LIMIT_HIT limit_mm={limit_mm:.3f} motors={hit_text}")
    manual_pose_state.last_motor_limit_active = tuple(
        bool(x) for x in current_active.tolist()
    )

    ctx.sim2real_bridge.motor_target_mm = target.copy()
    ctx.sim2real_bridge.motor_limit_active = current_active.copy()

    if ctx.dry_run:
        return True
    if actuator_tx is None:
        print("⚠️ [IL2] actuator_tx 未就绪，无法发送手动姿态目标。")
        return False

    return send_follow_target_mm(
        target,
        motor_mapper=control.motor_mapper,
        actuator_tx=actuator_tx,
        actuator_ids=control.actuator_ids,
        frame_name="F3_FOLLOW",
        critical=False,
    )


def compute_center_error(
    lumen_center: Optional[Tuple[int, int]],
    frame_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, int], Optional[int], Optional[int], Optional[float]]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    scope_center = (width // 2, height // 2)
    if lumen_center is None:
        return scope_center, None, None, None

    error_x_px = int(lumen_center[0]) - scope_center[0]
    error_y_px = int(lumen_center[1]) - scope_center[1]
    error_norm_px = math.hypot(error_x_px, error_y_px)
    return scope_center, error_x_px, error_y_px, error_norm_px


def resize_binary_mask(mask: Optional[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    height, width = int(target_shape[0]), int(target_shape[1])
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_arr.shape != (height, width):
        mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask_arr > 0).astype(np.uint8)


def process_mask_for_policy(raw_mask: Optional[np.ndarray]) -> np.ndarray:
    mask_binary = resize_binary_mask(raw_mask, (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE))
    return mask_binary.astype(np.uint8) * 255


def analyze_frame(
    frame_bgr: np.ndarray,
) -> tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray, float, str]:
    if get_lumen_center is None:
        raise RuntimeError(f"lumen_center_api unavailable: {_LUMEN_IMPORT_ERROR}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    center, mask, inference_sec = get_lumen_center(frame_rgb, return_time=True)
    mask_binary = resize_binary_mask(mask, frame_bgr.shape[:2])
    mask_u8 = mask_binary * 255
    status_text = "OK" if center is not None else "NO_LUMEN"
    inference_ms = float(inference_sec) * 1000.0
    return center, mask_binary, mask_u8, inference_ms, status_text


def build_overlay_frame(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    scope_center: Tuple[int, int],
    lumen_center: Optional[Tuple[int, int]],
    inference_ms: Optional[float],
    status_text: str,
    autopilot_on: bool,
    estop_latched: bool,
    policy_step_yaw: float,
    policy_step_pitch: float,
    feed_delta_pulses: int,
    manual_hold_active: bool = False,
) -> np.ndarray:
    display = np.asarray(frame_bgr, dtype=np.uint8).copy()
    mask_bool = mask_u8 > 0

    if np.any(mask_bool):
        green_overlay = display.copy()
        green_overlay[mask_bool] = (0, 255, 0)
        display = cv2.addWeighted(green_overlay, 0.35, display, 0.65, 0.0)

    cv2.drawMarker(
        display,
        scope_center,
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )
    if lumen_center is not None:
        cv2.circle(display, lumen_center, 6, (0, 0, 255), thickness=2)
        cv2.line(display, scope_center, lumen_center, (0, 255, 255), thickness=1)

    state_text = "AUTO" if autopilot_on else ("MANUAL-HOLD" if manual_hold_active else "MANUAL")
    if estop_latched:
        state_text += " | ESTOP"

    lines = [
        f"state: {state_text}",
        f"status: {status_text}",
        f"policy dyaw={policy_step_yaw:.5f} dpitch={policy_step_pitch:.5f}",
        f"manual hold={int(bool(manual_hold_active))}",
        f"feed delta={int(feed_delta_pulses)} pulses",
        "inference: n/a" if inference_ms is None else f"inference: {inference_ms:.2f} ms",
    ]
    panel_height = 24 + 20 * len(lines)
    cv2.rectangle(display, (4, 4), (520, panel_height), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (10, 24 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def write_metric_row(
    recorder: SessionRecorder,
    *,
    frame_idx: int,
    timestamp_sec: float,
    frame_bgr: np.ndarray,
    scope_center: Tuple[int, int],
    lumen_center: Optional[Tuple[int, int]],
    error_x_px: Optional[int],
    error_y_px: Optional[int],
    error_norm_px: Optional[float],
    inference_ms: Optional[float],
    status_text: str,
    autopilot_on: bool,
    policy_step_yaw: float,
    policy_step_pitch: float,
    bridge_command: str,
    estop_latched: bool,
    sample: GamepadSample,
    feed_delta_pulses: int,
    motor_target_mm: np.ndarray,
    manual_hold_active: bool = False,
    control_mode: str = "MANUAL",
) -> None:
    target_vec = np.asarray(motor_target_mm, dtype=np.float64).reshape(-1)
    recorder.writer.writerow(
        {
            "frame_idx": int(frame_idx),
            "timestamp_sec": f"{float(timestamp_sec):.6f}",
            "image_width": int(frame_bgr.shape[1]),
            "image_height": int(frame_bgr.shape[0]),
            "scope_center_x": int(scope_center[0]),
            "scope_center_y": int(scope_center[1]),
            "lumen_center_x": "" if lumen_center is None else int(lumen_center[0]),
            "lumen_center_y": "" if lumen_center is None else int(lumen_center[1]),
            "error_x_px": "" if error_x_px is None else int(error_x_px),
            "error_y_px": "" if error_y_px is None else int(error_y_px),
            "error_norm_px": "" if error_norm_px is None else f"{float(error_norm_px):.6f}",
            "inference_ms": "" if inference_ms is None else f"{float(inference_ms):.6f}",
            "status": status_text,
            "control_mode": str(control_mode),
            "autopilot_on": int(bool(autopilot_on)),
            "policy_step_yaw": f"{float(policy_step_yaw):.8f}",
            "policy_step_pitch": f"{float(policy_step_pitch):.8f}",
            "bridge_command": bridge_command,
            "estop_latched": int(bool(estop_latched)),
            "forward_pressed": int(bool(sample.forward_pressed)),
            "backward_pressed": int(bool(sample.backward_pressed)),
            "hold_pressed": int(sample_hold_pressed(sample)),
            "manual_hold_active": int(bool(manual_hold_active)),
            "feed_delta_pulses": int(feed_delta_pulses),
            "m1_target_mm": f"{float(target_vec[0]):.8f}",
            "m2_target_mm": f"{float(target_vec[1]):.8f}",
            "m3_target_mm": f"{float(target_vec[2]):.8f}",
            "m4_target_mm": f"{float(target_vec[3]):.8f}",
        }
    )
    recorder.csv_file.flush()


def save_frame_artifacts(
    recorder: SessionRecorder,
    frame_idx: int,
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    overlay_frame: np.ndarray,
) -> None:
    frame_name = f"frame_{frame_idx:06d}"
    raw_path = recorder.raw_dir / f"{frame_name}.jpg"
    mask_path = recorder.mask_dir / f"{frame_name}.png"
    overlay_path = recorder.overlay_dir / f"{frame_name}.jpg"
    cv2.imwrite(str(raw_path), frame_bgr)
    cv2.imwrite(str(mask_path), mask_u8)
    cv2.imwrite(str(overlay_path), overlay_frame)


def send_follow_target_mm(
    motor_target_mm: np.ndarray,
    *,
    motor_mapper: MotorMapper,
    actuator_tx: ActuatorTx,
    actuator_ids: Sequence[int],
    frame_name: str = "F3_FOLLOW",
    critical: bool = False,
) -> bool:
    try:
        target_counts = motor_mapper.mm_targets_to_counts(motor_target_mm)
    except Exception as exc:
        print(f"⚠️ [IL2] mm->count 映射失败: {exc}")
        return False

    return actuator_tx.send_follow_broadcast(
        actuator_ids,
        target_counts.tolist(),
        critical=critical,
        frame_name=frame_name,
    )


def feed_repeat_period_sec(feed_state: FeedRuntimeState) -> float:
    if feed_state.cfg is None:
        return 0.1
    return 1.0 / max(0.1, float(feed_state.cfg.repeat_hz))


def update_feed_buttons_from_sample(
    sample: GamepadSample,
    loop_state: FeedButtonLoopState,
    ctx: TeleopRuntimeContext,
    loop_start: float,
) -> None:
    feed_state = ctx.feed_state
    if not feed_state.enabled:
        return

    now_forward = bool(sample.forward_pressed)
    now_backward = bool(sample.backward_pressed)
    prev_forward = bool(loop_state.forward_pressed_prev)
    prev_backward = bool(loop_state.backward_pressed_prev)
    repeat_period = feed_repeat_period_sec(feed_state)

    loop_state.forward_pressed_prev = now_forward
    loop_state.backward_pressed_prev = now_backward

    if now_forward and now_backward:
        if not (prev_forward and prev_backward):
            print(">>> [IL2] FEED_DIRECTION_CONFLICT: 同时按下前进/后退，忽略本轮进给。")
        loop_state.next_send_ts = loop_start + repeat_period
        return

    direction_now: Optional[str]
    if now_forward:
        direction_now = "forward"
    elif now_backward:
        direction_now = "backward"
    else:
        direction_now = None

    direction_prev: Optional[str]
    if prev_forward and (not prev_backward):
        direction_prev = "forward"
    elif prev_backward and (not prev_forward):
        direction_prev = "backward"
    else:
        direction_prev = None

    if direction_now is None:
        loop_state.next_send_ts = loop_start
        return

    direction_changed = direction_now != direction_prev
    if direction_changed or loop_start >= loop_state.next_send_ts:
        _handle_feed_step_once(ctx, forward=(direction_now == "forward"))
        loop_state.next_send_ts = loop_start + repeat_period


def prefill_policy_buffer(cap: Any) -> deque[np.ndarray]:
    zero_mask = np.zeros((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE), dtype=np.uint8)
    prefill_mask = zero_mask.copy()
    if cv2 is not None and cap is not None:
        ok, frame = cap.read()
        if ok and frame is not None:
            try:
                _, mask_binary, _, _, _ = analyze_frame(frame)
                prefill_mask = process_mask_for_policy(mask_binary)
            except Exception:
                prefill_mask = zero_mask.copy()

    frame_buffer: deque[np.ndarray] = deque(maxlen=3)
    for _ in range(3):
        frame_buffer.append(prefill_mask.copy())
    return frame_buffer


def compute_loop_dt(loop_start: float, last_loop_ts: float, fallback_dt: float) -> tuple[float, float]:
    dt = loop_start - last_loop_ts
    if dt <= 0.0:
        dt = fallback_dt
    return min(dt, 0.2), loop_start


def print_estop_banner(reason: str) -> None:
    print(f"!!! IL2 ESTOP_LATCHED reason={reason} !!!")


def latch_estop(ctx: TeleopRuntimeContext, reason: str) -> None:
    if ctx.sim2real_bridge.estop_latched:
        return

    ctx.sim2real_bridge.estop_latched = True
    if ctx.actuator_monitor is not None:
        ctx.actuator_monitor.set_active(False)
    if ctx.actuator_tx is not None:
        ctx.actuator_tx.send_estop_all(ctx.actuator_ids, critical=True)
    if ctx.feed_state.enabled:
        ctx.feed_state.locked = True
        if not _feed_send_enable(ctx.feed_state, state=False):
            print("⚠️ [IL2] FEED disable on ESTOP failed.")
    print_estop_banner(reason)


def main() -> int:
    if cv2 is None:
        reason = "STARTUP_CV2_IMPORT_FAIL"
        print(f"Error: OpenCV 不可用。原因: {_CV2_IMPORT_ERROR}")
        _emit_exit_diagnostics(reason, exc=_CV2_IMPORT_ERROR)
        return 1
    if init_lumen_model is None or get_lumen_center is None:
        reason = "STARTUP_LUMEN_IMPORT_FAIL"
        print(f"Error: lumen_center_api 不可用。原因: {_LUMEN_IMPORT_ERROR}")
        _emit_exit_diagnostics(reason, exc=_LUMEN_IMPORT_ERROR)
        return 1
    if predict is None:
        reason = "STARTUP_POLICY_IMPORT_FAIL"
        print(f"Error: perception.api 不可用。原因: {_POLICY_IMPORT_ERROR}")
        _emit_exit_diagnostics(reason, exc=_POLICY_IMPORT_ERROR)
        return 1

    try:
        control = load_yaml_bound_control(SIM2REAL_CONFIG_PATH)
        teleop_cfg = load_teleop_config(SIM2REAL_CONFIG_PATH)
    except Exception as exc:
        reason = _classify_exception_reason("YAML_CONFIG_LOAD_FAIL", exc)
        print(f"❌ [IL2] YAML 配置加载失败: {exc}")
        _emit_exit_diagnostics(reason, exc=exc, traceback_text=traceback.format_exc())
        return 1

    try:
        print(">>> [IL2] 正在加载 U-Net 模型...")
        init_lumen_model()
        print(">>> [IL2] U-Net 初始化成功。")
    except Exception as exc:
        reason = _classify_exception_reason("LUMEN_MODEL_INIT_FAIL", exc)
        print(f"❌ [IL2] U-Net 初始化失败: {exc}")
        _emit_exit_diagnostics(reason, exc=exc, traceback_text=traceback.format_exc())
        return 1

    gamepad = PygameGamepadInput(teleop_cfg)
    actuator_tx: Optional[ActuatorTx] = None
    actuator_monitor: Optional[ActuatorMonitor] = None
    serial_link = None
    feed_state = FeedRuntimeState(
        enabled=False,
        cfg=None,
        serial_link=None,
        current_pulses=0,
        locked=False,
    )
    recorder: Optional[SessionRecorder] = None
    motor_recorder = MotorTargetRecorder(frames=[], targets=[])
    cap = None
    follow_tx_log_count = 0
    exit_reason = "UNKNOWN_EXIT"
    exit_exc: Optional[BaseException] = None
    exit_traceback = ""
    last_loop_state: dict[str, Any] = {
        "frame_idx": -1,
        "autopilot_on": False,
        "status": "BOOT",
        "bridge_command": "INIT",
    }
    exit_status = 0

    try:
        gamepad.open()
        print(f">>> [IL2] 手柄已连接: index={gamepad.controller_index} name={gamepad.controller_name}")

        actuator_tx, actuator_monitor, serial_link = _setup_actuator_io(
            runtime_cfg=control.runtime_cfg,
            sim2real_cfg=control.runtime_cfg.bridge,
            monitor_cfg=control.runtime_cfg.monitor,
            actuator_ids=control.actuator_ids,
            dry_run=False,
        )
        feed_state = _setup_feed_runtime(
            config_path=SIM2REAL_CONFIG_PATH,
            dry_run=False,
            actuator_serial_port=control.runtime_cfg.bridge.serial_port,
            check_port_conflict=True,
        )
        ctx = TeleopRuntimeContext(
            sim2real_bridge=control.sim2real_bridge,
            motor_mapper=control.motor_mapper,
            actuator_ids=control.actuator_ids,
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            monitor_settle_sec=control.runtime_cfg.monitor.response_timeout_sec,
            feed_state=feed_state,
            dry_run=False,
            allow_feed_without_actuator=False,
        )

        ok, reason = perform_reset_sequence(
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            motor_mapper=control.motor_mapper,
            actuator_ids=control.actuator_ids,
            sim2real_bridge=control.sim2real_bridge,
            monitor_settle_sec=control.runtime_cfg.monitor.response_timeout_sec,
            dry_run=False,
            feed_state=feed_state,
            allow_feed_without_actuator=False,
        )
        if not ok:
            exit_reason = f"BOOT_RESET_FAIL:{reason}"
            print(f"❌ [IL2] 启动复位失败: {reason}")
            return 1

        cap = cv2.VideoCapture(int(CAMERA_DEVICE_INDEX))
        if not cap.isOpened():
            exit_reason = "CAMERA_OPEN_FAIL"
            print(f"❌ [IL2] 无法打开设备 /dev/video{int(CAMERA_DEVICE_INDEX)}")
            return 1

        if SAVE_SESSION_ARTIFACTS:
            print(">>> [IL2] 数据记录将在按 5 开启自动驾驶后开始。")
        else:
            print(">>> [IL2] SAVE_SESSION_ARTIFACTS=False，跳过图像与 frame_metrics.csv 保存。")

        frame_buffer = prefill_policy_buffer(cap)
        autopilot_on = False
        frame_idx = 0
        record_frame_idx = 0
        read_failures = 0
        loop_period_sec = 1.0 / float(control.runtime_cfg.bridge.control_hz)
        last_loop_ts = time.monotonic()
        feed_loop_state = FeedButtonLoopState()
        manual_pose_state = ManualPoseLoopState()

        print(">>> [IL2] 启动完成。左摇杆手动控制姿态，B 锁定/解锁姿态；按 5 开启自动驾驶并开始记录，按 q 退出。")

        while True:
            loop_start = time.monotonic()
            dt, last_loop_ts = compute_loop_dt(loop_start, last_loop_ts, loop_period_sec)
            wall_time_sec = time.time()
            last_loop_state["frame_idx"] = frame_idx
            last_loop_state["autopilot_on"] = bool(autopilot_on)

            ctx.feed_state.last_delta_pulses = 0
            sample = gamepad.poll()
            update_feed_buttons_from_sample(sample, feed_loop_state, ctx, loop_start)

            monitor_fault = (
                actuator_monitor.get_fault()
                if actuator_monitor is not None and actuator_monitor.has_fault()
                else None
            )
            if monitor_fault is not None:
                autopilot_on = False
                exit_reason = f"MONITOR_{monitor_fault.reason}"
                last_loop_state["autopilot_on"] = False
                latch_estop(ctx, f"MONITOR_{monitor_fault.reason}")
                break

            if actuator_monitor is not None:
                actuator_monitor.set_active(not ctx.sim2real_bridge.estop_latched)

            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= MAX_CAMERA_READ_FAILURES:
                    autopilot_on = False
                    exit_reason = "CAMERA_READ_FAIL"
                    last_loop_state["autopilot_on"] = False
                    latch_estop(ctx, "CAMERA_READ_FAIL")
                    break
                time.sleep(0.02)
                continue
            read_failures = 0

            try:
                center_coords, mask_binary, mask_u8, inference_ms, status_text = analyze_frame(frame)
            except Exception as exc:
                autopilot_on = False
                exit_reason = _classify_exception_reason("PERCEPTION_FAIL", exc)
                exit_exc = exc
                exit_traceback = traceback.format_exc()
                last_loop_state["autopilot_on"] = False
                last_loop_state["status"] = "PERCEPTION_EXCEPTION"
                last_loop_state["bridge_command"] = "ERROR"
                print(f"❌ [IL2] U-Net 推理失败: {exc}")
                latch_estop(ctx, exit_reason)
                break

            policy_mask = process_mask_for_policy(mask_binary)
            frame_buffer.append(policy_mask)

            policy_step_yaw = 0.0
            policy_step_pitch = 0.0
            bridge_command = "IDLE"
            last_loop_state["status"] = status_text
            last_loop_state["bridge_command"] = bridge_command

            if autopilot_on:
                clear_manual_pose_hold(
                    manual_pose_state,
                    current_hold_pressed=sample_hold_pressed(sample),
                )
            elif not ctx.sim2real_bridge.estop_latched:
                manual_result = compute_manual_pose_result(
                    sample=sample,
                    teleop_cfg=teleop_cfg,
                    motor_limit_mm=ctx.sim2real_bridge.motor_limit_mm,
                    manual_pose_state=manual_pose_state,
                    current_motor_target_mm=ctx.sim2real_bridge.motor_target_mm,
                )
                bridge_command = manual_result.command
                last_loop_state["bridge_command"] = bridge_command
                send_ok = send_manual_pose_target_mm(
                    control=control,
                    ctx=ctx,
                    manual_pose_state=manual_pose_state,
                    motor_target_mm=manual_result.motor_target_mm,
                    motor_limit_active=manual_result.motor_limit_active,
                    actuator_tx=actuator_tx,
                )
                if not send_ok:
                    exit_reason = "SERIAL_TX_FAIL"
                    last_loop_state["autopilot_on"] = False
                    latch_estop(ctx, "SERIAL_TX_FAIL")
                    break
                follow_tx_log_count += 1
                if (follow_tx_log_count % FOLLOW_TX_LOG_EVERY_N) == 0:
                    target_counts = map_motor_target_to_counts(
                        control,
                        np.asarray(ctx.sim2real_bridge.motor_target_mm, dtype=np.float64),
                    )
                    count_text = " ".join(
                        f"{motor}(id={aid})={count}"
                        for motor, aid, count in zip(
                            control.runtime_cfg.bridge.motor_order,
                            control.actuator_ids,
                            target_counts.tolist(),
                        )
                    )
                    print(f"[IL2 TX] F3_FOLLOW seq={follow_tx_log_count} | {count_text}")

            if autopilot_on and (not ctx.sim2real_bridge.estop_latched):
                if center_coords is not None:
                    try:
                        action = predict(np.array(frame_buffer))
                    except Exception as exc:
                        autopilot_on = False
                        exit_reason = _classify_exception_reason("POLICY_FAIL", exc)
                        exit_exc = exc
                        exit_traceback = traceback.format_exc()
                        last_loop_state["autopilot_on"] = False
                        last_loop_state["bridge_command"] = "ERROR"
                        print(f"❌ [IL2] Policy 推理失败: {exc}")
                        latch_estop(ctx, exit_reason)
                        break

                    policy_step_yaw = float(action[0])
                    policy_step_pitch = float(action[1])
                    bridge_result = apply_policy_action(
                        control,
                        action=(policy_step_yaw, policy_step_pitch),
                        dt=dt,
                    )
                    bridge_command = bridge_result.command.value
                    last_loop_state["bridge_command"] = bridge_command

                    if bridge_result.command == BridgeCommand.ESTOP:
                        autopilot_on = False
                        exit_reason = bridge_result.reason or "ANGLE_LIMIT"
                        last_loop_state["autopilot_on"] = False
                        latch_estop(ctx, bridge_result.reason or "ANGLE_LIMIT")
                        break

                    if bridge_result.should_send:
                        assert actuator_tx is not None
                        send_ok = send_follow_target_mm(
                            bridge_result.motor_target_mm,
                            motor_mapper=control.motor_mapper,
                            actuator_tx=actuator_tx,
                            actuator_ids=control.actuator_ids,
                            frame_name="F3_FOLLOW",
                            critical=False,
                        )
                        if not send_ok:
                            autopilot_on = False
                            exit_reason = "SERIAL_TX_FAIL"
                            last_loop_state["autopilot_on"] = False
                            latch_estop(ctx, "SERIAL_TX_FAIL")
                            break
                        follow_tx_log_count += 1
                        if (follow_tx_log_count % FOLLOW_TX_LOG_EVERY_N) == 0:
                            target_counts = map_motor_target_to_counts(control, bridge_result.motor_target_mm)
                            count_text = " ".join(
                                f"{motor}(id={aid})={count}"
                                for motor, aid, count in zip(
                                    control.runtime_cfg.bridge.motor_order,
                                    control.actuator_ids,
                                    target_counts.tolist(),
                                )
                            )
                            print(f"[IL2 TX] F3_FOLLOW seq={follow_tx_log_count} | {count_text}")
                        motor_recorder.record(record_frame_idx, bridge_result.motor_target_mm)
                else:
                    bridge_command = "NO_LUMEN_HOLD"
                    last_loop_state["bridge_command"] = bridge_command

            scope_center, error_x_px, error_y_px, error_norm_px = compute_center_error(
                center_coords,
                frame.shape,
            )
            motor_target_mm = np.asarray(ctx.sim2real_bridge.motor_target_mm, dtype=np.float64)
            control_mode = control_mode_text(autopilot_on, manual_pose_state)
            overlay_frame = build_overlay_frame(
                frame_bgr=frame,
                mask_u8=mask_u8,
                scope_center=scope_center,
                lumen_center=center_coords,
                inference_ms=inference_ms,
                status_text=status_text,
                autopilot_on=autopilot_on,
                estop_latched=ctx.sim2real_bridge.estop_latched,
                policy_step_yaw=policy_step_yaw,
                policy_step_pitch=policy_step_pitch,
                feed_delta_pulses=int(ctx.feed_state.last_delta_pulses),
                manual_hold_active=manual_pose_state.hold_active,
            )

            recording_active = bool(autopilot_on)
            if should_record_frame(autopilot_on, recorder):
                assert recorder is not None
                save_frame_artifacts(recorder, record_frame_idx, frame, mask_u8, overlay_frame)
                write_metric_row(
                    recorder,
                    frame_idx=record_frame_idx,
                    timestamp_sec=wall_time_sec,
                    frame_bgr=frame,
                    scope_center=scope_center,
                    lumen_center=center_coords,
                    error_x_px=error_x_px,
                    error_y_px=error_y_px,
                    error_norm_px=error_norm_px,
                    inference_ms=inference_ms,
                    status_text=status_text,
                    control_mode=control_mode,
                    autopilot_on=autopilot_on,
                    policy_step_yaw=policy_step_yaw,
                    policy_step_pitch=policy_step_pitch,
                    bridge_command=bridge_command,
                    estop_latched=ctx.sim2real_bridge.estop_latched,
                    sample=sample,
                    feed_delta_pulses=int(ctx.feed_state.last_delta_pulses),
                    motor_target_mm=motor_target_mm,
                    manual_hold_active=manual_pose_state.hold_active,
                )
            if recording_active:
                record_frame_idx += 1

            cv2.imshow(WINDOW_NAME, overlay_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("5"):
                if ctx.sim2real_bridge.estop_latched:
                    print(">>> [IL2] ESTOP 已锁存，自动驾驶不可开启。")
                else:
                    was_autopilot_on = bool(autopilot_on)
                    autopilot_on = not autopilot_on
                    clear_manual_pose_hold(
                        manual_pose_state,
                        current_hold_pressed=sample_hold_pressed(sample),
                    )
                    last_loop_state["autopilot_on"] = bool(autopilot_on)
                    state_text = "开启" if autopilot_on else "关闭"
                    print(f">>> [IL2] 自动驾驶已{state_text}")
                    if autopilot_on and not was_autopilot_on:
                        recorder, recorder_started = maybe_start_session_recorder(
                            recorder,
                            output_dir=OUTPUT_DIR,
                            save_session_artifacts=SAVE_SESSION_ARTIFACTS,
                        )
                        if recorder_started and recorder is not None:
                            print(f">>> [IL2] 自动驾驶数据记录开始: {recorder.output_root}")
                        elif recorder is not None:
                            print(f">>> [IL2] 自动驾驶数据记录继续: {recorder.output_root}")
                        else:
                            print(">>> [IL2] 自动驾驶已开启；当前未保存图像与 frame_metrics.csv。")
            elif key == ord("q"):
                exit_reason = "USER_QUIT"
                print(">>> [IL2] 收到退出指令。")
                break

            frame_idx += 1
            sleep_sec = loop_period_sec - (time.monotonic() - loop_start)
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    except KeyboardInterrupt:
        exit_reason = "KEYBOARD_INTERRUPT"
        print("\n>>> [IL2] 检测到 Ctrl+C，正在退出...")
    except Exception as exc:
        exit_reason = _classify_exception_reason("UNEXPECTED_EXCEPTION", exc)
        exit_exc = exc
        exit_traceback = traceback.format_exc()
        exit_status = 1
        print(f"❌ [IL2] 未捕获异常: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close()
        motor_recorder.save(control.runtime_cfg.output if "control" in locals() else type("O", (), {"save_csv": False, "save_plot": False})())
        _cleanup_runtime(
            gamepad=gamepad,
            actuator_tx=actuator_tx,
            actuator_monitor=actuator_monitor,
            actuator_ids=control.actuator_ids if "control" in locals() else (),
            serial_link=serial_link,
            feed_state=feed_state,
        )
        _emit_exit_diagnostics(
            exit_reason,
            exc=exit_exc,
            traceback_text=exit_traceback,
            output_root=recorder.output_root if recorder is not None else None,
            loop_state=last_loop_state,
        )

    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
