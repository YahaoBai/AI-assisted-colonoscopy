#####################

#这个脚本只用来仿真

#####################



from __future__ import annotations

import csv
import math
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime environment dependent
    cv2 = None
    _CV2_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover - runtime environment dependent
    mujoco = None
    _MUJOCO_IMPORT_ERROR: Optional[Exception] = exc
else:
    _MUJOCO_IMPORT_ERROR = None

try:
    from pynput import keyboard
except Exception as exc:  # pragma: no cover - runtime environment dependent
    keyboard = None
    _KEYBOARD_IMPORT_ERROR: Optional[Exception] = exc
else:
    _KEYBOARD_IMPORT_ERROR = None

try:
    from perception.lumen_center_api import init as init_unet, get_lumen_center
except Exception as exc:  # pragma: no cover - runtime environment dependent
    init_unet = None
    get_lumen_center = None
    _LUMEN_IMPORT_ERROR: Optional[Exception] = exc
else:
    _LUMEN_IMPORT_ERROR = None

try:
    from perception.api import predict
except Exception as exc:  # pragma: no cover - runtime environment dependent
    predict = None
    _POLICY_IMPORT_ERROR: Optional[Exception] = exc
else:
    _POLICY_IMPORT_ERROR = None


IMG_WIDTH = 256
IMG_HEIGHT = 256
HALF_WIDTH = IMG_WIDTH * 0.5
HALF_HEIGHT = IMG_HEIGHT * 0.5
INV_HEIGHT = 1.0 / IMG_HEIGHT

MANUAL_MOVE_SPEED = 0.6 * 0.8
MANUAL_ROTATE_SPEED = 1.8 * 0.8
ZOOM_SPEED = 10.0
POLICY_IMAGE_SIZE = 256

FILTER_WINDOW = 5
TRIGGER_FRAMES = 5
RECOVERY_FRAMES = 30
SPEED_ERROR_THRESHOLD = 0.18
DEFAULT_OUTPUT_ROOT = Path("sim_capture")
CONTROLLER_NAME = "IL0 Auto"

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
    "record_mode",
    "autopilot_on",
    "policy_step_yaw",
    "policy_step_pitch",
]


@dataclass
class RuntimeKeyState:
    forward: bool = False
    pitch_up: bool = False
    pitch_down: bool = False
    yaw_left: bool = False
    yaw_right: bool = False
    zoom_in: bool = False
    zoom_out: bool = False
    auto_toggle_pressed: bool = False
    auto_toggle_requested: bool = False
    manual_toggle_pressed: bool = False
    manual_toggle_requested: bool = False
    quit_requested: bool = False


@dataclass
class SessionRecorder:
    mode: str
    output_root: Path
    raw_dir: Path
    mask_dir: Path
    overlay_dir: Path
    csv_path: Path
    csv_file: Any
    writer: csv.DictWriter

    @classmethod
    def create(cls, base_output_root: Path, mode: str) -> "SessionRecorder":
        output_root = resolve_session_output_root(base_output_root, mode)
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
            mode=str(mode),
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
class RecordingRuntime:
    active_mode: Optional[str] = None
    recorder: Optional[SessionRecorder] = None


def resolve_session_output_root(base_output_root: Path, mode: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_output_root).expanduser() / str(mode).strip().lower()
    candidate = base_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def close_active_session(runtime: RecordingRuntime) -> RecordingRuntime:
    if runtime.recorder is not None:
        runtime.recorder.close()
    runtime.recorder = None
    runtime.active_mode = None
    return runtime


def toggle_auto_recording(
    runtime: RecordingRuntime,
    *,
    autopilot_on: bool,
    base_output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[RecordingRuntime, bool, str]:
    if runtime.active_mode == "auto" and runtime.recorder is not None:
        output_root = runtime.recorder.output_root
        close_active_session(runtime)
        return runtime, False, f">>> [IL0] 自动录制已关闭: {output_root}"

    if runtime.active_mode == "manual" and runtime.recorder is not None:
        manual_root = runtime.recorder.output_root
        close_active_session(runtime)
        print(f">>> [IL0] 已关闭手动录制: {manual_root}")

    runtime.recorder = SessionRecorder.create(base_output_root, mode="auto")
    runtime.active_mode = "auto"
    return runtime, True, f">>> [IL0] 自动录制已开启: {runtime.recorder.output_root}"


def toggle_manual_recording(
    runtime: RecordingRuntime,
    *,
    autopilot_on: bool,
    base_output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[RecordingRuntime, bool, str]:
    if autopilot_on:
        return runtime, autopilot_on, ">>> [IL0] 自动模式中，忽略手动录制请求。"

    if runtime.active_mode == "manual" and runtime.recorder is not None:
        output_root = runtime.recorder.output_root
        close_active_session(runtime)
        return runtime, autopilot_on, f">>> [IL0] 手动录制已关闭: {output_root}"

    if runtime.active_mode == "auto" and runtime.recorder is not None:
        auto_root = runtime.recorder.output_root
        close_active_session(runtime)
        print(f">>> [IL0] 已关闭自动录制: {auto_root}")

    runtime.recorder = SessionRecorder.create(base_output_root, mode="manual")
    runtime.active_mode = "manual"
    return runtime, autopilot_on, f">>> [IL0] 手动录制已开启: {runtime.recorder.output_root}"


def compute_center_error(
    lumen_center: Optional[Tuple[int, int]],
    frame_shape: Tuple[int, ...],
) -> tuple[Tuple[int, int], Optional[int], Optional[int], Optional[float]]:
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
        assert cv2 is not None
        mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask_arr > 0).astype(np.uint8)


def process_mask_for_policy(raw_mask: Optional[np.ndarray]) -> np.ndarray:
    mask_binary = resize_binary_mask(raw_mask, (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE))
    return mask_binary.astype(np.uint8) * 255


def analyze_frame_rgb(
    frame_rgb: np.ndarray,
) -> tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray, float, str]:
    if get_lumen_center is None:
        raise RuntimeError(f"lumen_center_api unavailable: {_LUMEN_IMPORT_ERROR}")

    center, mask, inference_sec = get_lumen_center(frame_rgb, return_time=True)
    mask_binary = resize_binary_mask(mask, frame_rgb.shape[:2])
    mask_u8 = mask_binary * 255
    status_text = "OK" if center is not None else "NO_LUMEN"
    inference_ms = float(inference_sec) * 1000.0
    return center, mask_binary, mask_u8, inference_ms, status_text


def build_overlay_frame(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    scope_center: Tuple[int, int],
    lumen_center: Optional[Tuple[int, int]],
    error_x_px: Optional[int],
    error_y_px: Optional[int],
    error_norm_px: Optional[float],
    inference_ms: Optional[float],
    status_text: str,
    record_mode: Optional[str],
    autopilot_on: bool,
    step_yaw: float,
    step_pitch: float,
) -> np.ndarray:
    assert cv2 is not None
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

    record_text = "off" if record_mode is None else record_mode
    state_text = "AUTO" if autopilot_on else "MANUAL"
    lines = [
        f"state: {state_text}",
        f"record: {record_text}",
        f"status: {status_text}",
        f"step dyaw={step_yaw:.5f} dpitch={step_pitch:.5f}",
        "inference: n/a" if inference_ms is None else f"inference: {inference_ms:.2f} ms",
    ]
    if error_norm_px is not None and error_x_px is not None and error_y_px is not None:
        lines.append(f"err=({error_x_px}, {error_y_px}) |e|={error_norm_px:.2f}px")
    else:
        lines.append("err: n/a")

    panel_height = 24 + 20 * len(lines)
    cv2.rectangle(display, (4, 4), (560, panel_height), (0, 0, 0), thickness=-1)
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


def save_frame_artifacts(
    recorder: SessionRecorder,
    *,
    frame_idx: int,
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    overlay_frame: np.ndarray,
) -> None:
    assert cv2 is not None
    frame_name = f"frame_{frame_idx:06d}"
    cv2.imwrite(str(recorder.raw_dir / f"{frame_name}.jpg"), frame_bgr)
    cv2.imwrite(str(recorder.mask_dir / f"{frame_name}.png"), mask_u8)
    cv2.imwrite(str(recorder.overlay_dir / f"{frame_name}.jpg"), overlay_frame)


def write_metric_row(
    recorder: SessionRecorder,
    *,
    frame_idx: int,
    timestamp_sec: float,
    frame_shape: Tuple[int, ...],
    scope_center: Tuple[int, int],
    lumen_center: Optional[Tuple[int, int]],
    error_x_px: Optional[int],
    error_y_px: Optional[int],
    error_norm_px: Optional[float],
    inference_ms: Optional[float],
    status_text: str,
    record_mode: str,
    autopilot_on: bool,
    policy_step_yaw: float,
    policy_step_pitch: float,
) -> None:
    recorder.writer.writerow(
        {
            "frame_idx": int(frame_idx),
            "timestamp_sec": f"{float(timestamp_sec):.6f}",
            "image_width": int(frame_shape[1]),
            "image_height": int(frame_shape[0]),
            "scope_center_x": int(scope_center[0]),
            "scope_center_y": int(scope_center[1]),
            "lumen_center_x": "" if lumen_center is None else int(lumen_center[0]),
            "lumen_center_y": "" if lumen_center is None else int(lumen_center[1]),
            "error_x_px": "" if error_x_px is None else int(error_x_px),
            "error_y_px": "" if error_y_px is None else int(error_y_px),
            "error_norm_px": "" if error_norm_px is None else f"{float(error_norm_px):.6f}",
            "inference_ms": "" if inference_ms is None else f"{float(inference_ms):.6f}",
            "status": status_text,
            "record_mode": str(record_mode),
            "autopilot_on": int(bool(autopilot_on)),
            "policy_step_yaw": f"{float(policy_step_yaw):.8f}",
            "policy_step_pitch": f"{float(policy_step_pitch):.8f}",
        }
    )
    recorder.csv_file.flush()


def prefill_policy_buffer(renderer: Any, data: Any) -> deque[np.ndarray]:
    zero_mask = np.zeros((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE), dtype=np.uint8)
    prefill_mask = zero_mask.copy()
    try:
        renderer.update_scene(data, camera="endoscope_cam")
        frame_rgb = renderer.render()
        _, mask_binary, _, _, _ = analyze_frame_rgb(frame_rgb)
        prefill_mask = process_mask_for_policy(mask_binary)
    except Exception:
        prefill_mask = zero_mask.copy()

    frame_buffer: deque[np.ndarray] = deque(maxlen=3)
    for _ in range(3):
        frame_buffer.append(prefill_mask.copy())
    return frame_buffer


def _build_key_handlers(state: RuntimeKeyState):
    assert keyboard is not None

    def on_press(key: Any) -> None:
        if hasattr(key, "char"):
            try:
                if key.char == "1":
                    state.forward = True
                elif key.char == "+":
                    state.zoom_in = True
                elif key.char == "-":
                    state.zoom_out = True
                elif key.char == "5":
                    if not state.auto_toggle_pressed:
                        state.auto_toggle_pressed = True
                        state.auto_toggle_requested = True
                elif key.char == "6":
                    if not state.manual_toggle_pressed:
                        state.manual_toggle_pressed = True
                        state.manual_toggle_requested = True
                elif key.char == "q":
                    state.quit_requested = True
            except AttributeError:
                return
        else:
            if key == keyboard.Key.up:
                state.pitch_up = True
            elif key == keyboard.Key.down:
                state.pitch_down = True
            elif key == keyboard.Key.left:
                state.yaw_left = True
            elif key == keyboard.Key.right:
                state.yaw_right = True

    def on_release(key: Any) -> None:
        if hasattr(key, "char"):
            try:
                if key.char == "1":
                    state.forward = False
                elif key.char == "+":
                    state.zoom_in = False
                elif key.char == "-":
                    state.zoom_out = False
                elif key.char == "5":
                    state.auto_toggle_pressed = False
                elif key.char == "6":
                    state.manual_toggle_pressed = False
            except AttributeError:
                return
        else:
            if key == keyboard.Key.up:
                state.pitch_up = False
            elif key == keyboard.Key.down:
                state.pitch_down = False
            elif key == keyboard.Key.left:
                state.yaw_left = False
            elif key == keyboard.Key.right:
                state.yaw_right = False

    return on_press, on_release


def main() -> int:
    if cv2 is None:
        print(f"Error: OpenCV 不可用。原因: {_CV2_IMPORT_ERROR}")
        return 1
    if mujoco is None:
        print(f"Error: MuJoCo 不可用。原因: {_MUJOCO_IMPORT_ERROR}")
        return 1
    if keyboard is None:
        print(f"Error: pynput 键盘监听不可用。原因: {_KEYBOARD_IMPORT_ERROR}")
        return 1
    if init_unet is None or get_lumen_center is None:
        print(f"Error: U-Net 感知模块不可用。原因: {_LUMEN_IMPORT_ERROR}")
        return 1
    if predict is None:
        print(f"Error: IL policy 不可用。原因: {_POLICY_IMPORT_ERROR}")
        return 1

    try:
        print(">>> [IL0] 正在加载 Attention U-Net 感知模型...")
        init_unet("./checkpoints/attention_best_model.pth")
        print(">>> [IL0] U-Net 初始化成功。")
    except Exception as exc:
        print(f"❌ [IL0] U-Net 初始化失败: {exc}")
        return 1

    key_state = RuntimeKeyState()
    on_press, on_release = _build_key_handlers(key_state)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frame_buffer: deque[np.ndarray]
    history_x = deque(maxlen=FILTER_WINDOW)
    history_y = deque(maxlen=FILTER_WINDOW)
    recording_runtime = RecordingRuntime()
    frame_count = 0
    is_slowing_down = False
    speed_trigger_counter = 0
    speed_recovery_counter = 0

    try:
        if not Path("./assets/xml/colon_scene.xml").exists():
            model = mujoco.MjModel.from_xml_string(
                """
                <mujoco>
                  <worldbody>
                    <body name="camera_rig" pos="0 0 0" mocap="true">
                      <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
                      <camera name="endoscope_cam" mode="fixed" fovy="90"/>
                    </body>
                  </worldbody>
                </mujoco>
                """
            )
        else:
            model = mujoco.MjModel.from_xml_path("./assets/xml/colon_scene.xml")
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
        mocap_id = model.body("camera_rig").mocapid[0]
        camera_id = model.camera("endoscope_cam").id
    except Exception as exc:
        print(f"❌ [IL0] MuJoCo 初始化失败: {exc}")
        listener.stop()
        return 1

    print("\n" + "=" * 60)
    print("     结肠镜纯仿真系统 ")
    print("  [5] 自动导航 + 自动录制 开关")
    print("  [6] 手动录制 开关")
    print("=" * 60 + "\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.fixedcamid = camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            frame_buffer = prefill_policy_buffer(renderer, data)
            autopilot_on = False

            while viewer.is_running():
                step_start = time.time()
                dt = model.opt.timestep

                if key_state.auto_toggle_requested:
                    key_state.auto_toggle_requested = False
                    recording_runtime, autopilot_on, message = toggle_auto_recording(
                        recording_runtime,
                        autopilot_on=autopilot_on,
                    )
                    print(message)

                if key_state.manual_toggle_requested:
                    key_state.manual_toggle_requested = False
                    recording_runtime, autopilot_on, message = toggle_manual_recording(
                        recording_runtime,
                        autopilot_on=autopilot_on,
                    )
                    print(message)

                if key_state.quit_requested:
                    print(">>> [IL0] 收到退出指令。")
                    break

                step_pitch = 0.0
                step_yaw = 0.0
                local_displacement = np.zeros(3, dtype=np.float64)

                renderer.update_scene(data, camera="endoscope_cam")
                frame_rgb = np.asarray(renderer.render(), dtype=np.uint8)
                wall_time_sec = time.time()

                pipeline_start = time.time()
                try:
                    center_coords, mask_binary, mask_u8, inference_ms, status_text = analyze_frame_rgb(frame_rgb)
                except Exception as exc:
                    center_coords = None
                    mask_binary = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                    mask_u8 = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                    inference_ms = None
                    status_text = f"ANALYSIS_ERROR: {exc}"

                current_mask = process_mask_for_policy(mask_binary)
                frame_buffer.append(current_mask)

                filtered_norm_x = 0.0
                filtered_norm_y = 0.0
                total_norm_error = 0.0
                if center_coords is not None:
                    px_x, px_y = center_coords
                    raw_norm_x = (px_x - HALF_WIDTH) * INV_HEIGHT
                    raw_norm_y = (px_y - HALF_HEIGHT) * INV_HEIGHT
                    history_x.append(raw_norm_x)
                    history_y.append(raw_norm_y)
                    filtered_norm_x = float(np.median(np.asarray(history_x, dtype=np.float64)))
                    filtered_norm_y = float(np.median(np.asarray(history_y, dtype=np.float64)))
                    total_norm_error = math.hypot(filtered_norm_x, filtered_norm_y)

                if autopilot_on:
                    if center_coords is not None:
                        action = predict(np.asarray(frame_buffer))
                        step_yaw = float(action[0])
                        step_pitch = float(action[1])
                        _ = (time.time() - pipeline_start) * 1000.0

                        base_auto_speed = MANUAL_MOVE_SPEED * 0.85
                        if total_norm_error > SPEED_ERROR_THRESHOLD:
                            speed_trigger_counter += 1
                        else:
                            speed_trigger_counter = 0

                        if speed_trigger_counter >= TRIGGER_FRAMES:
                            current_speed = base_auto_speed * 0.4
                            speed_recovery_counter = RECOVERY_FRAMES
                            if not is_slowing_down:
                                print(
                                    f">>> [IL0] 限速触发: 连续 {TRIGGER_FRAMES} 帧确认高误差 | "
                                    f"推进速度降至 {current_speed:.3f}"
                                )
                                is_slowing_down = True
                        else:
                            if speed_recovery_counter > 0:
                                speed_recovery_counter -= 1
                                current_speed = base_auto_speed * 0.4
                            else:
                                current_speed = base_auto_speed
                                if is_slowing_down:
                                    print(f">>> [IL0] 限速解除 | 推进速度恢复 {current_speed:.3f}")
                                    is_slowing_down = False

                        local_displacement[2] -= current_speed * dt
                    else:
                        is_slowing_down = False
                        speed_trigger_counter = 0
                        speed_recovery_counter = 0
                else:
                    is_slowing_down = False
                    speed_trigger_counter = 0
                    speed_recovery_counter = 0
                    if key_state.forward:
                        local_displacement[2] -= MANUAL_MOVE_SPEED * dt
                    if key_state.pitch_up:
                        step_pitch += MANUAL_ROTATE_SPEED * dt
                    if key_state.pitch_down:
                        step_pitch -= MANUAL_ROTATE_SPEED * dt
                    if key_state.yaw_left:
                        step_yaw += MANUAL_ROTATE_SPEED * dt
                    if key_state.yaw_right:
                        step_yaw -= MANUAL_ROTATE_SPEED * dt

                scope_center, error_x_px, error_y_px, error_norm_px = compute_center_error(
                    lumen_center=center_coords,
                    frame_shape=frame_rgb.shape,
                )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                overlay_frame = build_overlay_frame(
                    frame_bgr=frame_bgr,
                    mask_u8=mask_u8,
                    scope_center=scope_center,
                    lumen_center=center_coords,
                    error_x_px=error_x_px,
                    error_y_px=error_y_px,
                    error_norm_px=error_norm_px,
                    inference_ms=inference_ms,
                    status_text=status_text,
                    record_mode=recording_runtime.active_mode,
                    autopilot_on=autopilot_on,
                    step_yaw=step_yaw,
                    step_pitch=step_pitch,
                )

                if recording_runtime.recorder is not None and recording_runtime.active_mode is not None:
                    save_frame_artifacts(
                        recording_runtime.recorder,
                        frame_idx=frame_count,
                        frame_bgr=frame_bgr,
                        mask_u8=mask_u8,
                        overlay_frame=overlay_frame,
                    )
                    write_metric_row(
                        recording_runtime.recorder,
                        frame_idx=frame_count,
                        timestamp_sec=wall_time_sec,
                        frame_shape=frame_rgb.shape,
                        scope_center=scope_center,
                        lumen_center=center_coords,
                        error_x_px=error_x_px,
                        error_y_px=error_y_px,
                        error_norm_px=error_norm_px,
                        inference_ms=inference_ms,
                        status_text=status_text,
                        record_mode=recording_runtime.active_mode,
                        autopilot_on=autopilot_on,
                        policy_step_yaw=step_yaw,
                        policy_step_pitch=step_pitch,
                    )

                if step_pitch != 0.0 or step_yaw != 0.0:
                    current_quat = data.mocap_quat[mocap_id].copy()
                    pitch_quat = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(
                        pitch_quat,
                        np.array([1.0, 0.0, 0.0]),
                        step_pitch,
                    )
                    yaw_quat = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(
                        yaw_quat,
                        np.array([0.0, 1.0, 0.0]),
                        step_yaw,
                    )
                    d_quat = np.zeros(4)
                    mujoco.mju_mulQuat(d_quat, yaw_quat, pitch_quat)
                    mujoco.mju_mulQuat(data.mocap_quat[mocap_id], current_quat, d_quat)

                if local_displacement[2] != 0.0:
                    current_quat = data.mocap_quat[mocap_id].copy()
                    world_disp = np.zeros(3)
                    mujoco.mju_rotVecQuat(world_disp, local_displacement, current_quat)
                    data.mocap_pos[mocap_id] += world_disp

                if key_state.zoom_in:
                    model.cam_fovy[camera_id] -= ZOOM_SPEED * dt
                if key_state.zoom_out:
                    model.cam_fovy[camera_id] += ZOOM_SPEED * dt

                mujoco.mj_step(model, data)
                viewer.sync()
                frame_count += 1
                time.sleep(max(0.0, dt - (time.time() - step_start)))

    except KeyboardInterrupt:
        print("\n>>> [IL0] 检测到 Ctrl+C，正在退出...")
    except Exception as exc:
        print(f"❌ [IL0] 未捕获异常: {exc}")
        traceback.print_exc()
        return 1
    finally:
        close_active_session(recording_runtime)
        listener.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
