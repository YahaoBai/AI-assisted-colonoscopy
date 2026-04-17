from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime environment dependent
    cv2 = None
    _CV2_IMPORT_ERROR: Optional[Exception] = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from perception.lumen_center_api import get_lumen_center, init as init_lumen_model
except Exception as exc:  # pragma: no cover - runtime environment dependent
    get_lumen_center = None
    init_lumen_model = None
    _LUMEN_IMPORT_ERROR: Optional[Exception] = exc
else:
    _LUMEN_IMPORT_ERROR = None


WINDOW_NAME = "USB Camera Stream"
DEFAULT_DEVICE_INDEX = 2
DEFAULT_OUTPUT_ROOT = Path("camera_capture")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read camera frames, save every frame, segment lumen, and compute center error."
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=DEFAULT_DEVICE_INDEX,
        help=f"Camera device index. Default: {DEFAULT_DEVICE_INDEX}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: camera_capture/<timestamp>/",
    )
    parser.add_argument(
        "--disable-analysis",
        action="store_true",
        help="Skip lumen segmentation/center extraction and only save raw frames.",
    )
    return parser


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


def resolve_output_root(output_dir: str) -> Path:
    if output_dir.strip():
        return Path(output_dir).expanduser()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def ensure_output_dirs(output_root: Path) -> tuple[Path, Path, Path, Path]:
    raw_dir = output_root / "raw"
    mask_dir = output_root / "mask"
    overlay_dir = output_root / "overlay"
    csv_path = output_root / "frame_metrics.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, mask_dir, overlay_dir, csv_path


def mask_to_u8(mask: Optional[np.ndarray], frame_shape: Tuple[int, int]) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)

    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.shape != (height, width):
        mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask_arr > 0).astype(np.uint8) * 255


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

    cv2.rectangle(display, (4, 4), (430, 72), (0, 0, 0), thickness=-1)
    cv2.putText(
        display,
        f"status: {status_text}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if lumen_center is None:
        error_text = "lumen center: None"
    else:
        error_text = (
            f"lumen=({lumen_center[0]}, {lumen_center[1]}) "
            f"err=({error_x_px}, {error_y_px}) |e|={error_norm_px:.2f}px"
        )
    cv2.putText(
        display,
        error_text,
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    infer_text = "inference: n/a" if inference_ms is None else f"inference: {inference_ms:.2f} ms"
    cv2.putText(
        display,
        infer_text,
        (10, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def analyze_frame(frame_bgr: np.ndarray) -> tuple[Optional[Tuple[int, int]], np.ndarray, Optional[float], str]:
    if get_lumen_center is None:
        raise RuntimeError(f"lumen_center_api unavailable: {_LUMEN_IMPORT_ERROR}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    center, mask, inference_sec = get_lumen_center(frame_rgb, return_time=True)
    mask_u8 = mask_to_u8(mask, frame_bgr.shape[:2])
    status_text = "OK" if center is not None else "NO_LUMEN"
    inference_ms = float(inference_sec) * 1000.0
    return center, mask_u8, inference_ms, status_text


def write_metric_row(
    writer: csv.DictWriter,
    csv_file,
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
) -> None:
    writer.writerow(
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
        }
    )
    csv_file.flush()


def main() -> int:
    args = build_arg_parser().parse_args()

    if cv2 is None:
        print(f"Error: OpenCV 不可用，无法启动相机采集。原因: {_CV2_IMPORT_ERROR}")
        return 1

    output_root = resolve_output_root(args.output_dir)
    raw_dir, mask_dir, overlay_dir, csv_path = ensure_output_dirs(output_root)

    analysis_enabled = not bool(args.disable_analysis)
    if analysis_enabled:
        if init_lumen_model is None:
            print(f"警告: 分割模型不可用，将退化为仅采集保存。原因: {_LUMEN_IMPORT_ERROR}")
            analysis_enabled = False
        else:
            try:
                print("正在加载腔道分割模型...")
                init_lumen_model()
                print("腔道分割模型加载成功。")
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                print(f"警告: 分割模型初始化失败，将退化为仅采集保存。原因: {exc}")
                analysis_enabled = False

    cap = cv2.VideoCapture(int(args.device_index))
    if not cap.isOpened():
        print(f"Error: 无法打开设备 /dev/video{int(args.device_index)}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"成功连接设备 /dev/video{int(args.device_index)}")
    print(f"当前采集参数: {width}x{height} @ {fps}FPS")
    print(f"输出目录: {output_root}")
    print(f"原始图像: {raw_dir}")
    print(f"分割掩码: {mask_dir}")
    print(f"叠加结果: {overlay_dir}")
    print(f"误差日志: {csv_path}")
    print("提示: 选中弹出的视频窗口，按下键盘上的小写字母 'q' 即可退出。")

    csv_file = csv_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
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
        ],
    )
    writer.writeheader()
    csv_file.flush()

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: 无法获取图像帧，数据流可能已中断。")
                break

            timestamp_sec = time.time()

            lumen_center: Optional[Tuple[int, int]] = None
            inference_ms: Optional[float] = None
            status_text = "RAW_ONLY"
            mask_u8 = np.zeros(frame.shape[:2], dtype=np.uint8)

            if analysis_enabled:
                try:
                    lumen_center, mask_u8, inference_ms, status_text = analyze_frame(frame)
                except Exception as exc:
                    status_text = f"ANALYSIS_ERROR: {exc}"
                    mask_u8 = np.zeros(frame.shape[:2], dtype=np.uint8)

            scope_center, error_x_px, error_y_px, error_norm_px = compute_center_error(
                lumen_center=lumen_center,
                frame_shape=frame.shape,
            )
            frame_name = f"frame_{frame_idx:06d}"
            overlay_frame = build_overlay_frame(
                frame_bgr=frame,
                mask_u8=mask_u8,
                scope_center=scope_center,
                lumen_center=lumen_center,
                error_x_px=error_x_px,
                error_y_px=error_y_px,
                error_norm_px=error_norm_px,
                inference_ms=inference_ms,
                status_text=status_text,
            )

            raw_path = raw_dir / f"{frame_name}.jpg"
            mask_path = mask_dir / f"{frame_name}.png"
            overlay_path = overlay_dir / f"{frame_name}.jpg"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(str(mask_path), mask_u8)
            cv2.imwrite(str(overlay_path), overlay_frame)

            write_metric_row(
                writer,
                csv_file,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                frame_shape=frame.shape,
                scope_center=scope_center,
                lumen_center=lumen_center,
                error_x_px=error_x_px,
                error_y_px=error_y_px,
                error_norm_px=error_norm_px,
                inference_ms=inference_ms,
                status_text=status_text,
            )

            cv2.imshow(WINDOW_NAME, overlay_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("收到退出指令，正在关闭...")
                break

            frame_idx += 1

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
