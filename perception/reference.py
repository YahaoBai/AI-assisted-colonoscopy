from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _same_type_as(reference: ArrayLike, x_np: np.ndarray) -> ArrayLike:
    if TORCH_AVAILABLE and isinstance(reference, torch.Tensor):
        return torch.as_tensor(
            x_np,
            dtype=reference.dtype,
            device=reference.device,
        )
    return x_np


@dataclass
class ActionNormConfig:
    """
    归一化配置。

    quantile:
        用于统计训练集尺度的分位数，比如 0.99 或 0.995
    clip_norm:
        归一化后是否裁剪到 [-1, 1]
    min_scale:
        防止尺度过小导致数值爆炸
    unit:
        标签单位，仅作说明用途，不参与计算
    """
    quantile: float = 0.995
    clip_norm: bool = True
    min_scale: float = 1e-6
    unit: str = "rad"


class ActionNormalizer:
    """
    对二维动作 [delta_yaw, delta_pitch] 做鲁棒归一化。

    训练时:
        - fit(train_actions)
        - normalize(train_actions)

    推理时:
        - normalize(raw_actions)   # 通常用于调试，不一定需要
        - denormalize(pred_norm_actions)

    这里默认动作 shape 为 [N, 2] 或 [2]
    第 0 维: yaw
    第 1 维: pitch
    """

    def __init__(self, config: Optional[ActionNormConfig] = None):
        self.config = config or ActionNormConfig()
        self.scale_yaw: Optional[float] = None
        self.scale_pitch: Optional[float] = None
        self.fitted: bool = False

    def fit(self, train_actions: ArrayLike) -> None:
        """
        从训练集动作中统计归一化尺度。
        使用 abs(action) 的分位数作为尺度。

        参数:
            train_actions: shape [N, 2]
        """
        arr = _to_numpy(train_actions).astype(np.float64)
        self._validate_action_shape(arr)

        yaw = np.abs(arr[:, 0])
        pitch = np.abs(arr[:, 1])

        q = self.config.quantile
        self.scale_yaw = max(float(np.quantile(yaw, q)), self.config.min_scale)
        self.scale_pitch = max(float(np.quantile(pitch, q)), self.config.min_scale)
        self.fitted = True

    def normalize(self, actions: ArrayLike) -> ArrayLike:
        """
        将真实动作归一化到大致 [-1, 1] 区间。
        """
        self._check_fitted()
        ref = actions
        arr = _to_numpy(actions).astype(np.float64)
        single = self._is_single_action(arr)
        arr2 = self._ensure_2d(arr)

        out = np.empty_like(arr2, dtype=np.float64)
        out[:, 0] = arr2[:, 0] / self.scale_yaw
        out[:, 1] = arr2[:, 1] / self.scale_pitch

        if self.config.clip_norm:
            out = np.clip(out, -1.0, 1.0)

        if single:
            out = out[0]
        return _same_type_as(ref, out)

    def denormalize(self, norm_actions: ArrayLike) -> ArrayLike:
        """
        将网络输出的归一化动作还原回真实物理单位。
        """
        self._check_fitted()
        ref = norm_actions
        arr = _to_numpy(norm_actions).astype(np.float64)
        single = self._is_single_action(arr)
        arr2 = self._ensure_2d(arr)

        out = np.empty_like(arr2, dtype=np.float64)
        out[:, 0] = arr2[:, 0] * self.scale_yaw
        out[:, 1] = arr2[:, 1] * self.scale_pitch

        if single:
            out = out[0]
        return _same_type_as(ref, out)

    def get_scales(self) -> Dict[str, float]:
        self._check_fitted()
        return {
            "scale_yaw": float(self.scale_yaw),
            "scale_pitch": float(self.scale_pitch),
        }

    def save_json(self, path: Union[str, Path]) -> None:
        """
        保存归一化器配置和尺度。
        """
        self._check_fitted()
        payload = {
            "config": asdict(self.config),
            "scale_yaw": float(self.scale_yaw),
            "scale_pitch": float(self.scale_pitch),
            "fitted": self.fitted,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "ActionNormalizer":
        """
        从 json 文件恢复归一化器。
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(config=ActionNormConfig(**payload["config"]))
        obj.scale_yaw = float(payload["scale_yaw"])
        obj.scale_pitch = float(payload["scale_pitch"])
        obj.fitted = bool(payload["fitted"])
        return obj

    def summary(self) -> str:
        self._check_fitted()
        return (
            f"ActionNormalizer(\n"
            f"  quantile={self.config.quantile},\n"
            f"  unit='{self.config.unit}',\n"
            f"  scale_yaw={self.scale_yaw:.8f},\n"
            f"  scale_pitch={self.scale_pitch:.8f}\n"
            f")"
        )

    @staticmethod
    def _validate_action_shape(arr: np.ndarray) -> None:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected action shape [N, 2], got {arr.shape}"
            )

    @staticmethod
    def _is_single_action(arr: np.ndarray) -> bool:
        return arr.ndim == 1 and arr.shape[0] == 2

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1 and arr.shape[0] == 2:
            return arr[None, :]
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        raise ValueError(f"Expected shape [2] or [N, 2], got {arr.shape}")

    def _check_fitted(self) -> None:
        if not self.fitted or self.scale_yaw is None or self.scale_pitch is None:
            raise RuntimeError("ActionNormalizer is not fitted yet.")


@dataclass
class SafetyConfig:
    """
    推理阶段安全配置。

    max_delta_yaw:
        单步最大允许 yaw 增量
    max_delta_pitch:
        单步最大允许 pitch 增量
    smooth_alpha:
        输出平滑系数，范围 (0, 1]
        越大越跟随当前输出，越小越平滑
    rate_limit_yaw:
        单步指令变化率限制（相对于上一时刻命令）
    rate_limit_pitch:
        单步指令变化率限制
    """
    max_delta_yaw: Optional[float] = None
    max_delta_pitch: Optional[float] = None
    smooth_alpha: float = 1.0
    rate_limit_yaw: Optional[float] = None
    rate_limit_pitch: Optional[float] = None


class ActionInferenceWrapper:
    """
    推理后处理器：
    1. 反归一化
    2. 输出平滑
    3. 变化率限制
    4. 安全裁剪

    注意：
    - 归一化器的尺度来自训练集，推理时固定不变
    - 安全限幅来自物理/控制约束
    """

    def __init__(self, normalizer: ActionNormalizer, safety: Optional[SafetyConfig] = None):
        self.normalizer = normalizer
        self.safety = safety or SafetyConfig()
        self.prev_cmd: Optional[np.ndarray] = None

    def reset(self) -> None:
        """
        在新 episode / 新视频 / 新轨迹开始前调用。
        """
        self.prev_cmd = None

    def step(self, pred_norm_action: ArrayLike) -> ArrayLike:
        """
        输入:
            pred_norm_action: shape [2]，通常是网络 tanh 输出
        输出:
            最终可发送的动作命令，shape [2]
        """
        ref = pred_norm_action
        pred_norm_np = _to_numpy(pred_norm_action).astype(np.float64)
        if pred_norm_np.shape != (2,):
            raise ValueError(f"Expected shape [2], got {pred_norm_np.shape}")

        # 1) 反归一化
        raw_action = _to_numpy(self.normalizer.denormalize(pred_norm_np)).astype(np.float64)

        # 2) 平滑
        smoothed = self._smooth(raw_action)

        # 3) 变化率限制
        rate_limited = self._apply_rate_limit(smoothed)

        # 4) 安全裁剪
        safe_action = self._apply_safety_clip(rate_limited)

        self.prev_cmd = safe_action.copy()
        return _same_type_as(ref, safe_action)

    def _smooth(self, action: np.ndarray) -> np.ndarray:
        alpha = float(self.safety.smooth_alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"smooth_alpha must be in (0, 1], got {alpha}")

        if self.prev_cmd is None:
            return action

        return alpha * action + (1.0 - alpha) * self.prev_cmd

    def _apply_rate_limit(self, action: np.ndarray) -> np.ndarray:
        if self.prev_cmd is None:
            return action

        out = action.copy()

        if self.safety.rate_limit_yaw is not None:
            dy = out[0] - self.prev_cmd[0]
            dy = np.clip(dy, -self.safety.rate_limit_yaw, self.safety.rate_limit_yaw)
            out[0] = self.prev_cmd[0] + dy

        if self.safety.rate_limit_pitch is not None:
            dp = out[1] - self.prev_cmd[1]
            dp = np.clip(dp, -self.safety.rate_limit_pitch, self.safety.rate_limit_pitch)
            out[1] = self.prev_cmd[1] + dp

        return out

    def _apply_safety_clip(self, action: np.ndarray) -> np.ndarray:
        out = action.copy()

        if self.safety.max_delta_yaw is not None:
            out[0] = np.clip(out[0], -self.safety.max_delta_yaw, self.safety.max_delta_yaw)

        if self.safety.max_delta_pitch is not None:
            out[1] = np.clip(out[1], -self.safety.max_delta_pitch, self.safety.max_delta_pitch)

        return out


def compute_dataset_stats(actions: ArrayLike, unit: str = "rad") -> Dict[str, float]:
    """
    方便你先看训练集标签分布。
    """
    arr = _to_numpy(actions).astype(np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape [N, 2], got {arr.shape}")

    yaw = np.abs(arr[:, 0])
    pitch = np.abs(arr[:, 1])

    stats = {
        "unit": unit,

        "yaw_mean_abs": float(np.mean(yaw)),
        "yaw_p95_abs": float(np.quantile(yaw, 0.95)),
        "yaw_p99_abs": float(np.quantile(yaw, 0.99)),
        "yaw_p995_abs": float(np.quantile(yaw, 0.995)),
        "yaw_max_abs": float(np.max(yaw)),

        "pitch_mean_abs": float(np.mean(pitch)),
        "pitch_p95_abs": float(np.quantile(pitch, 0.95)),
        "pitch_p99_abs": float(np.quantile(pitch, 0.99)),
        "pitch_p995_abs": float(np.quantile(pitch, 0.995)),
        "pitch_max_abs": float(np.max(pitch)),
    }
    return stats


def print_dataset_stats(actions: ArrayLike, unit: str = "rad") -> None:
    stats = compute_dataset_stats(actions, unit=unit)
    print("Dataset action stats:")
    for k, v in stats.items():
        if k == "unit":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.8f}")


# ----------------------------
# 下面是一个训练 / 推理使用示例
# ----------------------------

if __name__ == "__main__":
    # 例子：假设你的训练标签是 [N, 2]，单位为弧度
    # 这里只是造一些假数据演示
    rng = np.random.default_rng(42)

    n = 5000
    train_actions = np.zeros((n, 2), dtype=np.float64)

    # yaw 分布
    train_actions[:, 0] = rng.normal(loc=0.0, scale=deg2rad(0.35), size=n)

    # pitch 分布
    train_actions[:, 1] = rng.normal(loc=0.0, scale=deg2rad(0.25), size=n)

    # 加一点大动作样本
    idx = rng.choice(n, size=100, replace=False)
    train_actions[idx, 0] += rng.normal(0.0, deg2rad(0.8), size=len(idx))
    train_actions[idx, 1] += rng.normal(0.0, deg2rad(0.6), size=len(idx))

    print_dataset_stats(train_actions, unit="rad")

    # 1) 训练期：拟合归一化器
    normalizer = ActionNormalizer(
        ActionNormConfig(
            quantile=0.995,
            clip_norm=True,
            min_scale=1e-6,
            unit="rad",
        )
    )
    normalizer.fit(train_actions)
    print()
    print(normalizer.summary())

    # 2) 训练期：归一化标签
    train_actions_norm = normalizer.normalize(train_actions)
    print("\nNormalized sample:")
    print(train_actions_norm[:5])

    # 保存
    normalizer.save_json("action_normalizer.json")

    # 3) 推理期：加载归一化器
    loaded_normalizer = ActionNormalizer.load_json("action_normalizer.json")

    # 这里的安全阈值需要你将来根据真实“单步安全增量”来填
    # 现在先给个示意值
    safety = SafetyConfig(
        max_delta_yaw=deg2rad(1.2),      # 每步最大 yaw 增量
        max_delta_pitch=deg2rad(1.0),    # 每步最大 pitch 增量
        smooth_alpha=0.5,                # 越小越平滑
        rate_limit_yaw=deg2rad(0.5),     # 相邻命令最大变化
        rate_limit_pitch=deg2rad(0.4),
    )

    wrapper = ActionInferenceWrapper(loaded_normalizer, safety=safety)
    wrapper.reset()

    # 4) 假设这是网络输出的归一化动作（tanh 输出）
    pred_norm_seq = np.array([
        [0.20, -0.10],
        [0.45, -0.20],
        [0.85, -0.60],
        [1.10, -1.20],   # 即便超出，也会在后续物理裁剪中被处理
    ], dtype=np.float64)

    print("\nInference:")
    for i, pred_norm in enumerate(pred_norm_seq):
        cmd = wrapper.step(pred_norm)
        print(
            f"step={i:02d} | pred_norm={pred_norm} "
            f"| cmd_rad={cmd} "
            f"| cmd_deg={[rad2deg(cmd[0]), rad2deg(cmd[1])]}"
        )