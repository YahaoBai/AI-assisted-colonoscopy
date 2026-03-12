import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _same_type_as(reference, array: np.ndarray):
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)
    return array


@dataclass
class ActionNormConfig:
    quantile: float = 0.995
    clip_norm: bool = True
    min_scale: float = 1e-6
    unit: str = 'rad'


class ActionNormalizer:
    def __init__(
        self,
        norm_mode: str = 'robust_quantile',
        config: Optional[ActionNormConfig] = None,
        fixed_degree: float = 1.0,
    ):
        if norm_mode not in ('robust_quantile', 'fixed_degree'):
            raise ValueError(f'Unsupported norm_mode: {norm_mode}')
        self.norm_mode = norm_mode
        self.config = config or ActionNormConfig()
        self.fixed_degree = float(fixed_degree)

        self.scale_yaw: Optional[float] = None
        self.scale_pitch: Optional[float] = None
        self.fitted = False

    def fit(self, train_actions=None):
        if self.norm_mode == 'fixed_degree':
            fixed_scale = max(deg2rad(self.fixed_degree), self.config.min_scale)
            self.scale_yaw = float(fixed_scale)
            self.scale_pitch = float(fixed_scale)
            self.fitted = True
            return

        if train_actions is None:
            raise ValueError('train_actions is required for robust_quantile mode')

        arr = _to_numpy(train_actions).astype(np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f'Expected action shape [N,2], got {arr.shape}')

        yaw = np.abs(arr[:, 0])
        pitch = np.abs(arr[:, 1])
        q = float(self.config.quantile)

        self.scale_yaw = max(float(np.quantile(yaw, q)), self.config.min_scale)
        self.scale_pitch = max(float(np.quantile(pitch, q)), self.config.min_scale)
        self.fitted = True

    def normalize(self, actions):
        self._check_fitted()
        ref = actions
        arr = _to_numpy(actions).astype(np.float64)

        single = arr.ndim == 1
        if single:
            if arr.shape[0] != 2:
                raise ValueError(f'Expected shape [2], got {arr.shape}')
            arr2 = arr[None, :]
        else:
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f'Expected shape [N,2], got {arr.shape}')
            arr2 = arr

        out = np.empty_like(arr2, dtype=np.float64)
        out[:, 0] = arr2[:, 0] / self.scale_yaw
        out[:, 1] = arr2[:, 1] / self.scale_pitch

        if self.config.clip_norm:
            out = np.clip(out, -1.0, 1.0)

        if single:
            out = out[0]
        return _same_type_as(ref, out)

    def denormalize(self, norm_actions):
        self._check_fitted()
        ref = norm_actions
        arr = _to_numpy(norm_actions).astype(np.float64)

        single = arr.ndim == 1
        if single:
            if arr.shape[0] != 2:
                raise ValueError(f'Expected shape [2], got {arr.shape}')
            arr2 = arr[None, :]
        else:
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f'Expected shape [N,2], got {arr.shape}')
            arr2 = arr

        out = np.empty_like(arr2, dtype=np.float64)
        out[:, 0] = arr2[:, 0] * self.scale_yaw
        out[:, 1] = arr2[:, 1] * self.scale_pitch

        if single:
            out = out[0]
        return _same_type_as(ref, out)

    def normalize_torch(self, actions: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        scales = torch.tensor([self.scale_yaw, self.scale_pitch], dtype=actions.dtype, device=actions.device)
        out = actions / scales
        if self.config.clip_norm:
            out = torch.clamp(out, -1.0, 1.0)
        return out

    def denormalize_torch(self, norm_actions: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        scales = torch.tensor([self.scale_yaw, self.scale_pitch], dtype=norm_actions.dtype, device=norm_actions.device)
        return norm_actions * scales

    def to_checkpoint_dict(self):
        self._check_fitted()
        return {
            'norm_mode': self.norm_mode,
            'normalizer_config': asdict(self.config),
            'fixed_degree': float(self.fixed_degree),
            'scale_yaw': float(self.scale_yaw),
            'scale_pitch': float(self.scale_pitch),
        }

    @classmethod
    def from_checkpoint_dict(cls, payload):
        norm_mode = payload.get('norm_mode', payload.get('type', 'robust_quantile'))
        cfg_payload = payload.get('normalizer_config', {})
        config = ActionNormConfig(**cfg_payload) if cfg_payload else ActionNormConfig()
        fixed_degree = float(payload.get('fixed_degree', 1.0))

        obj = cls(norm_mode=norm_mode, config=config, fixed_degree=fixed_degree)
        if 'scale_yaw' in payload and 'scale_pitch' in payload:
            obj.scale_yaw = float(payload['scale_yaw'])
            obj.scale_pitch = float(payload['scale_pitch'])
            obj.fitted = True
        return obj

    def summary(self) -> str:
        self._check_fitted()
        return (
            'ActionNormalizer('\
            f'norm_mode={self.norm_mode}, '\
            f'quantile={self.config.quantile}, '\
            f'fixed_degree={self.fixed_degree}, '\
            f'scale_yaw={self.scale_yaw:.8f}, '\
            f'scale_pitch={self.scale_pitch:.8f})'
        )

    def _check_fitted(self):
        if not self.fitted or self.scale_yaw is None or self.scale_pitch is None:
            raise RuntimeError('ActionNormalizer is not fitted yet.')
