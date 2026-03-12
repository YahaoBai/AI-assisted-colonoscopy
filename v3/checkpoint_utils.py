import os
from typing import Any, Dict, Optional

import torch

from v3.norm import ActionNormalizer


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def extract_model_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = checkpoint.get('model_state_dict')
    if state_dict is not None:
        return state_dict

    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError('Cannot find model state dict in checkpoint.')


def build_checkpoint_payload(
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    normalizer: ActionNormalizer,
    loss_config: Dict[str, Any],
    epoch: int,
    val_loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    norm_dict = normalizer.to_checkpoint_dict()

    payload = {
        'model_state_dict': model.state_dict(),
        'model_config': dict(model_config),
        'normalization': dict(norm_dict),
        'norm_mode': norm_dict['norm_mode'],
        'normalizer_config': dict(norm_dict['normalizer_config']),
        'scale_yaw': float(norm_dict['scale_yaw']),
        'scale_pitch': float(norm_dict['scale_pitch']),
        'loss_config': dict(loss_config),
        'epoch': int(epoch),
    }
    if val_loss is not None:
        payload['val_loss'] = float(val_loss)
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    normalizer: ActionNormalizer,
    loss_config: Dict[str, Any],
    epoch: int,
    val_loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(path)
    payload = build_checkpoint_payload(
        model=model,
        model_config=model_config,
        normalizer=normalizer,
        loss_config=loss_config,
        epoch=epoch,
        val_loss=val_loss,
        extra=extra,
    )
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Any = 'cpu') -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_normalizer_from_checkpoint(checkpoint: Dict[str, Any]) -> ActionNormalizer:
    if 'normalization' in checkpoint:
        normalizer = ActionNormalizer.from_checkpoint_dict(checkpoint['normalization'])
        if normalizer.fitted:
            return normalizer

    if 'norm_mode' in checkpoint and 'normalizer_config' in checkpoint and 'scale_yaw' in checkpoint and 'scale_pitch' in checkpoint:
        payload = {
            'norm_mode': checkpoint['norm_mode'],
            'normalizer_config': checkpoint['normalizer_config'],
            'scale_yaw': checkpoint['scale_yaw'],
            'scale_pitch': checkpoint['scale_pitch'],
        }
        normalizer = ActionNormalizer.from_checkpoint_dict(payload)
        if normalizer.fitted:
            return normalizer

    if 'scale_yaw' in checkpoint and 'scale_pitch' in checkpoint:
        payload = {
            'norm_mode': 'fixed_degree',
            'normalizer_config': {'quantile': 0.995, 'clip_norm': True, 'min_scale': 1e-6, 'unit': 'rad'},
            'scale_yaw': checkpoint['scale_yaw'],
            'scale_pitch': checkpoint['scale_pitch'],
        }
        normalizer = ActionNormalizer.from_checkpoint_dict(payload)
        normalizer.fitted = True
        return normalizer

    normalizer = ActionNormalizer(norm_mode='fixed_degree', fixed_degree=1.0)
    normalizer.fit()
    return normalizer
