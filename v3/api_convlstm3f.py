"""V3 minimal API for MuJoCo/ROS integration.
Only one public function is exposed: predict(img_stack).
"""

import os
import numpy as np
import torch

from v3.checkpoint_utils import load_checkpoint, extract_model_state_dict, load_normalizer_from_checkpoint
from v3.model_convlstm3f import ConvLSTM3FPolicy


def predict(img_stack: np.ndarray) -> np.ndarray:
    """
    Input:
        img_stack: shape=(3, H, W), dtype uint8/float32 mask stack.
    Output:
        action: [delta_yaw, delta_pitch], shape=(2,), unit=rad.
    """
    if not hasattr(predict, '_initialized'):
        model_path = os.environ.get('IL_V3_MODEL_PATH', './checkpoints/convlstm3f_best.pth')
        device_name = os.environ.get('IL_V3_DEVICE', 'auto')

        if device_name == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_name)

        checkpoint = load_checkpoint(model_path, map_location=device)

        model_cfg = checkpoint.get('model_config', {})
        hidden_channels = model_cfg.get('hidden_channels', [128, 128])
        kernel_size = int(model_cfg.get('convlstm_kernel_size', 3))
        mlp_hidden_dim = int(model_cfg.get('mlp_hidden_dim', 128))

        model = ConvLSTM3FPolicy(
            hidden_channels=list(hidden_channels),
            convlstm_kernel_size=kernel_size,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        model.load_state_dict(extract_model_state_dict(checkpoint))
        model.to(device)
        model.eval()

        normalizer = load_normalizer_from_checkpoint(checkpoint)

        predict._model = model
        predict._normalizer = normalizer
        predict._device = device
        predict._initialized = True

    arr = np.asarray(img_stack)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f'Expected img_stack shape (3,H,W), got {arr.shape}')

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2).to(device=predict._device, dtype=torch.float32)

    with torch.no_grad():
        pred_norm = predict._model(x).cpu().numpy()[0]

    action = predict._normalizer.denormalize(pred_norm).astype(np.float32)
    return action
