"""
Colonoscopy Navigation API - 极速推理接口 (I/O与算子优化版)
"""

import numpy as np
import torch

MODEL_PATH = './checkpoints/colonoscopy_net_best.pth'
DEVICE = 'auto'

_model = None
_normalizer = None
_device = None


def _load_model():
    global _model, _normalizer, _device
    
    if _model is not None:
        return
    
    if DEVICE == 'auto':
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(DEVICE)
    
    from perception.models import ColonoscopyNet
    from perception.reference import ActionNormalizer, ActionNormConfig
    
    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    
    _normalizer = ActionNormalizer(ActionNormConfig(**checkpoint['normalizer_config']))
    _normalizer.scale_yaw = checkpoint['scale_yaw']
    _normalizer.scale_pitch = checkpoint['scale_pitch']
    _normalizer.fitted = True
    
    _model = ColonoscopyNet(use_depth_decoder=False)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.to(_device)
    _model.eval()


def predict(img_stack: np.ndarray) -> np.ndarray:
    """
    根据三帧图像预测动作
    
    Args:
        img_stack: 图像栈，shape=(3, H, W)，dtype=np.uint8
    
    Returns:
        action: 动作 [delta_yaw, delta_pitch]，单位弧度，shape=(2,)
    """
    global _model, _normalizer, _device
    
    if _model is None:
        _load_model()
    
    # [极速优化 1]: 显存带宽压榨。
    # 将体积最小的 uint8 直接送入 GPU，开启 non_blocking 异步传输
    # 然后在 GPU 端原位 (in-place) 完成 float 转换和除法归一化
    img_tensor = torch.from_numpy(img_stack).to(_device, non_blocking=True)
    img_tensor = img_tensor.unsqueeze(0).float().div_(255.0)
    
    # [极速优化 2]: 纯推理模式与半精度加速
    with torch.inference_mode():
        if _device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = _model(img_tensor)
        else:
            pred = _model(img_tensor)
            
    # 将结果拉回 CPU 并解包
    pred_np = pred.cpu().numpy()[0]
    
    # 动作反归一化 (调用 reference.py 中的现有逻辑)
    return _normalizer.denormalize(pred_np).astype(np.float32)