import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import cv2

MODEL_PATH = './checkpoints/best_model666.pth'
IMAGE_SIZE = 256
THRESHOLD = 0.5  
GUIDED_RADIUS = 9
GUIDED_EPS = 0.01
MORPHOLOGY_KERNEL = 5

_model = None
_device = None


def init(model_path=MODEL_PATH):
    global _model, _device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=_device)
    from perception.attention_unet import AttentionUNet
    _model = AttentionUNet(in_channels=3, out_channels=1)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.to(_device)
    _model.eval()


def guided_filter_postprocess(mask, image, radius=GUIDED_RADIUS, eps=GUIDED_EPS):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x**2 + sobel_y**2)
    edge = np.clip(edge / (edge.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    guide = np.stack([edge, edge, edge], axis=2).astype(np.float32)
    
    guide = guide.astype(np.float32)
    mask_float = mask.astype(np.float32)
    
    if guide.shape[:2] != mask.shape[:2]:
        guide = cv2.resize(guide, (mask.shape[1], mask.shape[0]))
    
    result = cv2.ximgproc.guidedFilter(guide=guide, src=mask_float, radius=radius, eps=eps)
    
    return result


def morphology_close(mask, kernel_size=MORPHOLOGY_KERNEL):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


# [极速优化 1]: 使用 OpenCV 替换 SciPy 的连通域提取
def largest_connected_component(mask):
    # OpenCV 要求输入类型为 uint8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:  # 如果只有背景
        return mask
    # 找到除背景(0)外，面积最大的连通域索引
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = (labels == max_label).astype(np.uint8)
    return result


# [极速优化 2]: 使用 OpenCV 轮廓法替换 SciPy 孔洞填充
def binary_fill_holes_cv2(mask):
    mask_255 = mask * 255
    # 查找轮廓及其层级关系
    contours, hierarchy = cv2.findContours(mask_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours)):
            # 如果存在父轮廓，说明该轮廓是内部孔洞，将其填白
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(mask_255, contours, i, 255, -1)
    return (mask_255 > 0).astype(np.uint8)


def get_lumen_center(image, return_time=False, 
                     use_guided_filter=True,
                     guided_radius=GUIDED_RADIUS,
                     guided_eps=GUIDED_EPS,
                     use_morphology=True,
                     morphology_kernel=MORPHOLOGY_KERNEL,
                     use_postprocessing=True,
                     threshold=THRESHOLD):
    global _model, _device
    
    if _model is None:
        init()
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    if isinstance(image, Image.Image):
        original_size = image.size 
        image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = np.array(image_resized)
    else:
        original_size = (image.shape[1], image.shape[0])
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image_np = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            image_np = image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_np).unsqueeze(0).to(_device)
    
    start_time = time.time()
    
    # [极速优化 3]: 启用 Inference Mode 和 FP16 半精度加速
    with torch.inference_mode():
        if _device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = _model(image_tensor)
        else:
            output = _model(image_tensor)
            
    inference_time = time.time() - start_time
    
    mask = output.squeeze().cpu().numpy()
    
    if use_guided_filter:
        mask_filtered = guided_filter_postprocess(
            mask, image_np, 
            radius=guided_radius, 
            eps=guided_eps
        )
        mask = mask_filtered
    
    mask_binary = (mask > threshold).astype(np.uint8)
    
    if use_morphology:
        mask_binary = morphology_close(mask_binary, kernel_size=morphology_kernel)
    
    if use_postprocessing:
        mask_binary = largest_connected_component(mask_binary)
        
    mask_binary = binary_fill_holes_cv2(mask_binary)
    
    mask_original_size = cv2.resize(mask_binary, original_size, interpolation=cv2.INTER_NEAREST)
    
    if mask_binary.sum() == 0:
        return (None, mask_original_size, inference_time) if return_time else (None, mask_original_size)
    
    # [极速优化 4]: 使用 OpenCV 替换 SciPy 的欧氏距离变换
    distance = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    max_idx = np.unravel_index(np.argmax(distance), distance.shape)
    center_y, center_x = max_idx
    
    scale_x = original_size[0] / IMAGE_SIZE
    scale_y = original_size[1] / IMAGE_SIZE
    center = (int(center_x * scale_x), int(center_y * scale_y))
    
    return (center, mask_original_size, inference_time) if return_time else (center, mask_original_size)


def get_segmentation_mask(image, return_time=False,
                          use_guided_filter=True,
                          guided_radius=GUIDED_RADIUS,
                          guided_eps=GUIDED_EPS,
                          use_morphology=True,
                          morphology_kernel=MORPHOLOGY_KERNEL,
                          use_postprocessing=True,
                          threshold=THRESHOLD):
    global _model, _device
    
    if _model is None:
        init()
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    if isinstance(image, Image.Image):
        original_size = image.size
        image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = np.array(image_resized)
    else:
        original_size = (image.shape[1], image.shape[0])
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image_np = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            image_np = image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_np).unsqueeze(0).to(_device)
    
    start_time = time.time()
    
    with torch.inference_mode():
        if _device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = _model(image_tensor)
        else:
            output = _model(image_tensor)
            
    inference_time = time.time() - start_time
    
    mask = output.squeeze().cpu().numpy()
    
    if use_guided_filter:
        mask_filtered = guided_filter_postprocess(
            mask, image_np, 
            radius=guided_radius, 
            eps=guided_eps
        )
        mask = mask_filtered
    
    mask_binary = (mask > threshold).astype(np.uint8)
    
    if use_morphology:
        mask_binary = morphology_close(mask_binary, kernel_size=morphology_kernel)
    
    if use_postprocessing:
        mask_binary = largest_connected_component(mask_binary)
        
    mask_binary = binary_fill_holes_cv2(mask_binary)
    
    return (mask_binary, inference_time) if return_time else mask_binary