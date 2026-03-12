import os
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Original3FrameDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        train_flag: bool = True,
        image_size: int = 256,
        hflip_p: float = 0.5,
        rotation_deg: float = 5.0,
        img_key: str = 'img',
        action_key: str = 'action',
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_flag = train_flag
        self.image_size = int(image_size)
        self.hflip_p = float(hflip_p) if train_flag else 0.0
        self.rotation_deg = float(rotation_deg) if train_flag else 0.0
        self.img_key = img_key
        self.action_key = action_key

        self.data_files = sorted([f for f in os.listdir(self.dataset_dir) if f.endswith('.npz')])
        self.dataset_size = len(self.data_files)

    def __len__(self):
        return self.dataset_size

    def _sample_geometric_params(self) -> Tuple[bool, float]:
        if not self.train_flag:
            return False, 0.0

        do_flip = np.random.rand() < self.hflip_p
        if self.rotation_deg > 0.0:
            angle_deg = float(np.random.uniform(-self.rotation_deg, self.rotation_deg))
        else:
            angle_deg = 0.0
        return do_flip, angle_deg

    @staticmethod
    def _transform_action(action: np.ndarray, do_flip: bool, angle_deg: float) -> np.ndarray:
        out = action.astype(np.float32).copy()

        if do_flip:
            out[0] = -out[0]

        if abs(angle_deg) > 1e-8:
            theta = np.deg2rad(angle_deg)
            c, s = np.cos(theta), np.sin(theta)
            yaw, pitch = out[0], out[1]
            out[0] = yaw * c - pitch * s
            out[1] = yaw * s + pitch * c

        return out

    def _transform_frame(self, frame: np.ndarray, do_flip: bool, angle_deg: float) -> torch.Tensor:
        out = frame

        if do_flip:
            out = cv2.flip(out, 1)

        if abs(angle_deg) > 1e-8:
            h, w = out.shape[:2]
            center = (w * 0.5, h * 0.5)
            matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            out = cv2.warpAffine(
                out,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if out.shape[0] != self.image_size or out.shape[1] != self.image_size:
            out = cv2.resize(out, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        out = out.astype(np.float32)
        if out.max() > 1.0:
            out = out / 255.0
        out = np.clip(out, 0.0, 1.0)
        return torch.from_numpy(out).unsqueeze(0)

    def __getitem__(self, idx: int):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.dataset_dir, file_name)
        data = np.load(file_path)

        if self.img_key not in data or self.action_key not in data:
            raise KeyError(f'Missing keys in {file_name}. Required: {self.img_key}, {self.action_key}')

        img_stack = data[self.img_key]
        action = data[self.action_key].astype(np.float32)

        if img_stack.ndim != 3 or img_stack.shape[0] != 3:
            raise ValueError(f'Expected img shape (3,H,W), got {img_stack.shape} in {file_name}')
        if action.shape != (2,):
            raise ValueError(f'Expected action shape (2,), got {action.shape} in {file_name}')

        do_flip, angle_deg = self._sample_geometric_params()
        action = self._transform_action(action, do_flip, angle_deg)

        frames = [self._transform_frame(img_stack[t], do_flip, angle_deg) for t in range(img_stack.shape[0])]
        frame_tensor = torch.stack(frames, dim=0)  # (T,1,H,W)

        return frame_tensor, action, file_name


def load_actions_by_indices(
    dataset_dir: str,
    data_files: List[str],
    indices: Iterable[int],
    action_key: str = 'action',
) -> np.ndarray:
    actions = []
    for idx in indices:
        path = os.path.join(dataset_dir, data_files[int(idx)])
        payload = np.load(path)
        if action_key not in payload:
            raise KeyError(f'Missing action key {action_key} in {path}')
        action = payload[action_key].astype(np.float32)
        if action.shape != (2,):
            raise ValueError(f'Expected action shape (2,), got {action.shape} in {path}')
        actions.append(action)

    if not actions:
        raise RuntimeError('No actions found for given indices')

    return np.stack(actions, axis=0)


def compute_action_statistics_orig3f(dataset_dir: str, action_key: str = 'action'):
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npz')])
    actions = []
    for name in files:
        payload = np.load(os.path.join(dataset_dir, name))
        if action_key in payload:
            action = payload[action_key].astype(np.float32)
            if action.shape == (2,):
                actions.append(action)

    if not actions:
        raise RuntimeError('No valid action records found.')

    arr = np.stack(actions, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    min_val = arr.min(axis=0)
    max_val = arr.max(axis=0)

    print('Action statistics (orig3f):')
    print(f'  Mean: {mean}')
    print(f'  Std: {std}')
    print(f'  Min: {min_val}')
    print(f'  Max: {max_val}')

    return min_val, max_val
