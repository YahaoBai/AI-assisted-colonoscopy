import argparse
import os
import time
from glob import glob

import numpy as np
import torch

from v3.checkpoint_utils import load_checkpoint, extract_model_state_dict, load_normalizer_from_checkpoint
from v3.model_convlstm3f import ConvLSTM3FPolicy


def get_args():
    parser = argparse.ArgumentParser(
        description='Inference for V3 ConvLSTM3F policy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', '--model-dir', type=str, default='checkpoints')
    parser.add_argument('--model-name', type=str, default='convlstm3f_best.pth')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to one .npz or a directory containing .npz files')
    parser.add_argument('--output-dir', type=str, default='inference_results_v3/convlstm3f')
    return parser.parse_args()


def get_input_files(input_path: str):
    if os.path.isfile(input_path):
        if not input_path.endswith('.npz'):
            raise ValueError(f'Input file must be .npz, got: {input_path}')
        return [input_path]

    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, '*.npz')))
        if not files:
            raise ValueError(f'No .npz files found in directory: {input_path}')
        return files

    raise ValueError(f'Input path does not exist: {input_path}')


def build_model_from_checkpoint(checkpoint):
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
    return model


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = os.path.join(args.model_dir, args.model_name)
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)

    model = build_model_from_checkpoint(checkpoint)
    model.to(device=device)
    model.eval()

    normalizer = load_normalizer_from_checkpoint(checkpoint)
    print(normalizer.summary())

    input_files = get_input_files(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)

    all_yaw_errors = []
    all_pitch_errors = []
    all_abs_errors = []
    results = []

    total_infer_time_sec = 0.0

    for idx, path in enumerate(input_files):
        print(f"[{idx + 1}/{len(input_files)}] {os.path.basename(path)}")
        data = np.load(path)

        if 'img' not in data or 'action' not in data:
            raise KeyError(f'Missing img/action in {path}')

        img_stack = data['img']
        action_gt = data['action'].astype(np.float32)

        if img_stack.ndim != 3 or img_stack.shape[0] != 3:
            raise ValueError(f'Expected img shape (3,H,W), got {img_stack.shape} in {path}')

        img_stack = img_stack.astype(np.float32)
        if img_stack.max() > 1.0:
            img_stack = img_stack / 255.0
        img_stack = np.clip(img_stack, 0.0, 1.0)

        x = torch.from_numpy(img_stack).float().unsqueeze(0).unsqueeze(2).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_norm = model(x).cpu().numpy()[0]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        total_infer_time_sec += (t1 - t0)

        pred_action = normalizer.denormalize(pred_norm).astype(np.float32)

        yaw_denom = max(abs(float(action_gt[0])), 1e-6)
        pitch_denom = max(abs(float(action_gt[1])), 1e-6)
        yaw_error = abs(float(action_gt[0] - pred_action[0])) / yaw_denom * 100.0
        pitch_error = abs(float(action_gt[1] - pred_action[1])) / pitch_denom * 100.0

        gt_norm = max(float(np.sqrt(action_gt[0] ** 2 + action_gt[1] ** 2)), 1e-6)
        diff_norm = float(np.sqrt((action_gt[0] - pred_action[0]) ** 2 + (action_gt[1] - pred_action[1]) ** 2))
        abs_error = diff_norm / gt_norm * 100.0

        all_yaw_errors.append(yaw_error)
        all_pitch_errors.append(pitch_error)
        all_abs_errors.append(abs_error)

        results.append({
            'file': os.path.basename(path),
            'gt_yaw': float(action_gt[0]),
            'gt_pitch': float(action_gt[1]),
            'pred_yaw': float(pred_action[0]),
            'pred_pitch': float(pred_action[1]),
            'yaw_error': yaw_error,
            'pitch_error': pitch_error,
            'abs_error': abs_error,
        })

    num_samples = len(input_files)
    avg_infer_time_ms = (total_infer_time_sec / num_samples * 1000.0) if num_samples > 0 else 0.0
    fps = (num_samples / total_infer_time_sec) if total_infer_time_sec > 0 else 0.0

    print('=' * 60)
    print(f'Total samples: {num_samples}')
    print(f"Yaw Error Mean (%): {np.mean(all_yaw_errors):.6f}")
    print(f"Pitch Error Mean (%): {np.mean(all_pitch_errors):.6f}")
    print(f"Abs Error Mean (%): {np.mean(all_abs_errors):.6f}")
    print(f'Avg Inference Time: {avg_infer_time_ms:.3f} ms/sample')
    print(f'FPS: {fps:.3f}')
    print('=' * 60)

    save_path = os.path.join(args.output_dir, 'inference_results.txt')
    with open(save_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('Inference Results Summary (V3 ConvLSTM3F)\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Total samples: {num_samples}\n')
        f.write(f'Avg Inference Time (ms/sample): {avg_infer_time_ms:.6f}\n')
        f.write(f'FPS: {fps:.6f}\n\n')

        for r in results:
            f.write(f"File: {r['file']}\n")
            f.write(f"  GT      - Yaw: {r['gt_yaw']:.6f}, Pitch: {r['gt_pitch']:.6f}\n")
            f.write(f"  Pred    - Yaw: {r['pred_yaw']:.6f}, Pitch: {r['pred_pitch']:.6f}\n")
            f.write(f"  Error(%) - Yaw: {r['yaw_error']:.6f}, Pitch: {r['pitch_error']:.6f}, Abs: {r['abs_error']:.6f}\n\n")

        f.write('Statistics\n')
        f.write(f"Yaw Mean (%): {np.mean(all_yaw_errors):.6f}\n")
        f.write(f"Pitch Mean (%): {np.mean(all_pitch_errors):.6f}\n")
        f.write(f"Abs Mean (%): {np.mean(all_abs_errors):.6f}\n")

    print(f'Saved results to {save_path}')


if __name__ == '__main__':
    main()
