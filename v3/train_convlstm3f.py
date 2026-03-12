import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from v3.checkpoint_utils import (
    extract_model_state_dict,
    load_checkpoint,
    load_normalizer_from_checkpoint,
    save_checkpoint,
)
from v3.dataset_orig3f import (
    Original3FrameDataset,
    compute_action_statistics_orig3f,
    load_actions_by_indices,
)
from v3.losses import smooth_l1_turn_weighted_loss
from v3.model_convlstm3f import ConvLSTM3FPolicy
from v3.norm import ActionNormConfig, ActionNormalizer


def parse_hidden_channels(s: str):
    values = [v.strip() for v in s.split(',') if v.strip()]
    if not values:
        raise ValueError('hidden_channels cannot be empty')
    return [int(v) for v in values]


def get_args():
    parser = argparse.ArgumentParser(
        description='Train V3 CNN+ConvLSTM policy on original 3-frame dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=8)
    parser.add_argument('-l', '--learning-rate', dest='lr', type=float, default=1e-4)
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.05)

    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir', type=str, default='ildata1')
    parser.add_argument('-m', '--model-dir', dest='model_dir', type=str, default='checkpoints_v3/convlstm3f')

    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--log-dir', type=str, default=None)

    parser.add_argument('--hidden-channels', type=str, default='128,128')
    parser.add_argument('--convlstm-kernel-size', type=int, default=3)
    parser.add_argument('--mlp-hidden-dim', type=int, default=128)

    parser.add_argument('--norm-mode', type=str, choices=['robust_quantile', 'fixed_degree'], default='robust_quantile')
    parser.add_argument('--norm-quantile', type=float, default=0.995)
    parser.add_argument('--fixed-degree', type=float, default=1.0)
    parser.add_argument('--clip-norm', dest='clip_norm', action='store_true')
    parser.add_argument('--no-clip-norm', dest='clip_norm', action='store_false')
    parser.set_defaults(clip_norm=True)

    parser.add_argument('--smoothl1-beta', type=float, default=0.1)
    parser.add_argument('--turn-threshold-deg', type=float, default=0.35)
    parser.add_argument('--turn-weight-alpha', type=float, default=3.0)
    parser.add_argument('--turn-weight-max', type=float, default=3.0)

    parser.add_argument('--rotation-deg', type=float, default=5.0)
    parser.add_argument('--hflip-p', type=float, default=0.5)

    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--load-checkpoint', type=str, default=None)
    parser.add_argument('--compute-stats', action='store_true')

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_datasets(args):
    full_train_dataset = Original3FrameDataset(
        dataset_dir=args.dataset_dir,
        train_flag=True,
        image_size=args.image_size,
        hflip_p=args.hflip_p,
        rotation_deg=args.rotation_deg,
    )
    full_val_dataset = Original3FrameDataset(
        dataset_dir=args.dataset_dir,
        train_flag=False,
        image_size=args.image_size,
    )

    if len(full_train_dataset) != len(full_val_dataset):
        raise RuntimeError('Train/val dataset length mismatch.')

    dataset_size = len(full_train_dataset)
    val_size = int(dataset_size * args.val)
    if dataset_size > 1:
        val_size = max(1, min(val_size, dataset_size - 1))
    else:
        val_size = 0

    train_size = dataset_size - val_size
    if train_size <= 0:
        raise ValueError('Train split is empty. Please reduce validation ratio.')

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices) if val_size > 0 else None

    return full_train_dataset, train_dataset, val_dataset, train_indices, val_indices


def build_normalizer(args, full_train_dataset, train_indices):
    config = ActionNormConfig(
        quantile=float(args.norm_quantile),
        clip_norm=bool(args.clip_norm),
        min_scale=1e-6,
        unit='rad',
    )
    normalizer = ActionNormalizer(
        norm_mode=args.norm_mode,
        config=config,
        fixed_degree=float(args.fixed_degree),
    )

    if args.norm_mode == 'robust_quantile':
        train_actions = load_actions_by_indices(
            dataset_dir=args.dataset_dir,
            data_files=full_train_dataset.data_files,
            indices=train_indices,
            action_key='action',
        )
        normalizer.fit(train_actions)
    else:
        normalizer.fit()

    return normalizer


def run_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    normalizer,
    smoothl1_beta,
    turn_threshold_deg,
    turn_weight_alpha,
    turn_weight_max,
    train_mode,
    writer=None,
    global_step=0,
    epoch=0,
    stage='train',
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_weighted = 0.0
    total_unweighted = 0.0
    total_turn_ratio = 0.0
    total_mean_weight = 0.0

    pbar = tqdm(dataloader, desc=f'{stage.capitalize()} Epoch {epoch}', leave=False)

    for batch in pbar:
        frames, actions, _ = batch
        frames = frames.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)

        target_norm = normalizer.normalize_torch(actions)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred_norm = model(frames)
            loss_dict = smooth_l1_turn_weighted_loss(
                pred_norm_actions=pred_norm,
                target_norm_actions=target_norm,
                target_raw_actions=actions,
                beta=smoothl1_beta,
                turn_threshold_deg=turn_threshold_deg,
                turn_weight_alpha=turn_weight_alpha,
                turn_weight_max=turn_weight_max,
            )
            loss = loss_dict['loss_weighted']

            if train_mode:
                loss.backward()
                optimizer.step()

        total_weighted += loss_dict['loss_weighted'].item()
        total_unweighted += loss_dict['loss_unweighted'].item()
        total_turn_ratio += loss_dict['turn_ratio'].item()
        total_mean_weight += loss_dict['mean_weight'].item()

        if train_mode:
            global_step += 1
            if writer is not None:
                writer.add_scalar('train/loss_weighted', loss_dict['loss_weighted'].item(), global_step=global_step)
                writer.add_scalar('train/loss_unweighted', loss_dict['loss_unweighted'].item(), global_step=global_step)
                writer.add_scalar('train/turn_ratio', loss_dict['turn_ratio'].item(), global_step=global_step)
                writer.add_scalar('train/mean_weight', loss_dict['mean_weight'].item(), global_step=global_step)

        pbar.set_postfix({
            'WLoss': f"{loss_dict['loss_weighted'].item():.4f}",
            'ULoss': f"{loss_dict['loss_unweighted'].item():.4f}",
            'Turn%': f"{loss_dict['turn_ratio'].item() * 100.0:.1f}",
            'Wmean': f"{loss_dict['mean_weight'].item():.3f}",
        })

    n = len(dataloader)
    metrics = {
        'loss_weighted': total_weighted / n,
        'loss_unweighted': total_unweighted / n,
        'turn_ratio': total_turn_ratio / n,
        'mean_weight': total_mean_weight / n,
    }

    if (not train_mode) and writer is not None:
        writer.add_scalar('val/loss_weighted', metrics['loss_weighted'], global_step=epoch)
        writer.add_scalar('val/loss_unweighted', metrics['loss_unweighted'], global_step=epoch)
        writer.add_scalar('val/turn_ratio', metrics['turn_ratio'], global_step=epoch)
        writer.add_scalar('val/mean_weight', metrics['mean_weight'], global_step=epoch)

    return metrics, global_step


def main():
    args = get_args()
    set_seed(args.seed)

    if args.compute_stats:
        compute_action_statistics_orig3f(args.dataset_dir)
        return

    hidden_channels = parse_hidden_channels(args.hidden_channels)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    full_train_dataset, train_dataset, val_dataset, train_indices, _ = build_datasets(args)
    print(f'Dataset size: {len(full_train_dataset)}')
    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset) if val_dataset is not None else 0}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = ConvLSTM3FPolicy(
        hidden_channels=hidden_channels,
        convlstm_kernel_size=args.convlstm_kernel_size,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    model.to(device=device)

    start_epoch = 0

    if args.load_checkpoint:
        checkpoint = load_checkpoint(args.load_checkpoint, map_location=device)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        normalizer = load_normalizer_from_checkpoint(checkpoint)
        start_epoch = int(checkpoint.get('epoch', 0))
        print(f'Loaded checkpoint: {args.load_checkpoint}, resume from epoch {start_epoch + 1}')
    else:
        normalizer = build_normalizer(args, full_train_dataset, train_indices)

    print(normalizer.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if args.tensorboard:
        if not TENSORBOARD_AVAILABLE:
            print('Warning: tensorboardX not installed. TensorBoard disabled.')
            writer = None
        else:
            log_dir = args.log_dir if args.log_dir else os.path.join(args.model_dir, 'logs')
            writer = SummaryWriter(log_dir=log_dir)
            print(f'TensorBoard logs -> {log_dir}')
    else:
        writer = None

    os.makedirs(args.model_dir, exist_ok=True)

    best_val = float('inf')
    global_step = 0

    model_config = {
        'arch': 'cnn_convlstm3f',
        'cnn_backbone': 'resnet18_layer2',
        'hidden_channels': list(hidden_channels),
        'convlstm_kernel_size': int(args.convlstm_kernel_size),
        'mlp_hidden_dim': int(args.mlp_hidden_dim),
        'input_frames': 3,
        'use_tanh': True,
    }
    loss_config = {
        'type': 'smoothl1_turn_weighted',
        'smoothl1_beta': float(args.smoothl1_beta),
        'turn_threshold_deg': float(args.turn_threshold_deg),
        'turn_weight_alpha': float(args.turn_weight_alpha),
        'turn_weight_max': float(args.turn_weight_max),
    }

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 50}")

        train_metrics, global_step = run_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            normalizer=normalizer,
            smoothl1_beta=args.smoothl1_beta,
            turn_threshold_deg=args.turn_threshold_deg,
            turn_weight_alpha=args.turn_weight_alpha,
            turn_weight_max=args.turn_weight_max,
            train_mode=True,
            writer=writer,
            global_step=global_step,
            epoch=epoch + 1,
            stage='train',
        )

        print(
            f"Train - WLoss: {train_metrics['loss_weighted']:.4f}, "
            f"ULoss: {train_metrics['loss_unweighted']:.4f}, "
            f"Turn%: {train_metrics['turn_ratio'] * 100.0:.2f}, "
            f"Wmean: {train_metrics['mean_weight']:.4f}"
        )

        if writer is not None:
            writer.add_scalar('train_epoch/loss_weighted', train_metrics['loss_weighted'], global_step=epoch + 1)
            writer.add_scalar('train_epoch/loss_unweighted', train_metrics['loss_unweighted'], global_step=epoch + 1)
            writer.add_scalar('train_epoch/turn_ratio', train_metrics['turn_ratio'], global_step=epoch + 1)
            writer.add_scalar('train_epoch/mean_weight', train_metrics['mean_weight'], global_step=epoch + 1)

        if val_loader is not None:
            val_metrics, _ = run_one_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=optimizer,
                device=device,
                normalizer=normalizer,
                smoothl1_beta=args.smoothl1_beta,
                turn_threshold_deg=args.turn_threshold_deg,
                turn_weight_alpha=args.turn_weight_alpha,
                turn_weight_max=args.turn_weight_max,
                train_mode=False,
                writer=writer,
                global_step=global_step,
                epoch=epoch + 1,
                stage='val',
            )
            current_val = val_metrics['loss_weighted']
            print(
                f"Val   - WLoss: {val_metrics['loss_weighted']:.4f}, "
                f"ULoss: {val_metrics['loss_unweighted']:.4f}, "
                f"Turn%: {val_metrics['turn_ratio'] * 100.0:.2f}, "
                f"Wmean: {val_metrics['mean_weight']:.4f}"
            )
        else:
            val_metrics = None
            current_val = train_metrics['loss_weighted']

        scheduler.step()

        if current_val < best_val:
            best_val = current_val
            best_path = os.path.join(args.model_dir, 'convlstm3f_best.pth')
            save_checkpoint(
                path=best_path,
                model=model,
                model_config=model_config,
                normalizer=normalizer,
                loss_config=loss_config,
                epoch=epoch + 1,
                val_loss=current_val,
                extra={
                    'augmentation_config': {
                        'hflip_p': float(args.hflip_p),
                        'rotation_deg': float(args.rotation_deg),
                    }
                },
            )
            print(f'Saved best checkpoint: {best_path}')

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.model_dir, f'convlstm3f_{epoch + 1}.pth')
            save_checkpoint(
                path=ckpt_path,
                model=model,
                model_config=model_config,
                normalizer=normalizer,
                loss_config=loss_config,
                epoch=epoch + 1,
                val_loss=current_val,
                extra={
                    'augmentation_config': {
                        'hflip_p': float(args.hflip_p),
                        'rotation_deg': float(args.rotation_deg),
                    }
                },
            )
            print(f'Saved periodic checkpoint: {ckpt_path}')

    if writer is not None:
        writer.close()

    print('\nTraining complete.')
    print(f'Best weighted val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
