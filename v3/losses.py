import torch
import torch.nn.functional as F

from v3.norm import deg2rad


def smooth_l1_turn_weighted_loss(
    pred_norm_actions: torch.Tensor,
    target_norm_actions: torch.Tensor,
    target_raw_actions: torch.Tensor,
    beta: float = 0.1,
    turn_threshold_deg: float = 0.35,
    turn_weight_alpha: float = 3.0,
    turn_weight_max: float = 3.0,
):
    per_dim_loss = F.smooth_l1_loss(
        pred_norm_actions,
        target_norm_actions,
        reduction='none',
        beta=beta,
    )
    per_sample_loss = per_dim_loss.mean(dim=1)

    amplitude_rad = torch.max(torch.abs(target_raw_actions[:, 0]), torch.abs(target_raw_actions[:, 1]))
    threshold_rad = deg2rad(turn_threshold_deg)

    weights = 1.0 + turn_weight_alpha * torch.relu(amplitude_rad - threshold_rad)
    if turn_weight_max is not None and turn_weight_max > 0.0:
        weights = torch.clamp(weights, max=turn_weight_max)
    weights = torch.clamp(weights, min=1.0)

    loss_weighted = (per_sample_loss * weights).mean()
    loss_unweighted = per_sample_loss.mean()
    turn_ratio = (amplitude_rad >= threshold_rad).float().mean()
    mean_weight = weights.mean()

    return {
        'loss_weighted': loss_weighted,
        'loss_unweighted': loss_unweighted,
        'turn_ratio': turn_ratio,
        'mean_weight': mean_weight,
    }
