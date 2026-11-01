"""Loss terms used to train the conditional VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Return mean single-scale SSIM for images in the [0, 1] range."""
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    prediction_variance = (
        F.avg_pool2d(prediction.square(), window_size, stride=1, padding=padding)
        - mean_prediction.square()
    )
    target_variance = (
        F.avg_pool2d(target.square(), window_size, stride=1, padding=padding)
        - mean_target.square()
    )
    covariance = (
        F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding)
        - mean_prediction * mean_target
    )

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
    denominator = (mean_prediction.square() + mean_target.square() + c1) * (
        prediction_variance + target_variance + c2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def cvae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    ssim_weight: float,
    edge_weight: float,
) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(reconstruction, target)
    ssim = 1.0 - structural_similarity(reconstruction, target)
    horizontal_edges = F.l1_loss(
        reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
    )
    vertical_edges = F.l1_loss(
        reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :],
        target[:, :, 1:, :] - target[:, :, :-1, :],
    )
    edge = 0.5 * (horizontal_edges + vertical_edges)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp(), dim=1).mean()
    total = l1 + ssim_weight * ssim + edge_weight * edge + beta * kl
    return {"total": total, "l1": l1, "ssim": ssim, "edge": edge, "kl": kl}
