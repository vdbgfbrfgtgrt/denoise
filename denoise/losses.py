"""Loss functions for DFGU-Net.

The paper uses a combination of mean squared error (MSE) and structural
similarity (SSIM) loss to balance pixel-wise accuracy and structural
preservation:

    L = alpha * MSE(x, y) + beta * (1 - SSIM(x, y))
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    g1 = _gaussian(window_size, 1.5)
    g2 = g1.unsqueeze(1) * g1.unsqueeze(0)
    return g2.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor,
              window: torch.Tensor, window_size: int,
              channel: int) -> torch.Tensor:
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


class SSIMLoss(nn.Module):
    """1 - SSIM loss averaged over the batch."""

    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", _create_window(window_size, channel))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ch = pred.shape[1]
        if ch != self.channel:
            self.channel = ch
            self.window = _create_window(self.window_size, ch).to(pred.device)
        ssim_map = _ssim_map(pred, target, self.window, self.window_size, ch)
        return 1.0 - ssim_map.mean()


class DFGULoss(nn.Module):
    """Combined MSE + (1 - SSIM) loss as described in the paper."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(window_size=window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        total = self.alpha * mse_loss + self.beta * ssim_loss
        return total, {"mse": mse_loss.item(),
                       "ssim": (1.0 - ssim_loss).item(),
                       "1-ssim": ssim_loss.item()}
