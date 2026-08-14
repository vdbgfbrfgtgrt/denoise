"""Utility functions for DFGU-Net.

Implements the harmonic mean filter (pre-denoising module) and assorted
helpers used by the denoising pipeline described in:
"Research on two-stage image denoising algorithm based on Feature Fusion
for neutron imaging".
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Pre-denoising module: Harmonic Mean Filter
# ---------------------------------------------------------------------------
def harmonic_mean_filter(images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply harmonic mean filtering to suppress white spot (salt) noise.

    Harmonic mean filter:
        g(i, j) = N^2 / sum_{window} 1 / f(i, j)

    where N is the window size. The harmonic mean filter is well suited to
    suppressing salt-type impulse noise while preserving the rest of the
    intensity distribution.

    Args:
        images: Tensor of shape (B, C, H, W), values in (0, 1].
        kernel_size: Window size N (paper uses 3x3 by default).

    Returns:
        Filtered tensor with the same shape as ``images``.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if images.min() <= 0:
        # Harmonic mean is undefined for zero pixels; clip to a small value.
        images = images.clamp_min(1e-6)

    pad = kernel_size // 2
    # Use average pooling on the inverse to obtain the arithmetic mean of 1/f.
    inv = 1.0 / images
    inv_mean = F.avg_pool2d(inv, kernel_size=kernel_size, stride=1, padding=pad)
    out = 1.0 / inv_mean.clamp_min(1e-12)
    return out


# ---------------------------------------------------------------------------
# Noise simulation (Poisson-Gaussian + white spot noise)
# ---------------------------------------------------------------------------
def add_white_spot_noise(images: torch.Tensor, ratio: float = 0.02,
                         intensity: float = 1.0) -> torch.Tensor:
    """Randomly replace pixels with high-intensity white spots (salt noise)."""
    out = images.clone()
    mask = (torch.rand_like(images) < ratio)
    out[mask] = intensity
    return out


def add_poisson_gaussian_noise(images: torch.Tensor, sigma: float = 0.02,
                               scale: float = 50.0) -> torch.Tensor:
    """Add Poisson-Gaussian mixed noise to simulate the neutron acquisition."""
    # Poisson component: scale up -> sample -> scale down.
    scaled = (images * scale).clamp_min(0)
    poisson = torch.poisson(scaled) / scale
    # Gaussian component.
    gaussian = torch.randn_like(images) * sigma
    noisy = poisson + gaussian
    return noisy.clamp(0.0, 1.0)


def simulate_neutron_noise(images: torch.Tensor, spot_ratio: float = 0.02,
                           spot_intensity: float = 1.0, sigma: float = 0.02,
                           poisson_scale: float = 50.0) -> torch.Tensor:
    """Full noise pipeline: white spot + Poisson-Gaussian mixture."""
    noisy = add_white_spot_noise(images, ratio=spot_ratio,
                                 intensity=spot_intensity)
    noisy = add_poisson_gaussian_noise(noisy, sigma=sigma, scale=poisson_scale)
    return noisy.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Image / checkpoint helpers
# ---------------------------------------------------------------------------
def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor in [0, 1] to a uint8 numpy array (H, W) or (H, W, C)."""
    arr = (images.clamp(0, 1) * 255.0).round().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def gpu_or_cpu() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr_simple(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """Tiny PSNR helper used for logging during training."""
    mse = F.mse_loss(x, y).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)
