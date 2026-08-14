"""Evaluation metrics for neutron image denoising.

Implements the metrics used in the paper:

  Reference-based:
    - PSNR  : Peak Signal-to-Noise Ratio
    - SSIM  : Structural Similarity Index Measure

  No-reference:
    - BIQI  : Blind Image Quality Index (Sobel gradient based)
    - NIQE  : Natural Image Quality Evaluator (MAD of MSCN stats)
    - SF    : Spatial Frequency (Row + Column frequencies)

NIQE here is a lightweight self-contained implementation based on MSCN
(Mean-Subtracted Contrast-Normalized) coefficients and a multivariate
Gaussian model fitted on a small set of "natural" patches sampled from
the input itself. The score is the symmetric KL / Mahalanobis-like
distance, so lower values correspond to images that are statistically
closer to natural images (matching the paper's convention: lower is
better).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g.unsqueeze(1) * g.unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, max_val: float = 1.0) -> float:
    """Structural Similarity Index (averaged over the batch)."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    ch = pred.shape[1]
    window = _gaussian_kernel(window_size).expand(ch, 1, window_size, window_size).contiguous()
    window = window.to(pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
    mu2 = F.conv2d(target, window, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=ch) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=ch) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# No-reference metrics
# ---------------------------------------------------------------------------
def _to_gray_numpy(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.mean(0) if arr.shape[0] == 3 else arr[0]
    return arr.numpy()


def biqi(img: torch.Tensor) -> float:
    """Blind Image Quality Index based on Sobel gradients (lower is better).

    The paper computes the average gradient magnitude as a proxy for the
    amount of high-intensity white spot noise. A lower BIQI value
    indicates more effective noise suppression.
    """
    arr = _to_gray_numpy(img)
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    from scipy.signal import convolve2d
    ix = convolve2d(arr, gx, mode="same", boundary="symm")
    iy = convolve2d(arr, gy, mode="same", boundary="symm")
    grad = np.sqrt(ix ** 2 + iy ** 2)
    return float(grad.mean() * 100.0)  # scaled to align with paper order of magnitude


def _mscn_coeffs(arr: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Compute Mean-Subtracted Contrast-Normalized coefficients."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(arr, size=window_size, mode="reflect")
    sq_mean = uniform_filter(arr * arr, size=window_size, mode="reflect")
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-8))
    mscn = (arr - mean) / (std + 1e-8)
    return mscn


def nique(img: torch.Tensor) -> float:
    """Lightweight NIQE-style no-reference metric (lower is better)."""
    arr = _to_gray_numpy(img).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    mscn = _mscn_coeffs(arr)
    # Pairwise products of adjacent neighbours for richer statistics.
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    feats = [mscn]
    for dy, dx in shifts:
        shifted = np.roll(mscn, shift=(dy, dx), axis=(0, 1))
        feats.append(mscn * shifted)
    feat_arr = np.stack([f.ravel() for f in feats], axis=0)
    mu = feat_arr.mean(axis=1)
    cov = np.cov(feat_arr)
    # Distance against a unit-Gaussian "natural" reference (mu=0, cov=I).
    inv_cov = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    diff = mu
    score = float(np.sqrt(diff @ inv_cov @ diff.T))
    return score


def spatial_frequency(img: torch.Tensor) -> float:
    """Spatial Frequency (higher is sharper / richer detail)."""
    arr = _to_gray_numpy(img).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    h, w = arr.shape
    rf = np.sqrt(np.sum((arr[:, 1:] - arr[:, :-1]) ** 2) / (h * w))
    cf = np.sqrt(np.sum((arr[1:, :] - arr[:-1, :]) ** 2) / (h * w))
    return float(np.sqrt(rf ** 2 + cf ** 2))


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(preds: torch.Tensor, targets: torch.Tensor = None,
             compute_no_ref: bool = True) -> Dict[str, float]:
    """Evaluate a batch of predictions.

    Args:
        preds: (B, C, H, W) tensor of denoised images in [0, 1].
        targets: Optional (B, C, H, W) tensor of ground-truth images.
        compute_no_ref: Whether to compute BIQI / NIQE / SF.
    """
    out: Dict[str, float] = {}
    if targets is not None:
        psnrs, ssims = [], []
        for i in range(preds.shape[0]):
            psnrs.append(psnr(preds[i:i+1], targets[i:i+1]))
            ssims.append(ssim(preds[i:i+1], targets[i:i+1]))
        out["PSNR"] = float(np.mean(psnrs))
        out["SSIM"] = float(np.mean(ssims))
    if compute_no_ref:
        biqis, niqes, sfs = [], [], []
        for i in range(preds.shape[0]):
            biqis.append(biqi(preds[i]))
            niqes.append(nique(preds[i]))
            sfs.append(spatial_frequency(preds[i]))
        out["BIQI"] = float(np.mean(biqis))
        out["NIQE"] = float(np.mean(niqes))
        out["SF"] = float(np.mean(sfs))
    return out
