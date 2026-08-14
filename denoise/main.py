"""DFGU-Net entry point.

Run a full experiment with::

    python3 main.py

Training defaults follow the paper:
  - Adam optimizer, initial LR = 0.001
  - batch size = 16, 200 epochs
  - LR decay x0.5 every 50 epochs
  - Loss = MSE + (1 - SSIM)
  - 80/20 train/val split (or use --val_root for a separate folder)

Use ``--mode eval`` to evaluate a checkpoint, and ``--mode infer`` to
denoise a folder of images and save the results.

Default dataset path is the path the user provided:
    /nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/train
Override with ``--data_root`` if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from dataset import NeutronDenoiseDataset, get_dataloaders
from losses import DFGULoss
from metrics import evaluate
from model import build_model
from utils import set_seed, save_checkpoint, gpu_or_cpu, to_uint8


DEFAULT_DATA_ROOT = "/nc1test1/zxr/neutron_experiment/EndoIR/dataset_neutron/train"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scalars(d: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in d.items()}


def _save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(to_uint8(tensor)).save(path)


# ---------------------------------------------------------------------------
# Training & validation
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: DFGULoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, log_every: int = 20):
    model.train()
    running_loss, running_psnr, n = 0.0, 0.0, 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        pred = model(noisy)
        loss, _ = criterion(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = noisy.shape[0]
        running_loss += loss.item() * bs
        running_psnr += _batch_psnr(pred, clean) * bs
        n += bs

        if (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d} | {it+1:04d}/{len(loader):04d}] "
                  f"loss={running_loss/n:.5f}  psnr={running_psnr/n:.3f}  "
                  f"({elapsed:.1f}s)", flush=True)
    return {
        "loss": running_loss / max(n, 1),
        "psnr": running_psnr / max(n, 1),
        "time": time.time() - t0,
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    all_metrics: Dict[str, List[float]] = {}
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        pred = model(noisy).clamp(0, 1)
        m = evaluate(pred, clean, compute_no_ref=True)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def _batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    from utils import psnr_simple
    return psnr_simple(pred.detach().clamp(0, 1), target.detach())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = gpu_or_cpu()
    print(f"Device: {device}")

    noise_params = {
        "spot_ratio": args.spot_ratio,
        "spot_intensity": args.spot_intensity,
        "sigma": args.sigma,
        "poisson_scale": args.poisson_scale,
    }

    print(f"Loading data from {args.data_root}")
    train_loader, val_loader = get_dataloaders(
        train_root=args.data_root,
        val_root=args.val_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_params=noise_params,
        val_split=args.val_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train samples: {len(train_loader.dataset)}  "
          f"Val samples: {len(val_loader.dataset)}")

    model = build_model(
        variant=args.variant,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        hmf_kernel_size=args.hmf_kernel,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model variant={args.variant}  parameters={n_params:.2f} M")

    criterion = DFGULoss(alpha=args.alpha, beta=args.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # LR decay x0.5 every 50 epochs (paper).
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_gamma,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_psnr = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion,
                                      optimizer, device, epoch,
                                      log_every=args.log_every)
        val_stats = validate(model, val_loader, device)
        scheduler.step()

        print(f"== Epoch {epoch:03d} ==  "
              f"train_loss={train_stats['loss']:.5f}  "
              f"val_PSNR={val_stats.get('PSNR', float('nan')):.3f}  "
              f"val_SSIM={val_stats.get('SSIM', float('nan')):.4f}  "
              f"NIQE={val_stats.get('NIQE', float('nan')):.4f}  "
              f"BIQI={val_stats.get('BIQI', float('nan')):.3f}  "
              f"SF={val_stats.get('SF', float('nan')):.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        history.append({"epoch": epoch,
                        "train": _scalars(train_stats),
                        "val": _scalars(val_stats)})

        # Save checkpoint.
        ckpt_path = os.path.join(args.out_dir, "latest.pth")
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
        }, ckpt_path)

        if val_stats.get("PSNR", 0.0) > best_psnr:
            best_psnr = val_stats.get("PSNR", 0.0)
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "val_metrics": _scalars(val_stats),
            }, os.path.join(args.out_dir, "best.pth"))

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training complete. Best val PSNR={best_psnr:.3f}")


# ---------------------------------------------------------------------------
# Evaluation on a checkpoint
# ---------------------------------------------------------------------------
def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = gpu_or_cpu()
    print(f"Device: {device}")

    noise_params = {
        "spot_ratio": args.spot_ratio,
        "spot_intensity": args.spot_intensity,
        "sigma": args.sigma,
        "poisson_scale": args.poisson_scale,
    }
    ds = NeutronDenoiseDataset(
        args.data_root, patch_size=None, augment=False, train=False,
        noise_params=noise_params, max_samples=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(
        variant=args.variant,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        hmf_kernel_size=args.hmf_kernel,
    ).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint from {args.resume}")

    all_metrics: Dict[str, List[float]] = {}
    for i, batch in enumerate(loader):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        with torch.no_grad():
            pred = model(noisy).clamp(0, 1)
        m = evaluate(pred, clean, compute_no_ref=True)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)
        if args.save_images:
            _save_image(pred[0], os.path.join(args.out_dir, "eval", f"{i:05d}.png"))
        print(f"[{i+1:04d}/{len(loader):04d}]  "
              + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)

    print("\n== Average metrics ==")
    for k, vs in all_metrics.items():
        print(f"  {k}: {np.mean(vs):.4f}  (std={np.std(vs):.4f})")


# ---------------------------------------------------------------------------
# Inference: denoise a folder of images
# ---------------------------------------------------------------------------
def run_infer(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = gpu_or_cpu()
    print(f"Device: {device}")

    ds = NeutronDenoiseDataset(
        args.data_root, patch_size=None, augment=False, train=False,
        noise_params={"spot_ratio": 0.0},  # do not add synthetic noise here
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(
        variant=args.variant,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels,
        hmf_kernel_size=args.hmf_kernel,
    ).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint from {args.resume}")
    else:
        print("WARNING: no --resume checkpoint provided; using random weights.")
    model.eval()

    out_dir = os.path.join(args.out_dir, "infer")
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(loader):
        noisy = batch["noisy"].to(device)
        with torch.no_grad():
            pred = model(noisy).clamp(0, 1)
        name = batch["name"][0]
        _save_image(pred[0], os.path.join(out_dir, name))
        print(f"[{i+1:04d}/{len(loader):04d}] saved {name}", flush=True)
    print(f"Done. Results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DFGU-Net: Dual-stage Feature-Gated U-Net for neutron image denoising"
    )
    p.add_argument("--mode", choices=["train", "eval", "infer"], default="train",
                   help="train / eval (with metrics) / infer (save denoised images)")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                   help="training / evaluation data folder")
    p.add_argument("--val_root", default=None,
                   help="optional separate validation folder")
    p.add_argument("--out_dir", default="./outputs",
                   help="where to save checkpoints, logs and images")
    p.add_argument("--resume", default=None,
                   help="checkpoint path for eval / infer / resume training")

    # Model
    p.add_argument("--variant", default="E",
                   choices=["A", "B", "C", "D", "E"],
                   help="model variant (E = full DFGU-Net)")
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--hmf_kernel", type=int, default=3,
                   help="harmonic mean filter kernel size (paper: 3)")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay_epochs", type=int, default=50)
    p.add_argument("--lr_decay_gamma", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.0, help="MSE weight")
    p.add_argument("--beta", type=float, default=1.0, help="SSIM loss weight")
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--max_samples", type=int, default=None,
                   help="optional cap on dataset size (for quick smoke tests)")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_images", action="store_true",
                   help="save per-image eval results (eval mode only)")

    # Noise simulation (used only when no paired noisy images are provided)
    p.add_argument("--spot_ratio", type=float, default=0.02,
                   help="white spot noise ratio")
    p.add_argument("--spot_intensity", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.02,
                   help="Gaussian noise sigma")
    p.add_argument("--poisson_scale", type=float, default=50.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("DFGU-Net: Dual-stage Feature-Gated U-Net for neutron image denoising")
    print("=" * 70)
    print(json.dumps(vars(args), indent=2))
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
