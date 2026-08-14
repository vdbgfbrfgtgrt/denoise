"""Dataset utilities for DFGU-Net training and evaluation.

The paper trains on the SIXray dataset (converted to grayscale) with
simulated neutron noise. The user's data root is expected to be a folder
of images. To stay flexible across different real-world layouts we try
several conventions:

  - paired layout  : <root>/clean  + <root>/noisy  (or gt/input, target/source)
  - flat layout    : <root>/*.png (treated as clean targets; noise is added
                     on-the-fly via the neutron noise simulator)
  - sub-dirs layout: <root>/<sub>/*.png  with sibling clean/noisy subdirs

If a clean/noisy pair is found, the noisy image is used as-is (no further
synthetic noise is added). Otherwise synthetic neutron noise is added to
the clean image on-the-fly.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils import simulate_neutron_noise


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXTENSIONS)


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if _is_image(f))


# Possible sub-directory name pairs for paired layouts.
CLEAN_NAMES = ("clean", "gt", "target", "label", "ground_truth", "GroundTruth")
NOISY_NAMES = ("noisy", "input", "source", "low", "degraded")


def _detect_paired_layout(root: str) -> Optional[Tuple[str, str]]:
    """Detect <root>/clean + <root>/noisy style layout."""
    entries = [d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))]
    lowers = {e.lower(): e for e in entries}
    clean_dir = next((lowers[n] for n in CLEAN_NAMES if n in lowers), None)
    noisy_dir = next((lowers[n] for n in NOISY_NAMES if n in lowers), None)
    if clean_dir and noisy_dir:
        return os.path.join(root, clean_dir), os.path.join(root, noisy_dir)
    return None


def _load_image(path: str) -> np.ndarray:
    """Load an image as a grayscale uint8 numpy array."""
    img = Image.open(path)
    if img.mode == "L":
        arr = np.array(img)
    elif img.mode == "I;16":
        arr = np.array(img).astype(np.float32)
        arr = (arr / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif img.mode == "I":
        arr = np.array(img).astype(np.float32)
        arr = (arr / arr.max() * 255.0).clip(0, 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
    else:
        arr = np.array(img.convert("L"))
    return arr


class NeutronDenoiseDataset(Dataset):
    """Dataset for neutron image denoising.

    Args:
        root: Path to the dataset folder. The loader tries paired-layout
              first, falling back to flat layout (clean images + on-the-fly
              neutron noise simulation).
        patch_size: Random crop size (paper uses 512x512). Use None or 0
                    to disable cropping.
        augment: Whether to apply random flips / rotations.
        train: True for training (with augmentation), False for testing.
        noise_params: dict of params passed to ``simulate_neutron_noise``.
                      Used only when no paired noisy image is available.
        max_samples: Optional cap on the number of samples.
    """

    def __init__(self,
                 root: str,
                 patch_size: Optional[int] = 512,
                 augment: bool = True,
                 train: bool = True,
                 noise_params: Optional[dict] = None,
                 max_samples: Optional[int] = None):
        super().__init__()
        self.root = root
        self.patch_size = patch_size
        self.augment = augment and train
        self.train = train
        self.noise_params = noise_params or {}

        paired = _detect_paired_layout(root)
        if paired is not None:
            clean_dir, noisy_dir = paired
            clean_imgs = _list_images(clean_dir)
            noisy_imgs = _list_images(noisy_dir)
            # Match by basename if possible, otherwise by index.
            self.paired = True
            self.items = []
            noisy_map = {os.path.splitext(os.path.basename(p))[0]: p for p in noisy_imgs}
            for c in clean_imgs:
                key = os.path.splitext(os.path.basename(c))[0]
                n = noisy_map.get(key) or noisy_imgs[min(len(self.items), len(noisy_imgs) - 1)] if noisy_imgs else None
                self.items.append((c, n))
            if not noisy_imgs:
                self.paired = False
        else:
            # Flat layout: images treated as clean targets.
            self.paired = False
            flat = _list_images(root)
            self.items = [(p, None) for p in flat]

        if max_samples is not None and max_samples > 0:
            self.items = self.items[:max_samples]

        if len(self.items) == 0:
            raise RuntimeError(
                f"No images found under {root!r}. "
                f"Expected either a paired layout (clean/noisy subdirs) "
                f"or a flat folder of images."
            )

    def __len__(self) -> int:
        return len(self.items)

    def _random_crop(self, clean: np.ndarray, noisy: np.ndarray,
                     ps: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = clean.shape[:2]
        if h < ps or w < ps:
            # Pad if smaller than patch.
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = clean.shape[:2]
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return clean[top:top + ps, left:left + ps], noisy[top:top + ps, left:left + ps]

    def _augment(self, clean: torch.Tensor, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            clean = torch.flip(clean, dims=[-1])
            noisy = torch.flip(noisy, dims=[-1])
        if random.random() < 0.5:
            clean = torch.flip(clean, dims=[-2])
            noisy = torch.flip(noisy, dims=[-2])
        k = random.choice([0, 1, 2, 3])
        if k:
            clean = torch.rot90(clean, k, dims=[-2, -1])
            noisy = torch.rot90(noisy, k, dims=[-2, -1])
        return clean, noisy

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

    def __getitem__(self, idx: int):
        clean_path, noisy_path = self.items[idx]
        clean = _load_image(clean_path)

        if self.paired and noisy_path is not None:
            noisy = _load_image(noisy_path)
        else:
            # Apply synthetic neutron noise to the clean image on the fly.
            clean_t = self._to_tensor(clean)
            noisy_t = simulate_neutron_noise(clean_t, **self.noise_params)
            noisy = (noisy_t.squeeze(0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Crop / pad.
        if self.patch_size and self.patch_size > 0:
            clean, noisy = self._random_crop(clean, noisy, self.patch_size)

        clean_t = self._to_tensor(clean)
        noisy_t = self._to_tensor(noisy)

        if self.augment:
            clean_t, noisy_t = self._augment(clean_t, noisy_t)

        return {"noisy": noisy_t, "clean": clean_t, "name": os.path.basename(clean_path)}


def get_dataloaders(train_root: str,
                    val_root: Optional[str] = None,
                    patch_size: int = 512,
                    batch_size: int = 16,
                    num_workers: int = 4,
                    noise_params: Optional[dict] = None,
                    val_split: float = 0.2,
                    seed: int = 42,
                    max_samples: Optional[int] = None):
    """Build training and validation DataLoaders.

    If ``val_root`` is None, an 80/20 split of ``train_root`` is used
    (matching the paper's experimental protocol).
    """
    noise_params = noise_params or {}
    full_dataset = NeutronDenoiseDataset(
        train_root, patch_size=patch_size, augment=True, train=True,
        noise_params=noise_params, max_samples=max_samples,
    )

    if val_root is not None and os.path.isdir(val_root):
        train_ds = full_dataset
        val_ds = NeutronDenoiseDataset(
            val_root, patch_size=None, augment=False, train=False,
            noise_params=noise_params,
        )
    else:
        # Hold-out split with fixed seed.
        n = len(full_dataset)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_val = max(1, int(round(n * val_split)))
        val_idx = set(perm[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]

        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        # For validation we disable augmentation by wrapping a new dataset.
        val_full = NeutronDenoiseDataset(
            train_root, patch_size=None, augment=False, train=False,
            noise_params=noise_params, max_samples=max_samples,
        )
        val_ds = torch.utils.data.Subset(val_full, list(val_idx))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
