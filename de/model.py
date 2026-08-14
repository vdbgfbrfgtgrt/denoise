"""DFGU-Net: Dual-stage Feature-Gated U-Net for neutron image denoising.

Architecture follows the paper "Research on two-stage image denoising
algorithm based on Feature Fusion for neutron imaging":

  1. Pre-denoising module  : harmonic mean filter (3x3)
  2. Stage-1 U-Net recovery: takes reference feature map F'' -> F1
  3. Gated Feature Fusion  : combines noisy features F' (reset gate)
                             with stage-1 features F1 (update gate),
                             GRU-inspired
  4. Stage-2 U-Net recovery: takes fused features -> clean image

Each basic recovery network is a U-Net whose levels contain two
consecutive efficient residual blocks. Each residual block consists of
two (Conv -> GroupNorm -> LeakyReLU) units with an identity skip.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import harmonic_mean_filter


# ---------------------------------------------------------------------------
# Spatial padding helpers
# ---------------------------------------------------------------------------
# The U-Net uses 3 stride-2 downsampling steps, so input spatial sizes must be
# multiples of 2^3 = 8 for the encoder skips and decoder upsamples to align.
# When validating on full images of arbitrary size we pad the input to the
# nearest multiple of 8 (reflect) and crop the output back to the original
# size. This keeps the architecture lossless for any resolution.
ALIGN_MULTIPLE = 8


def _pad_to_multiple(x: torch.Tensor, multiple: int = ALIGN_MULTIPLE
                     ) -> Tuple[torch.Tensor, int, int]:
    """Pad ``x`` spatially to a multiple of ``multiple`` (reflect mode).

    Returns (padded_tensor, original_h, original_w).
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    # F.pad order: (left, right, top, bottom).
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, h, w


def _crop_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Crop ``x`` back to (h, w)."""
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return x[..., :h, :w]


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    """Conv2d -> Normalization -> LeakyReLU unit."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1,
                 act: bool = True, norm: bool = True):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=pad, dilation=dilation, bias=not norm)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Efficient residual block: two Conv-Norm-LeakyReLU units + identity skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            ConvBNAct(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualBlocks(nn.Module):
    """Two consecutive residual blocks per U-Net level (paper spec)."""

    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Basic recovery network (U-Net backbone)
# ---------------------------------------------------------------------------
class BasicRecoveryNet(nn.Module):
    """U-Net based basic recovery network used in each stage.

    Feature schedule (paper):
        initial feature map : 32 x 512 x 512
        lowest feature map  : 128 x  64 x  64
    i.e. 3 downsampling steps with channel widths [32, 64, 128, 128].
    """

    def __init__(self, in_channels: int, out_channels: int = 1,
                 base_channels: int = 32, num_levels: int = 3,
                 use_input_proj: bool = True):
        super().__init__()
        self.use_input_proj = use_input_proj

        # Input projection (3x3 conv) producing the 32-channel feature map.
        self.input_proj = ConvBNAct(in_channels, base_channels, 3) if use_input_proj else None

        # Encoder
        enc_channels = []
        self.encoders = nn.ModuleList()
        ch = base_channels
        for level in range(num_levels):
            self.encoders.append(ResidualBlocks(ch))
            enc_channels.append(ch)
            ch = min(ch * 2, base_channels * 4)
        # Channel adjust after the last encoder before bottleneck (32->128).
        # We follow the schedule: 32 -> 64 -> 128 -> bottleneck 128.
        bottleneck_channels = base_channels * 4  # 128
        self.downsample_adjust = nn.ModuleList()
        ch = base_channels
        for level in range(num_levels):
            next_ch = min(ch * 2, bottleneck_channels)
            self.downsample_adjust.append(
                ConvBNAct(ch, next_ch, 3, stride=2)
            )
            ch = next_ch

        # Bottleneck (lowest level: 128 x 64 x 64)
        self.bottleneck = ResidualBlocks(bottleneck_channels)

        # Decoder
        self.upsample_adjust = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(enc_channels))  # [128, 64, 32] (top-down)
        ch = bottleneck_channels
        for level in range(num_levels):
            skip_ch = rev[level]
            up_ch = skip_ch
            # Upsample then reduce channels from `ch` to `up_ch`.
            self.upsample_adjust.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBNAct(ch, up_ch, 3),
                )
            )
            # After concat (up_ch + skip_ch = 2*up_ch) we first reduce to
            # up_ch and then apply two residual blocks (paper spec).
            self.decoders.append(nn.Sequential(
                ConvBNAct(up_ch * 2, up_ch, 3),
                ResidualBlocks(up_ch),
            ))
            ch = up_ch

        # Output head: 3x3 conv + Sigmoid (paper).
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to multiple of 8 to ensure skip connection alignment
        # (handles arbitrary image sizes during validation)
        x, orig_h, orig_w = _pad_to_multiple(x, ALIGN_MULTIPLE)

        if self.use_input_proj and self.input_proj is not None:
            x = self.input_proj(x)

        skips: List[torch.Tensor] = []
        for level, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            x = self.downsample_adjust[level](x)

        x = self.bottleneck(x)

        for level in range(len(self.decoders)):
            x = self.upsample_adjust[level](x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[level](x)

        x = self.out_conv(x)
        x = torch.sigmoid(x)
        # Crop back to original size
        return _crop_to(x, orig_h, orig_w)


# ---------------------------------------------------------------------------
# Gated Feature Fusion module (GRU-inspired)
# ---------------------------------------------------------------------------
class GatedFeatureFusion(nn.Module):
    """GRU-inspired gated feature fusion module.

    Following the paper:
        reset gate   r = sigmoid(Wr * F_noisy  + Ur * F_ref)
        update gate  z = sigmoid(Wz * F_ref    + Uz * F_noisy)
        candidate    h~ = tanh(W * F_ref + U * (r * F_noisy))
        fused        h = (1 - z) * F_noisy + z * h~

    where F_noisy plays the role of the noisy-image features F' (reset)
    and F_ref plays the role of the reference (stage-1) features F''
    (update).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.wr = nn.Conv2d(channels, channels, 3, padding=1)
        self.ur = nn.Conv2d(channels, channels, 3, padding=1)
        self.wz = nn.Conv2d(channels, channels, 3, padding=1)
        self.uz = nn.Conv2d(channels, channels, 3, padding=1)
        self.w = nn.Conv2d(channels, channels, 3, padding=1)
        self.u = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, f_noisy: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.wr(f_noisy) + self.ur(f_ref))
        z = torch.sigmoid(self.wz(f_ref) + self.uz(f_noisy))
        h_tilde = torch.tanh(self.w(f_ref) + self.u(r * f_noisy))
        h = (1.0 - z) * f_noisy + z * h_tilde
        return h


# ---------------------------------------------------------------------------
# Feature extraction used to derive F' from the noisy image
# ---------------------------------------------------------------------------
class NoisyFeatureExtractor(nn.Module):
    """Conv + a few residual blocks to obtain F' from the noisy image."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(in_channels, base_channels, 3),
            ResidualBlock(base_channels),
            ConvBNAct(base_channels, base_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# DFGU-Net: full two-stage architecture
# ---------------------------------------------------------------------------
class DFGUNet(nn.Module):
    """DFGU-Net: Dual-stage Feature-Gated U-Net.

    Pipeline:
        noisy x  --[harmonic mean filter]-->  reference x_ref
        x_ref    --[3x3 conv]-------------->  F'' (32-ch feature map)
        F''      --[Stage-1 U-Net]-------->   F1 (denoised feature / image)
        noisy x  --[NoisyFeatureExtractor]->  F' (32-ch feature map)
        F', F1   --[GFM]------------------>  fused feature F_fuse
        F_fuse   --[Stage-2 U-Net]-------->   clean image y_hat
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32,
                 hmf_kernel_size: int = 3,
                 use_pre_denoising: bool = True,
                 use_gfm: bool = True):
        super().__init__()
        self.use_pre_denoising = use_pre_denoising
        self.use_gfm = use_gfm

        self.hmf_kernel_size = hmf_kernel_size

        # 3x3 conv turning the (filtered) reference image into F''.
        self.ref_proj = ConvBNAct(in_channels, base_channels, 3)

        # Stage-1 basic recovery network operating on the reference feature map.
        self.stage1 = BasicRecoveryNet(in_channels=base_channels,
                                       out_channels=in_channels,
                                       base_channels=base_channels,
                                       use_input_proj=False)

        # Noisy feature extractor producing F'.
        self.noisy_extractor = NoisyFeatureExtractor(in_channels, base_channels)

        # GFM fuses F' and F1-derived features.
        self.gfm = GatedFeatureFusion(base_channels) if use_gfm else None

        # Stage-2 basic recovery network: from fused features to clean image.
        self.stage2 = BasicRecoveryNet(in_channels=base_channels,
                                       out_channels=out_channels,
                                       base_channels=base_channels,
                                       use_input_proj=False)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        # Pad input to a multiple of 8 so that the U-Net skip connections
        # align (3 stride-2 downsamples = factor 8). Crop back at the end.
        noisy, orig_h, orig_w = _pad_to_multiple(noisy, ALIGN_MULTIPLE)

        # 1) Pre-denoising: harmonic mean filter on the noisy image.
        if self.use_pre_denoising:
            reference = harmonic_mean_filter(noisy, kernel_size=self.hmf_kernel_size)
        else:
            reference = noisy

        # 2) Reference image -> F'' feature map -> Stage-1 recovery -> F1.
        f_ref = self.ref_proj(reference)            # F'' (32-ch)
        stage1_out = self.stage1(f_ref)             # denoised reference image
        # Re-encode stage1 output to 32-ch feature map F1.
        f1 = self.ref_proj(stage1_out)

        # 3) Noisy image -> F' feature map.
        f_noisy = self.noisy_extractor(noisy)       # F'

        # 4) Gated Feature Fusion (or simple concat fallback for ablation).
        if self.use_gfm:
            fused = self.gfm(f_noisy, f1)
        else:
            fused = f_noisy + f1

        # 5) Stage-2 recovery -> final clean image.
        y_hat = self.stage2(fused)
        return _crop_to(y_hat, orig_h, orig_w)


# ---------------------------------------------------------------------------
# Convenience: ablation variants (A, B, C, D, E) defined in the paper.
# ---------------------------------------------------------------------------
def build_model(variant: str = "E", **kwargs) -> nn.Module:
    """Build a model variant for ablation studies.

    A: single-stage (no pre-denoising, no GFM, no two-stage)
    B: two-stage without pre-denoising
    C: single-stage with pre-denoising
    D: two-stage with pre-denoising but no GFM (concat instead)
    E: full DFGU-Net
    """
    variant = variant.upper()
    if variant == "A":
        return SingleStageNet(use_pre_denoising=False, **kwargs)
    if variant == "C":
        return SingleStageNet(use_pre_denoising=True, **kwargs)
    if variant == "B":
        return DFGUNet(use_pre_denoising=False, use_gfm=True, **kwargs)
    if variant == "D":
        return DFGUNet(use_pre_denoising=True, use_gfm=False, **kwargs)
    if variant == "E":
        return DFGUNet(use_pre_denoising=True, use_gfm=True, **kwargs)
    raise ValueError(f"Unknown variant {variant!r}")


class SingleStageNet(nn.Module):
    """Variant A / C: a single basic recovery network (optionally with HMF)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, hmf_kernel_size: int = 3,
                 use_pre_denoising: bool = False):
        super().__init__()
        self.use_pre_denoising = use_pre_denoising
        self.hmf_kernel_size = hmf_kernel_size
        self.net = BasicRecoveryNet(in_channels=in_channels,
                                    out_channels=out_channels,
                                    base_channels=base_channels,
                                    use_input_proj=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        # Pad to a multiple of 8 (3 stride-2 downsamples), then crop back.
        noisy, orig_h, orig_w = _pad_to_multiple(noisy, ALIGN_MULTIPLE)
        if self.use_pre_denoising:
            noisy = harmonic_mean_filter(noisy, kernel_size=self.hmf_kernel_size)
        out = self.net(noisy)
        return _crop_to(out, orig_h, orig_w)


if __name__ == "__main__":
    # Quick shape sanity check (also exercises non-multiple-of-8 sizes).
    model = DFGUNet(in_channels=1, base_channels=32)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DFGU-Net parameters: {n_params:.2f} M")
    for h, w in [(128, 128), (512, 512), (551, 673), (1024, 1024)]:
        x = torch.randn(2, 1, h, w)
        y = model(x)
        assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
        print(f"  input {x.shape} -> output {y.shape}  OK")

    # Single-stage variant.
    s = SingleStageNet(in_channels=1, base_channels=32)
    x = torch.randn(1, 1, 551, 673)
    y = s(x)
    assert y.shape == x.shape
    print(f"  SingleStageNet input {x.shape} -> output {y.shape}  OK")
