"""CUT networks (pure PyTorch, no torchvision): ResNet generator, PatchGAN discriminator,
and the patch-sampling MLP for the contrastive (PatchNCE) loss.

Single-channel in/out — both domains are the (n_ax, n_lat) fan layout that reslice produces and
``calib.us_geometry.unwrap_fan`` maps real US onto. Kept compact and standard (CycleGAN/CUT
lineage) so the behaviour is well understood; the generator can return intermediate encoder
features for PatchNCE via ``encode_only``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _norm(c):
    return nn.InstanceNorm2d(c, affine=False, track_running_stats=False)


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _norm(dim),
        )

    def forward(self, x):
        return x + self.conv(x)


class ResnetGenerator(nn.Module):
    """CBCT-slice -> US generator. ``encode_only`` returns features at ``layers`` (for PatchNCE)."""

    def __init__(self, in_nc=1, out_nc=1, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_nc, ngf, 7), _norm(ngf), nn.ReLU(True)]
        mult = 1
        for _ in range(2):                                   # downsample x4
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                       _norm(ngf * mult * 2), nn.ReLU(True)]
            mult *= 2
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult)]
        for _ in range(2):                                   # upsample x4
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2,
                                          padding=1, output_padding=1),
                       _norm(ngf * mult // 2), nn.ReLU(True)]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_nc, 7), nn.Tanh()]
        self.model = nn.ModuleList(layers)

    def forward(self, x, layers=None, encode_only=False):
        if not layers:
            for m in self.model:
                x = m(x)
            return x
        feats, layers = [], set(layers)
        for i, m in enumerate(self.model):
            x = m(x)
            if i in layers:
                feats.append(x)
            if encode_only and i >= max(layers):
                return feats
        return feats


class NLayerDiscriminator(nn.Module):
    """PatchGAN: classifies overlapping patches of the (real or generated) US image."""

    def __init__(self, in_nc=1, ndf=64, n_layers=3):
        super().__init__()
        seq = [nn.Conv2d(in_nc, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            mult_prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, 4, stride=2, padding=1),
                    _norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, 4, stride=1, padding=1),
                _norm(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class PatchSampleMLP(nn.Module):
    """Sample ``num_patches`` locations per feature map, project (2-layer MLP), L2-normalise.

    MLPs are created lazily per layer (channel dims known only at first call). Returns the sampled
    features and the patch ids, so the same locations can be sampled from the paired feature map.
    """

    def __init__(self, nc=256):
        super().__init__()
        self.nc = nc
        self.mlps = nn.ModuleDict()

    def _mlp(self, key, in_c, device):
        if key not in self.mlps:
            self.mlps[key] = nn.Sequential(nn.Linear(in_c, self.nc), nn.ReLU(),
                                           nn.Linear(self.nc, self.nc)).to(device)
        return self.mlps[key]

    def forward(self, feats, num_patches=256, patch_ids=None):
        out, ids = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, HW, C)
            if patch_ids is not None:
                pid = patch_ids[i]
            else:
                n = min(num_patches, H * W)
                pid = torch.randperm(H * W, device=f.device)[:n]
            sampled = f[:, pid, :].reshape(-1, C)            # (B*n, C)
            sampled = self._mlp(str(i), C, f.device)(sampled)
            sampled = nn.functional.normalize(sampled, dim=1)
            out.append(sampled); ids.append(pid)
        return out, ids
