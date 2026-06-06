"""CUT model: ties the generator / discriminator / patch-MLP together and computes the
contrastive-unpaired-translation losses (adversarial + PatchNCE).

Loss = L_GAN(G, D) + lambda_NCE * PatchNCE(x, G(x))   [+ PatchNCE(y, G(y)) identity term].
PatchNCE pulls an output patch toward the input patch at the SAME location and pushes it from
others — structure preservation without pixel-aligned pairs, which is exactly why CUT fits our
cm-level registration (see docs/renderer.md). LSGAN (MSE) adversarial loss for stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .networks import ResnetGenerator, NLayerDiscriminator, PatchSampleMLP

NCE_LAYERS = (0, 4, 8, 12, 16)   # generator layer indices sampled for PatchNCE


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, feat_q, feat_k):
        # feat_q, feat_k: (N, C) sampled+projected features for one layer (same patch ids).
        N = feat_q.shape[0]
        feat_k = feat_k.detach()
        pos = (feat_q * feat_k).sum(1, keepdim=True)              # (N,1) positive logit
        neg = feat_q @ feat_k.t()                                 # (N,N) all pairs
        neg.fill_diagonal_(-1e9)                                  # exclude the positive
        logits = torch.cat([pos, neg], dim=1) / self.tau         # (N, 1+N)
        labels = torch.zeros(N, dtype=torch.long, device=feat_q.device)
        return self.ce(logits, labels)


class CUTModel(nn.Module):
    """Holds G, D, F and exposes loss computations; optimisers live in the training script."""

    def __init__(self, ngf=64, ndf=64, n_blocks=9, nce_dim=256, lambda_nce=1.0,
                 lambda_gan=1.0, nce_idt=True, num_patches=256):
        super().__init__()
        self.G = ResnetGenerator(1, 1, ngf, n_blocks)
        self.D = NLayerDiscriminator(1, ndf)
        self.F = PatchSampleMLP(nce_dim)
        self.nce = PatchNCELoss()
        self.lambda_nce, self.lambda_gan, self.nce_idt = lambda_nce, lambda_gan, nce_idt
        self.num_patches = num_patches
        self.mse = nn.MSELoss()

    # --- adversarial ------------------------------------------------------
    def d_loss(self, real_us, fake_us):
        pred_fake = self.D(fake_us.detach()); pred_real = self.D(real_us)
        return 0.5 * (self.mse(pred_fake, torch.zeros_like(pred_fake))
                      + self.mse(pred_real, torch.ones_like(pred_real)))

    def g_gan_loss(self, fake_us):
        pred = self.D(fake_us)
        return self.mse(pred, torch.ones_like(pred)) * self.lambda_gan

    # --- contrastive ------------------------------------------------------
    def patchnce(self, src, gen):
        """PatchNCE between source ``src`` and its translation ``gen`` (same generator encoder)."""
        feat_src = self.G(src, NCE_LAYERS, encode_only=True)
        feat_gen = self.G(gen, NCE_LAYERS, encode_only=True)
        k, ids = self.F(feat_src, self.num_patches)
        q, _ = self.F(feat_gen, self.num_patches, patch_ids=ids)
        loss = sum(self.nce(qi, ki) for qi, ki in zip(q, k)) / len(q)
        return loss * self.lambda_nce

    def g_loss(self, src_cbct, real_us):
        """Full generator loss. Returns (total, fake_us, parts-dict)."""
        fake = self.G(src_cbct)
        gan = self.g_gan_loss(fake)
        nce = self.patchnce(src_cbct, fake)
        total = gan + nce
        parts = {"gan": float(gan.detach()), "nce": float(nce.detach())}
        if self.nce_idt:                                          # identity-NCE on real US
            idt = self.G(real_us)
            nce_y = self.patchnce(real_us, idt)
            total = total + nce_y
            parts["nce_idt"] = float(nce_y.detach())
        return total, fake, parts
