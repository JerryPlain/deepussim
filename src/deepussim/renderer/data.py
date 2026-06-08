"""Unpaired two-domain data for CUT, both in the (n_ax, n_lat) fan layout.

Source domain = CBCT reslices (the anatomical structure); target domain = real US frames
unwrapped onto the same fan grid. CUT needs only the two *pools* (no pixel pairing), so the
cm-level registration never enters here. Build the pools once with ``scripts/prep_renderer_data.py``
(writes ``source_cbct.npz`` / ``target_us.npz``), then train off them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_unit(x: np.ndarray) -> np.ndarray:
    """Per-image normalise to [-1, 1] (Tanh generator range)."""
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


class FanPool(Dataset):
    """A single domain: stacked ``(N, n_ax, n_lat)`` images from an npz (key ``images``)."""

    def __init__(self, npz_path):
        self.images = np.load(npz_path)["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.from_numpy(_to_unit(self.images[i]))[None]   # (1, n_ax, n_lat)


class UnpairedFanDataset(Dataset):
    """Pairs a source image with a RANDOM target image (unpaired) — one CUT training item."""

    def __init__(self, source_npz, target_npz, seed=0):
        self.src = FanPool(source_npz)
        self.tgt = FanPool(target_npz)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        j = int(self.rng.integers(len(self.tgt)))
        return {"src": self.src[i], "tgt": self.tgt[j]}
