"""NeuralRenderer — wrap a trained CUT generator with the physics renderer's interface.

Drop-in for ``us.renderer.render``: takes a resliced CBCT intensity slice and returns a US-like
image in [0, 1], but the appearance comes from the learned generator instead of the 7-parameter
B-mode model. ``pipeline.scaleup.generate_dataset(renderer=NeuralRenderer(...))`` then writes
realistic US into the dataset while keeping every geometric product (pose, mask, force) identical.

torch is imported lazily here, so the physics path (and the rest of the CPU core) stays torch-free.
"""
from __future__ import annotations

import numpy as np

from .data import _to_unit


class NeuralRenderer:
    """Callable ``(intensity[, geom]) -> US image in [0,1]`` backed by a trained generator."""

    def __init__(self, ckpt_path, device=None):
        import torch
        from .networks import ResnetGenerator

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        a = ckpt["args"]
        self.G = ResnetGenerator(1, 1, a["ngf"], a["n_blocks"])
        self.G.load_state_dict(ckpt["G"])
        self.G.eval().to(self.device)
        self.epoch = ckpt.get("epoch")
        self._torch = torch

    def __call__(self, intensity, geom=None) -> np.ndarray:
        torch = self._torch
        x = _to_unit(intensity)                                  # CBCT slice -> [-1, 1]
        t = torch.from_numpy(x.astype("float32"))[None, None].to(self.device)
        with torch.no_grad():
            y = self.G(t)[0, 0].cpu().numpy()                    # generated US in [-1, 1]
        return ((y + 1.0) / 2.0).astype(np.float32)              # -> [0, 1], matches render()
