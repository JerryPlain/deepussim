"""Step 4 — calibrate the US renderer against registered real data.

Given real samples each carrying a probe pose already expressed in the CBCT frame
(see calib.registration), we reslice the CBCT at that pose, render with candidate
``RendererParams``, and minimise an image-similarity loss against the real US.

The loss (``render_loss``) is fully runnable. The optimiser in ``fit_renderer`` is a
derivative-free Nelder-Mead over a chosen subset of parameters — a reasonable default
given the renderer's speckle term is stochastic; swap for CMA-ES / a differentiable
renderer later. Treat speckle_sigma carefully (fix the seed during fitting so the loss
is deterministic, as done here).
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Sequence

import numpy as np

from ..data.volume import Volume
from ..data.record import Sample
from ..us.reslice import ProbeGeometry, reslice_volume
from ..us.renderer import RendererParams, render


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [-1, 1]; 1.0 == identical structure."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def render_loss(
    params: RendererParams,
    samples: Sequence[Sample],
    volume: Volume,
    geom: ProbeGeometry,
) -> float:
    """Mean ``1 - NCC`` between rendered and real US over the samples (0 == perfect).

    Each sample's ``pose`` must already be in the CBCT/volume frame.
    """
    if not samples:
        raise ValueError("no samples to fit against")
    total = 0.0
    for s in samples:
        intensity = reslice_volume(volume, s.pose, geom)
        sim = render(intensity, geom, params)
        total += 1.0 - ncc(sim, s.image)
    return total / len(samples)


def fit_renderer(
    samples: Sequence[Sample],
    volume: Volume,
    geom: ProbeGeometry,
    init: RendererParams | None = None,
    free: Sequence[str] = ("atten_db_cm_mhz", "tgc_db_cm", "reflect_scale", "gain_db"),
) -> tuple[RendererParams, float]:
    """Optimise the ``free`` parameters to minimise :func:`render_loss`.

    Returns the best ``RendererParams`` and the achieved loss.
    """
    from scipy.optimize import minimize

    init = init or RendererParams()
    base = asdict(init)
    x0 = np.array([base[name] for name in free], dtype=float)

    def unpack(x: np.ndarray) -> RendererParams:
        # Keep speckle deterministic during fitting.
        overrides = {name: float(v) for name, v in zip(free, x)}
        return replace(init, seed=0, **overrides)

    def objective(x: np.ndarray) -> float:
        return render_loss(unpack(x), samples, volume, geom)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 500})
    return unpack(res.x), float(res.fun)
