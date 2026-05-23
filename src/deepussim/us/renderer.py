"""First-pass US image-formation model (the acoustic half of Step 5).

This is intentionally a *simple, differentiable-in-spirit* B-mode model whose
parameters are what Step 4 calibrates against real US. It maps a resliced CBCT
intensity image (treated as a proxy for acoustic impedance Z) to a B-mode-like
envelope image in [0, 1]:

  1. interface reflection: amplitude reflectivity r = |dZ| / (Z_i + Z_{i+1}) along
     the axial (depth) direction — bright echoes where impedance changes;
  2. depth attenuation: round-trip ``2 * alpha * f * depth`` in dB, partially undone
     by time-gain compensation (TGC);
  3. speckle: multiplicative lognormal texture;
  4. log compression into a dynamic-range window, mapped to [0, 1].

It deliberately omits: refraction, shadowing/enhancement, beam-width blur, multipath.
Add those as the sim-to-real gap demands; keep params in RendererParams so the
calibration in calib.renderer_fit can reach them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reslice import ProbeGeometry


@dataclass
class RendererParams:
    freq_mhz: float = 5.0
    atten_db_cm_mhz: float = 0.5
    tgc_db_cm: float = 0.5
    reflect_scale: float = 1.0
    speckle_sigma: float = 0.5
    gain_db: float = 0.0
    dynamic_range_db: float = 50.0
    seed: int | None = 0


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def render(
    intensity: np.ndarray,
    geom: ProbeGeometry,
    params: RendererParams | None = None,
) -> np.ndarray:
    """Render a B-mode-like image from a resliced intensity plane (n_ax, n_lat)."""
    params = params or RendererParams()
    img = np.asarray(intensity, dtype=float)
    if img.shape != (geom.n_ax, geom.n_lat):
        raise ValueError(f"intensity {img.shape} != geometry ({geom.n_ax}, {geom.n_lat})")

    # Impedance proxy, normalised so reflectivity is scale-free.
    Z = _normalize(img) + 1e-3

    # (1) Axial interface reflectivity (amplitude).
    refl = np.zeros_like(Z)
    z_lo, z_hi = Z[:-1, :], Z[1:, :]
    refl[1:, :] = np.abs(z_hi - z_lo) / (z_hi + z_lo)
    refl *= params.reflect_scale

    # (2) Round-trip attenuation, TGC-compensated, per depth row.
    depth_cm = geom.axial_depths_mm() / 10.0
    atten_db = 2.0 * params.atten_db_cm_mhz * params.freq_mhz * depth_cm
    atten_db -= params.tgc_db_cm * depth_cm
    atten_lin = 10.0 ** (-atten_db / 20.0)
    env = refl * atten_lin[:, None]

    # (3) Multiplicative speckle.
    if params.speckle_sigma > 0:
        rng = np.random.default_rng(params.seed)
        env = env * rng.lognormal(0.0, params.speckle_sigma, size=env.shape)

    # (4) Log compression into the dynamic-range window.
    eps = 1e-6
    db = 20.0 * np.log10(env + eps)
    db -= db.max()
    db += params.gain_db
    out = (db + params.dynamic_range_db) / params.dynamic_range_db
    return np.clip(out, 0.0, 1.0)
