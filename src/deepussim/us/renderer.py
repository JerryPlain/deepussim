"""First-pass US image-formation model (the acoustic half of Step 5).

This is intentionally a *simple, differentiable-in-spirit* B-mode model whose
parameters are what Step 4 calibrates against real US. It maps a resliced CBCT
intensity image (treated as a proxy for acoustic impedance Z) to a B-mode-like
envelope image in [0, 1]:

  1. interface reflection: amplitude reflectivity r = |dZ| / (Z_i + Z_{i+1}) along
     the axial (depth) direction — bright echoes where impedance changes;
  2. depth attenuation: round-trip ``2 * alpha * f * depth`` in dB, partially undone
     by time-gain compensation (TGC);
  3. acoustic shadowing: cumulative transmission down each scanline — energy reflected
     at an interface no longer reaches deeper pixels, so strong reflectors cast shadows
     and the deep field darkens (``shadow_scale``; set 0 to disable);
  4. speckle: multiplicative lognormal texture;
  5. log compression into a dynamic-range window, mapped to [0, 1].

It deliberately omits: refraction, beam-width blur, multipath. Add those as the
sim-to-real gap demands; keep params in RendererParams so the calibration in
calib.renderer_fit can reach them.
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
    scatter_scale: float = 0.3       # diffuse tissue backscatter (fills homogeneous tissue); 0 = interfaces only
    shadow_scale: float = 1.0        # cumulative-transmission (acoustic shadowing); 0 disables
    beam_width_px: float = 0.0       # lateral (beam-width) blur; available but OFF by default — it
    #                                  smooths per-scanline striping yet shifted the deep-field
    #                                  physics worse (deep-conf AUC 0.56->0.79) and regressed the
    #                                  sim2real downstream (0.346->0.332), so keep it off.
    speckle_sigma: float = 0.5
    gain_db: float = 0.0
    dynamic_range_db: float = 50.0
    seed: int | None = 0


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def bmode(
    intensity: np.ndarray,
    depth_mm: np.ndarray,
    params: RendererParams | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Core B-mode model on an axial-major image: rows = depth, columns = scan lines.

    ``depth_mm`` is the per-row depth (shape ``(n_rows,)``). Works on any (depth, lateral)
    array — a resliced fan (via :func:`render`) or a display sector — so the acoustic model
    lives in one place. ``mask`` (optional) is the fan-content region: pass it when the input
    has a black surround (a display sector) so the content->background border is NOT treated as
    a giant interface (which would dominate log-compression and black out the tissue).
    """
    params = params or RendererParams()
    img = np.asarray(intensity, dtype=float)
    if mask is None:
        mask = img > (float(img.max()) * 0.02)
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(mask, iterations=2)          # drop the border ring from gradients

    # Impedance proxy, normalised so reflectivity is scale-free.
    Z = _normalize(img) + 1e-3

    # (1) Axial interface reflectivity (amplitude), suppressed at the fan border.
    refl = np.zeros_like(Z)
    z_lo, z_hi = Z[:-1, :], Z[1:, :]
    refl[1:, :] = np.abs(z_hi - z_lo) / (z_hi + z_lo)
    refl = np.clip(refl * params.reflect_scale, 0.0, 1.0) * inner

    # (1b) Diffuse tissue backscatter — homogeneous tissue echoes throughout (not just at
    # interfaces), proportional to local echogenicity. Without this the image is near-black.
    scatter = params.scatter_scale * Z * inner

    # (2) Round-trip depth attenuation, TGC-compensated (structure-independent).
    depth_cm = np.asarray(depth_mm, dtype=float) / 10.0
    atten_db = 2.0 * params.atten_db_cm_mhz * params.freq_mhz * depth_cm
    atten_db -= params.tgc_db_cm * depth_cm
    atten_lin = 10.0 ** (-atten_db / 20.0)

    # (3) Acoustic shadowing: cumulative one-way transmission down each scan line. Energy
    # reflected at an interface (~refl^2) is not transmitted deeper, so pixels below strong
    # reflectors go dark. Use the transmission ABOVE each pixel (exclusive), round-tripped.
    if params.shadow_scale > 0:
        step = np.clip(1.0 - params.shadow_scale * refl ** 2, 0.0, 1.0)
        trans = np.cumprod(step, axis=0)
        trans_excl = np.ones_like(trans)
        trans_excl[1:, :] = trans[:-1, :]
        shadow = trans_excl ** 2                       # down-and-back
    else:
        shadow = 1.0
    env = (refl + scatter) * atten_lin[:, None] * shadow

    # (3b) Beam-width lateral blur: a finite beam smooths neighbouring scan lines, removing the
    # per-column striping of an independent-scanline model (real US has this lateral coherence).
    if params.beam_width_px > 0:
        from scipy.ndimage import gaussian_filter1d
        env = gaussian_filter1d(env, params.beam_width_px, axis=1, mode="nearest")

    # (4) Multiplicative speckle.
    if params.speckle_sigma > 0:
        rng = np.random.default_rng(params.seed)
        env = env * rng.lognormal(0.0, params.speckle_sigma, size=env.shape)

    # (5) Log compression into the dynamic-range window.
    eps = 1e-6
    db = 20.0 * np.log10(env + eps)
    db -= db[inner].max() if inner.any() else db.max()  # normalise to the tissue peak, not border
    db += params.gain_db
    out = (db + params.dynamic_range_db) / params.dynamic_range_db
    return np.clip(out, 0.0, 1.0) * mask                # black surround outside the fan


def render(
    intensity: np.ndarray,
    geom: ProbeGeometry,
    params: RendererParams | None = None,
) -> np.ndarray:
    """Render a B-mode-like image from a resliced intensity fan (n_ax, n_lat)."""
    img = np.asarray(intensity, dtype=float)
    if img.shape != (geom.n_ax, geom.n_lat):
        raise ValueError(f"intensity {img.shape} != geometry ({geom.n_ax}, {geom.n_lat})")
    return bmode(img, geom.axial_depths_mm(), params)
