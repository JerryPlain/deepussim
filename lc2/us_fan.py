"""Fit the US probe fan from B-mode frames and unwrap a frame onto the reslice grid.

Self-contained reimplementation (numpy + scipy only). Verified identical to
``deepussim.calib.us_geometry``.

A convex-array B-mode image is a *sector*: scan lines fan out from a virtual apex behind the
transducer face, bounded by two straight side edges and inner (face) / outer (depth) arcs.
With the US pixel spacing (mm/px) this recovers the physical :class:`reslice.fan.ProbeGeometry`
so the CBCT can be resliced on the *same* fan as the real probe; :func:`unwrap_fan` puts a US
frame on that same polar grid so the two can be compared by LC2.

Fit steps (on a max-envelope over contact frames, see :func:`contact_envelope`):
1. threshold -> largest connected component -> fill holes (drops logo / border specks);
2. per-row left/right edges, fit lines to the *straight upper band* only (lower rows curve in
   along the outer arc) -> their intersection is the apex;
3. radii along the apex column: ``r0`` = apex->face arc, ``r1`` = apex->outer arc;
4. ``fov`` from the two side-line slopes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import lc2  # noqa: F401  (path setup for reslice)
from reslice.fan import ProbeGeometry


@dataclass
class FanFit:
    """Raw pixel-space fan fit (before applying the mm scale)."""

    apex_px: tuple[float, float]   # (x, y) virtual apex, usually y < 0 (above the frame)
    r0_px: float                   # apex -> transducer-face arc
    r1_px: float                   # apex -> outer (depth) arc
    fov_deg: float                 # total sector angle
    resid_px: float                # side-line fit residual (sanity)


def contact_envelope(images, contact=None, threshold: int = 20) -> np.ndarray:
    """Max-projection over (contact) frames: the sector that is *ever* bright = the fan.

    Filling speckle with the pixelwise max gives a solid bright sector that fits far more
    robustly than a single noisy frame. ``contact`` optionally restricts to in-contact frames.
    """
    imgs = np.asarray(images)
    if contact is not None:
        imgs = imgs[np.asarray(contact, dtype=bool)]
    if imgs.size == 0:
        raise ValueError("no frames to build the envelope from")
    return imgs.max(axis=0)


def _fan_mask(image: np.ndarray, threshold: int) -> np.ndarray:
    """Binary sector mask: threshold -> largest connected component -> fill holes."""
    from scipy import ndimage

    m = np.asarray(image) > threshold
    lab, n = ndimage.label(m)
    if n > 1:                                  # keep the fan, drop logo / border specks
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(m)


def fit_fan_pixels(image: np.ndarray, threshold: int = 20,
                   straight_band: tuple[float, float] = (0.08, 0.42)) -> FanFit:
    """Fit the convex sector of a B-mode image in pixel units (see module docstring)."""
    m = _fan_mask(image, threshold)
    H, W = m.shape
    rows = np.where(m.any(axis=1))[0]
    if rows.size < 10:
        raise ValueError("no fan detected (mask empty) — check the threshold")
    ytop, ybot = int(rows.min()), int(rows.max())
    hf = ybot - ytop

    left, right = {}, {}
    for y in range(H):
        on = np.where(m[y])[0]
        if on.size > 10:
            left[y], right[y] = int(on.min()), int(on.max())

    # Side edges are straight only in the upper band; lower rows curve in along the outer arc
    # and would bend the line fit (pushing the apex to infinity), so fit the upper band only.
    lo, hi = ytop + straight_band[0] * hf, ytop + straight_band[1] * hf
    ys = np.array([y for y in left if lo <= y <= hi], dtype=float)
    if ys.size < 5:
        raise ValueError("too few straight-edge rows to fit (bad band / mask)")
    aL, bL = np.polyfit(ys, [left[int(y)] for y in ys], 1)
    aR, bR = np.polyfit(ys, [right[int(y)] for y in ys], 1)
    if abs(aL - aR) < 1e-6:
        raise ValueError("side edges parallel — fan fit degenerate")

    ay = (bR - bL) / (aL - aR)                 # apex = intersection of the two side lines
    ax = aL * ay + bL
    col = int(round(ax))
    if not (0 <= col < W) or not m[:, col].any():
        raise ValueError(f"apex column {col} outside the fan — fit degenerate")
    on = np.where(m[:, col])[0]
    r0, r1 = float(on.min() - ay), float(on.max() - ay)
    fov = math.degrees(math.atan(aR) - math.atan(aL))

    resid = float(np.std([left[int(y)] - (aL * y + bL) for y in ys]
                         + [right[int(y)] - (aR * y + bR) for y in ys]))
    return FanFit((float(ax), float(ay)), r0, r1, abs(fov), resid)


def unwrap_fan(image: np.ndarray, fan: FanFit, n_ax: int = 512, n_lat: int = 256,
               order: int = 1) -> np.ndarray:
    """Resample a B-mode US frame onto the ``(n_ax, n_lat)`` polar grid the reslice produces.

    Each output cell ``(depth_index, scan_line)`` maps to a radius/angle in the fitted fan
    (``r`` from ``r0_px`` to ``r1_px``, ``theta`` over ``+/- fov/2``), then to a source pixel
    sampled bilinearly. Rows = depth, cols = scan line — the *same* layout as
    :func:`reslice.fan.reslice_fan`, so the two can be compared by LC2.
    """
    from scipy.ndimage import map_coordinates

    ax, ay = fan.apex_px
    depth = np.linspace(0.0, 1.0, n_ax)
    theta = np.deg2rad(np.linspace(-fan.fov_deg / 2.0, fan.fov_deg / 2.0, n_lat))
    r = fan.r0_px + np.outer(depth, np.ones(n_lat)) * (fan.r1_px - fan.r0_px)   # (n_ax, n_lat)
    th = np.outer(np.ones(n_ax), theta)
    px = ax + r * np.sin(th)
    py = ay + r * np.cos(th)
    sampled = map_coordinates(np.asarray(image, dtype=float), [py.ravel(), px.ravel()],
                              order=order, mode="constant", cval=0.0)
    return sampled.reshape(n_ax, n_lat)


def fit_fan_geometry(image: np.ndarray, us_spacing_mm: float, *, threshold: int = 20,
                     straight_band: tuple[float, float] = (0.08, 0.42),
                     n_lat: int = 256, n_ax: int = 512) -> ProbeGeometry:
    """Fit a physical :class:`reslice.fan.ProbeGeometry` from a B-mode frame/envelope + spacing.

    ``us_spacing_mm`` is the US pixel size (mm/px). ``n_lat``/``n_ax`` set the *output* reslice
    resolution (a sampling choice, not measured).
    """
    fit = fit_fan_pixels(image, threshold=threshold, straight_band=straight_band)
    return ProbeGeometry(
        radius_mm=fit.r0_px * us_spacing_mm,
        fov_deg=fit.fov_deg,
        depth_mm=(fit.r1_px - fit.r0_px) * us_spacing_mm,
        n_lat=n_lat,
        n_ax=n_ax,
    )
