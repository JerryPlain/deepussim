"""LC2 (Linear Correlation of Linear Combination) — the US<->CT similarity metric.

Self-contained reimplementation (numpy + scipy only). Verified bit-for-bit identical to
``deepussim.calib.lc2.lc2_similarity``.

US brightness is not a linear function of CT intensity (plain correlation fails across the
two modalities), but *locally* a US patch is well explained by a linear combination of the
CT intensity and its gradient magnitude (Wein et al. 2008):

    US  ~=  a * CT  +  b * |grad CT|  +  c        (a, b, c fit per local window)

LC2 = the fraction of US variance that combination explains, aggregated over the image and
weighted by each window's US variance (textured regions count, flat/black background does
not). LC2 in [0, 1]; higher = better aligned.

Implementation: per-pixel sliding-window least squares via box filters (``uniform_filter``).
Centering each window absorbs the intercept c, leaving a 2x2 normal-equation solve for (a, b);
the explained variance is then ``(a*cov(CT,US) + b*cov(|grad CT|,US)) / var(US)``.
"""
from __future__ import annotations

import numpy as np


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """|grad img| via central differences."""
    gy, gx = np.gradient(np.asarray(img, dtype=float))
    return np.hypot(gx, gy)


def lc2_map(us: np.ndarray, ct: np.ndarray, patch: int = 9, eps: float = 1e-6):
    """Per-pixel windowed LC2 and weights; returns ``(lc2_pixel, weight)`` (image-shaped).

    ``lc2_pixel`` is the explained-variance ratio in each window; ``weight`` is the window's
    US variance (used to weight the aggregate toward textured regions).
    """
    from scipy.ndimage import uniform_filter

    us = np.asarray(us, dtype=float)
    ct = np.asarray(ct, dtype=float)
    if us.shape != ct.shape:
        raise ValueError(f"us {us.shape} and ct {ct.shape} must match")

    # The two predictors of US brightness: CT intensity and its gradient magnitude.
    g = gradient_magnitude(ct)
    # Rescale both predictors to unit global variance so each tiny per-window 2x2 system stays
    # well-conditioned (a smooth CT volume has minute within-window covariances). LC2 is an
    # explained-variance *ratio*, so this global rescaling leaves the result unchanged.
    ct = (ct - ct.mean()) / (ct.std() + eps)
    g = (g - g.mean()) / (g.std() + eps)

    # Every quantity below is computed PER PIXEL over its surrounding patch, in one shot, with a
    # box filter: ``mean(x)[i,j]`` is the average of ``x`` over the patch centred at ``(i,j)``.
    def mean(x):
        return uniform_filter(x, size=patch, mode="nearest")

    # Per-window means of US, CT and gradient.
    Eu, Ec, Eg = mean(us), mean(ct), mean(g)
    # Per-window (co)variances: S_xy = E[x*y] - E[x]E[y]. Centering this way absorbs the
    # intercept c, so the local fit ``US ~= a*CT + b*G + c`` reduces to solving for (a, b).
    Scc = mean(ct * ct) - Ec * Ec          # var(CT)
    Sgg = mean(g * g) - Eg * Eg            # var(G)
    Scg = mean(ct * g) - Ec * Eg           # cov(CT, G)
    Scu = mean(ct * us) - Ec * Eu          # cov(CT, US)
    Sgu = mean(g * us) - Eg * Eu           # cov(G, US)
    Suu = mean(us * us) - Eu * Eu          # var(US)

    # Per-window least squares for (a, b): the 2x2 normal equations are
    #   [Scc Scg][a]   [Scu]
    #   [Scg Sgg][b] = [Sgu]
    # solved by Cramer's rule. Guard windows whose system is near-singular (flat CT).
    det = Scc * Sgg - Scg * Scg
    ok = np.abs(det) > eps
    det_safe = np.where(ok, det, 1.0)
    a = np.where(ok, (Sgg * Scu - Scg * Sgu) / det_safe, 0.0)
    b = np.where(ok, (Scc * Sgu - Scg * Scu) / det_safe, 0.0)

    # Variance of the fitted prediction = a*cov(CT,US) + b*cov(G,US); LC2 = that / var(US).
    explained = a * Scu + b * Sgu
    lc2_pixel = np.clip(explained / (Suu + eps), 0.0, 1.0)
    # Weight = local US variance: textured windows count, flat/black background is ignored.
    weight = np.clip(Suu, 0.0, None)
    return lc2_pixel, weight


def lc2_similarity(us: np.ndarray, ct: np.ndarray, patch: int = 9,
                   mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    """Scalar LC2 in [0, 1]: US-variance-weighted mean of the windowed explained variance.

    ``mask`` (optional, US-shaped bool) restricts aggregation to valid pixels (e.g. the fan
    interior, excluding the black surround).
    """
    lc2_pixel, weight = lc2_map(us, ct, patch=patch, eps=eps)
    if mask is not None:
        weight = weight * np.asarray(mask, dtype=float)
    wsum = float(weight.sum())
    if wsum <= eps:
        return 0.0
    return float((weight * lc2_pixel).sum() / wsum)
