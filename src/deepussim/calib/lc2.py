"""LC2 (Linear Correlation of Linear Combination) — the US↔CT registration similarity.

US brightness is not a linear function of CT intensity (NCC fails across the modalities), but
*locally* a US patch is well explained by a linear combination of the CT intensity and its
gradient magnitude (Wein et al. 2008):

    US  ≈  α · CT  +  β · |∇CT|  +  γ        (α, β, γ fit per local window)

LC2 is the fraction of US variance that combination explains, aggregated over the image
weighted by each window's US variance (so textured regions count, flat/black background does
not). LC2 ∈ [0, 1]; higher = better aligned. :func:`register_frame_lc2` then maximises it over a
*constrained* 6-DoF range around the calibration pose to register each real US frame to the CBCT.

Implementation: per-pixel sliding-window least squares via box filters (``uniform_filter``).
Centering each window absorbs the intercept γ, leaving a 2×2 normal-equation solve for (α, β);
the explained variance is then ``(α·cov(CT,US) + β·cov(|∇CT|,US)) / var(US)``.

References:
  - Wein et al., "Global registration of ultrasound to MRI using the LC2 metric…", MICCAI 2013.
  - Fuerst et al., "Automatic ultrasound–MRI registration… 2D and 3D LC2", MedIA 18(8), 2014.
  - Li et al. (TUM CAMP), "Robotic Ultrasound Makes CBCT Alive" — calibration-initialized,
    constrained-range LC2 rigid refinement; the workflow this module implements.
"""
from __future__ import annotations

import numpy as np


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """|∇img| via central differences."""
    gy, gx = np.gradient(np.asarray(img, dtype=float))
    return np.hypot(gx, gy)


def lc2_map(us: np.ndarray, ct: np.ndarray, patch: int = 9, eps: float = 1e-6):
    """Per-pixel windowed LC2 and weights. Returns ``(lc2_pixel, weight)`` (both image-shaped).

    ``lc2_pixel`` is the explained-variance in each window; ``weight`` is the window US variance.
    """
    from scipy.ndimage import uniform_filter

    us = np.asarray(us, dtype=float)
    ct = np.asarray(ct, dtype=float)
    if us.shape != ct.shape:
        raise ValueError(f"us {us.shape} and ct {ct.shape} must match")
    g = gradient_magnitude(ct)
    # Normalize the predictors to unit global variance so the per-window 2x2 stays well-scaled
    # (a smooth volume's within-window covariances are tiny; an absolute det guard would wrongly
    # zero valid windows). LC2 is an explained-variance *ratio*, invariant to this scaling.
    ct = (ct - ct.mean()) / (ct.std() + eps)
    g = (g - g.mean()) / (g.std() + eps)

    def mean(x):
        return uniform_filter(x, size=patch, mode="nearest")

    Eu, Ec, Eg = mean(us), mean(ct), mean(g)
    # centered second moments (covariances) per window
    Scc = mean(ct * ct) - Ec * Ec
    Sgg = mean(g * g) - Eg * Eg
    Scg = mean(ct * g) - Ec * Eg
    Scu = mean(ct * us) - Ec * Eu
    Sgu = mean(g * us) - Eg * Eu
    Suu = mean(us * us) - Eu * Eu

    det = Scc * Sgg - Scg * Scg
    ok = np.abs(det) > eps
    det_safe = np.where(ok, det, 1.0)
    a = np.where(ok, (Sgg * Scu - Scg * Sgu) / det_safe, 0.0)
    b = np.where(ok, (Scc * Sgu - Scg * Scu) / det_safe, 0.0)

    explained = a * Scu + b * Sgu                       # Var(prediction)
    lc2_pixel = np.clip(explained / (Suu + eps), 0.0, 1.0)
    weight = np.clip(Suu, 0.0, None)                    # weight textured windows, ignore flat
    return lc2_pixel, weight


def lc2_similarity(us: np.ndarray, ct: np.ndarray, patch: int = 9,
                   mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    """Scalar LC2 in [0, 1] — US-variance-weighted mean of the windowed explained variance.

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


def register_frame_lc2(us_polar, volume, geom, init_pose, *, max_trans_mm: float = 15.0,
                       max_rot_deg: float = 15.0, patch: int = 9, mask=None,
                       maxiter: int = 300):
    """Refine a US-frame pose by maximising LC2 against the CBCT, over a constrained range.

    ``us_polar`` is the real US frame already on the reslice fan grid (``calib.us_geometry.
    unwrap_fan``), ``volume`` the CBCT, ``geom`` the calibrated :class:`ProbeGeometry`, and
    ``init_pose`` the calibration pose ``T_cbct_from_probe`` (mm) — the replay/hand-eye estimate.
    A 6-DoF perturbation (rotvec, translation) in the *probe* frame is searched within
    ``±max_rot_deg`` / ``±max_trans_mm`` (Powell, bounded) — the calibration-initialized,
    constrained LC2 refinement of Wein/Li. Returns ``(refined_pose, lc2)``.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as Rot

    from ..us.reslice import reslice_volume

    init = np.asarray(init_pose, dtype=float)

    def pose_of(p):
        delta = np.eye(4)
        delta[:3, :3] = Rot.from_rotvec(p[:3]).as_matrix()
        delta[:3, 3] = p[3:]
        return init @ delta                     # perturb in the probe frame

    def neg_lc2(p):
        ct = reslice_volume(volume, pose_of(p), geom)
        return -lc2_similarity(us_polar, ct, patch=patch, mask=mask)

    rb = np.deg2rad(max_rot_deg)
    bounds = [(-rb, rb)] * 3 + [(-max_trans_mm, max_trans_mm)] * 3
    res = minimize(neg_lc2, np.zeros(6), method="Powell", bounds=bounds,
                   options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3})
    return pose_of(res.x), float(-res.fun)
