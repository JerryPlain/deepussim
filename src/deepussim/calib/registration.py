"""Step 3 — coordinate calibration (the one-time alignment, not per-frame matching).

With accurate FK probe poses, "registration to CBCT" is best done as a one-time
calibration of the coordinate chain rather than image-based matching of every frame:

    T_world_from_CBCT  =  T_world_from_phantom @ T_phantom_from_CBCT          (placement)
    pose_in_CBCT       =  inv(T_world_from_CBCT) @ T_world_from_probe         (per frame)

Each link is estimated once from corresponding fiducial points (e.g. CT-visible
markers touched by the probe tip), via :func:`rigid_register` (Umeyama / Kabsch,
rotation + translation, no scale). US<->CT image registration, if used at all, only
refines this — it is not the workhorse here.
"""
from __future__ import annotations

import numpy as np

from ..geometry import make_transform, compose, invert


def rigid_register(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Best-fit rigid transform mapping ``src`` points onto ``dst`` (Kabsch).

    Args:
        src, dst: (N, 3) corresponding point sets in the source/target frames.

    Returns:
        (T_dst_from_src, rmse) where applying T to src minimises ||T·src - dst||.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N, 3) and the same shape")
    if src.shape[0] < 3:
        raise ValueError("need at least 3 non-collinear correspondences")

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])  # reflection guard -> proper rotation
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c

    T = make_transform(R, t)
    residual = (R @ src.T).T + t - dst
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return T, rmse


def build_world_to_cbct(
    T_world_from_phantom: np.ndarray,
    T_phantom_from_cbct: np.ndarray,
) -> np.ndarray:
    """Compose the placement links into ``T_world_from_CBCT``."""
    return compose(T_world_from_phantom, T_phantom_from_cbct)


def probe_pose_in_cbct(
    T_world_from_cbct: np.ndarray,
    T_world_from_probe: np.ndarray,
) -> np.ndarray:
    """Express a world-frame probe pose in the CBCT frame, ready for reslicing."""
    return compose(invert(T_world_from_cbct), T_world_from_probe)
