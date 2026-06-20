"""LC2 registration: search for the pose correction that best matches CBCT to US.

A correction is a small rigid 6-DoF transform (rotation about a chosen centre + a
translation) applied in the phantom-centred CBCT frame:

    refined_pose = correction(params, centre) @ init_pose

* :func:`register_frame` searches one correction *per frame* (flexible, can overfit);
* :func:`register_global` searches ONE correction shared by *all* frames (robust).

Both maximise LC2 with a bounded Powell search, so the correction stays within the
expected calibration residual instead of drifting to an arbitrary look-alike pose.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import lc2  # noqa: F401  (path setup)
from reslice._geometry import make_transform, rot_x, rot_y, rot_z

from lc2.forward import Context, FrameTargets


def correction_matrix(params: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Rigid correction from ``[rx, ry, rz (deg), tx, ty, tz (mm)]``, rotating about ``center``.

    Rotating about ``center`` (a point in the slice region, not the far CBCT origin) keeps
    rotation and translation roughly independent, so the bounded search behaves well — a few
    degrees of rotation moves the slice by mm, not by tens of cm.
    """
    rx, ry, rz, tx, ty, tz = params
    R = rot_z(np.deg2rad(rz)) @ rot_y(np.deg2rad(ry)) @ rot_x(np.deg2rad(rx))
    # Read right-to-left: shift center to origin, rotate, shift back, then translate by t.
    return (
        make_transform(np.eye(3), [tx, ty, tz])
        @ make_transform(np.eye(3), center)
        @ make_transform(R, [0.0, 0.0, 0.0])
        @ make_transform(np.eye(3), -np.asarray(center, dtype=float))
    )


def _bounds(max_rot_deg: float, max_trans_mm: float):
    return [(-max_rot_deg, max_rot_deg)] * 3 + [(-max_trans_mm, max_trans_mm)] * 3


def register_frame(
    ctx: Context,
    frame: FrameTargets,
    max_trans_mm: float = 15.0,
    max_rot_deg: float = 15.0,
    maxiter: int = 60,
) -> dict:
    """Refine one frame independently. Returns before/after LC2, fan coverage, refined pose."""
    center = frame.init_pose_mm[:3, 3]

    def neg_lc2(p):
        pose = correction_matrix(p, center) @ frame.init_pose_mm
        return -ctx.score(pose, frame.us_polar)

    before = ctx.score(frame.init_pose_mm, frame.us_polar)
    res = minimize(
        neg_lc2, np.zeros(6), method="Powell",
        bounds=_bounds(max_rot_deg, max_trans_mm),
        options={"maxiter": maxiter, "xtol": 0.5, "ftol": 1e-3},
    )
    refined = correction_matrix(res.x, center) @ frame.init_pose_mm
    return {
        "index": frame.index,
        "lc2_before": before,
        "lc2_after": float(-res.fun),
        "inside_before": ctx.fraction_inside(frame.init_pose_mm),
        "inside_after": ctx.fraction_inside(refined),
        "params": res.x.tolist(),
        "refined_pose": refined,
    }


def register_global(
    ctx: Context,
    max_trans_mm: float = 15.0,
    max_rot_deg: float = 15.0,
    maxiter: int = 80,
) -> dict:
    """Refine ALL frames with one shared correction (maximise mean LC2). The robust mode."""
    # Rotate the shared correction about the centroid of all the probe positions.
    center = np.mean([f.init_pose_mm[:3, 3] for f in ctx.frames], axis=0)

    def mean_lc2(p):
        T = correction_matrix(p, center)
        return float(np.mean([ctx.score(T @ f.init_pose_mm, f.us_polar) for f in ctx.frames]))

    before = [ctx.score(f.init_pose_mm, f.us_polar) for f in ctx.frames]
    res = minimize(
        lambda p: -mean_lc2(p), np.zeros(6), method="Powell",
        bounds=_bounds(max_rot_deg, max_trans_mm),
        options={"maxiter": maxiter, "xtol": 0.5, "ftol": 1e-3},
    )
    T = correction_matrix(res.x, center)
    refined_poses = [T @ f.init_pose_mm for f in ctx.frames]
    after = [ctx.score(p, f.us_polar) for p, f in zip(refined_poses, ctx.frames)]
    return {
        "indices": [f.index for f in ctx.frames],
        "params": res.x.tolist(),
        "correction": T,
        "lc2_before": before,
        "lc2_after": after,
        "inside_before": [ctx.fraction_inside(f.init_pose_mm) for f in ctx.frames],
        "inside_after": [ctx.fraction_inside(p) for p in refined_poses],
        "refined_poses": refined_poses,
    }
