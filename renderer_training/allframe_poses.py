#!/usr/bin/env python
"""Reconstruct CBCT-space probe poses for EVERY frame of a sequence (not just the LC2 subset).

Only ~10 frames per sequence were LC2-registered and stored in ``pairs.npz``. But the global
LC2 correction is a single per-sweep rigid, and the phantom placement is shared, so every frame's
refined pose can be reconstructed:

    refined_pose_j = global_correction @ probe_pose_in_phantom_centered_mm(pose_from_sequence(j), T_wf)

``T_wf`` (world_from_phantom, the missing placement file) is recovered from any one registered
frame, whose refined pose and the correction are both stored:

    init_mm_k   = inv(correction) @ refined_pose_k          # stored
    T_wf        = pose_from_sequence(k) @ inv(init_m_k)     # constant across the sweep

Validated to reproduce the 150 stored refined poses to < 1e-3 mm / 1e-3 deg.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from reslice import pose as P
from reslice.pose import invert_rigid


def _mm_to_m(T: np.ndarray) -> np.ndarray:
    out = np.asarray(T, dtype=float).copy()
    out[:3, 3] /= 1000.0
    return out


def world_from_phantom(sequence: Path, lc2_npz: Path, probe_rot: float = 0.0) -> np.ndarray:
    """Recover ``T_world_from_phantom`` (metres) from the first registered frame of a sweep."""
    lc2 = np.load(lc2_npz)
    corr = lc2["correction"]
    fidx = np.asarray(lc2["indices"], int)
    refined = lc2["refined_poses"]
    init_mm = np.linalg.inv(corr) @ refined[0]
    twp = P.pose_from_sequence(str(sequence), int(fidx[0]), "ee", probe_rot)
    return twp @ invert_rigid(_mm_to_m(init_mm))


def all_frame_poses(sequence: Path, lc2_npz: Path, probe_rot: float = 0.0) -> np.ndarray:
    """Refined CBCT-space (mm) probe pose for every frame of the sequence -> (N, 4, 4)."""
    lc2 = np.load(lc2_npz)
    corr = lc2["correction"]
    twf = world_from_phantom(sequence, lc2_npz, probe_rot)
    n = len(np.load(sequence, allow_pickle=True)["poses"])
    out = np.empty((n, 4, 4))
    for j in range(n):
        twp = P.pose_from_sequence(str(sequence), j, "ee", probe_rot)
        out[j] = corr @ P.probe_pose_in_phantom_centered_mm(twp, twf)
    return out


def reprojection_error(sequence: Path, lc2_npz: Path, probe_rot: float = 0.0) -> tuple[float, float]:
    """Max (trans mm, rot deg) error reproducing the stored registered poses -- a self-check."""
    lc2 = np.load(lc2_npz)
    corr = lc2["correction"]; fidx = np.asarray(lc2["indices"], int); refined = lc2["refined_poses"]
    twf = world_from_phantom(sequence, lc2_npz, probe_rot)
    et, er = 0.0, 0.0
    for mi, f in enumerate(fidx):
        twp = P.pose_from_sequence(str(sequence), int(f), "ee", probe_rot)
        pred = corr @ P.probe_pose_in_phantom_centered_mm(twp, twf)
        et = max(et, float(np.linalg.norm(pred[:3, 3] - refined[mi][:3, 3])))
        c = (np.trace(pred[:3, :3].T @ refined[mi][:3, :3]) - 1) / 2
        er = max(er, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return et, er
