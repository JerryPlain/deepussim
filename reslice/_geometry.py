"""Small, self-contained rigid-transform helpers used across the reslice package.

Kept dependency-free (numpy only) so the package can also run inside the 3D Slicer
Python interpreter, which does not have the project library installed.

Convention: a 4x4 ``T_A_FROM_B`` maps a point expressed in frame B into frame A
(``p_A = T_A_FROM_B @ p_B``). Translations are whatever unit the caller uses (the
reslice code mixes metres for world poses and millimetres for volume coordinates and
is explicit about it at every boundary).
"""
from __future__ import annotations

import math

import numpy as np


def rot_x(theta: float) -> np.ndarray:
    """Rotation about +X by ``theta`` radians (3x3)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    """Rotation about +Y by ``theta`` radians (3x3)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    """Rotation about +Z by ``theta`` radians (3x3)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and a 3-vector."""
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, dtype=float)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def translation(offset) -> np.ndarray:
    """4x4 pure-translation transform."""
    return make_transform(np.eye(3), offset)


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right matrix product: ``compose(A, B, C) == A @ B @ C``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ np.asarray(T, dtype=float)
    return out


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (transpose the rotation, re-map the translation)."""
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v`` (raises on a zero-length input)."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n
