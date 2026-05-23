"""SE(3) transform utilities — the math primitives for Step 3 (registration).

Convention: a pose is a 4x4 homogeneous matrix ``T`` that maps a point from a
*source* frame into a *target* frame,

    p_target = T @ p_source,   p = [x, y, z, 1]^T.

We name transforms ``T_<target>_from_<source>`` so that composition reads left to
right: ``T_a_from_c = T_a_from_b @ T_b_from_c``.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def identity() -> Array:
    return np.eye(4)


def make_transform(R: Array, t) -> Array:
    """Assemble a 4x4 transform from a 3x3 rotation and a 3-vector translation."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert(T: Array) -> Array:
    """Inverse of a rigid transform (uses R^T, avoids a general matrix inverse)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose(*transforms: Array) -> Array:
    """Left-to-right composition: ``compose(A, B, C) == A @ B @ C``."""
    if not transforms:
        return identity()
    out = np.asarray(transforms[0], dtype=float)
    for T in transforms[1:]:
        out = out @ np.asarray(T, dtype=float)
    return out


def apply(T: Array, points: Array) -> Array:
    """Transform points. ``points`` is (N, 3) or (3,); returns the same shape."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
    out = (h @ np.asarray(T, dtype=float).T)[:, :3]
    return out[0] if single else out


def rot_x(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz(rx: float, ry: float, rz: float) -> Array:
    """Rotation from intrinsic X-Y-Z Euler angles (radians)."""
    return rot_x(rx) @ rot_y(ry) @ rot_z(rz)


def from_translation(t) -> Array:
    return make_transform(np.eye(3), t)


def is_rigid(T: Array, tol: float = 1e-6) -> bool:
    """Check that the upper-left 3x3 block is a proper rotation."""
    R = np.asarray(T, dtype=float)[:3, :3]
    return bool(
        np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol
    )
