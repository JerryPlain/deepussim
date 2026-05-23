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


# --- quaternions ----------------------------------------------------------
# Scalar-first (w, x, y, z) convention, matching Genesis' IK/pose API.

def quat_to_mat(q) -> Array:
    """Scalar-first unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def mat_to_quat(R) -> Array:
    """3x3 rotation matrix -> scalar-first unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=float)[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def pose_from_pos_quat(pos, quat) -> Array:
    """Assemble a 4x4 transform from a position and a scalar-first quaternion."""
    return make_transform(quat_to_mat(quat), pos)
