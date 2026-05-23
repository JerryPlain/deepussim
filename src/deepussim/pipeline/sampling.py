"""Probe-pose samplers (in the volume/CBCT frame) for scale-up.

These produce *geometric* poses for the no-sim path. When the Genesis scene is in the
loop, treat these as nominal targets — the achieved (reachable, in-contact) pose comes
back from ``UltrasoundScene.probe_pose()``.

Pose convention matches us.reslice: probe +z is axial (into tissue), +x lateral.
"""
from __future__ import annotations

import numpy as np

from ..geometry import make_transform, rot_x


def _aim_into_tissue(position, axial_dir) -> np.ndarray:
    """Build T_vol_from_probe with probe +z along ``axial_dir`` (into the volume)."""
    z = np.asarray(axial_dir, dtype=float)
    z = z / (np.linalg.norm(z) + 1e-12)
    # Pick a lateral axis not parallel to z.
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return make_transform(R, position)


def linear_sweep(start, end, n: int, axial_dir=(0.0, 0.0, -1.0)) -> list[np.ndarray]:
    """``n`` probe poses translating from ``start`` to ``end`` (mm), fixed orientation."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    return [_aim_into_tissue(start + t * (end - start), axial_dir) for t in ts]


def tilt_fan(position, n: int, max_tilt_deg: float = 20.0,
             axial_dir=(0.0, 0.0, -1.0)) -> list[np.ndarray]:
    """``n`` poses at a fixed point, fanning the probe through +/- ``max_tilt_deg``."""
    base = _aim_into_tissue(position, axial_dir)
    angles = np.deg2rad(np.linspace(-max_tilt_deg, max_tilt_deg, n))
    out = []
    for a in angles:
        R = base.copy()
        R[:3, :3] = base[:3, :3] @ rot_x(a)
        out.append(R)
    return out
