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


# --- surface-constrained trajectory (the real generator) ------------------
# The probe can only glide *along* the phantom surface, so poses are placed on the
# CBCT surface mesh: sample a point, estimate a smoothed normal, rest the probe on the
# surface with its axial (+z) axis pointing inward (perpendicular contact). The same
# pose stream then drives both the arm and the volume reslice (aligned by construction).


def _pose_on_surface(surface_pt, normal_out, sweep_dir, standoff_mm) -> np.ndarray:
    """Probe pose resting on the surface: +z axial inward, +x along the sweep direction.

    ``normal_out`` is the unit outward surface normal. The probe sits ``standoff_mm`` out
    along it; its axial axis points inward (``-normal_out``); the lateral (+x) axis follows
    ``sweep_dir`` projected into the imaging plane so the image width tracks the scan.
    """
    z = -np.asarray(normal_out, dtype=float)
    z = z / (np.linalg.norm(z) + 1e-12)
    d = np.asarray(sweep_dir, dtype=float)
    x = d - (d @ z) * z
    if np.linalg.norm(x) < 1e-6:  # sweep parallel to the normal: pick any in-plane axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - (ref @ z) * z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    pos = np.asarray(surface_pt, dtype=float) + standoff_mm * np.asarray(normal_out, dtype=float)
    return make_transform(R, pos)


def _smoothed_normal(tree, vertices, point, radius, n_fallback=30):
    """Outward-ish unit normal at ``point`` via PCA over neighbouring mesh vertices."""
    idx = tree.query_ball_point(point, radius)
    if len(idx) < 8:
        idx = tree.query(point, n_fallback)[1]
    Q = vertices[idx] - vertices[idx].mean(axis=0)
    _, vecs = np.linalg.eigh(Q.T @ Q)
    n = vecs[:, 0]  # smallest-variance direction = surface normal
    return n / (np.linalg.norm(n) + 1e-12)


def _pose_from_guide(tree, V, guide, sweep_dir, standoff_mm, smooth_radius_mm) -> np.ndarray:
    """One surface pose for a guide point: project to the surface, smooth normal, rest probe."""
    surface_pt = V[tree.query(guide)[1]]
    normal = _smoothed_normal(tree, V, surface_pt, smooth_radius_mm)
    if normal @ (guide - surface_pt) < 0:  # orient outward (toward the guide point)
        normal = -normal
    return _pose_on_surface(surface_pt, normal, sweep_dir, standoff_mm)


def _line_poses(tree, V, start, end, n, standoff_mm, smooth_radius_mm) -> list[np.ndarray]:
    sweep_dir = end - start
    return [_pose_from_guide(tree, V, start + t * sweep_dir, sweep_dir, standoff_mm,
                             smooth_radius_mm) for t in np.linspace(0.0, 1.0, n)]


def surface_sweep(mesh, start, end, n: int, standoff_mm: float = 2.0,
                  smooth_radius_mm: float = 8.0) -> list[np.ndarray]:
    """``n`` probe poses (``T_cbct_from_probe``, mm) gliding along the phantom surface.

    ``start``/``end`` are guide points in the CBCT mm frame placed *outside* the phantom on
    the approach side. Each is projected to the nearest surface point; a smoothed normal is
    estimated there (PCA over a ``smooth_radius_mm`` neighbourhood, oriented toward the guide
    point so it points outward) and a probe pose is built resting on the surface with the
    axial axis pointing inward. Returns the poses in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    return _line_poses(tree, V, np.asarray(start, float), np.asarray(end, float),
                       n, standoff_mm, smooth_radius_mm)


def surface_raster(mesh, axis: int = 0, span_frac: float = 0.6, cross_frac: float = 0.4,
                   n_lines: int = 5, n_per_line: int = 24, standoff_mm: float = 2.0,
                   smooth_radius_mm: float = 8.0, serpentine: bool = True,
                   clearance_mm: float = 20.0) -> list[np.ndarray]:
    """Cover a patch of the phantom's +z-top with ``n_lines`` parallel surface sweeps.

    Sweeps run along ``axis`` (0=x, 1=y) spanning ``span_frac`` of that extent, stepped across
    the other in-plane axis over ``cross_frac`` of its extent in ``n_lines`` lines of
    ``n_per_line`` poses each. With ``serpentine`` the lines alternate direction so the pose
    stream is a continuous lawnmower path. The KD-tree is built once and shared across lines.
    Returns ``n_lines * n_per_line`` poses (``T_cbct_from_probe``, mm) in scan order.
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices, dtype=float)
    tree = cKDTree(V)
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    centre = (lo + hi) / 2.0
    cross_axis = 1 - axis  # the other in-plane horizontal axis (0<->1); up is z (axis 2)
    half_along = (hi[axis] - lo[axis]) * span_frac / 2.0
    half_cross = (hi[cross_axis] - lo[cross_axis]) * cross_frac / 2.0
    z_top = hi[2] + clearance_mm
    cross_offsets = np.linspace(-half_cross, half_cross, n_lines) if n_lines > 1 else [0.0]

    poses: list[np.ndarray] = []
    for j, c in enumerate(cross_offsets):
        start = centre.copy(); end = centre.copy()
        start[2] = end[2] = z_top
        start[axis], end[axis] = centre[axis] - half_along, centre[axis] + half_along
        start[cross_axis] = end[cross_axis] = centre[cross_axis] + c
        line = _line_poses(tree, V, start, end, n_per_line, standoff_mm, smooth_radius_mm)
        if serpentine and j % 2 == 1:
            line = line[::-1]
        poses.extend(line)
    return poses


def top_sweep_endpoints(mesh, axis: int = 0, span_frac: float = 0.6,
                        clearance_mm: float = 20.0):
    """Convenience guide endpoints for a sweep across the mesh's +z-top along ``axis``.

    Returns ``(start, end)`` points held ``clearance_mm`` above the top of the bounding box,
    spanning ``span_frac`` of the extent along ``axis`` (0=x, 1=y), centred otherwise.
    """
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    centre = (lo + hi) / 2.0
    half = (hi[axis] - lo[axis]) * span_frac / 2.0
    z_top = hi[2] + clearance_mm
    start = centre.copy(); end = centre.copy()
    start[2] = end[2] = z_top
    start[axis] = centre[axis] - half
    end[axis] = centre[axis] + half
    return start, end
