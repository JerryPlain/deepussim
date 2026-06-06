"""Tests for probe-pose samplers, incl. the surface-constrained trajectory."""
import numpy as np
import pytest

from deepussim.pipeline.sampling import (
    linear_sweep, surface_sweep, surface_raster, contact_raster_ee,
)
from deepussim.geometry import is_rigid, make_transform


def test_linear_sweep_endpoints_and_orientation():
    poses = linear_sweep([0, 0, 10], [20, 0, 10], n=5, axial_dir=(0, 0, -1))
    assert len(poses) == 5
    assert np.allclose(poses[0][:3, 3], [0, 0, 10])
    assert np.allclose(poses[-1][:3, 3], [20, 0, 10])
    assert np.allclose(poses[0][:3, 2], [0, 0, -1])  # +z axial along axial_dir


def test_surface_sweep_sits_on_sphere_and_aims_inward():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    R = 50.0
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)  # centred at origin
    standoff = 2.0
    # Guide line outside the sphere; poses should land on it, axial pointing at the centre.
    poses = surface_sweep(mesh, [-30, 0, 70], [30, 0, 70], n=8, standoff_mm=standoff,
                          smooth_radius_mm=12.0)
    assert len(poses) == 8
    for T in poses:
        assert is_rigid(T, tol=1e-6)
        pos = T[:3, 3]
        # rests on the surface (sphere radius + standoff), within tolerance
        assert R - 1.0 < np.linalg.norm(pos) < R + standoff + 2.0
        # axial (+z) points inward = toward the centre (-pos direction)
        axial = T[:3, 2]
        assert axial @ (-pos / np.linalg.norm(pos)) > 0.9


def test_surface_sweep_is_ordered_along_the_sweep():
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    poses = surface_sweep(mesh, [-30, 0, 70], [30, 0, 70], n=6)
    xs = [T[0, 3] for T in poses]  # x increases monotonically along the sweep
    assert all(b >= a - 1e-6 for a, b in zip(xs, xs[1:]))


def test_surface_raster_covers_a_2d_patch():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    R, standoff = 50.0, 2.0
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)
    poses = surface_raster(mesh, axis=0, span_frac=0.5, cross_frac=0.5,
                           n_lines=4, n_per_line=5, standoff_mm=standoff)
    assert len(poses) == 4 * 5
    pos = np.array([T[:3, 3] for T in poses])
    # coverage spreads over BOTH in-plane axes (not a single line)
    assert np.ptp(pos[:, 0]) > 10.0 and np.ptp(pos[:, 1]) > 10.0
    for T in poses:
        p = T[:3, 3]
        assert R - 1.0 < np.linalg.norm(p) < R + standoff + 2.0   # on the sphere + standoff
        assert T[:3, 2] @ (-p / np.linalg.norm(p)) > 0.9          # axial points inward


def test_contact_raster_ee_orientation_follows_real_poses_not_mesh_normal():
    """The real probe presses straight DOWN; generated poses must inherit that orientation,
    NOT the local surface normal (which on a sphere points radially)."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    R, standoff = 50.0, 2.0
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)
    # Real contact poses near the +z cap, ALL aiming straight down (axial = -z) — the rig's
    # fixed downward press. Their positions form a small patch on the cap; their orientation
    # deliberately does NOT match the (radial) sphere normal except exactly at the pole.
    z = np.array([0.0, 0.0, -1.0]); x = np.array([1.0, 0.0, 0.0]); y = np.cross(z, x)
    down = make_transform(np.column_stack([x, y, z]), [0, 0, 0])   # right-handed, axial = -z
    # Patch on the +x SIDE of the sphere: there the surface normal points ~+x, but the real
    # poses still aim straight DOWN — so axial vs mesh-normal are unmistakably different.
    ys = np.linspace(-12, 12, 6); zs = np.linspace(-8, 8, 4)
    contact_poses = []
    for yy in ys:
        for zz in zs:
            xx = np.sqrt(max(R**2 - yy**2 - zz**2, 0.0))   # on the sphere, +x side
            T = down.copy(); T[:3, 3] = [xx, yy, zz]
            contact_poses.append(T)
    contact_poses = np.array(contact_poses)

    poses = contact_raster_ee(mesh, contact_poses, n_lines=4, n_per_line=5, standoff_mm=standoff)
    assert len(poses) == 4 * 5
    for T in poses:
        assert is_rigid(T, tol=1e-6)
        # axial follows the real downward press (-z), NOT the outward radial normal
        assert T[:3, 2] @ np.array([0, 0, -1]) > 0.99
        p = T[:3, 3]
        radial = p / np.linalg.norm(p)
        assert abs(T[:3, 2] @ radial) < 0.9          # off-pole: axial != surface normal
        assert np.linalg.norm(p) < R + standoff + 3.0  # rests near/on the surface
