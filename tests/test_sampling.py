"""Tests for probe-pose samplers, incl. the surface-constrained trajectory."""
import numpy as np
import pytest

from deepussim.pipeline.sampling import linear_sweep, surface_sweep
from deepussim.geometry import is_rigid


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
