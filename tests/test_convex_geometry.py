"""Geometry checks for the curvilinear (convex) probe model."""
import numpy as np

from deepussim.us.reslice import ProbeGeometry


def test_plane_grid_shape():
    g = ProbeGeometry(radius_mm=60, fov_deg=60, depth_mm=120, n_lat=64, n_ax=128)
    grid = g.plane_grid()
    assert grid.shape == (4, 128 * 64)
    assert np.allclose(grid[3], 1.0)  # homogeneous


def test_centre_line_is_straight_z_from_origin():
    g = ProbeGeometry(radius_mm=60, fov_deg=60, depth_mm=120, n_lat=65, n_ax=100)
    grid = g.plane_grid().reshape(4, g.n_ax, g.n_lat)
    centre = grid[:, :, g.n_lat // 2]  # the theta=0 scan line
    assert np.allclose(centre[0], 0.0, atol=1e-9)            # x == 0 along the centre beam
    assert np.allclose(centre[1], 0.0)                       # y == 0 (thin slice)
    assert np.isclose(centre[2, 0], 0.0, atol=1e-9)          # starts at the face (origin)
    assert np.isclose(centre[2, -1], g.depth_mm, atol=1e-6)  # ends at depth along the beam


def test_fan_widens_and_face_curves_back():
    g = ProbeGeometry(radius_mm=60, fov_deg=60, depth_mm=120, n_lat=64, n_ax=128)
    grid = g.plane_grid().reshape(4, g.n_ax, g.n_lat)
    near = grid[:, 0, :]   # face arc (depth 0)
    far = grid[:, -1, :]   # deepest arc
    # the fan is wider deeper than at the face
    assert np.ptp(far[0]) > np.ptp(near[0])
    # convex face: edge scan lines start *behind* the origin plane (z < 0)
    assert near[2].min() < -1e-6
    # symmetric about x=0
    assert np.isclose(near[0].min(), -near[0].max(), atol=1e-6)
