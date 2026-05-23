import numpy as np

from deepussim import geometry as g


def test_invert_roundtrip():
    T = g.make_transform(g.euler_xyz(0.3, -0.7, 1.1), [10.0, -5.0, 2.0])
    assert np.allclose(g.compose(T, g.invert(T)), np.eye(4), atol=1e-9)


def test_compose_matches_matmul():
    A = g.make_transform(g.rot_z(0.5), [1, 2, 3])
    B = g.make_transform(g.rot_x(-0.2), [0, 1, 0])
    assert np.allclose(g.compose(A, B), A @ B)


def test_apply_points_shape_and_value():
    T = g.from_translation([1.0, 2.0, 3.0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = g.apply(T, pts)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [1, 2, 3])
    # single point keeps 1D shape
    assert g.apply(T, [0, 0, 0]).shape == (3,)


def test_rotations_are_rigid():
    assert g.is_rigid(g.make_transform(g.euler_xyz(0.1, 0.2, 0.3), [0, 0, 0]))
    assert not g.is_rigid(g.make_transform(2.0 * np.eye(3), [0, 0, 0]))
