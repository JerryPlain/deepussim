import numpy as np

from deepussim import geometry as g


def test_quat_mat_roundtrip():
    for angles in [(0.1, 0.2, 0.3), (1.2, -0.5, 2.0), (-2.5, 0.7, -1.3)]:
        R = g.euler_xyz(*angles)
        q = g.mat_to_quat(R)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(g.quat_to_mat(q), R, atol=1e-9)


def test_identity_quat():
    q = g.mat_to_quat(np.eye(3))
    # identity rotation -> (w, x, y, z) = (1, 0, 0, 0) up to sign
    assert np.allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_quat_to_mat_is_rotation():
    q = g.mat_to_quat(g.euler_xyz(0.4, -1.1, 0.9))
    assert g.is_rigid(g.make_transform(g.quat_to_mat(q), [0, 0, 0]))


def test_pose_from_pos_quat():
    R = g.rot_z(0.6)
    q = g.mat_to_quat(R)
    T = g.pose_from_pos_quat([1.0, 2.0, 3.0], q)
    assert np.allclose(T[:3, :3], R, atol=1e-9)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
