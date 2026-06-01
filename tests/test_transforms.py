"""Sanity checks on the measured calibration transforms (doc v2 §4.2)."""
import numpy as np

from deepussim.calib.transforms import (
    T_PROBE_FROM_EE,
    T_EE_FROM_PROBE,
    T_PHANTOM_FROM_ROBOT,
    T_ROBOT_FROM_PHANTOM,
    T_WORLD_FROM_CBCT,
)
from deepussim.geometry import is_rigid, rot_z, invert


def test_measured_transforms_are_rigid():
    assert is_rigid(T_PROBE_FROM_EE, tol=1e-3)
    assert is_rigid(T_ROBOT_FROM_PHANTOM, tol=1e-3)


def test_handeye_is_rz45_and_standoff():
    assert np.allclose(T_PROBE_FROM_EE[:3, :3], rot_z(np.deg2rad(45.0)), atol=1e-4)
    assert np.allclose(T_PROBE_FROM_EE[:3, 3], [0.0, 0.0, -0.183])


def test_handeye_mount_is_the_inverse():
    # The chain's probe mount (E_T_U) is the inverse of the delivered hand-eye matrix.
    assert np.allclose(T_EE_FROM_PROBE @ T_PROBE_FROM_EE, np.eye(4), atol=1e-6)
    assert np.allclose(T_EE_FROM_PROBE, invert(T_PROBE_FROM_EE))


def test_phantom_robot_inverse_consistency():
    # The placement is given to 4 decimals, so its rotation is only ~1e-3 orthonormal;
    # invert() uses R^T, so the round-trip recovers identity to the measurement precision.
    assert np.allclose(T_ROBOT_FROM_PHANTOM @ T_PHANTOM_FROM_ROBOT, np.eye(4), atol=5e-3)


def test_world_from_cbct_is_the_placement():
    # Placement used to bridge robot/world <-> CBCT is the measured robot<-phantom matrix.
    assert np.allclose(T_WORLD_FROM_CBCT, T_ROBOT_FROM_PHANTOM)
