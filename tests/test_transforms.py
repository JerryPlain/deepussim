"""Sanity checks on the measured calibration transforms (doc v2 §4.2)."""
import numpy as np

from deepussim.calib.transforms import (
    ETU,
    T_PHANTOM_FROM_ROBOT,
    T_ROBOT_FROM_PHANTOM,
    probe_offset,
)
from deepussim.geometry import is_rigid, rot_z


def test_measured_transforms_are_rigid():
    assert is_rigid(ETU, tol=1e-3)
    assert is_rigid(T_PHANTOM_FROM_ROBOT, tol=1e-3)


def test_etu_is_rz45_and_standoff():
    assert np.allclose(ETU[:3, :3], rot_z(np.deg2rad(45.0)), atol=1e-4)
    assert np.allclose(ETU[:3, 3], [0.0, 0.0, -0.183])


def test_phantom_robot_inverse_consistency():
    # PhTR is given to 4 decimals, so its rotation is only ~1e-3 orthonormal; invert()
    # uses R^T, so the round-trip recovers identity to the measurement precision, not exactly.
    assert np.allclose(T_ROBOT_FROM_PHANTOM @ T_PHANTOM_FROM_ROBOT, np.eye(4), atol=5e-3)


def test_probe_offset_hypotheses_are_inverses():
    assert np.allclose(probe_offset("H1"), ETU)
    assert np.allclose(probe_offset("H2") @ probe_offset("H1"), np.eye(4), atol=1e-6)


def test_probe_offset_rejects_unknown_hypothesis():
    try:
        probe_offset("H3")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown hypothesis")
