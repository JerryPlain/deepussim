"""Unit tests for the rosbag-extraction pure logic (no bag / no rosbags lib needed)."""
import numpy as np

from deepussim.data.rosbag import (
    nearest_indices,
    pose_matrix_from_ros,
    to_grayscale,
)
from deepussim.geometry import is_rigid, rot_z


def test_nearest_indices_picks_closest():
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.array([-0.5, 0.4, 0.6, 2.9, 100.0])
    idx = nearest_indices(q, ref)
    assert idx.tolist() == [0, 0, 1, 3, 3]


def test_nearest_indices_ties_go_left():
    # exactly halfway -> the (<=) rule keeps the earlier sample
    assert nearest_indices(np.array([0.5]), np.array([0.0, 1.0])).tolist() == [0]


def test_pose_matrix_ros_quat_reorder_is_rigid():
    # ROS xyzw for a +90 deg rotation about z; translation in metres.
    s = np.sqrt(0.5)
    T = pose_matrix_from_ros((0.5, 0.45, 0.45), (0.0, 0.0, s, s))
    assert is_rigid(T)
    assert np.allclose(T[:3, 3], [0.5, 0.45, 0.45])
    assert np.allclose(T[:3, :3], rot_z(np.pi / 2), atol=1e-6)


def test_pose_identity_quat():
    T = pose_matrix_from_ros((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_to_grayscale_passthrough_and_rgb():
    mono = np.full((4, 5), 7, dtype=np.uint8)
    assert np.array_equal(to_grayscale(mono), mono)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 1] = 200  # pure green -> 0.587 * 200
    assert np.allclose(to_grayscale(rgb), round(0.587 * 200))
