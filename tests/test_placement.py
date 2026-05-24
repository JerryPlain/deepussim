import numpy as np

from deepussim import geometry as g
from deepussim.data.volume import Volume
from deepussim.calib.placement import (
    meters_to_mm,
    align_points_placement,
    align_centers_placement,
    sim_pose_to_cbct,
)


def test_meters_to_mm_scales_translation_only():
    T = g.make_transform(g.rot_z(0.7), [0.1, -0.2, 0.05])  # metres
    Tmm = meters_to_mm(T)
    assert np.allclose(Tmm[:3, 3], [100.0, -200.0, 50.0])  # mm
    assert np.allclose(Tmm[:3, :3], T[:3, :3])             # rotation unchanged


def test_align_points_maps_anchor_to_anchor():
    sim_anchor_m = [0.45, 0.0, 0.08]
    cbct_anchor_mm = [47.5, 47.5, 95.0]
    placement = align_points_placement(sim_anchor_m, cbct_anchor_mm)
    # A probe sitting exactly at the sim anchor must land on the CBCT anchor.
    pose_at_anchor = g.from_translation(sim_anchor_m)  # metres
    mapped = sim_pose_to_cbct(pose_at_anchor, placement)
    assert np.allclose(mapped[:3, 3], cbct_anchor_mm, atol=1e-6)


def test_align_centers_maps_phantom_centre_to_volume_centre():
    vol = Volume(np.zeros((96, 96, 96)), np.eye(4))  # 1mm spacing, identity affine
    placement = align_centers_placement([0.45, 0.0, 0.04], vol)
    mapped = sim_pose_to_cbct(g.from_translation([0.45, 0.0, 0.04]), placement)
    assert np.allclose(mapped[:3, 3], vol.center_world(), atol=1e-6)


def test_rotation_carries_through_units_and_placement():
    R = g.euler_xyz(0.3, -0.4, 1.0)
    pose_m = g.make_transform(R, [0.45, 0.01, 0.08])
    placement = align_points_placement([0.45, 0.0, 0.08], [47.5, 47.5, 95.0])
    mapped = sim_pose_to_cbct(pose_m, placement)
    # Identity placement rotation -> mapped rotation equals the probe's rotation.
    assert np.allclose(mapped[:3, :3], R, atol=1e-9)
    # Translation: (probe_m * 1000) shifted by the placement translation.
    expected = np.array([450.0, 10.0, 80.0]) + (np.array([47.5, 47.5, 95.0]) - np.array([450.0, 0.0, 80.0]))
    assert np.allclose(mapped[:3, 3], expected, atol=1e-6)
