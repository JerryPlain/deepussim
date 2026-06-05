"""The scale-up write gate: a pose is kept only if the probe both *contacted* and
*reached* its target. Contact alone is a false positive — an unreachable pose where IK
stalls far off but an arm link grazes the phantom still registers force, and reslicing
the (far-off) achieved pose would write an off-anatomy sample.
"""
import numpy as np
import pytest

from deepussim import geometry as g
from deepussim.data.volume import Volume
from deepussim.us.reslice import ProbeGeometry
from deepussim.calib.placement import meters_to_mm
from deepussim.pipeline.scaleup import pose_convergence, generate_dataset


# --- pose_convergence: error decomposed in the target frame ---------------------------

def test_convergence_zero_for_identical_pose():
    T = g.make_transform(g.rot_z(0.3), [0.5, 0.1, 0.2])
    lateral, axial, rot = pose_convergence(T, T)
    assert lateral == 0.0 and axial == 0.0 and rot == 0.0


def test_pure_axial_press_is_axial_not_lateral():
    # Push 20 mm along the target's +z (the intended press) -> axial only.
    T = g.from_translation([0.5, 0.0, 0.3])  # axial = world +z
    achieved = T.copy()
    achieved[:3, 3] = T[:3, 3] + np.array([0.0, 0.0, 0.020])
    lateral, axial, rot = pose_convergence(T, achieved)
    assert lateral < 1e-6
    assert axial == pytest.approx(20.0)  # mm
    assert rot < 1e-6


def test_lateral_miss_is_lateral_not_axial():
    T = g.from_translation([0.5, 0.0, 0.3])  # axial = world +z
    achieved = T.copy()
    achieved[:3, 3] = T[:3, 3] + np.array([0.015, 0.0, 0.0])  # 15 mm sideways
    lateral, axial, rot = pose_convergence(T, achieved)
    assert lateral == pytest.approx(15.0)
    assert axial < 1e-6


def test_orientation_error_reported_in_degrees():
    T = g.from_translation([0.5, 0.0, 0.3])
    achieved = g.compose(T, g.from_translation([0, 0, 0]))
    achieved[:3, :3] = g.rot_z(np.radians(12.0))[:3, :3]
    _, _, rot = pose_convergence(T, achieved)
    assert rot == pytest.approx(12.0)


def test_axial_decomposition_follows_a_tilted_target_axis():
    # If the target axial is tilted, the same world displacement splits differently:
    # a displacement *along* that tilted axis must read as axial, not lateral.
    T = g.make_transform(g.rot_x(np.radians(30.0)), [0.5, 0.0, 0.3])
    axis = T[:3, 2]
    achieved = T.copy()
    achieved[:3, 3] = T[:3, 3] + 0.018 * axis  # 18 mm along the (tilted) axial
    lateral, axial, rot = pose_convergence(T, achieved)
    assert lateral < 1e-6
    assert axial == pytest.approx(18.0)


# --- the gate inside generate_dataset, exercised with a fake scene --------------------

class _FakeScene:
    """Minimal stand-in for UltrasoundScene: returns a preset achieved pose + contact."""

    def __init__(self, achieved, contacted=True, force=(0.0, 0.0, 5.0)):
        self._achieved = np.asarray(achieved, dtype=float)
        self._contacted = contacted
        self._force = np.asarray(force, dtype=float)

    def servo_to_force(self, T_nom, target_n=5.0):
        return float(np.linalg.norm(self._force))

    def in_contact(self, threshold_n=0.1):
        return self._contacted

    def probe_pose(self):
        return self._achieved

    def contact_force(self):
        return self._force


def _toy_inputs():
    vol = Volume(np.random.default_rng(0).random((40, 40, 40)), np.eye(4))
    geom = ProbeGeometry(radius_mm=20.0, fov_deg=60.0, depth_mm=15.0, n_lat=16, n_ax=24)
    sim_to_cbct = meters_to_mm(g.identity())  # sim metres -> CBCT mm, no rotation/offset
    nominal = g.from_translation([0.020, 0.020, 0.020])  # target probe pose (m)
    return vol, geom, sim_to_cbct, nominal


def test_reached_and_contacting_pose_is_written(tmp_path):
    vol, geom, sim_to_cbct, nominal = _toy_inputs()
    scene = _FakeScene(achieved=nominal, contacted=True)  # landed exactly on target
    n = generate_dataset(tmp_path, vol, [nominal], geom, scene=scene,
                         sim_to_cbct=sim_to_cbct, force_target_n=5.0, progress=False)
    assert n == 1


def test_stalled_far_off_pose_is_dropped_despite_contact(tmp_path):
    vol, geom, sim_to_cbct, nominal = _toy_inputs()
    far = nominal.copy()
    far[:3, 3] += np.array([0.20, 0.0, 0.0])  # IK stalled 200 mm sideways
    scene = _FakeScene(achieved=far, contacted=True)  # a link grazes -> "contact"
    n = generate_dataset(tmp_path, vol, [nominal], geom, scene=scene,
                         sim_to_cbct=sim_to_cbct, force_target_n=5.0, progress=False)
    assert n == 0  # contact alone must not pass the gate


def test_non_contacting_pose_is_dropped(tmp_path):
    vol, geom, sim_to_cbct, nominal = _toy_inputs()
    scene = _FakeScene(achieved=nominal, contacted=False)
    n = generate_dataset(tmp_path, vol, [nominal], geom, scene=scene,
                         sim_to_cbct=sim_to_cbct, force_target_n=5.0, progress=False)
    assert n == 0


def test_axial_press_within_tolerance_is_kept(tmp_path):
    # A genuine 25 mm press along the axial is within reach_axial_mm (40) -> kept.
    vol, geom, sim_to_cbct, nominal = _toy_inputs()
    pressed = nominal.copy()
    pressed[:3, 3] += 0.025 * nominal[:3, 2]
    scene = _FakeScene(achieved=pressed, contacted=True)
    n = generate_dataset(tmp_path, vol, [nominal], geom, scene=scene,
                         sim_to_cbct=sim_to_cbct, force_target_n=5.0, progress=False)
    assert n == 1
