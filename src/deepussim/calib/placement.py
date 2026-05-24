"""Bridge sim-world poses (metres) into the CBCT frame (millimetres) for reslicing.

``scene.probe_pose()`` returns ``T_simworld_from_probe`` in **metres**, in the sim world
frame. ``reslice`` needs the probe pose in the **CBCT volume frame, in millimetres**.
Closing the sim -> reslice loop is therefore two things:

  1. units:     metres -> millimetres (scale the translation by 1000; rotation unchanged).
  2. placement: a rigid ``T_cbct_from_simworld`` (mm) fixing where the phantom sits in
                the CBCT frame relative to the sim world.

For synthetic/demo use, build the placement by mapping a chosen *sim anchor* point onto
a chosen *CBCT anchor* point (axes aligned, optional rotation). For real hardware this
is the Step-3 calibration: estimate it once from fiducials with
``calib.registration.rigid_register`` and pass the result in here.

Apply with :func:`sim_pose_to_cbct`.
"""
from __future__ import annotations

import numpy as np

from ..geometry import make_transform, compose
from ..data.volume import Volume

M_TO_MM = 1000.0


def meters_to_mm(T_pose_m: np.ndarray) -> np.ndarray:
    """Re-express a rigid pose's translation in mm; rotation is unchanged."""
    T = np.asarray(T_pose_m, dtype=float).copy()
    T[:3, 3] *= M_TO_MM
    return T


def align_points_placement(sim_anchor_m, cbct_anchor_mm, R_cbct_from_sim=None) -> np.ndarray:
    """``T_cbct_from_simworld`` (mm) mapping ``sim_anchor`` (m) onto ``cbct_anchor`` (mm).

    With identity rotation this is a pure translation. Pass a 3x3 ``R_cbct_from_sim`` to
    also reorient the sim axes into the CBCT frame.
    """
    R = np.eye(3) if R_cbct_from_sim is None else np.asarray(R_cbct_from_sim, dtype=float)
    sim_anchor_mm = np.asarray(sim_anchor_m, dtype=float) * M_TO_MM
    t = np.asarray(cbct_anchor_mm, dtype=float) - R @ sim_anchor_mm
    return make_transform(R, t)


def align_centers_placement(sim_phantom_pos_m, volume: Volume, R_cbct_from_sim=None) -> np.ndarray:
    """Convenience: map the sim phantom centre (m) onto the CBCT volume centre (mm)."""
    return align_points_placement(sim_phantom_pos_m, volume.center_world(), R_cbct_from_sim)


def sim_pose_to_cbct(T_simworld_from_probe_m, T_cbct_from_simworld_mm) -> np.ndarray:
    """Map a sim-world probe pose (m) into the CBCT frame (mm), ready for reslice."""
    return compose(np.asarray(T_cbct_from_simworld_mm, dtype=float),
                   meters_to_mm(T_simworld_from_probe_m))
