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

from ..geometry import make_transform, compose, rot_x, rot_z
from ..data.volume import Volume

M_TO_MM = 1000.0


def meters_to_mm(T_pose_m: np.ndarray) -> np.ndarray:
    """Re-express a rigid pose's translation in mm; rotation is unchanged."""
    T = np.asarray(T_pose_m, dtype=float).copy()
    T[:3, 3] *= M_TO_MM
    return T


def mm_to_meters(T_pose_mm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`meters_to_mm`: re-express a pose's translation in metres."""
    T = np.asarray(T_pose_mm, dtype=float).copy()
    T[:3, 3] /= M_TO_MM
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


def seat_phantom_placement(mesh, contact_origins_m, base, *, lie_down=True) -> np.ndarray:
    """``T_world_from_cbctm`` (rigid, m): seat the phantom into the robot world for reslicing.

    The measured robot<->CBCT matrix (``base`` = ``T_WORLD_FROM_CBCT``) is expressed in the CBCT
    *optical/scan* frame {c}, rolled from the DICOM-LPS frame of our exported ``intensity.nrrd``:
    applied as-is it stands the phantom on end, so the imaging fan grazes the surface tangentially.
    The real rig has the phantom LYING **belly-up** — the probe presses straight DOWN onto the
    up-facing anterior surface (verified: the real EE axial is world -z) and images into the body.
    The lie-down that reproduces this is ``Rx(90 deg) . Rz(180 deg)`` about the phantom centre:
    ``Rx(90)`` tips it off-end and ``Rz(180)`` flips it belly-up. (Validated in sim: 14/14 real
    poses reachable pressing from above, fan 82% inside tissue, vs the earlier belly-down
    ``Rx(90)`` alone which imaged out of the body at ~10% — see CHANGELOG 2026-06-04. LC2 is an
    unreliable arbiter here: the low-texture body makes it *prefer* the wrong belly-down graze, so
    the physical prior + in-tissue geometry decide the orientation, not LC2.) We bake that roll
    into one rigid transform from the CBCT-metre frame (= DICOM mm / 1000) to the robot world, then
    seat the surface onto the real contact cloud (a residual ~cm offset remains for LC2 to grind).

    ``mesh`` is the phantom surface (trimesh, CBCT mm); ``contact_origins_m`` are the probe-face
    positions in the robot/world frame (m) for the in-contact frames. Reslicing uses the SAME
    transform (``sim_to_cbct = meters_to_mm(invert(result))``), so mesh, volume and probe poses
    stay consistent.
    """
    import trimesh

    base = np.asarray(base, dtype=float)
    origins = np.asarray(contact_origins_m, dtype=float)
    c_m = np.asarray(mesh.bounds, dtype=float).mean(0) / M_TO_MM       # phantom centre (m)
    lie_R = rot_x(np.pi / 2.0) @ rot_z(np.pi)                         # tip off-end, then belly-up
    lie = make_transform(lie_R, [0.0, 0.0, 0.0]) if lie_down else np.eye(4)
    T = compose(base, lie, make_transform(np.eye(3), -c_m))           # lie-down about the centre
    placed = mesh.copy()
    placed.apply_transform(np.diag([1.0 / M_TO_MM] * 3 + [1.0]))      # mm mesh -> metres
    placed.apply_transform(T)
    close, _, _ = trimesh.proximity.ProximityQuery(placed).on_surface(origins)
    t_align = np.median(origins - close, axis=0)                      # seat surface onto contacts
    return compose(make_transform(np.eye(3), t_align), T)
