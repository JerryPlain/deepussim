"""Measured calibration transforms for the real DeepUSSim rig (doc v2 §4.2).

Two rigid links sitting at opposite ends of the US-pixel -> CBCT-voxel chain:

    p_C  =  C_T_R . R_T_E(t) . E_T_U . U_T_img . p_img         (doc §4.2)

where ``E_T_U`` (= ``T_EE_FROM_PROBE``) is the probe->end-effector mount and ``C_T_R`` is
bridged through the phantom. Both transforms below were ambiguous on delivery (the hand-eye
matrix could be read either way; the placement label turned out inverted) and have now been
**resolved by replay** — driving the probe along the real rosbag EE poses and selecting the
calibration that lands contact frames *on* the phantom surface and dark/non-contact frames
*off* it (see ``scripts/verify_replay.py``). The constants below already encode the resolved
directions, named for what they actually do; the raw measured matrices are the
``*_FROM_*`` constants and their inverses are derived.

Evidence (2026-05-31): under this calibration, contact frames land ~1-3 cm from the surface
and dark frames 13-21 cm off, on both real sequences; every alternative direction misses by
0.17-0.76 m. The residual cm-scale offset (probe-mesh origin vs the true transducer face,
plus a possible phantom shift between sequences) is what LC2 then grinds to millimetres.

Frame/units: translations in metres, rotations as plain 3x3. Convention matches
:mod:`deepussim.geometry`: ``T_a_from_b`` maps a point from frame b into frame a.
"""
from __future__ import annotations

import numpy as np

from ..geometry import invert

# Hand-eye, as delivered: maps end-effector-frame points into the probe frame, i.e. it IS
# T_probe_from_ee (the doc's U_T_E). Verified orthonormal, det +1, exactly
# Rz(45 deg) + (0, 0, -0.183 m) — the 45 deg in-plane offset and the 0.183 m standoff.
T_PROBE_FROM_EE = np.array([
    [0.7071068, -0.7071068, 0.0,  0.000],
    [0.7071068,  0.7071068, 0.0,  0.000],
    [0.0,        0.0,        1.0, -0.183],
    [0.0,        0.0,        0.0,  1.000],
])

# The probe mount the chain needs (E_T_U) and what SceneConfig.probe_offset expects.
T_EE_FROM_PROBE = invert(T_PROBE_FROM_EE)

# Robot<->phantom placement, as delivered: applying it to CBCT/phantom-frame points yields
# robot/world-frame points, i.e. it IS T_robot_from_phantom (== T_world_from_cbct, with
# world == robot base and the phantom frame == the CBCT frame). ~180 deg about z,
# t = (0.546, 0.393, 0.314) m.
T_ROBOT_FROM_PHANTOM = np.array([
    [-0.9989, -0.0158, -0.0452, 0.5460],
    [ 0.0162, -0.9998, -0.0091, 0.3930],
    [-0.0451, -0.0098,  0.9989, 0.3135],
    [ 0.0,     0.0,     0.0,    1.0000],
])

T_PHANTOM_FROM_ROBOT = invert(T_ROBOT_FROM_PHANTOM)

# Placement used when bridging the sim/robot world to the CBCT volume for reslicing.
#
# NB: this matrix is expressed in the CBCT *scan/optical* frame {c} (the LoopX frame Feng's
# calibration lives in), which is rolled 90 deg from the DICOM-LPS frame of our exported
# ``intensity.nrrd``. Applied to the DICOM volume as-is it stands the phantom on end, so the
# imaging fan grazes the surface tangentially (a ~90 deg error LC2 cannot grind). The real rig
# has the phantom LYING: an extra Rx(90 deg) about the phantom centre recovers it — validated by
# replaying the real EE poses into the volume (the fan then images *into* the tissue and LC2
# climbs on every contact frame). That roll + a contact-seat is applied by
# :func:`calib.seat_phantom_placement`; use it (not this raw matrix) to bridge poses into the
# DICOM volume. scripts/view_sim.py, run_scaleup.py and run_lc2.py all go through it.
T_WORLD_FROM_CBCT = T_ROBOT_FROM_PHANTOM
