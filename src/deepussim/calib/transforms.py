"""Measured calibration transforms for the real DeepUSSim rig (doc v2 §4.2).

================================  NAMING  ====================================
Convention (matches :mod:`deepussim.geometry`):

    T_A_FROM_B  maps a point FROM frame B  INTO frame A:   p_A = T_A_FROM_B @ p_B

Read the name right-to-left: "A from B" consumes B, produces A (so the *arrow*
is B -> A). Adjacent names cancel: T_A_FROM_B @ T_B_FROM_C == T_A_FROM_C.
Every matrix here is its inverse's twin: T_B_FROM_A == invert(T_A_FROM_B).

================================  THE 5 FRAMES  ==============================
There are only FIVE frames; two of them carry multiple names (this is the usual
source of confusion), so the aliases are spelled out once here:

    1. image    US image plane (a pixel)
    2. probe    US transducer
    3. ee       robot end-effector / flange
    4. robot    robot base            == WORLD   (we set world := robot base)
    5. cbct     the phantom's CT scan == PHANTOM == CAMERA
                (the CT frame is glued to the phantom; the 2026-06-12 tf labels
                 this same frame "camera" — all three names are ONE frame.)

Kinematic tree (what is rigidly attached to what — `robot` is the common root):

    robot (=world) --[ T_robot_from_ee(t), per-frame from the rosbag ]--> ee --> probe --> image
        \--[ T_ROBOT_FROM_PHANTOM, the placement ]--> cbct (=phantom=camera)

Data flow (a US pixel -> the CBCT voxel it images; cbct is the DESTINATION, not
the start). Read the product right-to-left; the middle frames cancel:

    p_cbct = T_cbct_from_robot . T_robot_from_ee(t) . T_ee_from_probe . T_probe_from_image . p_image
             └ invert(T_ROBOT_FROM_PHANTOM) ┘                                          (doc §4.2)

================================  THE 2 CALIBRATION CONSTANTS  ===============
Everything else is either a per-frame rosbag pose (T_robot_from_ee) or the probe
geometry (T_probe_from_image). Only two rigid links are *measured/calibrated*:

    * HAND-EYE  (ee <-> probe)   : how the probe is mounted on the flange.
    * PLACEMENT (robot <-> cbct) : where the phantom sits in the robot workspace.

Both were ambiguous on delivery and **resolved by replay** — driving the probe along
the real rosbag EE poses and selecting the calibration that lands contact frames *on*
the phantom surface and dark frames *off* it (``scripts/verify_replay.py``). The raw
measured matrices are the ``*_FROM_*`` constants; their inverses are derived.

Evidence (2026-05-31): under this calibration, contact frames land ~1-3 cm from the
surface and dark frames 13-21 cm off, on both real sequences; every alternative
direction misses by 0.17-0.76 m. The residual cm-scale offset is what LC2 grinds to mm.

Frame/units: translations in METRES, rotations as plain 3x3.
"""
from __future__ import annotations

import numpy as np

from ..geometry import invert

# --- HAND-EYE link (ee <-> probe) — UNCHANGED ------------------------------------------------
# T_PROBE_FROM_EE: ee/flange -> probe (maps flange-frame points into the probe frame).
# Verified orthonormal, det +1, exactly Rz(45 deg) + (0, 0, -0.183 m): the 45 deg in-plane
# offset and the 0.183 m standoff of the transducer from the flange.
T_PROBE_FROM_EE = np.array([
    [0.7071068, -0.7071068, 0.0,  0.000],
    [0.7071068,  0.7071068, 0.0,  0.000],
    [0.0,        0.0,        1.0, -0.183],
    [0.0,        0.0,        0.0,  1.000],
])

# T_EE_FROM_PROBE: probe -> ee/flange. The direction the chain uses; also SceneConfig.probe_offset.
T_EE_FROM_PROBE = invert(T_PROBE_FROM_EE)

# --- PLACEMENT link (robot <-> cbct/phantom) — UPDATED 2026-06-12 -----------------------------
# T_ROBOT_FROM_PHANTOM: cbct/phantom -> robot/world (where the phantom sits in the robot's
# workspace). The 2026-06-12 tf publishes exactly this: parent=franka (robot base),
# child=camera (the cbct/phantom frame), i.e. the pose of the phantom in the robot frame.
# t = (0.5445, 0.3551, 0.3156) m; rotation ~unchanged from the old matrix (diag -0.999/-0.999/
# +0.999), y-translation shifted ~38 mm (phantom repositioned). PROVISIONAL: like the old one,
# must be re-resolved by replay (scripts/verify_replay.py) against the NEW sequences, and the
# belly-up roll in seat_phantom_placement re-checked against the new CBCT's DICOM orientation.
T_ROBOT_FROM_PHANTOM = np.array([
    [-0.9988805840248178, -0.029483337597067026, -0.03699069697069639, 0.5445375623840216],
    [ 0.029688116851141373, -0.9995467109497567, -0.004998834600006676, 0.3551386900786097],
    [-0.03682654716469477, -0.006091422958769326,  0.9993031071653204, 0.31561909434512403],
    [ 0.0,                  0.0,                    0.0,                 1.0],
])
# Old (2026-05-31) matrix, kept for fallback / replay comparison:
#   [-0.9989, -0.0158, -0.0452, 0.5460], [0.0162, -0.9998, -0.0091, 0.3930],
#   [-0.0451, -0.0098,  0.9989, 0.3135], [0.0, 0.0, 0.0, 1.0]

# T_PHANTOM_FROM_ROBOT: robot/world -> cbct/phantom. The chain's last hop (data flow: put a
# robot-frame point into the CBCT volume) uses this = invert(T_ROBOT_FROM_PHANTOM).
T_PHANTOM_FROM_ROBOT = invert(T_ROBOT_FROM_PHANTOM)

# T_WORLD_FROM_CBCT: alias of T_ROBOT_FROM_PHANTOM (world == robot base, cbct == phantom). Used
# when bridging the sim/robot world to the CBCT volume for reslicing.
#
# NB: this matrix is expressed in the CBCT *scan/optical* frame {c}, rolled from the DICOM-LPS
# frame of our exported ``intensity.nrrd``. Applied to the DICOM volume as-is it stands the
# phantom on end. The real rig has it LYING **belly-up** (the probe presses straight down onto
# the up-facing anterior surface — the real EE axial is world -z). The lie-down that reproduces
# this is ``Rx(90 deg) . Rz(180 deg)`` about the phantom centre (Rx tips it off-end, Rz flips it
# belly-up); validated in sim (14/14 real poses reachable from above, fan 82% inside tissue) vs
# the earlier belly-down ``Rx(90)`` alone that imaged out of the body. LC2 is NOT the arbiter
# here — the low-texture body makes it prefer the wrong belly-down graze. That roll + a
# contact-seat is applied by :func:`calib.seat_phantom_placement`; use it (not this raw matrix)
# to bridge poses into the DICOM volume. view_sim.py, run_scaleup.py and run_lc2.py go through it.
T_WORLD_FROM_CBCT = T_ROBOT_FROM_PHANTOM
