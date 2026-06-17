"""Measured calibration transforms for the real DeepUSSim rig.

NAMING.  ``T_A_FROM_B`` maps a point from frame B into frame A:  ``p_A = T_A_FROM_B @ p_B``.
Read it right-to-left ("A from B" consumes B, produces A; the arrow is B -> A). Adjacent names
cancel (``T_A_FROM_B @ T_B_FROM_C == T_A_FROM_C``); inverses are twins
(``T_B_FROM_A == invert(T_A_FROM_B)``). Translations are in METRES.

FRAMES (only 5; some carry aliases — the usual source of confusion):

    image   US pixel plane
    probe   US transducer
    ee      robot end-effector / flange
    robot   robot base             == world
    cbct    the phantom's CT scan  == phantom == camera   (the CT frame is glued to the phantom)

A US pixel reaches the CBCT voxel it images along this chain (cbct is the DESTINATION;
read the product right-to-left):

    p_cbct = T_cbct_from_robot . T_robot_from_ee(t) . T_ee_from_probe . T_probe_from_image . p_image
             └ PLACEMENT (inv) ┘  └ rosbag, per-frame ┘  └ HAND-EYE ┘   └ probe geometry ┘

So only TWO links are calibrated; the rest are per-frame rosbag poses or probe geometry:

    HAND-EYE   ee <-> probe    how the probe is mounted on the flange
    PLACEMENT  robot <-> cbct  where the phantom sits in the robot workspace

Both were resolved by replay (``scripts/verify_replay.py``): the direction that lands contact
frames on the phantom surface and dark frames off it.
"""
from __future__ import annotations

import numpy as np

from ..geometry import invert

# === HAND-EYE  (ee <-> probe) — unchanged ====================================================
# ee/flange -> probe. Exactly Rz(45 deg) + a 0.183 m standoff of the transducer from the flange.
T_PROBE_FROM_EE = np.array([
    [0.7071068, -0.7071068, 0.0,  0.000],
    [0.7071068,  0.7071068, 0.0,  0.000],
    [0.0,        0.0,        1.0, -0.183],
    [0.0,        0.0,        0.0,  1.000],
])
T_EE_FROM_PROBE = invert(T_PROBE_FROM_EE)        # probe -> ee/flange (the direction the chain uses)

# === PLACEMENT  (robot <-> cbct) — updated 2026-06-12 ========================================
# cbct/phantom -> robot/world: the pose of the phantom in the robot workspace. Published as the
# tf parent=franka (robot base) -> child=camera (the cbct/phantom frame). PROVISIONAL until
# re-checked by replay on the new sequences (and the belly-up roll in seat_phantom_placement
# re-checked vs the new CBCT). Old 2026-05-31 value differed by 0.96 deg rot, ~38 mm in y.
T_ROBOT_FROM_PHANTOM = np.array([
    [-0.9988805840248178, -0.029483337597067026, -0.03699069697069639, 0.5445375623840216],
    [ 0.029688116851141373, -0.9995467109497567, -0.004998834600006676, 0.3551386900786097],
    [-0.03682654716469477, -0.006091422958769326,  0.9993031071653204, 0.31561909434512403],
    [ 0.0,                  0.0,                    0.0,                 1.0],
])
T_PHANTOM_FROM_ROBOT = invert(T_ROBOT_FROM_PHANTOM)   # robot/world -> cbct (the chain's last hop)
T_WORLD_FROM_CBCT    = T_ROBOT_FROM_PHANTOM           # alias (world == robot, cbct == phantom)

# NB: this matrix lives in the CBCT scan frame, rolled from the DICOM-LPS frame of intensity.nrrd,
# so do NOT apply it raw — calib.seat_phantom_placement adds the belly-up lie-down (Rx(90).Rz(180))
# + a contact-seat before bridging poses into the volume (used by run_scaleup / run_lc2 / view_sim).
