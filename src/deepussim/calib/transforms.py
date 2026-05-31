"""Measured calibration transforms for the real DeepUSSim rig (doc v2 §4.2).

Two rigid links delivered by the team, sitting at opposite ends of the
US-pixel -> CBCT-voxel chain:

    p_C  =  C_T_R . R_T_E(t) . ETU . U_T_img . p_img            (doc §4.2)

  ETU                  end-effector (FR3 link7 flange) <-> US probe center.
                       Rz(45 deg) + (0, 0, -0.183 m). The image-plane axes are rotated
                       45 deg from the flange x/y; the 0.183 m is the standoff/lever arm.
                       *** DIRECTION UNCONFIRMED (H1 vs H2) *** — Feng is unsure whether the
                       matrix maps probe->EE or EE->probe. Resolve by the replay task
                       (drive the probe mesh along the rosbag poses under both and see which
                       lands the probe ON the surface). Use :func:`probe_offset` to pick.

  T_PHANTOM_FROM_ROBOT robot base <-> phantom, i.e. the robot expressed in the phantom
                       frame ("PhTR"). ~180 deg about z, t = (0.546, 0.393, 0.314) m.
                       Direction confirmed. Bridges the arm world to the anatomy:
                       C_T_R = (phantom<->CBCT DICOM geometry) @ T_PHANTOM_FROM_ROBOT.

Frame/units: translations in metres, rotations as plain 3x3. Convention matches
:mod:`deepussim.geometry`: ``T_a_from_b`` maps a point from frame b into frame a.
"""
from __future__ import annotations

import numpy as np

from ..geometry import invert

# Feng Li's matrix — end-effector <-> probe center. Under hypothesis H1 this is read as
# T_ee_from_probe (probe-frame point -> EE frame); H2 reads it as the inverse. Verified
# orthonormal, det +1, exactly Rz(45 deg) + (0, 0, -0.183 m).
ETU = np.array([
    [0.7071068, -0.7071068, 0.0,  0.000],
    [0.7071068,  0.7071068, 0.0,  0.000],
    [0.0,        0.0,        1.0, -0.183],
    [0.0,        0.0,        0.0,  1.000],
])

# Feng哥's matrix — "the arm relative to the phantom": robot base expressed in the phantom
# frame (PhTR = T_phantom_from_robot). ~180 deg about z, t = (0.546, 0.393, 0.314) m.
T_PHANTOM_FROM_ROBOT = np.array([
    [-0.9989, -0.0158, -0.0452, 0.5460],
    [ 0.0162, -0.9998, -0.0091, 0.3930],
    [-0.0451, -0.0098,  0.9989, 0.3135],
    [ 0.0,     0.0,     0.0,    1.0000],
])

# Convenience: phantom origin expressed in the robot base frame.
T_ROBOT_FROM_PHANTOM = invert(T_PHANTOM_FROM_ROBOT)


def probe_offset(hypothesis: str = "H1") -> np.ndarray:
    """``T_ee_from_probe`` for :attr:`SceneConfig.probe_offset`, per the ETU hypothesis.

    H1: ETU already maps probe-frame -> EE (= T_ee_from_probe)  -> use ETU directly.
    H2: ETU maps EE -> probe (= T_probe_from_ee)                -> use invert(ETU).

    The replay task disambiguates these; until then both are one call apart.
    """
    h = hypothesis.upper()
    if h == "H1":
        return ETU.copy()
    if h == "H2":
        return invert(ETU)
    raise ValueError(f"hypothesis must be 'H1' or 'H2', got {hypothesis!r}")
