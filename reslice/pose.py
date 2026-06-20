"""Step 2 (geometry half): turn a recorded probe pose into an image plane in the
phantom-centred CBCT frame.

Pose chain (read right-to-left):

    T_world_from_probe = T_world_from_ee(t) @ T_EE_FROM_PROBE @ Rz(probe_offset)
    T_phantom_from_probe = inv(T_world_from_phantom) @ T_world_from_probe   (then m -> mm)

The ultrasound image plane is the probe's local X-Z plane:

* plane point  = probe origin;
* plane normal = probe local +Y (elevation);
* image lateral = probe local X, image axial (depth) = probe local Z.

The two calibrated links below match ``deepussim.calib.transforms`` exactly; they are
duplicated here so the package stays dependency-free.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from reslice._geometry import (
    compose,
    invert_rigid,
    make_transform,
    normalize,
    rot_x,
    rot_z,
)

# Hand-eye: probe frame -> robot end-effector/flange frame (Rz(-45 deg) + 0.183 m standoff).
T_EE_FROM_PROBE = np.array([
    [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
    [-0.7071067811865475, 0.7071067811865475, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.183],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

# Measured robot<-phantom placement (metres), before the belly-up lie-down roll.
T_WORLD_FROM_PHANTOM_MEASURED_M = np.array([
    [-0.9993213267731319, -0.004748964835081844, -0.036528525694109644, 0.5618980642296896],
    [0.004894960353026591, -0.9999803818526938, -0.0039083593598543695, 0.3711799666308714],
    [-0.03650924841094883, -0.004084512546023032, 0.9993249679347199, 0.3154777116310868],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

DEFAULT_PROBE_ROTATION_OFFSET_DEG = 0.0


def probe_rotation_offset(degrees: float) -> np.ndarray:
    """A pure rotation about the probe's local Z (spins the imaging plane azimuth)."""
    return make_transform(rot_z(math.radians(degrees)), [0.0, 0.0, 0.0])


def default_world_from_phantom_centered_m() -> np.ndarray:
    """Placement of the phantom-centred mesh in the robot world, in metres.

    The belly-up lie-down ``Rx(90) @ Rz(180)`` is applied so the probe presses straight
    down onto the up-facing anterior surface (matches the real rig).
    """
    lie = make_transform(rot_x(math.pi / 2.0) @ rot_z(math.pi), [0.0, 0.0, 0.0])
    return T_WORLD_FROM_PHANTOM_MEASURED_M @ lie


def pose_from_sequence(
    sequence: Path,
    frame: int,
    pose_kind: str,
    probe_rotation_deg: float,
) -> np.ndarray:
    """Read ``T_world_from_probe`` (metres) from a recorded sequence frame.

    ``pose_kind`` is ``"ee"`` (the npz stores end-effector poses; apply the hand-eye)
    or ``"probe"`` (the npz already stores probe poses).
    """
    seq = np.load(sequence, allow_pickle=True)
    poses = np.asarray(seq["poses"], dtype=float)
    if frame < 0 or frame >= len(poses):
        raise IndexError(f"frame {frame} out of range 0..{len(poses) - 1}")

    T_world_from_ee = poses[frame]
    spin = probe_rotation_offset(probe_rotation_deg)
    if pose_kind == "ee":
        return T_world_from_ee @ T_EE_FROM_PROBE @ spin
    if pose_kind == "probe":
        return T_world_from_ee @ spin
    raise ValueError("pose_kind must be 'ee' or 'probe'")


def meters_pose_to_mm(T_m: np.ndarray) -> np.ndarray:
    """Re-express a rigid pose's translation in mm; rotation unchanged."""
    out = np.asarray(T_m, dtype=float).copy()
    out[:3, 3] *= 1000.0
    return out


def probe_pose_in_phantom_centered_mm(
    T_world_from_probe_m: np.ndarray,
    T_world_from_phantom_m: np.ndarray,
) -> np.ndarray:
    """Map a world probe pose (m) into the phantom-centred CBCT frame (mm)."""
    T_phantom_from_world_m = invert_rigid(T_world_from_phantom_m)
    return meters_pose_to_mm(T_phantom_from_world_m @ T_world_from_probe_m)


def plane_from_probe_pose(
    T_phantom_from_probe_mm: np.ndarray,
    plane_mode: str = "probe-xz",
    plane_offset_y_mm: float = 0.0,
) -> dict[str, np.ndarray]:
    """Derive the image plane (point + unit axes) from the probe pose in phantom mm.

    ``plane_mode``:
      * ``"probe-xz"`` — image plane is probe X-Z, normal is probe +Y (the US convention);
      * ``"probe-xy"`` — image plane is probe X-Y, normal is probe +Z.
    ``plane_offset_y_mm`` shifts the plane point along probe +Y before sampling.
    """
    R = T_phantom_from_probe_mm[:3, :3]
    point = T_phantom_from_probe_mm[:3, 3] + float(plane_offset_y_mm) * normalize(R[:, 1])
    lateral = normalize(R[:, 0])
    if plane_mode == "probe-xz":
        normal = normalize(R[:, 1])
        axial = normalize(R[:, 2])
        meaning = "plane is probe local X-Z; normal is probe local +Y"
    elif plane_mode == "probe-xy":
        normal = normalize(R[:, 2])
        axial = normalize(R[:, 1])
        meaning = "plane is probe local X-Y; normal is probe local +Z"
    else:
        raise ValueError("plane_mode must be 'probe-xz' or 'probe-xy'")
    return {
        "point_centered_mm": point,
        "normal_centered": normal,
        "lateral_centered": lateral,
        "axial_centered": axial,
        "meaning": meaning,
        "plane_offset_y_mm": float(plane_offset_y_mm),
    }
