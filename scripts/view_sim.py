#!/usr/bin/env python
"""Open the Genesis viewer and watch the Franka scan/press the phantom (NOT headless).

This is the visual counterpart to scripts/smoke_sim.py: it opens an interactive window
(needs a display) and loops the probe back and forth across the phantom, pressing down
to make contact, so you can see the arm move. Close the window or Ctrl-C to stop.

    python scripts/view_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepussim.geometry import make_transform, mat_to_quat, rot_x
from deepussim.sim.scene import UltrasoundScene, SceneConfig


ROOT = Path(__file__).resolve().parents[1]
PHANTOM_MESH = ROOT / "data" / "phantom_mesh" / "segmentation_segment_1_m_centered.obj"
PHANTOM_SIZE = (0.31335004, 0.19479432, 0.23921130)

T_WORLD_FROM_PHANTOM_MEASURED = np.array([
    [-0.9989, -0.0158, -0.0452, 0.5460],
    [0.0162, -0.9998, -0.0091, 0.3930],
    [-0.0451, -0.0098, 0.9989, 0.3135],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)
T_PHANTOM_LIE_DOWN = make_transform(rot_x(np.pi / 2.0), [0.0, 0.0, 0.0])
T_WORLD_FROM_PHANTOM = T_WORLD_FROM_PHANTOM_MEASURED @ T_PHANTOM_LIE_DOWN
PHANTOM_TOP_AXIS = 1


def probe_pose_over_phantom(lateral_m: float, indent_m: float,
                            phantom_size: tuple[float, float, float]) -> np.ndarray:
    """Probe +z points into the phantom; position is just below the local top face."""
    R_phantom = T_WORLD_FROM_PHANTOM[:3, :3]
    center = T_WORLD_FROM_PHANTOM[:3, 3]
    x_axis = R_phantom[:, 0] / np.linalg.norm(R_phantom[:, 0])
    normal = R_phantom[:, PHANTOM_TOP_AXIS]
    normal = normal / np.linalg.norm(normal)

    probe_z = -normal
    probe_y = np.cross(probe_z, x_axis)
    probe_y = probe_y / np.linalg.norm(probe_y)
    probe_x = np.cross(probe_y, probe_z)
    R_probe = np.column_stack([probe_x, probe_y, probe_z])

    top_offset = phantom_size[PHANTOM_TOP_AXIS] / 2.0 - indent_m
    pos = center + lateral_m * x_axis + top_offset * normal
    return make_transform(R_probe, pos)


def main() -> None:
    phantom_size = PHANTOM_SIZE
    cfg = SceneConfig(
        backend="gpu",
        show_viewer=True,
        phantom_mesh=str(PHANTOM_MESH),
        phantom_pos=tuple(T_WORLD_FROM_PHANTOM[:3, 3]),
        phantom_quat=tuple(mat_to_quat(T_WORLD_FROM_PHANTOM[:3, :3])),
        phantom_size=phantom_size,
        camera_lookat=tuple(T_WORLD_FROM_PHANTOM[:3, 3]),
    )
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    indent_m = 0.02
    laterals = np.concatenate([
        np.linspace(-0.09, 0.09, 60),    # sweep across the phantom local x-axis
        np.linspace(0.09, -0.09, 60),
    ])

    print("[view_sim] viewer open - sweeping the probe across the placed phantom. Ctrl-C to stop.")
    viewer = getattr(scene._scene, "viewer", None)

    def viewer_alive() -> bool:
        return not (viewer is not None and hasattr(viewer, "is_alive")
                    and not viewer.is_alive())

    try:
        sweeps = 0
        while viewer_alive() and sweeps < 200:
            for lateral in laterals:
                scene.set_probe_pose(probe_pose_over_phantom(float(lateral), indent_m,
                                                             phantom_size))
                scene.step(3)
            sweeps += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
