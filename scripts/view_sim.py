#!/usr/bin/env python
"""Open the Genesis viewer and watch the Franka scan/press the phantom (NOT headless).

This is the visual counterpart to scripts/smoke_sim.py: it opens an interactive window
(needs a display) and loops the probe back and forth across the phantom, pressing down
to make contact, so you can see the arm move. Close the window or Ctrl-C to stop.

    python scripts/view_sim.py
"""
from __future__ import annotations

import numpy as np

from deepussim.geometry import pose_from_pos_quat
from deepussim.sim.scene import UltrasoundScene, SceneConfig


def main() -> None:
    cfg = SceneConfig(
        backend="gpu",
        show_viewer=True,
        phantom_pos=(0.45, 0.0, 0.04),
        phantom_size=(0.2, 0.2, 0.08),
    )
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    down = [0.0, 1.0, 0.0, 0.0]          # probe pointing straight down
    press_z = 0.07                        # probe-tip target: just into the box top (~0.08 m)
    xs = np.concatenate([np.linspace(0.36, 0.54, 60),    # sweep across the phantom
                         np.linspace(0.54, 0.36, 60)])

    print("[view_sim] viewer open — sweeping the probe across the phantom. Ctrl-C to stop.")
    viewer = getattr(scene._scene, "viewer", None)

    def viewer_alive() -> bool:
        return not (viewer is not None and hasattr(viewer, "is_alive")
                    and not viewer.is_alive())

    try:
        sweeps = 0
        while viewer_alive() and sweeps < 200:
            for x in xs:
                scene.set_probe_pose(pose_from_pos_quat([float(x), 0.0, press_z], down))
                scene.step(3)
            sweeps += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
