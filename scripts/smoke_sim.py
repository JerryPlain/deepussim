#!/usr/bin/env python
"""Headless smoke test for the Genesis scene (requires genesis-world + torch + a GPU).

Builds the Franka + box-phantom scene, lowers the probe until it contacts the phantom,
and prints the contact force + achieved probe pose. This is the manual check for
``deepussim.sim.scene`` — it is intentionally NOT part of the pytest suite because it
needs Genesis/GPU and spends ~60s compiling kernels on first run.

    python scripts/smoke_sim.py
"""
from __future__ import annotations

import numpy as np

from deepussim.geometry import pose_from_pos_quat, mat_to_quat
from deepussim.sim.scene import UltrasoundScene, SceneConfig


def main() -> None:
    # Box phantom top surface at z = pos.z + size.z/2 = 0.08 m, well within reach.
    cfg = SceneConfig(backend="gpu", show_viewer=False,
                      phantom_pos=(0.45, 0.0, 0.04), phantom_size=(0.2, 0.2, 0.08))
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    down = [0.0, 1.0, 0.0, 0.0]  # hand pointing straight down (scalar-first quat)
    scene.set_probe_pose(pose_from_pos_quat([0.45, 0.0, 0.30], down))
    scene.step(80)

    # Descend until the probe tip presses the phantom (box top ~0.08 m).
    contacted_at = None
    for z in np.arange(0.28, 0.04, -0.01):
        scene.set_probe_pose(pose_from_pos_quat([0.45, 0.0, float(z)], down))
        scene.step(25)
        if scene.in_contact(threshold_n=0.5):
            contacted_at = float(z)
            break

    f = scene.contact_force()
    p = scene.probe_pose()
    print("contacted at probe-z (m):", None if contacted_at is None else round(contacted_at, 3))
    print("contact force (N)      :", np.round(f, 3), "| |f| =", round(float(np.linalg.norm(f)), 2))
    print("in_contact             :", scene.in_contact(threshold_n=0.5))
    print("probe pos (m)          :", np.round(p[:3, 3], 4))
    print("probe quat (wxyz)      :", np.round(mat_to_quat(p[:3, :3]), 3))


if __name__ == "__main__":
    main()
