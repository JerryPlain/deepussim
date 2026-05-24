"""Step 5 — generate an aligned multi-modal dataset from many probe poses.

For each pose:
  - (optional) drive the Genesis scene, skip non-contacting poses, take the *achieved*
    pose and the contact force from physics;
  - reslice the intensity volume + render -> US image;
  - reslice the label volume (nearest-neighbour) -> anatomy mask;
  - write the Sample.

Runs end-to-end with ``scene=None`` (force omitted) given only volumes + poses, which
is the path exercised by ``scripts/run_scaleup.py`` on a synthetic phantom.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.volume import Volume
from ..data.record import Sample, DatasetWriter
from ..us.reslice import ProbeGeometry, reslice_volume
from ..us.renderer import RendererParams, render
from ..calib.placement import sim_pose_to_cbct


def generate_dataset(
    out_dir,
    volume: Volume,
    poses: Sequence[np.ndarray],
    geom: ProbeGeometry,
    params: RendererParams | None = None,
    label_volume: Volume | None = None,
    scene=None,
    sim_to_cbct: np.ndarray | None = None,
    settle_steps: int = 50,
    progress: bool = True,
) -> int:
    """Generate and write samples for ``poses``. Returns the number written.

    Frame of ``poses`` depends on ``scene``:
      - ``scene is None`` (no-sim): poses are reslice poses already in the CBCT frame (mm).
      - ``scene`` given: poses are *nominal probe targets in the sim world (m)*; the arm
        is driven there, non-contacting poses are dropped, and the *achieved* sim pose +
        contact force are taken from physics. The achieved pose is mapped into the CBCT
        frame via ``sim_to_cbct`` (T_cbct_from_simworld, mm) before reslicing — this is
        the sim->reslice bridge (see calib.placement). ``sim_to_cbct`` is required when
        ``scene`` is given.
    """
    if scene is not None and sim_to_cbct is None:
        raise ValueError("sim_to_cbct (T_cbct_from_simworld) is required when scene is given")
    params = params or RendererParams()
    iterator = poses
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(poses, desc="scale-up")
        except ImportError:
            pass

    with DatasetWriter(out_dir, meta={"geometry": geom.__dict__}) as writer:
        for idx, T_nominal in enumerate(iterator):
            force = None
            meta = {"index": idx, "source": "sim" if scene is not None else "reslice"}
            reslice_pose = np.asarray(T_nominal, dtype=float)

            if scene is not None:
                T_nom = np.asarray(T_nominal, dtype=float)
                if hasattr(scene, "servo_to_contact"):
                    contacted = scene.servo_to_contact(T_nom)
                else:
                    scene.set_probe_pose(T_nom)
                    scene.step(settle_steps)
                    contacted = scene.in_contact()
                if not contacted:
                    continue  # unreachable / no contact -> drop
                pose_sim = scene.probe_pose()                     # sim world, metres
                force = scene.contact_force()
                reslice_pose = sim_pose_to_cbct(pose_sim, sim_to_cbct)  # CBCT, mm
                meta["pose_sim_m"] = pose_sim.tolist()

            intensity = reslice_volume(volume, reslice_pose, geom, order=1)
            image = render(intensity, geom, params)

            mask = None
            if label_volume is not None:
                mask = reslice_volume(volume=label_volume, T_world_from_probe=reslice_pose,
                                      geom=geom, order=0).astype(np.int16)

            writer.add(Sample(image=image, pose=reslice_pose, mask=mask,
                              force=force, meta=meta))

        return len(writer)
