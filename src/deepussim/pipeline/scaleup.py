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


def generate_dataset(
    out_dir,
    volume: Volume,
    poses: Sequence[np.ndarray],
    geom: ProbeGeometry,
    params: RendererParams | None = None,
    label_volume: Volume | None = None,
    scene=None,
    settle_steps: int = 50,
    progress: bool = True,
) -> int:
    """Generate and write samples for ``poses``. Returns the number written.

    ``poses`` are nominal targets in the volume/CBCT frame. With a ``scene`` they are
    sent to the arm and the achieved in-contact pose is used instead.
    """
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
            T = np.asarray(T_nominal, dtype=float)

            if scene is not None:
                scene.set_probe_pose(T)
                scene.step(settle_steps)
                if not scene.in_contact():
                    continue  # unreachable / no contact -> drop
                T = scene.probe_pose()
                force = scene.contact_force()

            intensity = reslice_volume(volume, T, geom, order=1)
            image = render(intensity, geom, params)

            mask = None
            if label_volume is not None:
                mask = reslice_volume(label_volume, T, geom, order=0).astype(np.int16)

            meta = {"index": idx, "source": "sim" if scene is not None else "reslice"}
            writer.add(Sample(image=image, pose=T, mask=mask, force=force, meta=meta))

        return len(writer)
