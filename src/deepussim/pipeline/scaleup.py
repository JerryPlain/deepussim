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


def slice_coverage(intensity: np.ndarray) -> float:
    """Fraction of the resliced fan that imaged actual tissue (0 = empty, ~1 = full).

    An off-anatomy / out-of-volume reslice (e.g. an edge pose whose fan exits the body)
    comes back near-uniform — out-of-bounds samples are a constant, so the slice has almost
    no dynamic range. Measure the fraction of pixels rising above the slice's own floor;
    a (near-)uniform slice scores ~0 and can be dropped before it pollutes the dataset.
    """
    x = np.asarray(intensity, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:                       # uniform -> nothing imaged
        return 0.0
    return float(np.mean(x > lo + 0.05 * (hi - lo)))


def pose_convergence(T_target: np.ndarray, T_achieved: np.ndarray) -> tuple[float, float, float]:
    """How far the *achieved* probe pose landed from its *nominal target*.

    The error is decomposed in the target's own frame, because the servo *intends* to
    press along the probe axial (``servo_to_force`` modulates depth by up to its clip):

      - ``lateral_mm`` — translation error perpendicular to the target axial (+z). This is
        the IK in-plane miss; it must be tiny for the slice to image the intended spot.
      - ``axial_mm``   — |translation error| along the target axial. Legitimately non-zero
        (the press depth), so this is allowed a wide tolerance, not required to be ~0.
      - ``rot_deg``    — angle between the achieved and target orientations.

    Inputs are 4x4 poses in a common frame (sim world); ``*_mm`` are in that frame's units
    times 1000 (so metres -> mm). A far-stalled / unreachable pose blows up lateral_mm and
    rot_deg while a genuine press only grows axial_mm.
    """
    T_target = np.asarray(T_target, dtype=float)
    T_achieved = np.asarray(T_achieved, dtype=float)
    axial = T_target[:3, 2] / (np.linalg.norm(T_target[:3, 2]) + 1e-12)  # +z = into tissue
    d = T_achieved[:3, 3] - T_target[:3, 3]
    d_axial = float(d @ axial)
    lateral = float(np.linalg.norm(d - d_axial * axial))
    # geodesic angle between rotations: arccos((tr(Ra^T Rb) - 1) / 2)
    cos = (np.trace(T_target[:3, :3].T @ T_achieved[:3, :3]) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return lateral * 1000.0, abs(d_axial) * 1000.0, rot_deg


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
    force_target_n: float | None = None,
    reach_lateral_mm: float = 8.0,
    reach_axial_mm: float = 40.0,
    reach_rot_deg: float = 10.0,
    min_coverage: float = 0.05,
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

    A pose is written only if the probe both **made contact** and **reached its target**:
    contact alone is a false-positive gate — an unreachable pose where IK stalls far away
    but some arm link grazes the phantom still registers force, and reslicing the (far-off)
    *achieved* pose would write an off-anatomy sample. ``reach_*`` bound the achieved-vs-
    target deviation (see :func:`pose_convergence`): ``reach_lateral_mm`` / ``reach_rot_deg``
    must be small (IK landed where we aimed, pointed right), while ``reach_axial_mm`` is wide
    (the servo presses along the axial on purpose). Separately, a pose whose resliced fan
    imaged (almost) no tissue — coverage below ``min_coverage`` (see :func:`slice_coverage`),
    e.g. an edge pose whose fan exits the body — is dropped before writing. Dropped poses are
    counted and logged by reason (no-contact / not-reached / empty) rather than silently kept.
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

    dropped = {"no_contact": 0, "not_reached": 0, "empty": 0}  # auditable drop reasons
    with DatasetWriter(out_dir, meta={"geometry": geom.__dict__}) as writer:
        # for each candidate pose, drive the sim (if given), reslice, render, and write a Sample
        for idx, T_nominal in enumerate(iterator):
            force = None
            meta = {"index": idx, "source": "sim" if scene is not None else "reslice"}
            # default reslice pose is the nominal target (sim world, m) mapped into CBCT (mm) if sim given, else just the pose (CBCT, mm)
            reslice_pose = np.asarray(T_nominal, dtype=float)

            if scene is not None:
                T_nom = np.asarray(T_nominal, dtype=float) # if sim: nominal target pose in sim world (m)

                # move the sim probe there, check contact, get the achieved pose + force from the sim physics
                if force_target_n is not None and hasattr(scene, "servo_to_force"):
                    scene.servo_to_force(T_nom, target_n=force_target_n)  # hold a realistic force
                    contacted = scene.in_contact()
                elif hasattr(scene, "servo_to_contact"):
                    contacted = scene.servo_to_contact(T_nom)
                else:
                    scene.set_probe_pose(T_nom)
                    scene.step(settle_steps)
                    contacted = scene.in_contact()
                if not contacted:
                    dropped["no_contact"] += 1
                    continue  # no force on any link -> drop

                pose_sim = scene.probe_pose() # achieved probe pose in sim world (m)
                # GATE: contact alone is a false positive — an unreachable pose where IK stalls
                # far off but an arm link grazes the phantom still registers force. Require the
                # probe to have actually reached its target (small lateral/rot miss; axial press
                # allowed) before trusting the achieved pose to reslice.
                lateral_mm, axial_mm, rot_deg = pose_convergence(T_nom, pose_sim)
                meta["reach"] = {"lateral_mm": lateral_mm, "axial_mm": axial_mm, "rot_deg": rot_deg}
                if (lateral_mm > reach_lateral_mm or axial_mm > reach_axial_mm
                        or rot_deg > reach_rot_deg):
                    dropped["not_reached"] += 1
                    continue  # probe stalled / off-target -> would write an off-anatomy sample

                force = scene.contact_force() # Newtons, if available (else None)
                reslice_pose = sim_pose_to_cbct(pose_sim, sim_to_cbct)  # CBCT, mm
                meta["pose_sim_m"] = pose_sim.tolist()
            
            # reslice the intensity at the reslice pose; drop poses whose fan imaged (almost)
            # no tissue — e.g. an edge pose whose fan exits the body — before rendering/writing.
            intensity = reslice_volume(volume, reslice_pose, geom, order=1)
            if slice_coverage(intensity) < min_coverage:
                dropped["empty"] += 1
                continue
            image = render(intensity, geom, params) # render the resliced intensity into a US image (with depth-dependent brightness, etc)

            # reslice the label volume at the same pose, if given, to get a mask of the anatomy visible in the image (nearest-neighbour to preserve discrete labels)
            mask = None
            if label_volume is not None:
                mask = reslice_volume(volume=label_volume, T_world_from_probe=reslice_pose,
                                      geom=geom, order=0).astype(np.int16)
            
            # save the Sample: image, pose, mask, force, and metadata (including sim vs reslice source and sim pose if from sim)
            # - force: from sim physics, if available (else None)
            # - image: rendered from the resliced intensity volume
            # - pose: the reslice pose in CBCT frame (mm) — this is the "ground truth" for learning, and is the nominal target pose mapped into CBCT if from sim, else just the pose if from reslice 
            # - mask: resliced from the label volume at the same pose, if given (else None)
            # - meta: at least the index and source; if from sim also the achieved sim pose in metres for reference

            writer.add(Sample(image=image, pose=reslice_pose, mask=mask,
                              force=force, meta=meta))

        n = len(writer)
        n_dropped = sum(dropped.values())
        if n_dropped:
            print(f"scale-up: wrote {n}/{n + n_dropped} poses "
                  f"(dropped {dropped['no_contact']} no-contact, "
                  f"{dropped['not_reached']} not-reached, {dropped['empty']} empty)")
        return n
