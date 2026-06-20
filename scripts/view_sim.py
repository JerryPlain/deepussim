#!/usr/bin/env python
"""Replay a recorded probe trajectory in the Genesis viewer to check phantom placement.

This is a visual sanity tool: point ``--seq`` at any recorded sequence ``.npz``
(``scan1`` .. ``scan15``) and watch the FR3 drive the probe along the REAL trajectory
that was recorded during that scan, over the calibrated phantom. Use it to eyeball, per
sequence, whether the placement seats the probe on the phantom surface (riding the
surface = good; floating above / buried inside / off to one side = bad).

    conda activate deepussim
    python scripts/view_sim.py --seq data/sequences/scan1.npz
    python scripts/view_sim.py --seq data/sequences/scan7.npz --stride 3 --loops 5

The viewer opens a window (needs a display) and loops the trajectory. Close the window
or Ctrl-C to stop. Pass --headless to run without a window (e.g. a quick load check).

------------------------------------------------------------------------------------
COORDINATE CHAIN (all rigid transforms; ``T_A_FROM_B`` maps a point B -> A)

    world / robot base ── measured calib ──> cbct / phantom
          ▲                                        │ lie-down belly-up (Rx90·Rz180)
          │ rosbag, per frame                      │ + seat onto the real contact cloud
        ee / flange ── hand-eye (T_EE_FROM_PROBE) ──> probe (US transducer)

Two things are placed in the scene:

  • PHANTOM  — seat_phantom_on_contacts(): start from the measured robot<-cbct calib
    (``T_WORLD_FROM_CBCT``), lie the phantom belly-up, then snap its surface onto the
    cloud of recorded probe-contact points so the surface sits where the probe pressed.
    A fixed Z nudge (``--z-offset-m``) finishes the depth seating. By default the seat is
    computed from the replayed sequence's own contacts; pass ``--seat-seq`` to seat from a
    different (e.g. better-covered) sequence and reuse that placement.

  • PROBE    — probe_poses_from_sequence(): for each in-contact frame take the recorded
    end-effector pose and apply the hand-eye ``T_EE_FROM_PROBE`` to get the true probe
    pose, then drive the arm there by IK (UltrasoundScene.set_probe_pose).

NOTE on ``--probe-rz-deg`` (the 45° you may have noticed): it spins the probe about its
OWN long axis (the beam / probe local Z). That does NOT move the probe tip, so it has no
effect on contact/placement — it only re-orients the probe MODEL in the viewer. The real
45° mount angle is already inside ``T_EE_FROM_PROBE``; the slicer (slicer_3.0) images at
probe-rz = 0 and that is the geometric imaging plane. The viewer defaults to -45 only so
the rendered probe model looks right. Set ``--probe-rz-deg 0`` to see the bare geometric
pose. This knob is cosmetic for placement checking; it does not change where the probe is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepussim.calib.transforms import T_EE_FROM_PROBE, T_WORLD_FROM_CBCT
from deepussim.geometry import compose, make_transform, mat_to_quat, rot_x, rot_z
from deepussim.sim.scene import SceneConfig, UltrasoundScene


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQ = ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_MESH = ROOT / "data" / "phantom_mesh" / "segmentation_segment_1_m_centered.obj"

# Fixed Z nudge applied after the contact snap to finish seating the belly-up phantom onto the
# probe cloud. This value is what visibly seats the probe on the surface in the viewer; keep it.
DEFAULT_Z_OFFSET_M = -0.170656

# Probe model spin about its own axis — see the module docstring. Cosmetic for the viewer;
# the slicer images at 0.0. Exposed as --probe-rz-deg so you can compare.
DEFAULT_PROBE_RZ_DEG = -45.0


def load_sequence(seq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ee_poses, contact)`` from a recorded sequence ``.npz``.

    ``ee_poses`` is ``(N, 4, 4)`` ``T_world_from_ee`` per frame; ``contact`` is a bool
    mask marking frames where the probe was touching the phantom (bright US image).
    """
    data = np.load(seq_path, allow_pickle=True)
    ee_poses = np.asarray(data["poses"], dtype=float)
    contact = np.asarray(data["contact"], dtype=bool)
    return ee_poses, contact


def seat_phantom_on_contacts(
    ee_poses: np.ndarray,
    contact: np.ndarray,
    mesh,
    z_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Seat the phantom into the robot world for this sequence.

    Returns ``(T_world_from_phantom, t_align)``: the rigid pose to place the phantom mesh
    at, and the translation that was applied to snap it onto the contact cloud (printed so
    you can compare seatings across sequences).
    """
    from scipy.spatial import cKDTree

    # Probe-face positions in the world frame for the in-contact frames: ee pose then the
    # hand-eye gives the true probe pose; its origin is the probe face that touched tissue.
    origins = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses[contact]])

    # Measured robot<-cbct calibration, then lie the phantom belly-up so the up-facing
    # anterior surface meets a probe pressing straight down (matches the real rig).
    lie = make_transform(rot_x(np.pi / 2.0) @ rot_z(np.pi), [0.0, 0.0, 0.0])
    T_world_from_phantom = compose(T_WORLD_FROM_CBCT, lie)

    # Place the (centred, metre-unit) mesh and shift it so its surface sits under the probe
    # contact points. Median over the per-contact nearest-vertex offsets is robust to the
    # odd frame that was not quite on the surface.
    verts_h = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    placed = (T_world_from_phantom @ verts_h.T).T[:, :3]
    nearest = placed[cKDTree(placed).query(origins)[1]]
    t_align = np.median(origins - nearest, axis=0)
    t_align[2] += z_offset_m  # residual depth seating (see DEFAULT_Z_OFFSET_M)

    T_world_from_phantom = compose(make_transform(np.eye(3), t_align), T_world_from_phantom)
    return T_world_from_phantom, t_align


def probe_poses_from_sequence(
    ee_poses: np.ndarray,
    contact: np.ndarray,
    probe_rz_deg: float,
    stride: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Probe poses to replay: ``ee · hand-eye · Rz(probe_rz)`` for each in-contact frame.

    Returns ``(probe_poses, frame_indices)``. ``probe_rz`` only spins the probe about its
    own axis (cosmetic — see the module docstring).
    """
    spin = make_transform(rot_z(np.deg2rad(probe_rz_deg)), [0.0, 0.0, 0.0])
    frames = np.arange(len(ee_poses))[contact][:: max(1, stride)]
    poses = [compose(compose(ee_poses[i], T_EE_FROM_PROBE), spin) for i in frames]
    return poses, frames


def demo_sweep_poses(
    T_world_from_phantom: np.ndarray,
    phantom_size: np.ndarray,
    indent_m: float,
) -> list[np.ndarray]:
    """Old synthetic back-and-forth sweep (``--demo-sweep``), kept for a quick rig check.

    Builds an idealized probe that points straight into the up-facing surface and slides
    along the phantom's local x-axis. Uses no recorded data — it only shows the arm can
    reach and press the placed phantom.
    """
    R = T_world_from_phantom[:3, :3]
    center = T_world_from_phantom[:3, 3]
    top_axis = 1  # anterior-posterior axis; the up-facing side is -this after the lie-down
    x_axis = R[:, 0] / np.linalg.norm(R[:, 0])
    normal = -R[:, top_axis] / np.linalg.norm(R[:, top_axis])

    laterals = np.concatenate([np.linspace(-0.09, 0.09, 60), np.linspace(0.09, -0.09, 60)])
    top_offset = phantom_size[top_axis] / 2.0 - indent_m

    poses = []
    for lateral in laterals:
        probe_z = -normal                       # beam points into the phantom
        probe_y = np.cross(probe_z, x_axis)
        probe_y /= np.linalg.norm(probe_y)
        probe_x = np.cross(probe_y, probe_z)
        R_probe = np.column_stack([probe_x, probe_y, probe_z])
        pos = center + float(lateral) * x_axis + top_offset * normal
        poses.append(make_transform(R_probe, pos))
    return poses


def viewer_camera(T_world_from_phantom: np.ndarray) -> tuple[tuple, tuple]:
    """A camera pose looking at the phantom roughly along its surface normal."""
    center = T_world_from_phantom[:3, 3]
    R = T_world_from_phantom[:3, :3]
    x_axis = R[:, 0] / np.linalg.norm(R[:, 0])
    normal = -R[:, 1] / np.linalg.norm(R[:, 1])
    pos = center + 0.75 * normal - 0.45 * x_axis + np.array([0.0, 0.0, 0.25])
    return tuple(pos), tuple(center)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, default=DEFAULT_SEQ,
                    help="sequence .npz to replay (poses/contact). Try any of data/sequences/scan*.npz")
    ap.add_argument("--seat-seq", type=Path, default=None,
                    help="optional: seat the phantom from this sequence's contacts instead of --seq")
    ap.add_argument("--mesh", type=Path, default=DEFAULT_MESH,
                    help="centred phantom surface mesh (metre units) shown in the scene")
    ap.add_argument("--probe-rz-deg", type=float, default=DEFAULT_PROBE_RZ_DEG,
                    help="spin the probe model about its own axis (cosmetic; slicer uses 0)")
    ap.add_argument("--z-offset-m", type=float, default=DEFAULT_Z_OFFSET_M,
                    help="residual depth-seating nudge along world Z after the contact snap")
    ap.add_argument("--stride", type=int, default=1, help="replay every Nth in-contact frame")
    ap.add_argument("--steps-per-frame", type=int, default=8, help="Genesis steps held per pose")
    ap.add_argument("--loops", type=int, default=100, help="number of replay loops")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--headless", action="store_true", help="run without a viewer window")
    ap.add_argument("--demo-sweep", action="store_true",
                    help="ignore the sequence and run the old synthetic sweep instead")
    args = ap.parse_args()

    import trimesh

    ee_poses, contact = load_sequence(args.seq)            # trajectory to replay
    n_contact = int(contact.sum())
    if n_contact == 0:
        raise SystemExit(f"[view_sim] {args.seq} has no in-contact frames to replay")

    # Placement: seat from the replayed sequence's own contacts, unless --seat-seq overrides.
    if args.seat_seq is not None:
        seat_ee, seat_contact = load_sequence(args.seat_seq)
    else:
        seat_ee, seat_contact = ee_poses, contact

    mesh = trimesh.load(args.mesh, process=False)
    T_world_from_phantom, t_align = seat_phantom_on_contacts(
        seat_ee, seat_contact, mesh, args.z_offset_m
    )

    if args.demo_sweep:
        phantom_size = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
        probe_poses = demo_sweep_poses(T_world_from_phantom, phantom_size, indent_m=0.02)
        frames = np.arange(len(probe_poses))
    else:
        probe_poses, frames = probe_poses_from_sequence(
            ee_poses, contact, args.probe_rz_deg, args.stride
        )

    camera_pos, camera_lookat = viewer_camera(T_world_from_phantom)
    cfg = SceneConfig(
        backend=args.backend,
        show_viewer=not args.headless,
        phantom_mesh=str(args.mesh),
        phantom_pos=tuple(T_world_from_phantom[:3, 3]),
        phantom_quat=tuple(mat_to_quat(T_world_from_phantom[:3, :3])),
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
    )
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    print(f"[view_sim] seq            : {args.seq}")
    print(f"[view_sim] seat-seq       : {args.seat_seq or '(same as seq)'}")
    print(f"[view_sim] in-contact     : {n_contact}/{len(ee_poses)} frames"
          f"{'' if args.demo_sweep else f', replaying {len(probe_poses)} (stride {args.stride})'}")
    print(f"[view_sim] seat t_align m : {np.round(t_align, 4)}")
    print(f"[view_sim] probe Rz deg   : {'n/a (demo sweep)' if args.demo_sweep else args.probe_rz_deg}")
    print("[view_sim] viewer open — probe should ride the phantom surface. Close window / Ctrl-C to stop.")

    viewer = getattr(scene._scene, "viewer", None)

    def viewer_alive() -> bool:
        return not (viewer is not None and hasattr(viewer, "is_alive") and not viewer.is_alive())

    try:
        for _ in range(args.loops):
            if not viewer_alive():
                break
            for T in probe_poses:
                scene.set_probe_pose(T)
                scene.step(args.steps_per_frame)
                if not viewer_alive():
                    break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
