#!/usr/bin/env python
"""Replay scan1's recorded probe trajectory in the Genesis viewer."""
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
DEFAULT_SEQ = ROOT / "data" / "sequences_20260612" / "scan1.npz"
DEFAULT_MESH = ROOT / "data" / "phantom_mesh" / "segmentation_segment_1_m_centered.obj"
PHANTOM_POSITION_OFFSET_M = np.array([0.0, 0.0, -0.170656])
PROBE_ROTATION_OFFSET = make_transform(rot_z(np.deg2rad(-45.0)), [0.0, 0.0, 0.0])


def placed_phantom_from_scan1(seq_path: Path, mesh_path: Path):
    """Match the centered mesh placement used by the slicer scripts."""
    import trimesh
    from scipy.spatial import cKDTree

    d = np.load(seq_path, allow_pickle=True)
    ee_poses = np.asarray(d["poses"], dtype=float)
    contact = np.asarray(d["contact"], dtype=bool)
    origins = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_poses[contact]])

    lie = make_transform(rot_x(np.pi / 2.0) @ rot_z(np.pi), [0.0, 0.0, 0.0])
    T_world_from_cbctm = compose(T_WORLD_FROM_CBCT, lie)

    mesh = trimesh.load(mesh_path, process=False)
    vertices_h = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    placed_vertices = (T_world_from_cbctm @ vertices_h.T).T[:, :3]
    nearest = placed_vertices[cKDTree(placed_vertices).query(origins)[1]]
    t_align = np.median(origins - nearest, axis=0)
    t_align += PHANTOM_POSITION_OFFSET_M
    T_world_from_cbctm = compose(make_transform(np.eye(3), t_align), T_world_from_cbctm)
    return T_world_from_cbctm, t_align


def viewer_camera(T_world_from_cbctm: np.ndarray):
    center = T_world_from_cbctm[:3, 3]
    R = T_world_from_cbctm[:3, :3]
    x_axis = R[:, 0] / np.linalg.norm(R[:, 0])
    normal = -R[:, 1] / np.linalg.norm(R[:, 1])
    pos = center + 0.75 * normal - 0.45 * x_axis + np.array([0.0, 0.0, 0.25])
    return tuple(pos), tuple(center)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default=str(DEFAULT_SEQ), help="sequence .npz with poses/contact")
    ap.add_argument("--mesh", default=str(DEFAULT_MESH), help="centered phantom mesh used in Genesis")
    ap.add_argument("--stride", type=int, default=1, help="play every Nth frame")
    ap.add_argument("--steps-per-frame", type=int, default=12, help="Genesis steps to hold each pose")
    ap.add_argument("--loops", type=int, default=200, help="number of replay loops")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    args = ap.parse_args()

    seq_path = Path(args.seq)
    mesh_path = Path(args.mesh)
    d = np.load(seq_path, allow_pickle=True)
    ee_poses = np.asarray(d["poses"], dtype=float)
    contact = np.asarray(d["contact"], dtype=bool)
    frames = np.arange(len(ee_poses))[contact][:: max(1, args.stride)]

    T_world_from_cbctm, t_align = placed_phantom_from_scan1(seq_path, mesh_path)
    probe_poses = [
        compose(compose(ee_poses[i], T_EE_FROM_PROBE), PROBE_ROTATION_OFFSET)
        for i in frames
    ]
    camera_pos, camera_lookat = viewer_camera(T_world_from_cbctm)
    cfg = SceneConfig(
        backend=args.backend,
        show_viewer=True,
        phantom_mesh=str(mesh_path),
        phantom_pos=tuple(T_world_from_cbctm[:3, 3]),
        phantom_quat=tuple(mat_to_quat(T_world_from_cbctm[:3, :3])),
        camera_pos=camera_pos,
        camera_lookat=camera_lookat,
    )
    scene = UltrasoundScene(cfg).build()
    scene.reset()

    print(f"[view_scan1] seq: {seq_path}")
    print(f"[view_scan1] replaying {len(probe_poses)} contact poses from {len(ee_poses)} frames")
    print(f"[view_scan1] seating translation m: {t_align}")
    print("[view_scan1] Genesis viewer open. Close the window or Ctrl-C to stop.")

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
