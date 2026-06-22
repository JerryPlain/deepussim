#!/usr/bin/env python
"""Generate new trajectories by offsetting existing LC2-refined rosbag trajectories.

This is the conservative coverage-expansion generator. It keeps the motion shape and probe
orientation of real trajectories, but translates whole trajectories along principal coverage
directions in the belly-up robot/world frame, then maps them back to CBCT mm for reslice. Candidate offsets are scored by:

* novelty: distance to the nearest existing LC2 pose;
* validity: fraction of the CBCT fan inside the volume.

The output is a standard trajectory archive with key ``poses_cbct_mm``.

Example:
    python renderer_training/generate_offset_trajectories.py \
        --out data/trajectories/novel_offset_valid.npz \
        --offsets-mm -30 -20 -10 10 20 30 \
        --n-trajectories 12 --per-trajectory 32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from renderer_training.analyze_pose_coverage import _natural_scan_key, load_lc2_poses
from renderer_training.generate_novel_trajectories import _inside_ratios, _score_candidates




def _load_world_from_phantom(path: Path) -> np.ndarray:
    from reslice.io import load_transform_4x4
    from reslice.pose import default_world_from_phantom_centered_m

    return load_transform_4x4(path) if path.exists() else default_world_from_phantom_centered_m()


def _poses_mm_to_world_m(poses_mm: np.ndarray, T_world_from_phantom: np.ndarray) -> np.ndarray:
    out = np.asarray(poses_mm, dtype=float).copy()
    out[:, :3, 3] /= 1000.0
    return np.einsum("ij,njk->nik", T_world_from_phantom, out)


def _poses_world_m_to_mm(poses_world: np.ndarray, T_world_from_phantom: np.ndarray) -> np.ndarray:
    from reslice._geometry import invert_rigid

    T_phantom_from_world = invert_rigid(T_world_from_phantom)
    out = np.einsum("ij,njk->nik", T_phantom_from_world, np.asarray(poses_world, dtype=float))
    out[:, :3, 3] *= 1000.0
    return out


def _pca_axes(points: np.ndarray) -> np.ndarray:
    X = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt


def _resample_poses(poses: np.ndarray, n: int) -> np.ndarray:
    if len(poses) == n:
        return poses.copy()
    from scipy.spatial.transform import Rotation, Slerp

    src_t = np.linspace(0.0, 1.0, len(poses))
    dst_t = np.linspace(0.0, 1.0, n)
    out = np.repeat(np.eye(4)[None], n, axis=0)
    out[:, :3, 3] = np.column_stack(
        [np.interp(dst_t, src_t, poses[:, j, 3]) for j in range(3)]
    )
    slerp = Slerp(src_t, Rotation.from_matrix(poses[:, :3, :3]))
    out[:, :3, :3] = slerp(dst_t).as_matrix()
    return out


def _candidate_offsets(axes: np.ndarray, offsets_mm: list[float], include_depth: bool) -> list[np.ndarray]:
    # Use the two high-variance position directions by default. They are the most interpretable
    # "nearby but not already scanned" translations. Depth offsets are optional because they can
    # easily push the fan out of volume or out of contact.
    dirs = [axes[0], axes[1]]
    if include_depth:
        dirs.append(axes[2])
    out = []
    for d in dirs:
        for mm in offsets_mm:
            if abs(mm) > 1e-9:
                out.append(np.asarray(d, dtype=float) * float(mm))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "trajectories" / "novel_offset_valid.npz")
    ap.add_argument("--offsets-mm", nargs="+", type=float, default=[-30, -20, -10, 10, 20, 30])
    ap.add_argument("--include-depth-offsets", action="store_true")
    ap.add_argument("--n-trajectories", type=int, default=12)
    ap.add_argument("--per-trajectory", type=int, default=32)
    ap.add_argument("--min-inside", type=float, default=0.85)
    ap.add_argument("--min-valid-fraction", type=float, default=0.9)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--angle-weight", type=float, default=0.25)
    ap.add_argument("--world-from-phantom", type=Path, default=REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt",
                    help="T_world_from_phantom_liedown in metres; falls back to reslice.pose default if missing")
    ap.add_argument("--offset-frame", choices=["world", "cbct"], default="world",
                    help="frame where PCA/offsets are defined. world = belly-up robot frame; cbct = legacy phantom frame")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json")
    ap.add_argument("--volume-path", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--ref-sequence", type=Path, default=REPO_ROOT / "data" / "sequences" / "scan1.npz")
    ap.add_argument("--us-spacing", type=float, default=0.166112957)
    args = ap.parse_args()

    existing_poses, rows = load_lc2_poses(args.lc2_dir, ["scan*_global.npz"])
    T_world_from_phantom = _load_world_from_phantom(args.world_from_phantom)
    existing_world = _poses_mm_to_world_m(existing_poses, T_world_from_phantom)
    if args.offset_frame == "world":
        positions_for_pca = existing_world[:, :3, 3] * 1000.0
    else:
        positions_for_pca = existing_poses[:, :3, 3]
    axes = _pca_axes(positions_for_pca)
    offsets = _candidate_offsets(axes, args.offsets_mm, args.include_depth_offsets)

    by_sequence: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_sequence.setdefault(row["sequence"], []).append(i)

    candidate_groups: list[np.ndarray] = []
    metadata: list[dict] = []
    for seq in sorted(by_sequence, key=lambda s: _natural_scan_key(Path(s))):
        base = existing_poses[by_sequence[seq]]
        base = _resample_poses(base, args.per_trajectory)
        for j, offset in enumerate(offsets):
            if args.offset_frame == "world":
                shifted_world = _poses_mm_to_world_m(base, T_world_from_phantom)
                shifted_world[:, :3, 3] += offset / 1000.0
                shifted = _poses_world_m_to_mm(shifted_world, T_world_from_phantom)
            else:
                shifted = base.copy()
                shifted[:, :3, 3] += offset
            candidate_groups.append(shifted)
            metadata.append(
                {
                    "source_sequence": seq,
                    "offset_id": j,
                    "offset_x_mm": float(offset[0]),
                    "offset_y_mm": float(offset[1]),
                    "offset_z_mm": float(offset[2]),
                }
            )

    candidates = np.concatenate(candidate_groups, axis=0)
    group_id = np.concatenate(
        [np.full(len(g), i, dtype=np.int32) for i, g in enumerate(candidate_groups)]
    )
    score, nearest_mm, axial_angle, nearest_idx = _score_candidates(
        candidates,
        existing_poses,
        position_weight=args.position_weight,
        angle_weight=args.angle_weight,
    )
    inside = _inside_ratios(args, candidates)

    group_rows: list[dict] = []
    for gid in range(len(candidate_groups)):
        mask = group_id == gid
        valid = inside[mask] >= args.min_inside
        group_rows.append(
            {
                "trajectory_id": gid,
                **metadata[gid],
                "n_poses": int(mask.sum()),
                "n_valid": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "mean_score": float(score[mask].mean()),
                "mean_nearest_existing_mm": float(nearest_mm[mask].mean()),
                "min_nearest_existing_mm": float(nearest_mm[mask].min()),
                "mean_axial_angle_deg": float(axial_angle[mask].mean()),
                "mean_inside": float(inside[mask].mean()),
                "min_inside": float(inside[mask].min()),
            }
        )

    selectable = [
        r
        for r in group_rows
        if r["valid_fraction"] >= args.min_valid_fraction and r["min_inside"] >= args.min_inside
    ]
    if len(selectable) < args.n_trajectories:
        selectable = [r for r in group_rows if r["valid_fraction"] >= args.min_valid_fraction]
    if len(selectable) < args.n_trajectories:
        raise SystemExit(
            f"only {len(selectable)} trajectories pass validity filters; lower --min-inside "
            "or --min-valid-fraction, or use smaller offsets"
        )
    selected = sorted(selectable, key=lambda r: (r["mean_score"], r["mean_inside"]), reverse=True)[
        : args.n_trajectories
    ]
    selected_ids = np.array([r["trajectory_id"] for r in selected], dtype=np.int32)
    keep = np.isin(group_id, selected_ids)
    order = np.argsort(group_id[keep], kind="stable")

    selected_poses = candidates[keep][order]
    selected_group = group_id[keep][order]
    selected_score = score[keep][order]
    selected_nearest = nearest_mm[keep][order]
    selected_angle = axial_angle[keep][order]
    selected_inside = inside[keep][order]
    selected_nearest_idx = nearest_idx[keep][order]

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        poses_cbct_mm=selected_poses,
        trajectory_id=selected_group,
        score=selected_score.astype(np.float32),
        nearest_existing_mm=selected_nearest.astype(np.float32),
        axial_angle_to_nearest_deg=selected_angle.astype(np.float32),
        inside_ratio=selected_inside.astype(np.float32),
        nearest_existing_index=selected_nearest_idx.astype(np.int32),
        selected_candidate_ids=selected_ids,
        pca_axes=axes,
        offset_frame=args.offset_frame,
        T_world_from_phantom=T_world_from_phantom,
        existing_pose_count=len(existing_poses),
    )

    stem = out.with_suffix("")
    pose_rows = []
    for i, pose in enumerate(selected_poses):
        p = pose[:3, 3]
        pose_rows.append(
            {
                "pose_index": i,
                "trajectory_id": int(selected_group[i]),
                "x_mm": float(p[0]),
                "y_mm": float(p[1]),
                "z_mm": float(p[2]),
                "score": float(selected_score[i]),
                "nearest_existing_mm": float(selected_nearest[i]),
                "axial_angle_to_nearest_deg": float(selected_angle[i]),
                "inside_ratio": float(selected_inside[i]),
                "nearest_existing_index": int(selected_nearest_idx[i]),
            }
        )
    _write_csv(stem.with_name(stem.name + "_poses.csv"), pose_rows)
    _write_csv(stem.with_name(stem.name + "_candidate_summary.csv"), group_rows)

    info = {
        "out": str(out),
        "n_selected_poses": int(len(selected_poses)),
        "selected_candidate_ids": selected_ids.tolist(),
        "mean_nearest_existing_mm": float(selected_nearest.mean()),
        "min_inside": float(selected_inside.min()),
        "mean_inside": float(selected_inside.mean()),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    stem.with_name(stem.name + "_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
