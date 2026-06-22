#!/usr/bin/env python
"""Generate new trajectory poses that complement existing rosbag/LC2 coverage.

This script builds candidate surface trajectories with the existing deepussim samplers, scores
each line/curve by novelty against LC2-refined real poses, filters candidates whose CBCT fan is
mostly outside the volume, and writes a trajectory file compatible with existing tooling:

    data/trajectories/novel_surface_curves.npz  (key: poses_cbct_mm)

The default is conservative: surface-curves leave the scanned patch spatially, but rotations are
borrowed from nearest real contact/LC2 poses by the existing sampler. So the renderer sees new
CBCT slice locations without a sudden out-of-distribution probe orientation jump.

Example:
    python renderer_training/generate_novel_trajectories.py \
        --out data/trajectories/novel_surface_curves.npz \
        --n-trajectories 6 --candidates 15 --per-line 32
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

from renderer_training.analyze_pose_coverage import load_lc2_poses


DEFAULT_REPORT = REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_REF_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_MESH = REPO_ROOT / "data" / "cbct_20260612" / "phantom_surface.stl"
DEFAULT_US_SPACING = 0.166112957


def _build_candidates(args, mesh, contact_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from deepussim.pipeline.sampling import (
        contact_raster_ee,
        side_anchor_curves,
        surface_curves_from_points,
        surface_raster,
    )

    if args.mode == "surface-curves":
        curves = side_anchor_curves(
            mesh,
            contact_poses,
            n_curves=args.candidates,
            n_anchors=args.curve_anchors,
            reach=args.curve_reach,
            along_frac=args.span_frac,
        )
        poses = np.asarray(
            surface_curves_from_points(
                mesh,
                curves,
                contact_poses=contact_poses,
                points_per_curve=args.per_line,
                standoff_mm=args.standoff_mm,
            ),
            dtype=float,
        )
    elif args.mode == "contact-raster":
        poses = np.asarray(
            contact_raster_ee(
                mesh,
                contact_poses,
                n_lines=args.candidates,
                n_per_line=args.per_line,
                expand=args.expand,
                standoff_mm=args.standoff_mm,
            ),
            dtype=float,
        )
    else:
        poses = np.asarray(
            surface_raster(
                mesh,
                axis=args.sweep_axis,
                span_frac=args.span_frac,
                cross_frac=args.cross_frac,
                n_lines=args.candidates,
                n_per_line=args.per_line,
                standoff_mm=args.standoff_mm,
            ),
            dtype=float,
        )
    group_id = np.repeat(np.arange(args.candidates, dtype=np.int32), args.per_line)
    if len(group_id) != len(poses):
        raise RuntimeError(f"candidate grouping mismatch: {len(group_id)} ids for {len(poses)} poses")
    return poses, group_id


def _angle_to_existing_deg(candidate_axes: np.ndarray, existing_axes: np.ndarray, nearest_idx: np.ndarray) -> np.ndarray:
    ref = existing_axes[nearest_idx]
    dot = np.sum(candidate_axes * ref, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


def _score_candidates(
    candidate_poses: np.ndarray,
    existing_poses: np.ndarray,
    position_weight: float,
    angle_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    existing_pos = existing_poses[:, :3, 3]
    candidate_pos = candidate_poses[:, :3, 3]
    tree = cKDTree(existing_pos)
    nearest_mm, nearest_idx = tree.query(candidate_pos)
    axial_angle = _angle_to_existing_deg(candidate_poses[:, :3, 2], existing_poses[:, :3, 2], nearest_idx)
    score = position_weight * nearest_mm + angle_weight * axial_angle
    return score, nearest_mm, axial_angle, nearest_idx.astype(np.int32)


def _inside_ratios(args, poses: np.ndarray) -> np.ndarray:
    from lc2.forward import Context, fit_us_fan
    from reslice.io import load_volume_data

    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame_key = "recon" if Path(args.volume_path).suffix.lower() == ".mhd" else "dicom"
        affine = np.array(
            report["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"],
            dtype=float,
        )
    else:
        import SimpleITK as sitk
        from reslice.frame import affine_from_sitk

        # Default build_frame uses phantom_origin_lps_mm = [0, 0, 0], so the LPS affine is
        # already the phantom-centred affine when the report has not been materialised.
        affine = affine_from_sitk(sitk.ReadImage(str(args.volume_path)))
    volume = load_volume_data(Path(args.volume_path))
    ref = np.load(args.ref_sequence, allow_pickle=True)
    _, geom = fit_us_fan(ref["images"], ref["contact"], args.us_spacing)
    ctx = Context(volume=volume, affine_centered=affine, geom=geom)
    return np.asarray([ctx.fraction_inside(p) for p in poses], dtype=np.float32)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "trajectories" / "novel_surface_curves.npz")
    ap.add_argument("--mode", choices=["surface-curves", "surface-raster", "contact-raster"], default="surface-curves")
    ap.add_argument("--n-trajectories", type=int, default=6, help="number of candidate lines/curves to keep")
    ap.add_argument("--candidates", type=int, default=15, help="number of candidate lines/curves to generate")
    ap.add_argument("--per-line", type=int, default=32)
    ap.add_argument("--standoff-mm", type=float, default=2.0)
    ap.add_argument("--span-frac", type=float, default=1.0)
    ap.add_argument("--cross-frac", type=float, default=0.8)
    ap.add_argument("--curve-reach", type=float, default=2.0)
    ap.add_argument("--curve-anchors", type=int, default=6)
    ap.add_argument("--expand", type=float, default=1.3)
    ap.add_argument("--sweep-axis", type=int, default=0)
    ap.add_argument("--min-inside", type=float, default=0.85)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--angle-weight", type=float, default=0.25, help="score mm-equivalent per axial degree")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF_SEQUENCE)
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING)
    args = ap.parse_args()

    import trimesh

    existing_poses, _ = load_lc2_poses(args.lc2_dir, ["scan*_global.npz"])
    mesh = trimesh.load(args.mesh, process=False)
    candidates, group_id = _build_candidates(args, mesh, existing_poses)
    score, nearest_mm, axial_angle, nearest_idx = _score_candidates(
        candidates,
        existing_poses,
        position_weight=args.position_weight,
        angle_weight=args.angle_weight,
    )
    inside = _inside_ratios(args, candidates)

    group_rows: list[dict] = []
    for gid in sorted(np.unique(group_id)):
        mask = group_id == gid
        valid = inside[mask] >= args.min_inside
        group_rows.append(
            {
                "trajectory_id": int(gid),
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
        row
        for row in group_rows
        if row["valid_fraction"] >= 0.8 and row["min_inside"] >= args.min_inside
    ]
    if len(selectable) < args.n_trajectories:
        selectable = [row for row in group_rows if row["valid_fraction"] >= 0.8]
    if len(selectable) < args.n_trajectories:
        selectable = group_rows
    selected = sorted(selectable, key=lambda r: (r["mean_score"], r["mean_inside"]), reverse=True)[
        : args.n_trajectories
    ]
    selected_ids = np.array([row["trajectory_id"] for row in selected], dtype=np.int32)
    keep = np.isin(group_id, selected_ids)

    selected_poses = candidates[keep]
    selected_group = group_id[keep]
    order = np.argsort(selected_group, kind="stable")
    selected_poses = selected_poses[order]
    selected_group = selected_group[order]
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
        nearest_existing_index=selected_nearest_idx,
        selected_candidate_ids=selected_ids,
        mode=args.mode,
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
        "mode": args.mode,
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
