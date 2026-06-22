#!/usr/bin/env python
"""Analyze coverage of existing rosbag/LC2 probe trajectories.

The renderer scale-up plan needs poses we did not already collect. This script turns the
existing LC2-refined poses into a compact coverage product:

    coverage_existing.npz   pose stack + per-pose metrics
    coverage_summary.csv    one row per sequence
    coverage_poses.csv      one row per pose

Example:
    python renderer_training/analyze_pose_coverage.py \
        --lc2-dir data/lc2 \
        --out data/trajectory_coverage/existing
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


def _natural_scan_key(path: Path) -> tuple[int, str]:
    stem = path.stem.replace("_global", "")
    if stem.startswith("scan") and stem[4:].isdigit():
        return int(stem[4:]), stem
    return 10**9, stem


def _pick(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"none of {names} found; archive has {data.files}")


def load_lc2_poses(lc2_dir: Path, patterns: list[str]) -> tuple[np.ndarray, list[dict]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(lc2_dir.glob(pattern))
    paths = sorted(set(paths), key=_natural_scan_key)
    if not paths:
        raise SystemExit(f"no LC2 files matched {patterns} in {lc2_dir}")

    poses: list[np.ndarray] = []
    rows: list[dict] = []
    for path in paths:
        data = np.load(path)
        p = np.asarray(_pick(data, "global_refined_poses", "refined_poses"), dtype=float)
        idx = np.asarray(_pick(data, "frame_index", "indices"), dtype=int)
        lc2_before = np.asarray(_pick(data, "global_lc2_before", "lc2_before"), dtype=float)
        lc2_after = np.asarray(_pick(data, "global_lc2_after", "lc2_after"), dtype=float)
        inside_before = np.asarray(_pick(data, "global_inside_before", "inside_before"), dtype=float)
        inside_after = np.asarray(_pick(data, "global_inside_after", "inside_after"), dtype=float)
        if len(p) != len(idx):
            raise ValueError(f"{path}: poses/frame_index length mismatch")
        for i in range(len(p)):
            poses.append(p[i])
            rows.append(
                {
                    "sequence": path.name.replace("_global.npz", ".npz"),
                    "lc2_file": str(path),
                    "frame_index": int(idx[i]),
                    "lc2_before": float(lc2_before[i]),
                    "lc2_after": float(lc2_after[i]),
                    "inside_before": float(inside_before[i]),
                    "inside_after": float(inside_after[i]),
                }
            )
    return np.asarray(poses, dtype=float), rows


def _nearest_metrics(points: np.ndarray, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    k = min(2, len(points))
    dist, _ = tree.query(points, k=k)
    nearest = dist[:, 1] if k == 2 else np.full(len(points), np.nan)
    density = np.asarray([len(x) - 1 for x in tree.query_ball_point(points, radius_mm)], dtype=int)
    return nearest, density


def _bin_ids(points: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    ijk = np.floor((points - lo) / span * bins).astype(int)
    ijk = np.clip(ijk, 0, bins - 1)
    flat = ijk[:, 0] * bins * bins + ijk[:, 1] * bins + ijk[:, 2]
    return ijk, flat, np.stack([lo, hi], axis=0)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--patterns", nargs="+", default=["scan*_global.npz"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "trajectory_coverage" / "existing")
    ap.add_argument("--density-radius-mm", type=float, default=10.0)
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    poses, rows = load_lc2_poses(args.lc2_dir, args.patterns)
    points = poses[:, :3, 3]
    lateral = poses[:, :3, 0]
    axial = poses[:, :3, 2]
    nearest, density = _nearest_metrics(points, args.density_radius_mm)
    bin_ijk, bin_id, bounds = _bin_ids(points, args.bins)

    pose_rows: list[dict] = []
    density_key = f"density_r{args.density_radius_mm:g}mm"
    for i, row in enumerate(rows):
        p = points[i]
        pose_rows.append(
            {
                **row,
                "pose_index": i,
                "x_mm": float(p[0]),
                "y_mm": float(p[1]),
                "z_mm": float(p[2]),
                "axial_x": float(axial[i, 0]),
                "axial_y": float(axial[i, 1]),
                "axial_z": float(axial[i, 2]),
                "lateral_x": float(lateral[i, 0]),
                "lateral_y": float(lateral[i, 1]),
                "lateral_z": float(lateral[i, 2]),
                "nearest_existing_mm": float(nearest[i]),
                density_key: int(density[i]),
                "bin_x": int(bin_ijk[i, 0]),
                "bin_y": int(bin_ijk[i, 1]),
                "bin_z": int(bin_ijk[i, 2]),
                "bin_id": int(bin_id[i]),
            }
        )

    summary_rows: list[dict] = []
    for seq in sorted({r["sequence"] for r in pose_rows}, key=lambda s: _natural_scan_key(Path(s))):
        sel = [r for r in pose_rows if r["sequence"] == seq]
        P = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in sel], dtype=float)
        summary_rows.append(
            {
                "sequence": seq,
                "n_poses": len(sel),
                "x_min": float(P[:, 0].min()),
                "x_max": float(P[:, 0].max()),
                "y_min": float(P[:, 1].min()),
                "y_max": float(P[:, 1].max()),
                "z_min": float(P[:, 2].min()),
                "z_max": float(P[:, 2].max()),
                "mean_lc2_before": float(np.mean([r["lc2_before"] for r in sel])),
                "mean_lc2_after": float(np.mean([r["lc2_after"] for r in sel])),
                "mean_inside_after": float(np.mean([r["inside_after"] for r in sel])),
                "occupied_bins": len({r["bin_id"] for r in sel}),
            }
        )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "coverage_poses.csv", pose_rows)
    _write_csv(out / "coverage_summary.csv", summary_rows)
    np.savez_compressed(
        out / "coverage_existing.npz",
        poses=poses,
        positions=points,
        axial=axial,
        lateral=lateral,
        nearest_existing_mm=nearest,
        density=density,
        bin_ijk=bin_ijk,
        bin_id=bin_id,
        bounds_mm=bounds,
        density_radius_mm=float(args.density_radius_mm),
        bins=int(args.bins),
        sequence=np.array([r["sequence"] for r in rows]),
        frame_index=np.array([r["frame_index"] for r in rows], dtype=np.int32),
    )

    info = {
        "n_poses": int(len(poses)),
        "n_sequences": int(len(summary_rows)),
        "occupied_bins": int(len(np.unique(bin_id))),
        "total_bins": int(args.bins**3),
        "position_min_mm": points.min(axis=0).round(3).tolist(),
        "position_max_mm": points.max(axis=0).round(3).tolist(),
        "median_nearest_existing_mm": float(np.nanmedian(nearest)),
        "density_radius_mm": float(args.density_radius_mm),
    }
    (out / "coverage_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
