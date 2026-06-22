#!/usr/bin/env python
"""Generate high-quality novel trajectories by *fanning/tilting* the probe on real scan lines.

Rationale (why tilt, not translation):
  The CBCT volume is small and the rosbag already swept most of the in-volume surface, so any
  *translation* large enough to be novel pushes the fan out of the volume. Probe *orientation* is
  the untapped axis: rocking/fanning the probe at an in-volume contact point makes the fan cut
  through new anatomy (genuinely novel slice content) while

    * staying inside the volume  (position barely moves)  -> renderable
    * staying in-distribution     (orientation = a real pose +/- a bounded tilt, never a geometric
                                   surface normal -- the real probe presses straight down)
    * staying executable          (each (line, tilt) pair is a continuous fanned sweep)

So novelty here is measured in ORIENTATION (beam-angle vs the nearest real poses), not position,
and optionally verified in CONTENT space (1 - NCC between the candidate's CBCT fan and the nearest
real pose's fan).

Output: a standard trajectory archive with key ``poses_cbct_mm`` (+ tilt / novelty / validity meta).

Example:
    apptainer/run.sh python renderer_training/generate_tilt_trajectories.py \
        --out data/trajectories/novel_tilt_valid.npz \
        --lat-tilts-deg -24 -16 -8 8 16 24 --elev-tilts-deg -12 0 12 \
        --per-line 16 --n-trajectories 16 --min-inside 0.9
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from renderer_training.analyze_pose_coverage import _natural_scan_key, load_lc2_poses

DEFAULT_REPORT = REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_REF_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_US_SPACING = 0.166112957


# --------------------------------------------------------------------------- geometry / context
def _build_context(args):
    """CBCT volume + phantom-centred affine + fitted probe fan, for validity & content scoring."""
    from lc2.forward import Context, fit_us_fan
    from reslice.io import load_volume_data

    report_path = Path(args.report)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame_key = "recon" if Path(args.volume_path).suffix.lower() == ".mhd" else "dicom"
        affine = np.array(report["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"], dtype=float)
    else:
        import SimpleITK as sitk
        from reslice.frame import affine_from_sitk
        affine = affine_from_sitk(sitk.ReadImage(str(args.volume_path)))
    volume = load_volume_data(Path(args.volume_path))
    ref = np.load(args.ref_sequence, allow_pickle=True)
    _, geom = fit_us_fan(ref["images"], ref["contact"], args.us_spacing)
    return Context(volume=volume, affine_centered=affine, geom=geom)


def _resample_poses(poses: np.ndarray, n: int) -> np.ndarray:
    """Resample a pose stack to ``n`` smooth poses (lerp position, slerp rotation)."""
    if len(poses) == n:
        return poses.copy()
    from scipy.spatial.transform import Rotation, Slerp

    src_t = np.linspace(0.0, 1.0, len(poses))
    dst_t = np.linspace(0.0, 1.0, n)
    out = np.repeat(np.eye(4)[None], n, axis=0)
    out[:, :3, 3] = np.column_stack([np.interp(dst_t, src_t, poses[:, j, 3]) for j in range(3)])
    out[:, :3, :3] = Slerp(src_t, Rotation.from_matrix(poses[:, :3, :3]))(dst_t).as_matrix()
    return out


def _tilt(pose: np.ndarray, lat_deg: float, elev_deg: float) -> np.ndarray:
    """Rock the probe about its own lateral (x) and elevation (y) axes; keep the contact position.

    Rotating in the probe frame keeps the orientation a bounded delta off a *real* pose (so the
    renderer still sees an in-distribution down-press), while the beam now cuts new anatomy.
    """
    from deepussim.geometry import rot_x, rot_y

    out = pose.copy()
    out[:3, :3] = pose[:3, :3] @ rot_x(np.deg2rad(lat_deg)) @ rot_y(np.deg2rad(elev_deg))
    return out


# --------------------------------------------------------------------------- novelty metrics
def _beam_novelty_deg(cand: np.ndarray, real_pos: np.ndarray, real_z: np.ndarray,
                      radius_mm: float) -> tuple[float, float]:
    """Min beam-angle (deg) to any real pose within ``radius_mm`` -> orientation novelty at this spot.

    Returns (novelty_deg, nearest_mm). Falls back to the single nearest real pose if none in radius.
    """
    from scipy.spatial import cKDTree

    if not hasattr(_beam_novelty_deg, "_tree") or _beam_novelty_deg._key is not real_pos:
        _beam_novelty_deg._tree = cKDTree(real_pos)
        _beam_novelty_deg._key = real_pos
    tree = _beam_novelty_deg._tree
    p = cand[:3, 3]
    z = cand[:3, 2] / (np.linalg.norm(cand[:3, 2]) + 1e-12)
    idx = tree.query_ball_point(p, radius_mm)
    nearest_mm, nidx = tree.query(p)
    if not idx:
        idx = [int(nidx)]
    zr = real_z[idx]
    dots = np.clip(zr @ z, -1.0, 1.0)
    return float(np.degrees(np.arccos(dots.max()))), float(nearest_mm)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation over the jointly valid (nonzero) fan region."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = (a != 0) & (b != 0)
    if m.sum() < 16:
        return 0.0
    av, bv = a[m] - a[m].mean(), b[m] - b[m].mean()
    denom = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / denom) if denom > 1e-12 else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "trajectories" / "novel_tilt_valid.npz")
    ap.add_argument("--lat-tilts-deg", nargs="+", type=float, default=[-24, -16, -8, 8, 16, 24],
                    help="in-plane fan angles about the probe lateral (x) axis")
    ap.add_argument("--elev-tilts-deg", nargs="+", type=float, default=[-12, 0, 12],
                    help="out-of-plane rock angles about the probe elevation (y) axis")
    ap.add_argument("--per-line", type=int, default=16, help="poses per fanned sweep (resampled real line)")
    ap.add_argument("--n-trajectories", type=int, default=16, help="number of (line, tilt) sweeps to keep")
    ap.add_argument("--radius-mm", type=float, default=8.0, help="neighbourhood for orientation-novelty")
    ap.add_argument("--min-inside", type=float, default=0.9, help="per-pose: min fraction of fan inside volume")
    ap.add_argument("--min-valid-fraction", type=float, default=0.8, help="keep sweep if this frac of poses valid")
    ap.add_argument("--content-novelty", action="store_true", default=True,
                    help="also score 1-NCC vs the nearest real pose's CBCT fan (verification)")
    ap.add_argument("--no-content-novelty", dest="content_novelty", action="store_false")
    ap.add_argument("--max-lines", type=int, default=None, help="debug: only use the first N real sequences")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF_SEQUENCE)
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING)
    args = ap.parse_args()

    existing_poses, rows = load_lc2_poses(args.lc2_dir, ["scan*_global.npz"])
    real_pos = existing_poses[:, :3, 3].copy()
    real_z = existing_poses[:, :3, 2] / (np.linalg.norm(existing_poses[:, :3, 2], axis=1, keepdims=True) + 1e-12)
    ctx = _build_context(args)

    # group real poses into per-sequence skeleton lines (in-volume, in-distribution by construction)
    by_seq: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_seq.setdefault(r["sequence"], []).append(i)
    seqs = sorted(by_seq, key=lambda s: _natural_scan_key(Path(s)))
    if args.max_lines:
        seqs = seqs[: args.max_lines]

    tilts = [(la, el) for la in args.lat_tilts_deg for el in args.elev_tilts_deg
             if not (abs(la) < 1e-6 and abs(el) < 1e-6)]

    # report the real beam-tilt spread (angle of each real beam to the mean real beam) for context
    mean_beam = real_z.mean(0); mean_beam /= np.linalg.norm(mean_beam) + 1e-12
    real_beam_off = np.degrees(np.arccos(np.clip(real_z @ mean_beam, -1, 1)))

    cand_poses: list[np.ndarray] = []
    group_id: list[int] = []
    meta: list[dict] = []
    gid = 0
    for seq in seqs:
        base = _resample_poses(existing_poses[by_seq[seq]], args.per_line)
        for (la, el) in tilts:
            for k in range(len(base)):
                cand_poses.append(_tilt(base[k], la, el))
                group_id.append(gid)
            meta.append({"source_sequence": seq, "lat_deg": la, "elev_deg": el})
            gid += 1
    cand_poses = np.asarray(cand_poses)
    group_id = np.asarray(group_id, dtype=np.int32)
    n_groups = gid
    print(f"built {len(cand_poses)} candidate poses across {n_groups} fanned sweeps "
          f"({len(seqs)} lines x {len(tilts)} tilts)", flush=True)

    inside = np.empty(len(cand_poses), dtype=np.float32)
    novelty = np.empty(len(cand_poses), dtype=np.float32)
    nearest = np.empty(len(cand_poses), dtype=np.float32)
    for i, pose in enumerate(cand_poses):
        inside[i] = ctx.fraction_inside(pose)
        novelty[i], nearest[i] = _beam_novelty_deg(pose, real_pos, real_z, args.radius_mm)
        if (i + 1) % 500 == 0:
            print(f"  scored {i + 1}/{len(cand_poses)}", flush=True)

    content = np.full(len(cand_poses), np.nan, dtype=np.float32)
    if args.content_novelty:
        from scipy.spatial import cKDTree
        tree = cKDTree(real_pos)
        real_fan_cache: dict[int, np.ndarray] = {}
        valid_idx = np.where(inside >= args.min_inside)[0]
        print(f"content-novelty on {len(valid_idx)} valid poses ...", flush=True)
        for c, i in enumerate(valid_idx):
            _, j = tree.query(cand_poses[i][:3, 3])
            if j not in real_fan_cache:
                real_fan_cache[j] = ctx.cbct_fan(existing_poses[j])
            content[i] = 1.0 - _ncc(ctx.cbct_fan(cand_poses[i]), real_fan_cache[j])
            if (c + 1) % 200 == 0:
                print(f"  content {c + 1}/{len(valid_idx)}", flush=True)

    group_rows: list[dict] = []
    for g in range(n_groups):
        mask = group_id == g
        valid = inside[mask] >= args.min_inside
        group_rows.append({
            "trajectory_id": g, **meta[g],
            "n_poses": int(mask.sum()), "n_valid": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "mean_beam_novelty_deg": float(novelty[mask].mean()),
            "mean_inside": float(inside[mask].mean()), "min_inside": float(inside[mask].min()),
            "mean_content_novelty": float(np.nanmean(content[mask])) if args.content_novelty else float("nan"),
        })

    selectable = [r for r in group_rows if r["valid_fraction"] >= args.min_valid_fraction]
    if len(selectable) < args.n_trajectories:
        raise SystemExit(f"only {len(selectable)} sweeps pass valid_fraction>={args.min_valid_fraction}; "
                         "lower --min-inside / --min-valid-fraction or reduce tilt magnitudes")
    selected = sorted(selectable, key=lambda r: r["mean_beam_novelty_deg"], reverse=True)[: args.n_trajectories]
    selected_ids = np.array([r["trajectory_id"] for r in selected], dtype=np.int32)

    # keep only the in-volume poses of the selected sweeps (per-pose validity clipping)
    keep = np.isin(group_id, selected_ids) & (inside >= args.min_inside)
    order = np.lexsort((np.arange(len(group_id))[keep], group_id[keep]))
    sel_poses = cand_poses[keep][order]
    sel_group = group_id[keep][order]
    sel_inside = inside[keep][order]
    sel_novel = novelty[keep][order]
    sel_near = nearest[keep][order]
    sel_content = content[keep][order]

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, poses_cbct_mm=sel_poses, trajectory_id=sel_group,
        inside_ratio=sel_inside.astype(np.float32),
        beam_novelty_deg=sel_novel.astype(np.float32),
        nearest_existing_mm=sel_near.astype(np.float32),
        content_novelty=sel_content.astype(np.float32),
        selected_candidate_ids=selected_ids, existing_pose_count=len(existing_poses),
        real_beam_offset_deg=real_beam_off.astype(np.float32),
    )

    stem = out.with_suffix("")
    _write_csv(stem.with_name(stem.name + "_poses.csv"), [
        {"pose_index": i, "trajectory_id": int(sel_group[i]),
         "x_mm": float(sel_poses[i, 0, 3]), "y_mm": float(sel_poses[i, 1, 3]), "z_mm": float(sel_poses[i, 2, 3]),
         "inside_ratio": float(sel_inside[i]), "beam_novelty_deg": float(sel_novel[i]),
         "nearest_existing_mm": float(sel_near[i]), "content_novelty": float(sel_content[i])}
        for i in range(len(sel_poses))])
    _write_csv(stem.with_name(stem.name + "_candidate_summary.csv"), group_rows)

    info = {
        "out": str(out), "method": "tilt-fan on real scan lines, per-pose in-volume clipped",
        "n_selected_poses": int(len(sel_poses)), "n_selected_sweeps": int(len(selected_ids)),
        "tilts_deg": tilts,
        "real_beam_offset_deg": {"mean": float(real_beam_off.mean()), "p95": float(np.percentile(real_beam_off, 95))},
        "selected_mean_beam_novelty_deg": float(sel_novel.mean()),
        "selected_mean_inside": float(sel_inside.mean()), "selected_min_inside": float(sel_inside.min()),
        "selected_mean_content_novelty": float(np.nanmean(sel_content)) if args.content_novelty else None,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    stem.with_name(stem.name + "_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
