#!/usr/bin/env python
"""Generate LC2-refined CBCT/US training pairs for renderer experiments.

This script freezes the data product we want after registration:

    {CBCT display-sector at multi-frame LC2 refined pose, raw real US frame}

The CBCT sector is generated with the same `cbct_sector_zoom(...)` display path used by
`plot_script.plots_lc2.compare` for the figures that were visually checked. Do not use
`Context.cbct_fan()` / `unwrap_fan()` here: that polar-grid path is useful for LC2 scoring,
but it did not match the display-space pairs we need for renderer training.

It also writes the legacy CUT pools (`source_cbct.npz` and `target_us.npz`) so the existing
unpaired renderer training can consume the same LC2-refined data without changes.

Example:
    python renderer_training/pair_generation.py \
        --sequences data/sequences/scan*.npz \
        --lc2-dir data/lc2 \
        --out data/renderer_lc2_pairs
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

DEFAULT_REPORT = REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_PLACEMENT = REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_REF_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_US_SPACING = 0.166112957


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


def _lc2_path_for(sequence: Path, lc2_dir: Path) -> Path:
    return lc2_dir / f"{sequence.stem}_global.npz"


def _load_cbct_frame(volume_path: Path, report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from reslice.io import load_volume_data

    volume = load_volume_data(Path(volume_path))
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame_key = "recon" if Path(volume_path).suffix.lower() == ".mhd" else "dicom"
        affine = np.array(
            report["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"],
            dtype=float,
        )
    else:
        import SimpleITK as sitk
        from reslice.frame import affine_from_sitk

        print(f"warning: missing {report_path}; using image affine fallback", flush=True)
        affine = affine_from_sitk(sitk.ReadImage(str(volume_path)))
    return volume, affine


def _append_pairs_for_sequence(
    sequence: Path,
    lc2_npz: Path,
    volume: np.ndarray,
    affine_centered: np.ndarray,
    rows: list[dict],
    cbct_images: list[np.ndarray],
    us_images: list[np.ndarray],
    poses: list[np.ndarray],
) -> None:
    from plot_script.plots_reslice.compare import cbct_sector_zoom

    lc2 = np.load(lc2_npz)
    frame_index = np.asarray(_pick(lc2, "frame_index", "indices"), dtype=int)
    refined_poses = np.asarray(_pick(lc2, "global_refined_poses", "refined_poses"), dtype=float)
    lc2_before = np.asarray(_pick(lc2, "global_lc2_before", "lc2_before"), dtype=float)
    lc2_after = np.asarray(_pick(lc2, "global_lc2_after", "lc2_after"), dtype=float)
    inside_before = np.asarray(_pick(lc2, "global_inside_before", "inside_before"), dtype=float)
    inside_after = np.asarray(_pick(lc2, "global_inside_after", "inside_after"), dtype=float)

    if len(refined_poses) != len(frame_index):
        raise RuntimeError(f"pose/frame length mismatch in {lc2_npz}: {len(refined_poses)} vs {len(frame_index)}")

    seq = np.load(sequence, allow_pickle=True)
    raw_images = seq["images"]
    n_frames = len(frame_index)
    for i, fr in enumerate(frame_index):
        us = np.asarray(raw_images[int(fr)], dtype=np.float32)
        cbct = cbct_sector_zoom(volume, affine_centered, refined_poses[i], us.shape[:2]).astype(np.float32)
        cbct_images.append(cbct)
        us_images.append(us)
        poses.append(refined_poses[i].astype(np.float64))
        rows.append(
            {
                "pair_index": len(rows),
                "sequence": sequence.name,
                "frame_index": int(fr),
                "lc2_before": float(lc2_before[i]),
                "lc2_after": float(lc2_after[i]),
                "inside_before": float(inside_before[i]),
                "inside_after": float(inside_after[i]),
                "space": "display_sector",
                "cbct_source": "plot_script.plots_reslice.compare.cbct_sector_zoom",
                "us_source": "raw_sequence_frame",
                "lc2_npz": str(lc2_npz),
            }
        )
    print(f"{sequence.name}: added {n_frames} display-sector LC2-refined pairs from {lc2_npz.name}")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", nargs="+", type=Path,
                    default=sorted((REPO_ROOT / "data" / "sequences").glob("scan*.npz"), key=_natural_scan_key))
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--world-from-phantom", type=Path, default=DEFAULT_PLACEMENT)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF_SEQUENCE,
                    help="sequence used to fit the shared US fan geometry")
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "renderer_lc2_pairs")
    ap.add_argument("--allow-missing-lc2", action="store_true",
                    help="skip sequences without <stem>_global.npz instead of failing")
    args = ap.parse_args()

    sequences = sorted([Path(p) for p in args.sequences], key=_natural_scan_key)
    if not sequences:
        raise SystemExit("no sequences provided")

    missing = [p for p in sequences if not _lc2_path_for(p, args.lc2_dir).exists()]
    if missing and not args.allow_missing_lc2:
        msg = "\n".join(f"  {p} -> {_lc2_path_for(p, args.lc2_dir)}" for p in missing)
        raise SystemExit(f"missing LC2 global files:\n{msg}")

    volume, affine_centered = _load_cbct_frame(args.volume_path, args.report)

    rows: list[dict] = []
    cbct_images: list[np.ndarray] = []
    us_images: list[np.ndarray] = []
    poses: list[np.ndarray] = []

    for sequence in sequences:
        lc2_npz = _lc2_path_for(sequence, args.lc2_dir)
        if not lc2_npz.exists():
            print(f"skip {sequence.name}: missing {lc2_npz}")
            continue
        _append_pairs_for_sequence(
            sequence=sequence,
            lc2_npz=lc2_npz,
            volume=volume,
            affine_centered=affine_centered,
            rows=rows,
            cbct_images=cbct_images,
            us_images=us_images,
            poses=poses,
        )

    if not rows:
        raise SystemExit("no pairs generated")

    cbct = np.stack(cbct_images).astype(np.float32)
    us = np.stack(us_images).astype(np.float32)
    refined_poses = np.stack(poses).astype(np.float64)
    frame_index = np.array([r["frame_index"] for r in rows], dtype=np.int32)
    sequence = np.array([r["sequence"] for r in rows])
    lc2_before = np.array([r["lc2_before"] for r in rows], dtype=np.float32)
    lc2_after = np.array([r["lc2_after"] for r in rows], dtype=np.float32)
    inside_before = np.array([r["inside_before"] for r in rows], dtype=np.float32)
    inside_after = np.array([r["inside_after"] for r in rows], dtype=np.float32)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "pairs.npz",
        cbct=cbct,
        us=us,
        refined_poses=refined_poses,
        sequence=sequence,
        frame_index=frame_index,
        lc2_before=lc2_before,
        lc2_after=lc2_after,
        inside_before=inside_before,
        inside_after=inside_after,
    )
    np.savez_compressed(out / "source_cbct.npz", images=cbct)
    np.savez_compressed(out / "target_us.npz", images=us)
    _write_manifest(out / "manifest.csv", rows)

    print(f"wrote {out / 'pairs.npz'}")
    print(f"wrote {out / 'source_cbct.npz'} + {out / 'target_us.npz'}")
    print(
        f"pairs={len(rows)} display_shape={cbct.shape[1:]} "
        f"LC2 {lc2_before.mean():.3f}->{lc2_after.mean():.3f} "
        f"inside {inside_before.mean() * 100:.1f}%->{inside_after.mean() * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
