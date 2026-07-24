#!/usr/bin/env python
"""Stratified random US/pose/CBCT-reslice samples from extracted ROS bags.

The extracted ``scan*.npz`` files are lossless working copies of the image, pose and
contact streams in the corresponding ``scan*.bag`` files.  This script samples only
contact frames, covers every available scan once, and distributes any remaining samples
across distinct scans before reusing one.  CBCT is directly sampled with the fitted real-US
inner arc, perpendicular side rays, and bottom-tangent concentric outer arc.  Rows and
columns share one physical mm/px value; there is no final crop-and-resize operation.
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

from plot_script.plots_reslice.compare import _normalize_display  # noqa: E402
from reslice import pose as P  # noqa: E402
from reslice.io import load_transform_4x4, load_volume_data  # noqa: E402
from preview_us_normal_reslice_region import (  # noqa: E402
    US_SPACING_MM,
    cbct_bottom_tangent_reslice,
)


def _scan_number(path: Path) -> int:
    return int("".join(ch for ch in path.stem if ch.isdigit()))


def _sample_plan(paths: list[Path], n: int, rng: np.random.Generator) -> list[Path]:
    if n < len(paths):
        return list(rng.choice(paths, size=n, replace=False))
    plan = list(paths)
    remaining = n - len(paths)
    while remaining:
        take = min(remaining, len(paths))
        plan.extend(rng.choice(paths, size=take, replace=False).tolist())
        remaining -= take
    return plan


def _choose_frame(data: np.lib.npyio.NpzFile, used: set[int], rng: np.random.Generator) -> int:
    n_frames = len(data["images"])
    contact = (
        np.asarray(data["contact"], dtype=bool)
        if "contact" in data.files
        else np.ones(n_frames, dtype=bool)
    )
    candidates = np.flatnonzero(contact)
    candidates = np.asarray([i for i in candidates if int(i) not in used], dtype=int)
    if not len(candidates):
        raise RuntimeError("no unused contact frames available")
    return int(rng.choice(candidates))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences-dir", type=Path, required=True)
    ap.add_argument("--bag-dir", type=Path, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--placement", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--inner-arc-threshold", type=float, default=30.0)
    ap.add_argument("--us-spacing-mm", type=float, default=US_SPACING_MM)
    args = ap.parse_args()

    paths = sorted(args.sequences_dir.glob("scan*.npz"), key=_scan_number)
    if not paths:
        raise SystemExit(f"no scan*.npz in {args.sequences_dir}")
    if args.n < 1:
        raise SystemExit("--n must be positive")

    missing_bags = [p.stem for p in paths if not (args.bag_dir / f"{p.stem}.bag").exists()]
    if missing_bags:
        raise SystemExit(f"missing source bags for: {', '.join(missing_bags)}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    affine = np.asarray(
        report["phantom_centered_frame"]["recon_affine_centered_from_ijk_mm"],
        dtype=float,
    )
    volume = load_volume_data(args.volume)
    world_from_phantom = load_transform_4x4(args.placement)

    rng = np.random.default_rng(args.seed)
    plan = _sample_plan(paths, args.n, rng)
    used_by_scan: dict[str, set[int]] = {p.stem: set() for p in paths}
    records: list[dict] = []
    us_images: list[np.ndarray] = []
    cbct_images: list[np.ndarray] = []

    args.out.mkdir(parents=True, exist_ok=True)
    individual_dir = args.out / "samples"
    individual_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sample_id, sequence in enumerate(plan, start=1):
        data = np.load(sequence, allow_pickle=True)
        frame = _choose_frame(data, used_by_scan[sequence.stem], rng)
        used_by_scan[sequence.stem].add(frame)

        us = np.asarray(data["images"][frame])
        world_from_ee = np.asarray(data["poses"][frame], dtype=float)
        world_from_probe = world_from_ee @ P.T_EE_FROM_PROBE
        phantom_from_probe_mm = P.probe_pose_in_phantom_centered_mm(
            world_from_probe, world_from_phantom
        )
        cbct, geometry, reslice_debug = cbct_bottom_tangent_reslice(
            volume,
            affine,
            phantom_from_probe_mm,
            us,
            inner_arc_threshold=args.inner_arc_threshold,
            us_spacing_mm=args.us_spacing_mm,
        )

        bag = args.bag_dir / f"{sequence.stem}.bag"
        record = {
            "sample_id": sample_id,
            "bag": str(bag),
            "sequence": str(sequence),
            "frame": frame,
            "contact": True,
            "T_world_from_ee_m": world_from_ee.tolist(),
            "T_world_from_probe_m": world_from_probe.tolist(),
            "T_phantom_from_probe_mm": phantom_from_probe_mm.tolist(),
            "inner_arc_center_xy_px": np.asarray(geometry["centre"]).tolist(),
            "inner_radius_px": float(geometry["inner_radius"]),
            "outer_radius_px": float(geometry["outer_radius"]),
            "inner_arc_median_error_px": float(geometry["median_residual"]),
            "bottom_tangent_xy_px": np.asarray(geometry["bottom_tangent"]).tolist(),
            "mm_per_pixel": float(reslice_debug["mm_per_pixel"]),
            "surface_pixel_rc": reslice_debug["surface_pixel_rc"],
            "post_reslice_resize": False,
        }
        records.append(record)
        us_images.append(us)
        cbct_images.append(cbct.astype(np.float32))

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
        axes[0].imshow(_normalize_display(us), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Real US")
        axes[1].imshow(cbct, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(
            f"CBCT bottom-tangent reslice ({float(reslice_debug['mm_per_pixel']):.4f} mm/px)"
        )
        for ax in axes:
            ax.axis("off")
        xyz = world_from_ee[:3, 3]
        fig.suptitle(
            f"#{sample_id:02d} {bag.name} frame {frame} | "
            f"EE xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(individual_dir / f"{sample_id:02d}_{sequence.stem}_f{frame}.png", dpi=150)
        plt.close(fig)

    ncols = 4
    nrows = int(np.ceil(args.n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2.65 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record, us, cbct in zip(axes.ravel(), records, us_images, cbct_images):
        left = _normalize_display(us)
        right = np.clip(cbct, 0.0, 1.0)
        divider = np.ones((left.shape[0], 8), dtype=np.float32)
        pair = np.concatenate([left, divider, right], axis=1)
        ax.imshow(pair, cmap="gray", vmin=0, vmax=1)
        ax.axvline(left.shape[1] + 3.5, color="white", linewidth=1)
        xyz = np.asarray(record["T_world_from_ee_m"])[:3, 3]
        ax.set_title(
            f"#{record['sample_id']:02d} {Path(record['bag']).name}  frame {record['frame']}\n"
            f"EE xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m",
            fontsize=9,
        )
        ax.text(0.01, 0.98, "real US", color="yellow", fontsize=8,
                ha="left", va="top", transform=ax.transAxes)
        ax.text(0.99, 0.98, "CBCT reslice", color="yellow", fontsize=8,
                ha="right", va="top", transform=ax.transAxes)
    fig.suptitle(
        f"{args.n} stratified random contact samples from {len(paths)} ROS bags "
        f"(bottom-tangent, no resize; seed={args.seed})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out / "montage_20_samples.png", dpi=160)
    plt.close(fig)

    (args.out / "samples.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "inner_arc_threshold": args.inner_arc_threshold,
                "us_spacing_mm": args.us_spacing_mm,
                "geometry": "US inner arc + perpendicular rays + bottom-tangent outer arc",
                "post_reslice_resize": False,
                "sampling": "one contact frame per bag plus five additional distinct bags",
                "volume": str(args.volume),
                "report": str(args.report),
                "placement": str(args.placement),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (args.out / "samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id", "bag", "frame", "ee_x_m", "ee_y_m", "ee_z_m",
                "probe_x_mm", "probe_y_mm", "probe_z_mm",
            ],
        )
        writer.writeheader()
        for record in records:
            ee = np.asarray(record["T_world_from_ee_m"])[:3, 3]
            probe = np.asarray(record["T_phantom_from_probe_mm"])[:3, 3]
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "bag": Path(record["bag"]).name,
                    "frame": record["frame"],
                    "ee_x_m": ee[0], "ee_y_m": ee[1], "ee_z_m": ee[2],
                    "probe_x_mm": probe[0], "probe_y_mm": probe[1], "probe_z_mm": probe[2],
                }
            )

    np.savez_compressed(
        args.out / "samples_20.npz",
        us=np.stack(us_images),
        cbct=np.stack(cbct_images),
        T_world_from_ee_m=np.stack([r["T_world_from_ee_m"] for r in records]),
        T_phantom_from_probe_mm=np.stack([r["T_phantom_from_probe_mm"] for r in records]),
        scan=np.asarray([Path(r["bag"]).stem for r in records]),
        frame=np.asarray([r["frame"] for r in records], dtype=np.int32),
    )
    print(f"wrote {len(records)} samples from {len(set(Path(r['bag']).name for r in records))} bags")
    print(args.out / "montage_20_samples.png")


if __name__ == "__main__":
    main()
