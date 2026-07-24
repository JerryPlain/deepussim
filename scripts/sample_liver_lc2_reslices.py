#!/usr/bin/env python
"""Select liver-positive frames across ROS bags and compare robot/LC2 CBCT reslices."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lc2.forward import Context, FrameTargets  # noqa: E402
from lc2.register import register_frame  # noqa: E402
from lc2.us_fan import FanFit, unwrap_fan  # noqa: E402
from plot_script.plots_reslice.compare import _normalize_display  # noqa: E402
from reslice import pose as P  # noqa: E402
from reslice.fan import ProbeGeometry, reslice_fan  # noqa: E402
from reslice.io import load_transform_4x4, load_volume_data  # noqa: E402
from preview_us_normal_reslice_region import (  # noqa: E402
    US_SPACING_MM,
    cbct_bottom_tangent_reslice,
    display_pixel_to_plane_point,
    fit_inner_arc,
    region_mask,
    sample_cbct_on_us_grid,
)

LIVER_ID = 2


def _scan_number(path: Path) -> int:
    return int("".join(ch for ch in path.stem if ch.isdigit()))


def _fan_models(
    geometry: dict[str, np.ndarray | float], spacing_mm: float
) -> tuple[FanFit, ProbeGeometry]:
    centre = np.asarray(geometry["centre"], dtype=float)
    fov_deg = float(
        np.rad2deg(float(geometry["angle_right"]) - float(geometry["angle_left"]))
    )
    fan = FanFit(
        apex_px=(float(centre[0]), float(centre[1])),
        r0_px=float(geometry["inner_radius"]),
        r1_px=float(geometry["outer_radius"]),
        fov_deg=fov_deg,
        resid_px=float(geometry["median_residual"]),
    )
    geom = ProbeGeometry(
        radius_mm=fan.r0_px * spacing_mm,
        fov_deg=fan.fov_deg,
        depth_mm=(fan.r1_px - fan.r0_px) * spacing_mm,
        n_lat=256,
        n_ax=512,
    )
    return fan, geom


def _pose_from_data(data: np.lib.npyio.NpzFile, frame: int, placement: np.ndarray) -> np.ndarray:
    world_from_ee = np.asarray(data["poses"][frame], dtype=float)
    world_from_probe = world_from_ee @ P.T_EE_FROM_PROBE
    return P.probe_pose_in_phantom_centered_mm(world_from_probe, placement)


def _candidate_pool(
    paths: list[Path],
    labels: np.ndarray,
    affine: np.ndarray,
    placement: np.ndarray,
    geometry: ProbeGeometry,
    candidates_per_scan: int,
) -> list[dict]:
    candidates: list[dict] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            contact = np.asarray(data["contact"], dtype=bool)
            indices = np.flatnonzero(contact)
            selected = indices[
                np.linspace(
                    0,
                    len(indices) - 1,
                    min(candidates_per_scan, len(indices)),
                    dtype=int,
                )
            ]
            scan_rows = []
            for frame in np.unique(selected):
                pose = _pose_from_data(data, int(frame), placement)
                label_fan = reslice_fan(labels, affine, pose, geometry, order=0)
                coverage = float(np.mean(np.rint(label_fan).astype(np.int16) == LIVER_ID))
                row = {"sequence": path, "frame": int(frame), "liver_cov_polar": coverage}
                candidates.append(row)
                scan_rows.append(row)
        maximum = max(row["liver_cov_polar"] for row in scan_rows)
        print(
            f"[candidate] {path.stem}: {len(scan_rows)} checked, "
            f"max liver={100.0 * maximum:.1f}%",
            flush=True,
        )
    return candidates


def _select_plan(
    candidates: list[dict],
    paths: list[Path],
    n: int,
    min_liver_cov: float,
    rng: np.random.Generator,
) -> list[dict]:
    selected: list[dict] = []
    used: set[tuple[str, int]] = set()

    # First cover every scan that has a liver-positive candidate. Randomise within its
    # eight strongest candidates so the result is liver-rich without always taking maxima.
    for path in paths:
        eligible = [
            row
            for row in candidates
            if row["sequence"] == path and row["liver_cov_polar"] >= min_liver_cov
        ]
        eligible.sort(key=lambda row: row["liver_cov_polar"], reverse=True)
        if eligible:
            row = eligible[int(rng.integers(0, min(8, len(eligible))))]
            selected.append(row)
            used.add((path.stem, int(row["frame"])))
        if len(selected) == n:
            return selected

    # Fill the remaining slots with liver-positive candidates, avoiding near-duplicate
    # frames from the same scan.
    remaining = [row for row in candidates if row["liver_cov_polar"] >= min_liver_cov]
    rng.shuffle(remaining)
    for row in remaining:
        key = (row["sequence"].stem, int(row["frame"]))
        if key in used:
            continue
        same_scan = [x for x in selected if x["sequence"] == row["sequence"]]
        if any(abs(int(x["frame"]) - int(row["frame"])) < 12 for x in same_scan):
            continue
        selected.append(row)
        used.add(key)
        if len(selected) == n:
            return selected

    if len(selected) < n:
        raise RuntimeError(
            f"only {len(selected)} diverse liver-positive samples found; "
            f"lower --min-liver-cov or increase --candidates-per-scan"
        )
    return selected


def _direct_liver_mask(
    labels: np.ndarray,
    affine: np.ndarray,
    pose: np.ndarray,
    shape: tuple[int, int],
    geometry: dict[str, np.ndarray | float],
    surface_pixel_rc: list[float],
    spacing_mm: float,
) -> np.ndarray:
    plane = P.plane_from_probe_pose(pose, "probe-xz", 0.0)
    surface_point = display_pixel_to_plane_point(
        np.asarray(surface_pixel_rc, dtype=float), plane, shape
    )
    sampled, valid, _ = sample_cbct_on_us_grid(
        labels,
        affine,
        plane,
        surface_point,
        shape,
        geometry,
        mm_per_pixel=spacing_mm,
        order=0,
    )
    return (
        (np.rint(sampled).astype(np.int16) == LIVER_ID)
        & valid
        & region_mask(shape, geometry)
    )


def _rgb_with_liver(image: np.ndarray, liver: np.ndarray | None = None) -> np.ndarray:
    gray = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if liver is not None and np.any(liver):
        edge = liver ^ ndimage.binary_erosion(liver, iterations=2)
        rgb[edge] = (1.0, 0.35, 0.0)
    return rgb


def _make_triptych(us: np.ndarray, initial: np.ndarray, refined: np.ndarray, liver: np.ndarray):
    panels = [
        _rgb_with_liver(_normalize_display(us), liver),
        _rgb_with_liver(initial),
        _rgb_with_liver(refined, liver),
    ]
    divider = np.ones((us.shape[0], 7, 3), dtype=np.float32)
    return np.concatenate([panels[0], divider, panels[1], divider, panels[2]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences-dir", type=Path, required=True)
    ap.add_argument("--bag-dir", type=Path, required=True)
    ap.add_argument("--volume", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--placement", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--reference-scan", default="scan5")
    ap.add_argument("--reference-frame", type=int, default=126)
    ap.add_argument("--candidates-per-scan", type=int, default=28)
    ap.add_argument("--min-liver-cov", type=float, default=0.10)
    ap.add_argument("--us-spacing-mm", type=float, default=US_SPACING_MM)
    ap.add_argument("--inner-arc-threshold", type=float, default=30.0)
    ap.add_argument("--max-trans-mm", type=float, default=8.0)
    ap.add_argument("--max-rot-deg", type=float, default=8.0)
    ap.add_argument("--maxiter", type=int, default=50)
    args = ap.parse_args()

    paths = sorted(args.sequences_dir.glob("scan*.npz"), key=_scan_number)
    if not paths:
        raise SystemExit(f"no scan*.npz under {args.sequences_dir}")
    missing = [path.stem for path in paths if not (args.bag_dir / f"{path.stem}.bag").exists()]
    if missing:
        raise SystemExit(f"missing source bags: {', '.join(missing)}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    affine = np.asarray(
        report["phantom_centered_frame"]["recon_affine_centered_from_ijk_mm"], dtype=float
    )
    volume = load_volume_data(args.volume)
    labels = load_volume_data(args.labels)
    placement = load_transform_4x4(args.placement)

    reference_path = args.sequences_dir / f"{args.reference_scan}.npz"
    with np.load(reference_path, allow_pickle=True) as reference:
        reference_us = np.asarray(reference["images"][args.reference_frame])
    reference_display = fit_inner_arc(reference_us, threshold=args.inner_arc_threshold)
    _, reference_polar = _fan_models(reference_display, args.us_spacing_mm)

    candidates = _candidate_pool(
        paths,
        labels,
        affine,
        placement,
        reference_polar,
        args.candidates_per_scan,
    )
    rng = np.random.default_rng(args.seed)
    plan = _select_plan(candidates, paths, args.n, args.min_liver_cov, rng)
    rng.shuffle(plan)
    print(
        f"[plan] selected {len(plan)} liver-positive frames from "
        f"{len(set(row['sequence'].stem for row in plan))} bags",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    triptychs: list[np.ndarray] = []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sample_id, choice in enumerate(plan, start=1):
        sequence_path = Path(choice["sequence"])
        frame = int(choice["frame"])
        with np.load(sequence_path, allow_pickle=True) as data:
            us = np.asarray(data["images"][frame])
            world_from_ee = np.asarray(data["poses"][frame], dtype=float)
            init_pose = _pose_from_data(data, frame, placement)

        display_geometry = fit_inner_arc(us, threshold=args.inner_arc_threshold)
        fan, polar_geometry = _fan_models(display_geometry, args.us_spacing_mm)
        us_polar = unwrap_fan(us, fan, n_ax=polar_geometry.n_ax, n_lat=polar_geometry.n_lat)
        target = FrameTargets(index=frame, us_polar=us_polar, init_pose_mm=init_pose)
        context = Context(
            volume=volume, affine_centered=affine, geom=polar_geometry, frames=[target]
        )
        result = register_frame(
            context,
            target,
            max_trans_mm=args.max_trans_mm,
            max_rot_deg=args.max_rot_deg,
            maxiter=args.maxiter,
        )
        refined_pose = np.asarray(result["refined_pose"], dtype=float)

        initial_cbct, initial_geometry, initial_debug = cbct_bottom_tangent_reslice(
            volume,
            affine,
            init_pose,
            us,
            inner_arc_threshold=args.inner_arc_threshold,
            us_spacing_mm=args.us_spacing_mm,
        )
        refined_cbct, refined_geometry, refined_debug = cbct_bottom_tangent_reslice(
            volume,
            affine,
            refined_pose,
            us,
            inner_arc_threshold=args.inner_arc_threshold,
            us_spacing_mm=args.us_spacing_mm,
        )
        initial_liver = _direct_liver_mask(
            labels,
            affine,
            init_pose,
            us.shape[:2],
            initial_geometry,
            initial_debug["surface_pixel_rc"],
            args.us_spacing_mm,
        )
        refined_liver = _direct_liver_mask(
            labels,
            affine,
            refined_pose,
            us.shape[:2],
            refined_geometry,
            refined_debug["surface_pixel_rc"],
            args.us_spacing_mm,
        )
        liver_before = float(initial_liver.mean())
        liver_after = float(refined_liver.mean())

        origin_shift = float(np.linalg.norm(refined_pose[:3, 3] - init_pose[:3, 3]))
        relative_rotation = refined_pose[:3, :3] @ init_pose[:3, :3].T
        rotation_deg = float(np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude()))
        record = {
            "sample_id": sample_id,
            "bag": str(args.bag_dir / f"{sequence_path.stem}.bag"),
            "sequence": str(sequence_path),
            "frame": frame,
            "selection_liver_cov_polar": float(choice["liver_cov_polar"]),
            "liver_cov_initial_display": liver_before,
            "liver_cov_refined_display": liver_after,
            "lc2_before": float(result["lc2_before"]),
            "lc2_after": float(result["lc2_after"]),
            "inside_before": float(result["inside_before"]),
            "inside_after": float(result["inside_after"]),
            "correction_rx_ry_rz_deg_tx_ty_tz_mm": result["params"],
            "probe_origin_shift_mm": origin_shift,
            "relative_rotation_deg": rotation_deg,
            "T_world_from_ee_m": world_from_ee.tolist(),
            "init_T_phantom_from_probe_mm": init_pose.tolist(),
            "refined_T_phantom_from_probe_mm": refined_pose.tolist(),
        }
        records.append(record)
        triptychs.append(_make_triptych(us, initial_cbct, refined_cbct, refined_liver))

        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.45))
        axes[0].imshow(_normalize_display(us), cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(initial_cbct, cmap="gray", vmin=0, vmax=1)
        axes[2].imshow(refined_cbct, cmap="gray", vmin=0, vmax=1)
        for ax in (axes[0], axes[2]):
            if refined_liver.any():
                ax.contour(refined_liver.astype(float), [0.5], colors="#ff5900", linewidths=1.0)
        axes[0].set_title("Real US + refined liver projection")
        axes[1].set_title(f"Robot pose | LC2 {float(result['lc2_before']):.3f}")
        axes[2].set_title(
            f"LC2 refined {float(result['lc2_after']):.3f} | liver {100*liver_after:.1f}%"
        )
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"#{sample_id:02d} {sequence_path.stem}.bag frame {frame} | "
            f"delta={origin_shift:.1f} mm / {rotation_deg:.1f} deg | "
            f"inside {100*float(result['inside_before']):.0f}% -> "
            f"{100*float(result['inside_after']):.0f}%",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            samples_dir / f"{sample_id:02d}_{sequence_path.stem}_f{frame}.png", dpi=145
        )
        plt.close(fig)
        print(
            f"[lc2 {sample_id:02d}/{len(plan)}] {sequence_path.stem} f{frame}: "
            f"{float(result['lc2_before']):.3f}->{float(result['lc2_after']):.3f}, "
            f"inside {100*float(result['inside_before']):.1f}%->"
            f"{100*float(result['inside_after']):.1f}%, liver={100*liver_after:.1f}%",
            flush=True,
        )

    ncols = 4
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3.45 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, record in zip(axes.ravel(), triptychs, records):
        ax.imshow(image)
        ax.set_title(
            f"#{record['sample_id']:02d} {Path(record['bag']).name} f{record['frame']} | "
            f"liver {100*record['liver_cov_refined_display']:.1f}%\n"
            f"LC2 {record['lc2_before']:.3f}->{record['lc2_after']:.3f} | "
            f"inside {100*record['inside_before']:.0f}->{100*record['inside_after']:.0f}% | "
            f"d={record['probe_origin_shift_mm']:.1f}mm/{record['relative_rotation_deg']:.1f}deg",
            fontsize=8,
        )
        ax.text(0.01, 0.98, "real US + liver", color="yellow", fontsize=7,
                ha="left", va="top", transform=ax.transAxes)
        ax.text(0.50, 0.98, "robot CBCT", color="yellow", fontsize=7,
                ha="center", va="top", transform=ax.transAxes)
        ax.text(0.99, 0.98, "LC2 CBCT + liver", color="yellow", fontsize=7,
                ha="right", va="top", transform=ax.transAxes)
    fig.suptitle(
        f"{len(records)} liver-positive samples | orange = refined CBCT liver label | "
        f"US/CBCT {args.us_spacing_mm:.9f} mm/px | per-frame bounded LC2",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    montage = args.out / "montage_20_liver_lc2.png"
    fig.savefig(montage, dpi=145)
    plt.close(fig)

    payload = {
        "seed": args.seed,
        "sampling": "liver-positive, stratified across bags",
        "liver_id": LIVER_ID,
        "min_liver_cov_selection": args.min_liver_cov,
        "us_spacing_mm": args.us_spacing_mm,
        "placement": str(args.placement),
        "post_reslice_resize": False,
        "records": records,
    }
    (args.out / "samples.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id", "bag", "frame", "selection_liver_cov_polar",
            "liver_cov_initial_display", "liver_cov_refined_display", "lc2_before",
            "lc2_after", "inside_before", "inside_after", "probe_origin_shift_mm",
            "relative_rotation_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(montage, flush=True)


if __name__ == "__main__":
    main()
