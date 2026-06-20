"""Figure helpers for the reslice package: real US vs resliced CBCT fan sectors.

Builds the same-scale comparison images used to eyeball CBCT<->US alignment. Reuses the
``reslice`` package for all geometry (no logic duplicated here).

    python -m plot_script.plots_reslice.compare --sequence data/sequences/scan5.npz --frames 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so ``reslice`` resolves when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reslice import pose as P
from reslice import sector as sec
from reslice.io import load_transform_4x4, load_volume_data
from reslice.sampling import reslice_rectangular_plane

# Defaults matching the project's 2026-06-12 setup.
DEFAULT_REPORT = _REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = _REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_PLACEMENT = _REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_OUT_DIR = _REPO_ROOT / "figures" / "reslice"

# Fan fitted from the real US (fit_us_geometry on scan1).
FAN = dict(depth_mm=93.0, fov_deg=57.0, near_mm=15.0)
SLICE_W, SLICE_H = 360.0, 300.0
AXIAL_SIGN, LATERAL_SIGN = -1.0, 1.0


def _normalize_display(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def cbct_sector_zoom(volume, affine_centered, T_phantom_from_probe_mm, target_shape) -> np.ndarray:
    """Resliced CBCT fan sector, cropped+zoomed to ``target_shape`` (US size)."""
    plane = P.plane_from_probe_pose(T_phantom_from_probe_mm, "probe-xz", 0.0)
    rect, valid = reslice_rectangular_plane(
        volume, affine_centered, plane, width_mm=SLICE_W, height_mm=SLICE_H,
        n_rows=target_shape[0], n_cols=target_shape[1],
        axial_sign=AXIAL_SIGN, lateral_sign=LATERAL_SIGN,
    )
    rect, valid = np.rot90(rect, 2), np.rot90(valid, 2)
    _, thr, _ = sec.detect_content_top_row(rect, valid, threshold=None, min_pixels=8)
    probe_px = sec.project_point_to_display_pixel(
        T_phantom_from_probe_mm[:3, 3], plane, width_mm=SLICE_W, height_mm=SLICE_H,
        rows=rect.shape[0], cols=rect.shape[1], axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN, display_rot180=True,
    )
    depth_dir = sec.project_direction_to_display_rc(
        T_phantom_from_probe_mm[:3, 2], plane, axial_sign=AXIAL_SIGN,
        lateral_sign=LATERAL_SIGN, display_rot180=True,
    )
    apex, _ = sec.apex_from_pose_and_edge(
        rect, valid, threshold=thr, probe_pixel_rc=probe_px,
        depth_direction_rc=depth_dir, max_line_distance_px=5.0,
    )
    sector, mask, dbg = sec.apply_sector(
        rect, valid, top_margin_rows=2, apex_col_fraction=0.5,
        apex_pixel_rc=apex, depth_direction_rc=depth_dir,
        content_threshold=None, content_min_pixels=8,
        width_mm=SLICE_W, height_mm=SLICE_H, **FAN,
    )
    crop_mask = sec.sector_mask_in_display_image(
        rect.shape, dbg["apex_row"], dbg["apex_col"], np.asarray(dbg["depth_direction_rc"]),
        dbg["fov_deg"], dbg["depth_mm"], 0.0, dbg["mm_per_row"], dbg["mm_per_col"],
    )
    zoom, _ = sec.crop_and_zoom_sector(
        sec.normalize_image(sector), mask, crop_mask, tuple(target_shape), margin_px=18
    )
    return zoom


def compare_grid(
    sequence: Path,
    n_frames: int = 4,
    report: Path = DEFAULT_REPORT,
    volume_path: Path = DEFAULT_VOLUME,
    placement: Path = DEFAULT_PLACEMENT,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    """Save a ``(n_frames x [US | CBCT sector])`` grid for one sequence."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(Path(report).read_text(encoding="utf-8"))
    frame_key = "recon" if Path(volume_path).suffix.lower() == ".mhd" else "dicom"
    affine = np.array(rep["phantom_centered_frame"][f"{frame_key}_affine_centered_from_ijk_mm"])
    volume = load_volume_data(Path(volume_path))
    T_world_from_phantom = load_transform_4x4(Path(placement))

    z = np.load(sequence, allow_pickle=True)
    images, contact = z["images"], np.asarray(z["contact"], dtype=bool)
    ci = np.where(contact)[0]
    frames = ci[np.linspace(0, len(ci) - 1, n_frames, dtype=int)]

    fig, ax = plt.subplots(n_frames, 2, figsize=(6, 3 * n_frames), squeeze=False)
    for r, fr in enumerate(frames):
        us = images[int(fr)]
        Twp = P.pose_from_sequence(Path(sequence), int(fr), "ee", 0.0)
        Tpp = P.probe_pose_in_phantom_centered_mm(Twp, T_world_from_phantom)
        zoom = cbct_sector_zoom(volume, affine, Tpp, us.shape[:2])
        ax[r, 0].imshow(_normalize_display(us), cmap="gray")
        ax[r, 0].set_title(f"{Path(sequence).stem} f{fr} US", fontsize=9)
        ax[r, 1].imshow(zoom, cmap="gray")
        ax[r, 1].set_title("CBCT sector", fontsize=9)
        ax[r, 0].axis("off"); ax[r, 1].axis("off")
    fig.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"compare_{Path(sequence).stem}_grid.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    path = compare_grid(args.sequence, n_frames=args.frames, out_dir=args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
