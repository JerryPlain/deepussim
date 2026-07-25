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
from reslice.us_display import DEFAULT_FAN_JSON, UsDisplayFan, render_on_us_grid

# Defaults matching the project's 2026-06-12 setup.
DEFAULT_REPORT = _REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = _REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_PLACEMENT = _REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_OUT_DIR = _REPO_ROOT / "figures" / "2_cbct_us_reslice_check"

# The one authoritative US display fan, fitted by ``calib/fit_us_fan.py``. Loaded once.
_FAN: UsDisplayFan | None = None


def _get_fan() -> UsDisplayFan:
    """Lazily load (and cache) the fitted US display fan."""
    global _FAN
    if _FAN is None:
        _FAN = UsDisplayFan.load(DEFAULT_FAN_JSON)
    return _FAN


# Legacy: derived from the fitted fan so anything still reading these stays correct. The old
# hand-tuned values (near_mm=15) were wrong -- see tests/test_display_alignment.py and
# figures/10_fan_geometry_mismatch/. The reslice-rectangular display path they fed is retired.
FAN = dict(depth_mm=_get_fan().depth_mm, fov_deg=_get_fan().fov_deg, near_mm=_get_fan().radius_mm)


def _normalize_display(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def cbct_sector_zoom(volume, affine_centered, T_phantom_from_probe_mm, target_shape) -> np.ndarray:
    """CBCT sampled directly onto the real US display grid, normalised to ``[0, 1]``.

    Pixel-aligned to the US by construction (see ``reslice.us_display.render_on_us_grid``):
    no crop, no anisotropic resize. ``target_shape`` must equal the fitted fan shape (the real
    US frame size); a differing shape is resized bilinearly as a fallback.
    """
    fan = _get_fan()
    img, _valid = render_on_us_grid(volume, affine_centered, T_phantom_from_probe_mm, fan, order=1)
    img = sec.normalize_image(img)
    if tuple(target_shape) != img.shape:
        from scipy.ndimage import zoom as _zoom

        img = _zoom(img, (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1]), order=1)
    return img.astype(np.float32)


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
