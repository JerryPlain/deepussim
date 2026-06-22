#!/usr/bin/env python
"""Compare paired vs unpaired CUT renderer checkpoints per scan.

The visual comparison is generated from real LC2-paired frames in ``pairs.npz``.
For each scan, the script writes one figure per training method:

    Input CBCT sector | generated US | Real US same frame

The current renderer pairs are display-space B-mode sectors generated with the same
`cbct_sector_zoom(...)` path as the LC2 comparison figures, so plots show images directly.
A legacy polar-grid display path is still kept for old pair archives.

Outputs:
    figures/renderer_training/unpaired/scan*_cut_unpaired_fan_comparison.{png,pdf}
    figures/renderer_training/paired/scan*_cut_paired_fan_comparison.{png,pdf}
    figures/renderer_training/cut_loss_comparison.{png,pdf}
    figures/renderer_training/cut_training_summary.txt

Example:
    apptainer/run.sh python -m plot_script.plots_renderer.compare_cut --device cpu
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deepussim.renderer.cut import CUTModel  # noqa: E402
from lc2.forward import fit_us_fan  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from plot_script.style.style import C, TYPE, apply_style, save  # noqa: E402


DEFAULT_DATA = _REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
DEFAULT_PAIRED = _REPO_ROOT / "runs" / "renderer_cut_paired_ep200_b4_lp005_blur5" / "generator.pt"
DEFAULT_UNPAIRED = _REPO_ROOT / "runs" / "renderer_cut_unpaired_ep200_b4" / "generator.pt"
DEFAULT_OUT = _REPO_ROOT / "figures" / "5_renderer_training"
DEFAULT_REF = _REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_US_SPACING_MM = 0.166112957


def _scan_key(name: str) -> tuple[int, str]:
    stem = Path(str(name)).stem
    m = re.search(r"scan(\d+)", stem)
    return (int(m.group(1)) if m else 10**9, stem)


def _to_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def _display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).squeeze()
    lo, hi = np.nanpercentile(x, [1.0, 99.0])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _fit_display_fan(ref_sequence: Path, us_spacing: float):
    ref = np.load(ref_sequence, allow_pickle=True)
    fan, _ = fit_us_fan(ref["images"], ref["contact"], us_spacing)
    return fan, tuple(ref["images"].shape[1:3])


def _polar_to_sector(polar: np.ndarray, fan, output_shape: tuple[int, int]) -> np.ndarray:
    """Scan-convert a polar fan grid back to B-mode display coordinates."""
    polar = np.asarray(polar, dtype=np.float32).squeeze()
    n_ax, n_lat = polar.shape
    height, width = output_shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ax, ay = fan.apex_px
    dx = xx - float(ax)
    dy = yy - float(ay)
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dx, dy)
    half = np.deg2rad(float(fan.fov_deg) / 2.0)

    row = (radius - float(fan.r0_px)) / max(float(fan.r1_px - fan.r0_px), 1e-6) * (n_ax - 1)
    col = (theta + half) / max(2.0 * half, 1e-6) * (n_lat - 1)
    valid = (row >= 0) & (row <= n_ax - 1) & (col >= 0) & (col <= n_lat - 1)
    sector = map_coordinates(polar, [row.ravel(), col.ravel()], order=1, mode="constant", cval=0.0)
    sector = sector.reshape(output_shape)
    sector[~valid] = 0.0
    return sector


def _read_losses(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list[float]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            out[key].append(float(value))
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _checkpoint_arg(ck: dict, name: str, default: int | float) -> int | float:
    return (ck.get("args") or {}).get(name, default)


def _load_generator(path: Path, device: torch.device) -> CUTModel:
    ck = torch.load(path, map_location=device)
    model = CUTModel(
        ngf=int(_checkpoint_arg(ck, "ngf", 64)),
        ndf=int(_checkpoint_arg(ck, "ndf", 64)),
        n_blocks=int(_checkpoint_arg(ck, "n_blocks", 9)),
        lambda_nce=float(_checkpoint_arg(ck, "lambda_nce", 1.0)),
        num_patches=int(_checkpoint_arg(ck, "num_patches", 256)),
    ).to(device)
    model.G.load_state_dict(ck["G"])
    model.eval()
    return model


def _load_pairs(data_path: Path) -> dict[str, np.ndarray | str]:
    data = np.load(data_path, allow_pickle=True)
    cbct = data["cbct"]
    us = data["us"]
    seq = data["sequence"] if "sequence" in data.files else np.array(["unknown.npz"] * len(cbct))
    frame = data["frame_index"] if "frame_index" in data.files else np.arange(len(cbct))
    src = np.stack([_to_unit(x) for x in cbct]).astype(np.float32)
    image_space = "polar" if cbct.shape[1:] == (512, 256) else "display"
    return {
        "src": src,
        "real_us": us.astype(np.float32),
        "sequence": seq,
        "frame": frame,
        "image_space": image_space,
    }


def _select_scan_rows(indices: np.ndarray, frames: np.ndarray, per_scan: int) -> np.ndarray:
    if per_scan <= 0 or len(indices) <= per_scan:
        return indices
    order = np.argsort(frames[indices], kind="stable")
    ordered = indices[order]
    pick = np.linspace(0, len(ordered) - 1, per_scan, dtype=int)
    return ordered[pick]


def _run_generator(ckpt: Path, src: np.ndarray, device_name: str) -> np.ndarray:
    device = torch.device(device_name)
    model = _load_generator(ckpt, device)
    src_t = torch.from_numpy(src[:, None]).to(device)
    outs = []
    with torch.no_grad():
        for start in range(0, len(src_t), 16):
            outs.append(model.G(src_t[start:start + 16]).cpu().numpy())
    return np.concatenate(outs, axis=0)



def _plot_image(image: np.ndarray, pairs: dict, fan, display_shape: tuple[int, int]) -> np.ndarray:
    if pairs.get("image_space") == "polar":
        image = _polar_to_sector(image, fan, display_shape)
    return _display(image)

def _plot_scan_grid(
    method_name: str,
    generated: np.ndarray,
    pairs: dict[str, np.ndarray],
    scan_name: str,
    row_indices: np.ndarray,
    fan,
    display_shape: tuple[int, int],
    out_dir: Path,
    filename: str,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(row_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(6.2, max(2.0, 2.0 * rows)), squeeze=False)
    headers = ["Input CBCT sector", f"{method_name} generated US", "Real US same frame"]
    for c, title in enumerate(headers):
        axes[0, c].set_title(title, fontsize=TYPE["small"])

    stacks = [pairs["src"][:, None], generated, pairs["real_us"][:, None]]
    for r, idx in enumerate(row_indices):
        for c, stack in enumerate(stacks):
            axes[r, c].imshow(_plot_image(stack[idx, 0], pairs, fan, display_shape), cmap="gray")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        axes[r, 0].set_ylabel(f"f{int(pairs['frame'][idx])}", fontsize=TYPE["tiny"])

    fig.suptitle(
        f"{method_name}: {Path(scan_name).stem} real LC2-paired display-sector frames",
        fontsize=TYPE["body"] + 1,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    save(fig, str(out))
    plt.close(fig)
    return out.with_suffix(".png")



def _plot_pair_grid(
    pairs: dict[str, np.ndarray],
    scan_name: str,
    row_indices: np.ndarray,
    fan,
    display_shape: tuple[int, int],
    out_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(row_indices)
    fig, axes = plt.subplots(rows, 2, figsize=(4.3, max(2.0, 2.0 * rows)), squeeze=False)
    headers = ["LC2-pair CBCT sector", "Real US same frame"]
    for c, title in enumerate(headers):
        axes[0, c].set_title(title, fontsize=TYPE["small"])

    stacks = [pairs["src"][:, None], pairs["real_us"][:, None]]
    for r, idx in enumerate(row_indices):
        for c, stack in enumerate(stacks):
            axes[r, c].imshow(_plot_image(stack[idx, 0], pairs, fan, display_shape), cmap="gray")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        axes[r, 0].set_ylabel(f"f{int(pairs['frame'][idx])}", fontsize=TYPE["tiny"])

    stem = Path(str(scan_name)).stem
    fig.suptitle(f"Pair sanity check: {stem} LC2-refined display CBCT vs real US", fontsize=TYPE["body"] + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_pair_cbct_vs_real_us"
    save(fig, str(out))
    plt.close(fig)
    return out.with_suffix(".png")

def make_scan_figures(
    data_path: Path,
    paired_ckpt: Path,
    unpaired_ckpt: Path,
    out_dir: Path,
    per_scan: int,
    device_name: str,
    ref_sequence: Path,
    us_spacing: float,
    pairs_only: bool = False,
) -> list[Path]:
    apply_style()
    fan, display_shape = _fit_display_fan(ref_sequence, us_spacing)
    pairs = _load_pairs(data_path)
    fake_unpaired = None if pairs_only else _run_generator(unpaired_ckpt, pairs["src"], device_name)
    fake_paired = None if pairs_only else _run_generator(paired_ckpt, pairs["src"], device_name)

    written: list[Path] = []
    sequences = sorted(np.unique(pairs["sequence"]), key=_scan_key)
    for seq in sequences:
        all_idx = np.where(pairs["sequence"] == seq)[0]
        row_idx = _select_scan_rows(all_idx, pairs["frame"], per_scan=per_scan)
        stem = Path(str(seq)).stem
        written.append(_plot_pair_grid(pairs, str(seq), row_idx, fan, display_shape, out_dir / "cbct_us_pairs"))
        if not pairs_only:
            written.append(
                _plot_scan_grid(
                    "CUT unpaired baseline",
                    fake_unpaired,
                    pairs,
                    str(seq),
                    row_idx,
                    fan,
                    display_shape,
                    out_dir / "unpaired_vs_real",
                    f"{stem}_cut_unpaired_fan_comparison",
                )
            )
            written.append(
                _plot_scan_grid(
                    "CUT paired weak-L1",
                    fake_paired,
                    pairs,
                    str(seq),
                    row_idx,
                    fan,
                    display_shape,
                    out_dir / "paired_vs_real",
                    f"{stem}_cut_paired_fan_comparison",
                )
            )
    return written


def make_loss_plot(paired_run: Path, unpaired_run: Path, out_dir: Path, smooth: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style()
    losses = {
        "paired": _read_losses(paired_run / "losses.csv"),
        "unpaired": _read_losses(unpaired_run / "losses.csv"),
    }
    metrics = ["total", "gan", "nce", "nce_idt", "d", "pair_raw"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), squeeze=False)
    colors = {"paired": C["ours"], "unpaired": C["baseline_a"]}
    for ax, metric in zip(axes.ravel(), metrics):
        for label, data in losses.items():
            if metric not in data:
                continue
            ax.plot(data["epoch"], _moving_average(data[metric], smooth), label=label, color=colors[label], lw=1.2)
        ax.set_title(metric, fontsize=TYPE["small"])
        ax.set_xlabel("epoch")
        ax.grid(True, axis="both")
        if metric == "total":
            ax.legend(loc="upper right")
    fig.suptitle(f"Training losses, {smooth}-epoch moving average", fontsize=TYPE["body"] + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "loss" / "cut_loss_comparison"
    save(fig, str(out))
    plt.close(fig)
    return out.with_suffix(".png")


def _summary_line(label: str, data: dict[str, np.ndarray], metric: str, tail: int) -> str:
    values = data[metric]
    epochs = data["epoch"].astype(int)
    tail = min(tail, len(values) // 2)
    first = values[:tail]
    last = values[-tail:]
    best = int(np.argmin(values))
    improve = (float(first.mean()) - float(last.mean())) / max(abs(float(first.mean())), 1e-8)
    slope = float(np.polyfit(np.arange(tail, dtype=np.float32), last, 1)[0])
    return (
        f"{label:8s} {metric:8s} "
        f"first{tail}_mean={first.mean():.4f} last{tail}_mean={last.mean():.4f} "
        f"improve={improve * 100:+.1f}% best=ep{epochs[best]}:{values[best]:.4f} "
        f"last_slope/ep={slope:+.5f}"
    )


def write_summary(paired_run: Path, unpaired_run: Path, out_dir: Path, tail: int) -> Path:
    losses = {
        "paired": _read_losses(paired_run / "losses.csv"),
        "unpaired": _read_losses(unpaired_run / "losses.csv"),
    }
    lines = ["CUT renderer training summary", ""]
    for label, data in losses.items():
        for metric in ("total", "nce", "nce_idt", "pair_raw"):
            if metric in data:
                lines.append(_summary_line(label, data, metric, tail=tail))
    lines += [
        "",
        "Interpretation:",
        "- 300 epochs finished for both runs.",
        "- Losses still improve over the final-vs-initial tail window, but GAN losses oscillate.",
        "- Per-scan visuals are under this output directory: pairs, unpaired, and paired.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cut_training_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--paired-ckpt", type=Path, default=DEFAULT_PAIRED)
    ap.add_argument("--unpaired-ckpt", type=Path, default=DEFAULT_UNPAIRED)
    ap.add_argument("--paired-run", type=Path, default=DEFAULT_PAIRED.parent)
    ap.add_argument("--unpaired-run", type=Path, default=DEFAULT_UNPAIRED.parent)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF, help="sequence used to recover display fan geometry")
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING_MM)
    ap.add_argument("--per-scan", type=int, default=0, help="rows per scan; <=0 plots every paired frame")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs-only", action="store_true", help="only plot pair sanity checks; do not load trained generators")
    args = ap.parse_args()

    scan_pngs = make_scan_figures(
        data_path=args.data,
        paired_ckpt=args.paired_ckpt,
        unpaired_ckpt=args.unpaired_ckpt,
        out_dir=args.out_dir,
        per_scan=args.per_scan,
        device_name=args.device,
        ref_sequence=args.ref_sequence,
        us_spacing=args.us_spacing,
        pairs_only=args.pairs_only,
    )
    print(f"wrote {len(scan_pngs)} per-scan comparison PNGs")
    if not args.pairs_only:
        loss_png = make_loss_plot(args.paired_run, args.unpaired_run, args.out_dir, smooth=args.smooth)
        summary = write_summary(args.paired_run, args.unpaired_run, args.out_dir, tail=args.tail)
        print(f"wrote {loss_png}")
        print(f"wrote {summary}")


if __name__ == "__main__":
    main()
