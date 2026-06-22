#!/usr/bin/env python
"""Render pseudo-US images from CBCT poses with a trained CUT renderer.

This is the inference-side companion to the CUT renderer training scripts. It can either:

1. load a trajectory archive with ``poses_cbct_mm`` and slice CBCT sectors at those poses, or
2. load a pre-sliced CBCT sector archive with an ``images``/``cbct`` array.

The output archive stores both the generator input and output so every generated US frame can be
checked later:

    cbct_sector         display-space CBCT sector before normalisation
    source_cbct_unit    per-image min/max normalised CBCT in [-1, 1]
    generated_us_unit   raw generator output in [-1, 1]
    generated_us        clipped display image in [0, 1]

Example:
    apptainer/run.sh python renderer_training/render_us_from_poses.py \
        --trajectory data/trajectories/novel_offset_world_valid.npz \
        --checkpoint runs/renderer_cut_paired_display_ep300_b2_lp005/generator.pt \
        --out data/rendered_us/novel_offset_world_valid_paired_ep300_b2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepussim.renderer.cut import CUTModel  # noqa: E402


DEFAULT_REPORT = REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_REF_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_CKPT = REPO_ROOT / "runs" / "renderer_cut_paired_display_ep300_b2_lp005" / "generator.pt"


def _pick(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"none of {names} found; archive has keys: {data.files}")


def _to_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def _display_from_unit(x: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)


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


def _load_cbct_frame(volume_path: Path, report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from reslice.io import load_volume_data

    volume = load_volume_data(volume_path)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame_key = "recon" if volume_path.suffix.lower() == ".mhd" else "dicom"
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


def _target_shape(ref_sequence: Path, explicit_shape: list[int] | None) -> tuple[int, int]:
    if explicit_shape is not None:
        if len(explicit_shape) != 2:
            raise ValueError("--target-shape expects exactly two ints: height width")
        return int(explicit_shape[0]), int(explicit_shape[1])
    ref = np.load(ref_sequence, allow_pickle=True)
    return tuple(int(v) for v in ref["images"].shape[1:3])


def _slice_cbct_from_trajectory(
    trajectory: Path,
    volume_path: Path,
    report_path: Path,
    target_shape: tuple[int, int],
    max_poses: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from plot_script.plots_reslice.compare import cbct_sector_zoom

    traj = np.load(trajectory, allow_pickle=True)
    poses = np.asarray(_pick(traj, "poses_cbct_mm", "poses", "refined_poses"), dtype=np.float64)
    if max_poses is not None:
        poses = poses[:max_poses]

    volume, affine_centered = _load_cbct_frame(volume_path, report_path)
    cbct = np.stack(
        [cbct_sector_zoom(volume, affine_centered, pose, target_shape).astype(np.float32) for pose in poses]
    )

    meta: dict[str, np.ndarray] = {"poses_cbct_mm": poses}
    for key in (
        "trajectory_id",
        "inside_ratio",
        "nearest_existing_mm",
        "axial_angle_to_nearest_deg",
        "score",
    ):
        if key in traj.files:
            value = np.asarray(traj[key])
            meta[key] = value[: len(poses)]
    return cbct, meta


def _load_cbct_archive(cbct_npz: Path, max_poses: int | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(cbct_npz, allow_pickle=True)
    image_key = _pick_image_key(data)
    original_n = len(data[image_key])
    cbct = np.asarray(data[image_key], dtype=np.float32)
    if max_poses is not None:
        cbct = cbct[:max_poses]
    meta: dict[str, np.ndarray] = {}
    for key in data.files:
        if key == image_key:
            continue
        value = np.asarray(data[key])
        if value.ndim > 0 and len(value) == original_n:
            meta[key] = value[: len(cbct)]
    return cbct, meta


def _pick_image_key(data: np.lib.npyio.NpzFile) -> str:
    for key in ("images", "cbct", "cbct_sector"):
        if key in data.files:
            return key
    raise KeyError(f"no image key found in {data.files}")


def _run_generator(
    checkpoint: Path,
    source_unit: np.ndarray,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    device = torch.device(device_name)
    model = _load_generator(checkpoint, device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(source_unit), batch_size):
            batch = torch.from_numpy(source_unit[start : start + batch_size, None]).to(device)
            out.append(model.G(batch).cpu().numpy()[:, 0])
            print(
                f"rendered {min(start + batch_size, len(source_unit))}/{len(source_unit)}",
                flush=True,
            )
    return np.concatenate(out, axis=0).astype(np.float32)


def _write_manifest(path: Path, n: int, meta: dict[str, np.ndarray]) -> None:
    fields = ["index"]
    for key, value in meta.items():
        if np.asarray(value).ndim <= 1 and len(value) == n:
            fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            row = {"index": i}
            for key in fields[1:]:
                value = np.asarray(meta[key])[i]
                row[key] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def _write_preview(
    path: Path,
    cbct_sector: np.ndarray,
    generated_us: np.ndarray,
    n_preview: int,
) -> None:
    if n_preview <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n_preview, len(cbct_sector))
    indices = np.linspace(0, len(cbct_sector) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 2, figsize=(4.8, max(2.0, 1.8 * n)), squeeze=False)
    axes[0, 0].set_title("CBCT sector")
    axes[0, 1].set_title("Generated US")
    for row, idx in enumerate(indices):
        cbct = _display_from_unit(_to_unit(cbct_sector[idx]))
        axes[row, 0].imshow(cbct, cmap="gray")
        axes[row, 1].imshow(generated_us[idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(f"#{idx}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path, help="trajectory npz with poses_cbct_mm")
    source.add_argument("--cbct-npz", type=Path, help="pre-sliced CBCT sector npz with images/cbct key")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--ref-sequence", type=Path, default=DEFAULT_REF_SEQUENCE)
    ap.add_argument("--target-shape", nargs=2, type=int, default=None, metavar=("H", "W"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-poses", type=int, default=None, help="debug limit")
    ap.add_argument("--preview", type=int, default=12, help="number of evenly spaced rows in preview PNG/PDF")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.trajectory:
        shape = _target_shape(args.ref_sequence, args.target_shape)
        cbct_sector, meta = _slice_cbct_from_trajectory(
            trajectory=args.trajectory,
            volume_path=args.volume_path,
            report_path=args.report,
            target_shape=shape,
            max_poses=args.max_poses,
        )
        source_path = args.trajectory
        source_kind = "trajectory"
    else:
        cbct_sector, meta = _load_cbct_archive(args.cbct_npz, max_poses=args.max_poses)
        source_path = args.cbct_npz
        source_kind = "cbct_npz"

    source_cbct_unit = np.stack([_to_unit(x) for x in cbct_sector]).astype(np.float32)
    print(
        f"source={source_path} ({source_kind}) | checkpoint={args.checkpoint} | "
        f"n={len(source_cbct_unit)} shape={source_cbct_unit.shape[1:]} device={args.device}",
        flush=True,
    )
    generated_us_unit = _run_generator(
        checkpoint=args.checkpoint,
        source_unit=source_cbct_unit,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    generated_us = _display_from_unit(generated_us_unit)

    archive = out / "rendered_us.npz"
    np.savez_compressed(
        archive,
        cbct_sector=cbct_sector.astype(np.float32),
        source_cbct_unit=source_cbct_unit,
        generated_us_unit=generated_us_unit,
        generated_us=generated_us.astype(np.float32),
        **meta,
    )
    _write_manifest(out / "manifest.csv", len(generated_us), meta)
    _write_preview(out / "preview_generated_us.png", cbct_sector, generated_us, args.preview)

    info = {
        "source": str(source_path),
        "source_kind": source_kind,
        "checkpoint": str(args.checkpoint),
        "out": str(out),
        "archive": str(archive),
        "n_images": int(len(generated_us)),
        "image_shape": [int(v) for v in generated_us.shape[1:]],
        "generated_us_range": [float(generated_us.min()), float(generated_us.max())],
        "generated_us_unit_range": [float(generated_us_unit.min()), float(generated_us_unit.max())],
    }
    (out / "render_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
