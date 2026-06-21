"""Visualise a generated scale-up dataset (the run_scaleup output dir).

Reads <dataset>/sample_*.npz (image, pose, mask, force) + index.json and emits LaTeX-styled
figures (PNG+PDF) under figures/<dataset-name>/:

    force.{png,pdf}      contact-force distribution + per-sample force (sim datasets)
    coverage.{png,pdf}   probe positions (top/side), coloured by force — the scanned patch
    samples.{png,pdf}    a montage: CBCT slice / rendered US / anatomy-mask overlay

    python plot_script/plot_dataset.py data/ds_sim                 # + --volume for slice context
    python plot_script/plot_dataset.py data/ds_sim --volume data/cbct_20260612/intensity.nrrd

A quick data-quality check after a scale-up run: are forces near the target, is coverage the
real patch, do masks look sane. (The US image is the placeholder renderer until B1 lands.)
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))              # style/ lives beside this script
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))  # import deepussim without install
from style.style import apply_style, figure, save, C, TYPE  # noqa: E402
apply_style()

ROOT = Path(__file__).parents[1]


def load_dataset(d: Path):
    files = sorted(d.glob("sample_*.npz"))
    imgs, poses, masks, forces = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        imgs.append(z["image"]); poses.append(z["pose"])
        masks.append(z["mask"] if "mask" in z.files else None)
        forces.append(float(np.linalg.norm(z["force"])) if "force" in z.files else np.nan)
    return files, imgs, np.array(poses), masks, np.array(forces)


def fig_force(forces, out, target=None):
    fig, axes = figure(ncols=2, width="double", height=2.2)
    axes[0].hist(forces, bins=20, color=C["ours"])
    if target is not None:
        axes[0].axvline(target, color=C["bad"], lw=1.0, ls="--", label=f"target {target:g} N")
        axes[0].legend()
    axes[0].set_xlabel("contact force (N)"); axes[0].set_ylabel("samples")
    axes[0].set_title(f"force distribution (median {np.nanmedian(forces):.1f} N)", fontsize=TYPE["small"])
    axes[1].plot(np.arange(len(forces)), forces, lw=0.8, color=C["ours"], marker="o", ms=2)
    if target is not None:
        axes[1].axhline(target, color=C["bad"], lw=1.0, ls="--")
    axes[1].set_xlabel("sample index (scan order)"); axes[1].set_ylabel("contact force (N)")
    axes[1].set_title("force per pose", fontsize=TYPE["small"]); axes[1].grid(True, axis="both")
    save(fig, out); plt.close(fig)


def fig_coverage(poses, forces, out):
    pos = poses[:, :3, 3]   # CBCT mm
    fig, axes = figure(ncols=2, width="double", height=2.8)
    for ax, (a, b, ylab, ttl) in zip(axes, [(0, 1, "Y", "top (X-Y)"), (0, 2, "Z", "side (X-Z)")]):
        sc = ax.scatter(pos[:, a], pos[:, b], c=forces, cmap="viridis", s=14)
        ax.set_xlabel("X (mm)"); ax.set_ylabel(f"{ylab} (mm)")
        ax.set_title(f"probe positions, {ttl}", fontsize=TYPE["small"])
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, axis="both")
    cb = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02); cb.set_label("force (N)", fontsize=TYPE["small"])
    save(fig, out); plt.close(fig)


def fig_samples(files, imgs, poses, masks, out, volume=None, geom=None):
    from deepussim.us.reslice import reslice_volume
    pick = np.linspace(0, len(files) - 1, 4, dtype=int)
    nrows = 3 if volume is not None else 2
    fig, axes = plt.subplots(nrows, 4, figsize=(7.2, 2.05 * nrows)); fig.set_constrained_layout(True)
    axes = np.atleast_2d(axes)
    for col, i in enumerate(pick):
        r = 0
        raw = None
        if volume is not None:
            raw = reslice_volume(volume, poses[i], geom, order=1)
            axes[r, col].imshow(raw, cmap="gray", aspect="auto"); axes[r, col].set_title(f"sample {i}", fontsize=TYPE["small"]); r += 1
        axes[r, col].imshow(imgs[i], cmap="gray", aspect="auto", vmin=0, vmax=1)
        if volume is None:
            axes[r, col].set_title(f"sample {i}", fontsize=TYPE["small"])
        r += 1
        base = raw if raw is not None else imgs[i]
        axes[r, col].imshow(base, cmap="gray", aspect="auto")
        if masks[i] is not None:
            m = np.ma.masked_where(masks[i] == 0, masks[i])
            axes[r, col].imshow(m, cmap="tab10", aspect="auto", alpha=0.55, interpolation="nearest")
        for rr in range(nrows):
            axes[rr, col].set_xticks([]); axes[rr, col].set_yticks([]); axes[rr, col].grid(False)
    rlabels = (["CBCT slice", "rendered US (placeholder)", "anatomy mask"] if volume is not None
               else ["rendered US (placeholder)", "anatomy mask"])
    for rr, lb in enumerate(rlabels):
        axes[rr, 0].set_ylabel(lb, fontsize=TYPE["small"])
    save(fig, out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="dataset dir (run_scaleup --out), e.g. data/ds_sim")
    ap.add_argument("--volume", help="CBCT intensity volume for a raw-slice context row")
    ap.add_argument("--target-n", type=float, default=5.0, help="force-servo target for the marker")
    args = ap.parse_args()

    d = Path(args.dataset)
    files, imgs, poses, masks, forces = load_dataset(d)
    print(f"{d}: {len(files)} samples | force N median={np.nanmedian(forces):.2f} "
          f"min={np.nanmin(forces):.2f} max={np.nanmax(forces):.2f}")
    out_dir = ROOT / "figures" / d.name; out_dir.mkdir(parents=True, exist_ok=True)

    volume = geom = None
    if args.volume:
        from deepussim.data.volume import load_volume
        from deepussim.us.reslice import ProbeGeometry
        import json
        volume = load_volume(args.volume)
        gmeta = json.loads((d / "index.json").read_text()).get("meta", {}).get("geometry", {})
        geom = ProbeGeometry(**{k: gmeta[k] for k in
                                ("radius_mm", "fov_deg", "depth_mm", "n_lat", "n_ax") if k in gmeta})

    if np.isfinite(forces).any():
        fig_force(forces, str(out_dir / "force"), target=args.target_n)
        fig_coverage(poses, forces, str(out_dir / "coverage"))
    fig_samples(files, imgs, poses, masks, str(out_dir / "samples"), volume=volume, geom=geom)
    print(f"  figures -> {out_dir}")


if __name__ == "__main__":
    main()
