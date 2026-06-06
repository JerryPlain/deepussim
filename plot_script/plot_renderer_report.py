"""Comprehensive learned-renderer (CUT) training report.

Produces, under figures/renderer/:
  losses.{png,pdf}       d / gan / nce / nce_idt vs epoch (training health)
  progression.{png,pdf}  the SAME fixed example's generated US across checkpoints (is it learning?)
  final_samples.{png,pdf} src / fake / tgt grid at the last checkpoint (the go/no-go read)

Loss curves come from runs/<run>/losses.csv if present, else parsed from the SLURM log
(--log deepussim-renderer-<jobid>.out). Sample panels come from samples_ep*.npz.

    python plot_script/plot_renderer_report.py runs/renderer_cut \
        --log deepussim-renderer-3706862.out
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style.style import apply_style, figure, save, C, TYPE   # noqa: E402
apply_style()

ROOT = Path(__file__).parents[1]
OUT = ROOT / "figures" / "renderer"
_EP = re.compile(r"epoch\s+(\d+)/\d+\s+(.*?)\s+\(")


def parse_losses(run_dir: Path, log: Path | None):
    csv = run_dir / "losses.csv"
    if csv.exists():
        rows = [l.split(",") for l in csv.read_text().splitlines() if l]
        head = rows[0]; data = {k: [] for k in head}
        for r in rows[1:]:
            for k, v in zip(head, r):
                data[k].append(float(v))
        return data
    if not log or not log.exists():
        return None
    data = {"epoch": []}
    for line in log.read_text().splitlines():
        m = _EP.search(line)
        if not m:
            continue
        data["epoch"].append(int(m.group(1)))
        for kv in m.group(2).split():
            if "=" in kv:
                k, v = kv.split("=")
                data.setdefault(k, []).append(float(v))
    return data if data["epoch"] else None


def _norm(im):
    lo, hi = float(im.min()), float(im.max())
    return (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)


def fig_losses(data):
    fig, ax = figure(width="double", height=2.4)
    ep = data["epoch"]
    for k, col in [("d", C["baseline_a"]), ("gan", C["ours"]),
                   ("nce", C["baseline_b"]), ("nce_idt", C["baseline_c"])]:
        if k in data and len(data[k]) == len(ep):
            ax.plot(ep, data[k], lw=1.0, color=col, label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(ncol=4, loc="upper right")
    ax.set_title("CUT training losses"); ax.grid(True, axis="both")
    save(fig, str(OUT / "losses")); plt.close(fig)


def fig_progression(snaps):
    eps = [int(p.stem.replace("samples_ep", "")) for p in snaps]
    pick = [snaps[i] for i in np.linspace(0, len(snaps) - 1, min(5, len(snaps))).astype(int)]
    fig, axes = plt.subplots(1, len(pick), figsize=(1.7 * len(pick) + 0.4, 2.4))
    fig.set_constrained_layout(True); axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pick):
        fake = np.load(p)["fake"][0, 0]
        ax.imshow(_norm(fake), cmap="gray", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"epoch {int(p.stem.replace('samples_ep',''))}", fontsize=TYPE["small"])
    fig.suptitle("generated US for one fixed input, across training", fontsize=TYPE["body"])
    save(fig, str(OUT / "progression")); plt.close(fig)


def fig_final(snaps):
    d = np.load(snaps[-1]); src, fake, tgt = d["src"], d["fake"], d["tgt"]
    n = min(4, src.shape[0]); ep = snaps[-1].stem.replace("samples_ep", "")
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n + 0.6, 5.4)); fig.set_constrained_layout(True)
    axes = np.atleast_2d(axes)
    for r, (lb, st) in enumerate([("CBCT slice", src), ("generated US", fake),
                                  ("real US (unpaired)", tgt)]):
        for c in range(n):
            axes[r, c].imshow(_norm(st[c, 0]), cmap="gray", aspect="auto")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
        axes[r, 0].set_ylabel(lb, fontsize=TYPE["small"])
    fig.suptitle(f"learned renderer — final (epoch {ep})", fontsize=TYPE["body"] + 1)
    save(fig, str(OUT / "final_samples")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run dir (with samples_ep*.npz), e.g. runs/renderer_cut")
    ap.add_argument("--log", help="SLURM log to parse losses from if losses.csv is absent")
    args = ap.parse_args()
    run = Path(args.run)
    OUT.mkdir(parents=True, exist_ok=True)

    snaps = sorted(run.glob("samples_ep*.npz"))
    losses = parse_losses(run, Path(args.log) if args.log else None)
    if losses:
        fig_losses(losses); print(f"losses: {len(losses['epoch'])} epochs")
    else:
        print("no losses (no losses.csv / parsable --log)")
    if snaps:
        fig_progression(snaps); fig_final(snaps)
        print(f"samples: {len(snaps)} checkpoints -> {OUT}")
    else:
        print(f"no samples_ep*.npz in {run}")


if __name__ == "__main__":
    main()
