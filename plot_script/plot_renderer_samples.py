"""Visualise learned-renderer (CUT) training samples: the src/fake/tgt stacks dumped by
scripts/train_renderer.py (samples_ep*.npz).

Rows = CBCT slice (input) / generated US (fake) / real US (target, unpaired reference);
columns = a few examples. The go/no-go read: does `fake` look like real US while keeping the
CBCT structure? Emits PNG+PDF under figures/renderer/.

    python plot_script/plot_renderer_samples.py runs/renderer_cut            # latest checkpoint
    python plot_script/plot_renderer_samples.py runs/renderer_cut/samples_ep0050.npz
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))              # style/ beside this script
from style.style import apply_style, save, TYPE            # noqa: E402
apply_style()

ROOT = Path(__file__).parents[1]


def _resolve(arg: Path) -> Path:
    if arg.is_file():
        return arg
    snaps = sorted(arg.glob("samples_ep*.npz"))
    if not snaps:
        raise SystemExit(f"no samples_ep*.npz in {arg}")
    return snaps[-1]                                         # latest epoch


def _norm(img):
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def main():
    arg = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/renderer_cut")
    npz = _resolve(arg)
    d = np.load(npz)
    src, fake, tgt = d["src"], d["fake"], d["tgt"]          # (B,1,n_ax,n_lat) in [-1,1]
    n = min(4, src.shape[0])
    ep = npz.stem.replace("samples_ep", "")

    rows = [("CBCT slice (input)", src), ("generated US (fake)", fake),
            ("real US (target, unpaired)", tgt)]
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n + 0.6, 5.4)); fig.set_constrained_layout(True)
    axes = np.atleast_2d(axes)
    for r, (label, stack) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(_norm(stack[c, 0]), cmap="gray", aspect="auto")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
            if r == 0:
                axes[r, c].set_title(f"#{c}", fontsize=TYPE["small"])
        axes[r, 0].set_ylabel(label, fontsize=TYPE["small"])
    fig.suptitle(f"learned renderer (CUT) — epoch {ep}", fontsize=TYPE["body"] + 1)

    out = ROOT / "figures" / "renderer"; out.mkdir(parents=True, exist_ok=True)
    save(fig, str(out / f"samples_ep{ep}"))
    plt.close(fig)
    print(f"{npz.name}: src/fake/tgt {tuple(src.shape)} -> {out}/samples_ep{ep}.png")


if __name__ == "__main__":
    main()
