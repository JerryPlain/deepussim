#!/usr/bin/env python
"""Does physics-based shadowing give the sim realistic deep-field confidence?

Runs the acoustic B-mode model (deepussim.us.renderer.bmode) on the SAME CBCT sectors the CUT
renderer used, with and without shadowing, and compares deep-field confidence against real US
and the CUT-generated US. If shadowing works, the physics sim's deep-conf AUC-vs-real should
drop toward 0.5 (realistic), unlike the CUT gen (~0.97).

    module load python/3.12-base
    python renderer_training/physics_sim_check.py --n 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deepussim.us.renderer import bmode, RendererParams                  # noqa: E402
from renderer_training.confidence_realism import _frame_scores, _auc     # noqa: E402
from style.style import figure, save, C, TYPE                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", type=Path,
                    default=REPO_ROOT / "data" / "rendered_us" / "novel_tilt_paired" / "rendered_us.npz")
    ap.add_argument("--scores", type=Path,
                    default=REPO_ROOT / "figures" / "13_confidence_realism" / "scores.npz",
                    help="cached real/gen deep-conf from confidence_realism.py")
    ap.add_argument("--depth-mm", type=float, default=93.0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "14_physics_shadow")
    args = ap.parse_args()

    d = np.load(args.render, allow_pickle=True)
    cbct = d["cbct_sector"]; cut_gen = d["generated_us"]
    n = min(args.n, len(cbct))
    depth = np.linspace(0.0, args.depth_mm, cbct.shape[1])

    cached = np.load(args.scores)
    r_deep = cached["r_deep"]                              # real US deep-conf (from confidence_realism)
    g_deep = cached["g_deep"]                              # CUT-gen deep-conf

    phys_ns, phys_sh = [], []
    for i in range(n):
        ns = bmode(cbct[i], depth, RendererParams(shadow_scale=0.0))
        sh = bmode(cbct[i], depth, RendererParams(shadow_scale=1.0))
        phys_ns.append(_frame_scores(ns)[1]); phys_sh.append(_frame_scores(sh)[1])
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{n}", flush=True)
    phys_ns, phys_sh = np.array(phys_ns), np.array(phys_sh)

    auc = {"CUT gen": _auc(g_deep, r_deep),
           "phys no-shadow": _auc(phys_ns, r_deep),
           "phys +shadow": _auc(phys_sh, r_deep)}
    print("\ndeep-field confidence, mean (AUC vs real; 0.5=realistic):")
    print(f"  real           {np.nanmean(r_deep):.3f}")
    for name, arr in [("CUT gen", g_deep), ("phys no-shadow", phys_ns), ("phys +shadow", phys_sh)]:
        print(f"  {name:14s} {np.nanmean(arr):.3f}   AUC={auc[name]:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "phys_scores.npz", r_deep=r_deep, g_deep=g_deep,
             phys_ns=phys_ns, phys_sh=phys_sh, auc=np.array(list(auc.values())))
    import matplotlib; matplotlib.use("Agg")
    fig, (axH, axI) = figure(ncols=2, width="double", height=2.5)
    bins = np.linspace(0, 1, 26)
    axH.hist(r_deep[~np.isnan(r_deep)], bins=bins, density=True, alpha=0.7, color=C["baseline_a"], label="real")
    axH.hist(g_deep[~np.isnan(g_deep)], bins=bins, density=True, alpha=0.6, color=C["neutral"],
             label=f"CUT gen (AUC {auc['CUT gen']:.2f})")
    axH.hist(phys_sh[~np.isnan(phys_sh)], bins=bins, density=True, alpha=0.6, color=C["ours"],
             label=f"phys+shadow (AUC {auc['phys +shadow']:.2f})")
    axH.set_xlabel("deep-field confidence"); axH.set_ylabel("density")
    axH.set_title("Physics rendering reproduces real deep-field confidence")
    axH.grid(True, axis="y"); axH.grid(False, axis="x"); axH.legend(loc="upper right", fontsize=TYPE["tiny"])

    # sample: cbct sector -> phys no-shadow -> phys +shadow (eyeball the shadows)
    j = 0
    axI.axis("off")
    from matplotlib import gridspec
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=axI.get_subplotspec(), wspace=0.05)
    titles = ["CBCT sector", "phys (no shadow)", "phys (+shadow)"]
    imgs = [cbct[j], bmode(cbct[j], depth, RendererParams(shadow_scale=0.0)),
            bmode(cbct[j], depth, RendererParams(shadow_scale=1.0))]
    for k, (im, t) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(sub[k]); ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=TYPE["small"]); ax.axis("off")
    save(fig, str(args.out / "physics_shadow"))
    print(f"wrote {args.out}/physics_shadow.pdf + .png")


if __name__ == "__main__":
    main()
