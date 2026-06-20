"""CLI: LC2-refine a sequence's probe poses (per-frame and/or global).

    python -m lc2.run                                   # global, 2026-06-12 scan1
    python -m lc2.run --method per-frame --n 8
    python -m lc2.run --method both --sequence data/sequences/scan5.npz --n 12

Reports LC2 before/after AND the fan tissue-coverage (%inside) before/after — the latter
is the anti-graze guard: a per-frame run that lifts LC2 while *dropping* coverage is gaming
the metric, not aligning. Refined poses + scores are saved to an ``.npz``.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "lc2"

import argparse
import json
from pathlib import Path

import numpy as np

import lc2  # noqa: F401  (path setup for reslice)
from lc2.forward import prepare
from lc2.register import register_frame, register_global

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "reslice" / "outputs" / "frame_origin000" / "physical_frame_report.json"
DEFAULT_VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"
DEFAULT_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_PLACEMENT = REPO_ROOT / "reslice" / "outputs" / "world_from_phantom_liedown.txt"
DEFAULT_US_SPACING = 0.166112957   # mm/px (Feng's calibration)


def _summary(tag: str, before, after, inside_before, inside_after) -> None:
    before, after = np.asarray(before), np.asarray(after)
    ib, ia = np.asarray(inside_before) * 100, np.asarray(inside_after) * 100
    print(f"[{tag}] LC2   mean {before.mean():.3f} -> {after.mean():.3f}  "
          f"(improved {int((after > before).sum())}/{len(after)})")
    print(f"[{tag}] inside mean {ib.mean():.0f}% -> {ia.mean():.0f}%  "
          f"(if LC2 up but inside down => gaming the metric)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("global", "per-frame", "both"), default="global")
    ap.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--volume-path", type=Path, default=DEFAULT_VOLUME)
    ap.add_argument("--world-from-phantom", type=Path, default=DEFAULT_PLACEMENT)
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING)
    ap.add_argument("--n", type=int, default=12, help="number of in-contact frames to use")
    ap.add_argument("--max-trans-mm", type=float, default=15.0)
    ap.add_argument("--max-rot-deg", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=None, help="save refined poses + scores (.npz)")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    ctx = prepare(
        args.sequence, report, args.volume_path, args.world_from_phantom,
        args.us_spacing, args.n,
    )
    print(f"[lc2] sequence {args.sequence.name}: {len(ctx.frames)} frames  "
          f"fan radius={ctx.geom.radius_mm:.1f} fov={ctx.geom.fov_deg:.1f} depth={ctx.geom.depth_mm:.1f}")

    out: dict = {"sequence": str(args.sequence)}

    if args.method in ("global", "both"):
        g = register_global(ctx, max_trans_mm=args.max_trans_mm, max_rot_deg=args.max_rot_deg)
        _summary("global", g["lc2_before"], g["lc2_after"], g["inside_before"], g["inside_after"])
        print(f"[global] correction: rot(deg)={np.round(g['params'][:3], 2)}  "
              f"trans(mm)={np.round(g['params'][3:], 1)}")
        out.update(global_params=np.array(g["params"]), global_correction=g["correction"],
                   global_refined_poses=np.array(g["refined_poses"]),
                   global_lc2_before=np.array(g["lc2_before"]), global_lc2_after=np.array(g["lc2_after"]),
                   global_inside_before=np.array(g["inside_before"]), global_inside_after=np.array(g["inside_after"]),
                   frame_index=np.array(g["indices"]))

    if args.method in ("per-frame", "both"):
        results = [register_frame(ctx, f, max_trans_mm=args.max_trans_mm, max_rot_deg=args.max_rot_deg)
                   for f in ctx.frames]
        _summary("per-frame", [r["lc2_before"] for r in results], [r["lc2_after"] for r in results],
                 [r["inside_before"] for r in results], [r["inside_after"] for r in results])
        out.update(perframe_refined_poses=np.array([r["refined_pose"] for r in results]),
                   perframe_lc2_before=np.array([r["lc2_before"] for r in results]),
                   perframe_lc2_after=np.array([r["lc2_after"] for r in results]),
                   perframe_inside_before=np.array([r["inside_before"] for r in results]),
                   perframe_inside_after=np.array([r["inside_after"] for r in results]),
                   frame_index=np.array([r["index"] for r in results]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **out)
        print(f"[lc2] saved -> {args.out}")


if __name__ == "__main__":
    main()
