#!/usr/bin/env python
"""Estimate the convex `ProbeGeometry` from real US frames + the scanner pixel spacing.

Builds a max-envelope over the in-contact frames of the given extracted sequence(s) (the sector
that is ever bright = the imaging fan), fits the convex sector, and prints the physical
`ProbeGeometry` (radius_mm / fov_deg / depth_mm) to paste into `configs/renderer.yaml`.
`--us-spacing` is the scanner's US pixel size in mm/px (from the device / calibration).

    python scripts/fit_us_geometry.py --seq data/sequences/phantom.npz data/sequences/phantom1.npz \
        --us-spacing 0.166112957 --overlay sim_shots/us_fan_outline.png
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from deepussim.calib.us_geometry import contact_envelope, fit_fan_geometry, fit_fan_pixels


def _save_outline(env, fit, path):
    """Overlay the fitted fan outline (sides + arcs) on the envelope for a visual check."""
    from PIL import Image

    H, W = env.shape
    ov = np.stack([env.astype(np.uint8)] * 3, axis=-1)
    ax, ay = fit.apex_px

    def dot(px, py, c):
        if 0 <= py < H and 0 <= px < W:
            ov[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = c

    for t in np.linspace(-fit.fov_deg / 2, fit.fov_deg / 2, 220):
        for rr in (fit.r0_px, fit.r1_px):                      # inner + outer arcs (green)
            dot(int(round(ax + rr * math.sin(math.radians(t)))),
                int(round(ay + rr * math.cos(math.radians(t)))), [0, 255, 0])
    for rr in np.linspace(fit.r0_px, fit.r1_px, 320):          # side edges (yellow)
        for t in (-fit.fov_deg / 2, fit.fov_deg / 2):
            dot(int(round(ax + rr * math.sin(math.radians(t)))),
                int(round(ay + rr * math.cos(math.radians(t)))), [255, 255, 0])
    Image.fromarray(ov).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", nargs="+", required=True, help="extracted .npz sequence(s)")
    ap.add_argument("--us-spacing", type=float, required=True, help="US pixel size (mm/px)")
    ap.add_argument("--threshold", type=int, default=20, help="fan-vs-background threshold")
    ap.add_argument("--n-lat", type=int, default=256, help="output scan lines")
    ap.add_argument("--n-ax", type=int, default=512, help="output samples per line")
    ap.add_argument("--overlay", help="save a fan-outline overlay PNG here")
    args = ap.parse_args()

    env = np.maximum.reduce([
        contact_envelope(np.load(s, allow_pickle=True)["images"],
                         np.load(s, allow_pickle=True)["contact"], args.threshold)
        for s in args.seq
    ])
    fit = fit_fan_pixels(env, threshold=args.threshold)
    geom = fit_fan_geometry(env, args.us_spacing, threshold=args.threshold,
                            n_lat=args.n_lat, n_ax=args.n_ax)
    print(f"fan fit: apex(px)=({fit.apex_px[0]:.0f},{fit.apex_px[1]:.0f}) "
          f"r0={fit.r0_px:.0f} r1={fit.r1_px:.0f} fov={fit.fov_deg:.1f} resid={fit.resid_px:.1f}px")
    print("\nconfigs/renderer.yaml geometry:")
    print(f"  radius_mm: {geom.radius_mm:.1f}")
    print(f"  fov_deg: {geom.fov_deg:.1f}")
    print(f"  depth_mm: {geom.depth_mm:.1f}")
    print(f"  n_lat: {geom.n_lat}")
    print(f"  n_ax: {geom.n_ax}")
    if args.overlay:
        _save_outline(env, fit, args.overlay)
        print(f"\nsaved overlay -> {args.overlay}")


if __name__ == "__main__":
    main()
