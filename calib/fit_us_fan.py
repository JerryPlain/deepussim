#!/usr/bin/env python
"""Fit the probe's display fan once and freeze it to ``calib/us_fan.json``.

The fan is a property of the probe + depth setting, not of a scan, so it is fitted ONCE from
a clean reference sequence and reused everywhere (this is what ``pair_generation.py``'s
``--ref-sequence`` / ``--us-spacing`` arguments were always meant to feed, but never did).

    module load python/3.12-base
    python calib/fit_us_fan.py --sequence data/sequences/scan1.npz

Writes a small tracked JSON so the geometry is reviewable in version control, unlike
everything under the gitignored ``/data/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reslice.us_display import DEFAULT_FAN_JSON, UsDisplayFan   # noqa: E402

DEFAULT_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
DEFAULT_US_SPACING = 0.166112957        # matches renderer_training/pair_generation.py


def fit(sequence: Path, us_spacing_mm: float) -> tuple[UsDisplayFan, np.ndarray]:
    """Fit the display fan from a sequence's contact-frame envelope."""
    from lc2.us_fan import contact_envelope, fit_fan_pixels

    seq = np.load(sequence, allow_pickle=True)
    env = contact_envelope(seq["images"], np.asarray(seq["contact"], dtype=bool))
    fan_fit = fit_fan_pixels(env)
    fan = UsDisplayFan.from_fan_fit(
        fan_fit, us_spacing_mm, env.shape, source=f"{sequence.name}//contact_envelope"
    )
    return fan, env


def _report(fan: UsDisplayFan, env: np.ndarray) -> float:
    """Print the fit and return its IoU against the thresholded envelope."""
    from scipy import ndimage

    m = env > 20
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    sup = fan.support_mask()
    return float((sup & m).sum() / (sup | m).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE,
                    help="clean reference sequence to fit from")
    ap.add_argument("--us-spacing", type=float, default=DEFAULT_US_SPACING,
                    help="US display pixel size in mm")
    ap.add_argument("--out", type=Path, default=DEFAULT_FAN_JSON)
    args = ap.parse_args()

    if not args.sequence.exists():
        raise SystemExit(f"missing sequence: {args.sequence}")

    fan, env = fit(args.sequence, args.us_spacing)
    iou = _report(fan, env)

    print(f"fitted from {fan.source}  frame {fan.rows}x{fan.cols}")
    print(f"  apex_px      : ({fan.apex_x_px:.1f}, {fan.apex_y_px:.1f})"
          f"   {'(above frame)' if fan.apex_y_px < 0 else '(INSIDE frame -- suspicious)'}")
    print(f"  r0_px / r1_px: {fan.r0_px:.1f} / {fan.r1_px:.1f}")
    print(f"  fov_deg      : {fan.fov_deg:.2f}")
    print(f"  resid_px     : {fan.resid_px:.2f}")
    print(f"  -> radius_mm : {fan.radius_mm:.2f}    (ProbeGeometry.radius_mm / sector near_mm)")
    print(f"  -> depth_mm  : {fan.depth_mm:.2f}")
    print(f"  support IoU vs envelope(>20): {iou:.3f}")
    print(f"     (speckle/threshold ceiling -- see tests/test_display_alignment.py)")

    out = fan.save(args.out)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
