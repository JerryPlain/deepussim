"""Acceptance gate: the CBCT display sector must land on the real US fan.

RED as of 2026-07-24 (git tag ``pre-fan-fix``). The CBCT display sector in ``pairs.npz`` --
and therefore every segmentation mask projected through it -- is not pixel-aligned to the
real US frame. See ``figures/10_fan_geometry_mismatch/``.

Why these tests and not the old check
-------------------------------------
``project_labels_to_us.py`` self-checks ``corr(reslice, pairs['cbct']) == 1.000``. That is a
tautology: it proves the reslice reproduces the *stored sector*, never that the sector matches
the *US*. These tests compare against the fan fitted from the real B-mode instead.

Choice of reference
-------------------
The reference is the **analytic fan** rasterised from ``fit_us_fan`` -- not a thresholded US
intensity mask. Measured on scan1, the analytic fan only reaches IoU 0.736 against the US
contact envelope (>20) and 0.30 against a single frame, because B-mode content is speckle and
dark anatomy drops below any threshold. So an intensity mask cannot support a tight bound;
two geometric constructs can. ``ENVELOPE_IOU_FLOOR`` keeps a loose sanity check on the
intensity side to catch gross errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

US_SPACING_MM = 0.166112957          # matches pair_generation.DEFAULT_US_SPACING
REF_SEQUENCE = REPO_ROOT / "data" / "sequences" / "scan1.npz"
PAIRS = REPO_ROOT / "data" / "renderer_lc2_pairs" / "pairs.npz"
VOLUME = REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd"

# Two geometric supports should agree almost exactly once the fix lands.
SECTOR_IOU_MIN = 0.98
# Ceiling imposed by speckle/thresholding, measured for the analytic fan itself (0.736).
ENVELOPE_IOU_FLOOR = 0.70


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"missing data dependency: {path}")


@pytest.fixture(scope="module")
def fitted_fan():
    """The one authoritative probe fan, fitted from the reference sequence."""
    _require(REF_SEQUENCE)
    pytest.importorskip("scipy")
    from lc2.forward import fit_us_fan

    seq = np.load(REF_SEQUENCE, allow_pickle=True)
    fan, geom = fit_us_fan(seq["images"], seq["contact"], US_SPACING_MM)
    return fan, geom, seq


@pytest.fixture(scope="module")
def rendered_sectors():
    """CBCT sectors rendered *by the current code* at the LC2 poses of the ref sequence.

    Tests the rendering path directly rather than the stored ``pairs.npz`` (which is only
    rebuilt in Phase 4), so the fix is verified immediately and stays a valid regression after.
    """
    _require(PAIRS)
    _require(VOLUME)
    pytest.importorskip("SimpleITK")
    from plot_script.plots_reslice.compare import cbct_sector_zoom
    from renderer_training.pair_generation import _load_cbct_frame, DEFAULT_REPORT

    pairs = np.load(PAIRS, allow_pickle=True)
    seqs = np.asarray(pairs["sequence"]).astype(str)
    sel = np.where(seqs == REF_SEQUENCE.name)[0]
    if sel.size == 0:
        pytest.skip("reference sequence not present in pairs.npz")
    vol, affine = _load_cbct_frame(VOLUME, DEFAULT_REPORT)
    shape = pairs["us"][sel[0]].shape[:2]
    return [cbct_sector_zoom(vol, affine, pairs["refined_poses"][i], shape) for i in sel]


def fan_support_mask(shape, fan) -> np.ndarray:
    """Rasterise the fitted sector: apex + [r0, r1] radii + +/- fov/2."""
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    dx, dy = xx - fan.apex_px[0], yy - fan.apex_px[1]
    r = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dx, dy))          # angle from +y (down), matches unwrap_fan
    return (r >= fan.r0_px) & (r <= fan.r1_px) & (np.abs(theta) <= fan.fov_deg / 2.0)


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n > 1:                                        # drop the on-screen logo / border specks
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(mask)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


# --------------------------------------------------------------------------------------
# 1. the hardcoded display constants must equal the fan actually fitted from the US
# --------------------------------------------------------------------------------------

def test_display_fan_depth_and_fov_match_fitted(fitted_fan):
    """Control: these two were genuinely fitted and are already correct."""
    _, geom, _ = fitted_fan
    from plot_script.plots_reslice.compare import FAN

    assert FAN["depth_mm"] == pytest.approx(geom.depth_mm, abs=2.0)
    assert FAN["fov_deg"] == pytest.approx(geom.fov_deg, abs=2.0)


def test_display_fan_near_mm_matches_fitted_apex_radius(fitted_fan):
    """RED: near_mm=15.0 but the fitted virtual-apex radius is ~64.7 mm (4.3x off).

    near_mm is the inner radius of ``sector_mask_in_display_image`` -- i.e. the apex-to-face
    distance -- which is exactly ``ProbeGeometry.radius_mm``. Getting it wrong puts the sector
    apex inside the frame instead of ~348 px above it, so the sector opens far too fast.
    """
    _, geom, _ = fitted_fan
    from plot_script.plots_reslice.compare import FAN

    assert FAN["near_mm"] == pytest.approx(geom.radius_mm, rel=0.05), (
        f"display near_mm={FAN['near_mm']} vs fitted radius_mm={geom.radius_mm:.1f}"
    )


# --------------------------------------------------------------------------------------
# 2. the stored CBCT sector must occupy the real US fan
# --------------------------------------------------------------------------------------

def test_cbct_sector_support_matches_us_fan(fitted_fan, rendered_sectors):
    """The rendered CBCT sector must fill the fitted US fan (was ~0.30 pre-fix)."""
    fan, _, _ = fitted_fan
    ref = fan_support_mask(rendered_sectors[0].shape, fan)
    scores = [iou(ref, sec > 1e-6) for sec in rendered_sectors]
    assert float(np.median(scores)) > SECTOR_IOU_MIN, (
        f"CBCT sector support vs fitted US fan: median IoU={np.median(scores):.3f} "
        f"(min {np.min(scores):.3f}) over {len(scores)} frames"
    )


def test_cbct_sector_covers_the_near_field(fitted_fan, rendered_sectors):
    """The sector must reach the top fan rows, where the US images the capsule (was 0.0 pre-fix).

    A separate assertion from IoU because this is the failure that actually costs us labels:
    near-field anatomy was silently absent from every projected mask.
    """
    fan, _, _ = fitted_fan
    ref = fan_support_mask(rendered_sectors[0].shape, fan)
    fan_rows = np.where(ref.any(axis=1))[0]
    top_band = slice(int(fan_rows.min()), int(fan_rows.min()) + 80)

    covered = [
        (ref[top_band] & (sec[top_band] > 1e-6)).sum() / max(ref[top_band].sum(), 1)
        for sec in rendered_sectors
    ]
    assert float(np.median(covered)) > 0.80, (
        f"near-field coverage in the first 80 fan rows: median={np.median(covered):.3f}"
    )


def test_analytic_fan_still_tracks_the_us_envelope(fitted_fan):
    """Guard on the fan fit itself: if this drifts, the reference above is untrustworthy."""
    fan, _, seq = fitted_fan
    env = seq["images"][np.asarray(seq["contact"], dtype=bool)].max(axis=0)
    ref = fan_support_mask(env.shape, fan)
    assert iou(ref, _largest_filled(env > 20)) > ENVELOPE_IOU_FLOOR
