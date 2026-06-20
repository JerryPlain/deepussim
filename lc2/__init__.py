"""LC2 pose refinement on top of the `reslice` slicing logic.

The `reslice` package slices the CBCT at the calibration pose (no US, no optimization).
This package wraps that slicing in an LC2 optimisation: repeatedly reslice while nudging
the pose, scoring each reslice against the real US with the LC2 similarity, to grind the
~cm calibration residual.

Two registration modes (see `register.py`):
  * per-frame  — an independent 6-DoF nudge for each frame (can overfit / graze the surface);
  * global     — ONE small 6-DoF correction shared by all frames (robust, the recommended one).

Self-contained: CBCT fan reslice from `reslice.fan`; the US fan-fit / unwrap (`us_fan.py`)
and the LC2 metric (`metric.py`) are reimplemented here (numpy + scipy only), verified
identical to the original `deepussim` versions. No `deepussim` dependency. CLI:
`python -m lc2.run`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so the sibling `reslice` package resolves.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
