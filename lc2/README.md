# lc2

LC2 pose refinement built on top of the `reslice` slicing logic.

`reslice` slices the CBCT at the calibration pose — no US, no optimization. `lc2` wraps
that slicing in an optimisation: it repeatedly reslices while nudging the pose, scoring
each reslice against the real US with the **LC2** (Linear Correlation of Linear
Combination) similarity, to grind the ~cm calibration residual toward mm.

```
   pose --> [ reslice CBCT on the US fan ] --> LC2 vs unwrapped US --> high enough?
     ^                                                                     | no
     +------------------- bounded pose nudge <-----------------------------+
```

## Two modes

| mode | what it searches | character |
|---|---|---|
| `per-frame` | one 6-DoF nudge **per frame** | flexible; **can overfit / graze the surface** |
| `global` | **one** 6-DoF correction shared by all frames | constrained, robust — **recommended** |

The bounded Powell search (±15 mm, ±15° by default) keeps the correction inside the
expected calibration residual, so it refines the real pose instead of drifting to an
arbitrary look-alike.

## Anti-gaming guard

LC2 is an imperfect proxy: on low-texture data it can be maximised by grazing the surface
(a high score from a *worse* pose). So `lc2.run` reports, alongside LC2, the fan
**tissue coverage (`%inside`)** before/after. **LC2 up but `%inside` down = gaming, not
alignment** — trust only corrections that raise LC2 while keeping coverage. `global` is
far less prone to this than `per-frame`.

## What it uses (self-contained — no `deepussim` dependency)

* **CBCT slicing** — `reslice.fan.reslice_fan` (same geometry as `reslice`, fan layout so
  it lines up with the US).
* **US fan-fit / unwrap** — `lc2.us_fan` (reimplemented here, verified identical to the
  original `deepussim.calib.us_geometry`).
* **LC2 metric** — `lc2.metric` (reimplemented here, verified bit-for-bit identical to
  `deepussim.calib.lc2.lc2_similarity`).
* **Init pose** — `reslice.pose` calibration chain with the `reslice` default placement,
  so LC2 refines *from the same pose `reslice` would slice at*.

## Usage

```bash
conda activate deepussim

# Global correction (recommended), 12 frames of scan1:
python -m lc2.run --method global --n 12 --out data/lc2/scan1_global.npz

# Compare per-frame vs global on one sequence:
python -m lc2.run --method both --sequence data/sequences/scan5.npz --n 12
```

Key flags: `--method {global,per-frame,both}`, `--sequence`, `--n` (frames),
`--us-spacing` (mm/px, default 0.166112957), `--max-trans-mm` / `--max-rot-deg` (search
bounds), `--out` (save refined poses + scores).

## Note

This refines poses; it does not yet fold the correction back into the calibration
(`deepussim.calib.transforms`) or rebuild the dataset — those are follow-on steps. Always
sanity-check a refinement against `%inside` (and a visual reslice) before trusting it.
