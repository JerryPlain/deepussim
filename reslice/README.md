# reslice

Clean, geometry-faithful reimplementation of the CBCT→US reslicing workflow (formerly
`slicer_3.0`). Same maths — verified identical to `slicer_3.0` (pose and rectangular
reslice differ by 0.0) — but factored into small modules with English docstrings and
no Windows-path / 3D-Slicer-display cruft.

This package does **pure geometric reslicing** (probe pose → CBCT slice). It does **not**
do LC2 / image registration — that lives in the sibling `lc2/` package, built on top of this one.

## Modules

| file | role |
|---|---|
| `_geometry.py` | rigid-transform helpers (numpy-only, self-contained) |
| `io.py` | load a CBCT volume (`.mhd`/`.nrrd`/DICOM dir) and 4×4 transforms |
| `frame.py` | **step 1** — voxel→mm affine + phantom-centred frame |
| `pose.py` | **step 2a** — hand-eye/placement chain → image plane |
| `sampling.py` | **step 2b** — reslice a rectangular plane out of the volume |
| `sector.py` | **step 4** — surface edge, fan apex, sector mask, crop+zoom |
| `build_frame.py` | CLI for step 1 |
| `slice.py` | CLI for steps 2+4 (pose → fan sector + US compare) |

## Coordinate convention

- Raw volume frame is DICOM/LPS millimetres: `+X = Left`, `+Y = Posterior`, `+Z = Superior`.
- The *phantom-centred* frame keeps those axes and only shifts the origin to the phantom
  local origin (default LPS `[0,0,0]`).
- The US image plane is the probe local **X-Z** plane: lateral = probe X, axial (depth) =
  probe Z, normal = probe +Y.

## Usage

```bash
conda activate deepussim

# Step 1 — build the CBCT frame (once per DICOM/recon pair):
python -m reslice.build_frame \
  --dicom-dir data/cbct_20260612/DICOM1 \
  --recon data/cbct_20260612/CBCT.mhd \
  --out-dir reslice/outputs/frame_origin000

# Step 2+4 — reslice a fan sector from a probe pose (defaults: 2026-06-12 scan1, frame 17):
python -m reslice.slice --depth-mm 93 --fov-deg 57 --near-mm 15
```

`--depth-mm` / `--fov-deg` / `--near-mm` set the fan shape; fit them to your US with
`scripts/fit_us_geometry.py`. `--world-from-phantom` overrides the phantom placement;
the default `reslice/outputs/world_from_phantom_liedown.txt` is the calibrated 2026-06-12
placement (`T_WORLD_FROM_CBCT @ Rx(90)·Rz(180)`).

## Figures

Comparison figures (real US vs resliced CBCT sector) are produced by
`plot_script/plots_reslice/compare.py` and written under `figures/reslice/`:

```bash
python -m plot_script.plots_reslice.compare --sequence data/sequences/scan5.npz --frames 4
```

## Notes

- `data/` is gitignored; the CBCT volume/DICOM are expected under
  `data/cbct_20260612/` and the probe sequences under `data/sequences/` (symlink your
  own copies there).
- To reslice a different DICOM/recon + rosbag day, pass the matching `--dicom-dir`/
  `--recon`, `--sequence`, `--volume-path` and `--world-from-phantom` together so the
  pairing stays on one date.
