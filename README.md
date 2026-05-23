# deepussim

Genesis-based **ultrasound (US) simulation**: reslice a CBCT volume of a phantom,
render simulated US images at arbitrary probe poses, calibrate the renderer against
robot-collected real data, then scale up data generation in simulation.

## Pipeline (data flow)

```
1. CBCT scan of phantom        ->  3D intensity volume  + segmented label volume
                                   (geometric reference  AND  anatomy ground truth)
2. Real collection (Franka)    ->  (US image, probe pose [FK], contact force) tuples
3. Registration to CBCT        ->  one-time coordinate calibration:
                                   T_world->CBCT = T_phantom->CBCT . T_base->phantom . T_flange->US
                                   then FK propagates every frame into the CBCT frame
4. Sim calibration             ->  fit the US renderer params so reslice+render ~ real US
5. Scale-up in simulation      ->  for many poses: reslice(intensity)->render->US image,
                                   reslice(label)->anatomy mask, Genesis physics->contact force
```

Two facts that the scaffold encodes deliberately:
- **Anatomy masks are free**: reslicing the *label* volume at the same pose as the
  intensity volume yields the segmentation ground truth — no manual labelling per frame.
- **Force comes from physics, not CBCT**: contact force is produced by the Genesis
  contact dynamics of the probe pressing on the phantom, not from the volume. CBCT only
  yields image + mask. Rigid reslicing also omits tissue deformation — a known residual
  sim-to-real gap.

## Layout

```
src/deepussim/
  geometry.py        SE(3) transforms (Step 3 math primitives)
  data/              CBCT / label volume IO, dataset records
  us/                reslice + US image-formation renderer (calibration params live here)
  calib/             point-based rigid registration + renderer parameter fitting
  sim/               Genesis scene: Franka + phantom, contact force, trajectory (lazy import)
  pipeline/          pose sampling + scale-up dataset generation
configs/             renderer / phantom / trajectory parameters
scripts/             run_scaleup.py, run_real_collection.py, make_synthetic_phantom.py
tests/               geometry / reslice / renderer unit tests
```

## Setup

Genesis (`genesis-world`) supports **Python 3.10–3.12** (not 3.13). Use a conda env:

```bash
conda create -n deepussim python=3.11 -y
conda activate deepussim
pip install -e ".[dev]"
# pin the resolved Genesis version afterwards:
pip freeze | grep genesis-world
```

The geometric core (`geometry`, `us.reslice`, `us.renderer`, `calib.registration`)
only needs numpy/scipy and runs without Genesis installed. The `sim` package imports
Genesis lazily, so the rest of the package is importable without it.

## Quick start (no real data, no Genesis)

```bash
python scripts/make_synthetic_phantom.py --out data/phantom        # synthetic CBCT + labels
python scripts/run_scaleup.py --volume data/phantom/intensity.nii.gz \
    --labels data/phantom/labels.nii.gz --out data/synth_ds --n 64 # generate a dataset
pytest -q
```

## Status

Scaffold. Runnable: geometry, reslice, renderer (first-pass acoustic model), rigid
registration, no-sim scale-up. Skeletons pending implementation: `sim.scene` (verify
against your installed Genesis API), `calib.renderer_fit` (loss is runnable; optimiser
choice TBD), real hardware capture.
