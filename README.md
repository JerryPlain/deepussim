# DeepUSSim

**DeepUSSim** investigates whether a pretrained ultrasound *foundation-model encoder* is a
more robust visual representation for robotic ultrasound (US) than a model trained from
scratch. Answering this requires large, spatially aligned, labelled US data — prohibitively
expensive to acquire on hardware (the present study has only two real scan sequences).
DeepUSSim instead *calibrates a generator* — "CBCT volume + probe pose → US image + anatomy
mask" — from a small amount of real robot data, then samples probe poses freely in
simulation to synthesize aligned multimodal data at scale.

The system is organized in two stages. **Stage 1 (real-to-sim)** builds and calibrates the
generator: it fuses a CBCT volume of the phantom with two real robot-collected US sequences,
recovering both the *geometry* (which CBCT slice each US frame corresponds to) and the
*appearance* (how a CBCT slice should look as US). **Stage 2 (sim-to-real)** consumes the
generated data to evaluate encoders and to drive the downstream tasks (segmentation,
navigation). One-line compression: *CBCT supplies the anatomy ground truth; real US supplies
the modality constraint (geometry via LC2, appearance via a learned renderer); simulation
supplies the scale — which is then used to ask which representation transfers best.*

## Pipeline

Real US enters Stage 1 at exactly two points — **geometric registration** and **render
supervision** — and never elsewhere; everything downstream is synthesized.

```mermaid
flowchart TB
    CBCT[("CBCT DICOM<br/>(phantom)")]
    USREAL[("Real US<br/>2 ROS1 rosbags")]
    PROBE[["Probe mesh<br/>(supplied)"]]

    subgraph STAGE1["Stage 1 · real-to-sim — build and calibrate the data generator"]
        direction TB
        ASSETS["Assets (3D Slicer / VTK)<br/>intensity volume · surface mesh · label"]
        CHAIN["Geometric chain (poses → CBCT voxels)<br/>p_C = C_T_R · R_T_E(t) · E_T_U · U_T_img · p_img"]
        LC2["LC2 registration<br/>grind the cm residual → aligned {US ↔ CBCT slice} pairs"]
        REND["Appearance renderer<br/>CBCT slice → US-like image"]
        TRAJ["Trajectory<br/>surface point cloud → probe poses"]
        SCALE["Scale-up (Genesis physics + reslice + render)<br/>US image · pose · contact force · anatomy mask"]
    end

    subgraph STAGE2["Stage 2 · sim-to-real — answer the research question"]
        direction TB
        ENC["Encoder evaluation<br/>foundation vs. from-scratch"]
        SEG["Segmentation (which anatomy)"]
        NAV["Navigation (probe → target)"]
    end

    CBCT -->|3D Slicer| ASSETS
    USREAL -. registration .-> LC2
    USREAL -. render supervision .-> REND
    PROBE --> SCALE
    ASSETS --> CHAIN --> LC2 --> REND --> SCALE
    ASSETS --> TRAJ --> SCALE
    SCALE --> ENC --> SEG --> NAV

    classDef s1 fill:#e6f4f1,stroke:#2a9d8f,color:#14323b;
    classDef s2 fill:#e7eefc,stroke:#3a6ea5,color:#16233f;
    classDef inp fill:#fff3e0,stroke:#e07a1f,color:#5a3a00;
    class ASSETS,CHAIN,LC2,REND,TRAJ,SCALE s1;
    class ENC,SEG,NAV s2;
    class CBCT,USREAL,PROBE inp;
```

### Geometric backbone

Every US pixel is mapped to a physical position inside the CBCT volume by the rigid chain
(with `A_T_B` denoting "take a point from frame *B* into frame *A*"):

```
p_C  =  C_T_R · R_T_E(t) · E_T_U · U_T_img · p_img
```

| Segment | Meaning | Source / status |
|---|---|---|
| `U_T_img` | US intrinsics (pixel → fan mm) | convex fan **fitted** from the real B-mode + Feng's `us_spacing` (`calib.us_geometry.fit_fan_geometry`) |
| `E_T_U` | probe ↔ end-effector (hand-eye) | `Rz(45°)`, −0.183 m on z; mount = `T_EE_FROM_PROBE` (the measured matrix is its inverse, `T_PROBE_FROM_EE`), resolved by replay |
| `R_T_E(t)` | end-effector pose (forward kinematics) | per-frame, from the rosbag pose topic |
| `C_T_R` | CBCT ↔ robot | measured `PhTR` (robot↔phantom) **+ a belly-up lie-down** (`Rx90·Rz180`): `PhTR` lives in the CBCT scan frame `{c}`; the phantom is tipped *lying belly-up* and seated onto the contacts by `calib.seat_phantom_placement` |

The cm-scale residual that accumulates along this chain is not resliced directly; **LC2**
(Linear Correlation of Linear Combination) registers each real US frame to the CBCT by image
content, emitting the `{US ↔ slice}` pairs that supervise the renderer. LC2 fixes *pose*
(where); the renderer fixes *appearance* (how it looks) — the two are kept on separate axes
and never fused into one loss. On the present low-texture phantom LC2 reliably removes the
gross error and aligns the surface, but saturates before mm precision (see Status).

### Trajectory generation

The probe can only glide *along* the phantom surface (it cannot pierce it), so scan
trajectories are constrained to that surface. Each pose is built from the CBCT surface mesh:

1. **Sample** points across the surface — the candidate spots where the probe sits.
   `pipeline.sampling.surface_raster` lays parallel guide lines over the phantom top and
   projects each to the nearest mesh vertex (`surface_sweep` is the single-line case).
2. **Estimate a smoothed surface normal** at each point (`_smoothed_normal`: PCA over a
   neighbourhood, to avoid the staircase noise of the threshold-extracted mesh).
3. **Build a probe pose** (`_pose_on_surface`): position on the surface (with a small outward
   standoff so the probe rests on it rather than inside), axial axis along the inward normal
   (perpendicular contact, as in SonoGym); the in-plane rotation is a free, consistent convention.
4. **Order** the points into a smooth (serpentine) scan path.

The code is [`pipeline/sampling.py`](src/deepussim/pipeline/sampling.py); it is exercised on a
sphere in [`tests/test_sampling.py`](tests/test_sampling.py) (poses rest on the surface +
standoff, axial · inward-normal ≈ 1, raster covers a 2-D patch). On the real phantom a 120-pose
raster sits at the 2 mm standoff with axial · inward-normal ≈ 0.99.

The resulting pose stream (`T_cbct_from_probe`, mm) is used **twice from one source** — fed
directly to reslice the volume, *and* mapped through the seated placement
(`calib.seat_phantom_placement`, the inverse of the reslice bridge) into the sim world to drive
the arm and press the Genesis probe into contact. Because the arm path and the reslice plane
come from the *same* poses, "where the probe is" and "which slice we image" are aligned by
construction — the same property that makes the anatomy masks free.

**Trajectory sources** feed the same machinery (the `--trajectory` flag of
[`run_scaleup.py`](scripts/run_scaleup.py)): **replay** (sim only) drives the real rosbag EE poses
— reachable and on-surface by construction, the faithful reference; **generated** (`surface` /
`raster` / `contact`) samples novel poses for scale-up, mapped into the sim to drive the arm
(out-of-reach poses are dropped).

> ⚠️ **Known limitation (this phantom).** The supplied surface mesh is **not watertight**, so its
> normals are unreliable — the mesh-normal-based generators (`surface_sweep`, `surface_raster`,
> `contact_raster`) can mis-orient the probe (real-axial · mesh-normal ≈ −0.26 where it should be
> +1). For now the **reliable** trajectory is `replay`; a robust generated trajectory needs either
> a watertight mesh (fill holes / remesh) or to orient poses from the real EE poses rather than the
> mesh. [`scripts/view_trajectory.py`](scripts/view_trajectory.py) renders a saved/generated
> trajectory (`--trajectory-file` or `--trajectory contact`) for inspection.

A generated trajectory is **deterministic** from (mesh + params), so it need not be stored — the
achieved poses are written into the dataset anyway; `--save-trajectory <path.npz>` optionally
dumps it (`T_cbct_from_probe`, mm) for inspection or to reuse the identical poses across runs.
[`scripts/view_trajectory.py`](scripts/view_trajectory.py) renders one (generate from the mesh,
or load a saved `.npz`) — poses coloured by scan order with inward axial arrows, plus the
standoff / axial·normal it sits at — and saves the figure.

### Design invariants

- **Anatomy masks are free.** Reslicing the *label* volume at the same pose as the intensity
  volume yields per-frame segmentation ground truth — no manual labelling.
- **Force comes from physics, not the CBCT.** Contact force is produced by Genesis contact
  dynamics of the probe pressing on a *soft* (tissue-compliant) phantom and force-servoed to a
  realistic target (~few N, `UltrasoundScene.servo_to_force` + `SceneConfig.contact_timeconst`);
  the CBCT yields only image + mask. Reslicing is still rigid (no tissue deformation) — a known
  residual sim-to-real gap.
- **Mesh ≠ volume.** Genesis (a physics engine) consumes the *surface mesh* to make the
  probe glide on the phantom; reslicing consumes the *volume*. The CBCT DICOM yields three
  distinct products (volume, surface mesh, label) that must not be conflated.
- **Three models must not be conflated.** The synthesis/renderer (a Stage 1 tool), the visual
  encoder under evaluation (the research subject), and the control policy (downstream) are
  separate; conflating "is the synthesis good" with "is the representation good" voids the
  conclusion.
- **Dark frames are tagged, never dropped.** Lift-off / non-contact US frames are labelled
  `contact = 0` (negatives for the dataset, lift-off supervision for the renderer) rather than
  silently discarded.

> **Resolved by replay (probe mount + placement direction).** Driving the probe along the real
> rosbag poses and measuring its distance to the phantom surface picks the calibration
> unambiguously ([`scripts/verify_replay.py`](scripts/verify_replay.py)): the mount is
> `T_EE_FROM_PROBE` (the delivered hand-eye matrix `T_PROBE_FROM_EE` is its inverse), and the
> measured robot↔phantom matrix is used as CBCT→world (`T_WORLD_FROM_CBCT`). Under it, contact
> frames land ~1–3 cm from the surface while non-contact ("dark") frames sit 13–21 cm off
> (lift-off) — a two-sided check of the whole chain on both sequences.

> **Resolved: the CBCT-frame orientation (belly-up).** That matrix fixes the probe *position* but
> is expressed in the CBCT scan/optical frame `{c}`, rolled from the DICOM-LPS frame of our
> exported `intensity.nrrd` — applied raw it stands the phantom on end. The real rig has it
> *lying* **belly-up**: the probe presses straight down onto the up-facing anterior surface (the
> real EE axial is world `-z`) and images into the body. The lie-down that reproduces this is
> `Rx(90°)·Rz(180°)` about the phantom centre (`Rx` tips it off-end, `Rz` flips it belly-up),
> centralised in `calib.seat_phantom_placement`. Validated in sim: **14/14 real poses reachable
> pressing from above, fan 82% inside tissue**, vs the earlier belly-down `Rx(90°)` alone that
> imaged out of the body (~10%). **LC2 is not the arbiter here** — the low-texture body makes it
> *prefer* the wrong belly-down graze, so the physical prior + in-tissue geometry decide the
> orientation. (Superseded the belly-down placement of commit `2769501`; see CHANGELOG 2026-06-04.)

## Layout

```
src/deepussim/
  geometry.py        SE(3) transforms + quaternion helpers (geometric primitives)
  data/              volume IO (NIfTI / NRRD / DICOM), dataset records, rosbag extraction
  us/                reslice + US image-formation renderer (calibration params live here)
  calib/             fan-geometry fit (us_geometry), LC2 registration (lc2), lie-down placement
                     (placement.seat_phantom_placement), rigid registration, renderer fitting,
                     measured ETU/PhTR transforms
  sim/               Genesis scene: FR3 + phantom, contact force, trajectory (lazy import)
  assets/franka_fr3/ vendored Franka Research 3 MJCF + meshes (MuJoCo Menagerie)
  pipeline/          pose sampling (surface_sweep/raster) + scale-up dataset generation
configs/             renderer / phantom / trajectory parameters
scripts/             run_scaleup.py, run_lc2.py, fit_us_geometry.py, extract_rosbags.py,
                     verify_replay.py, view_sim.py, view_trajectory.py,
                     run_real_collection.py, make_synthetic_phantom.py, smoke_sim.py
docs/                data_collection.md (field checklist), data_layout.md (data/ tree + how to fetch)
tests/               geometry / quaternion / reslice / renderer / placement / rosbag / transforms unit tests
```

Data files (CBCT, rosbags, derived NRRD/STL) are **not** in git — see
[`docs/data_layout.md`](docs/data_layout.md) for the `data/` tree and how to fetch it.

## Robot model

The sim uses the **Franka Research 3 (FR3)** — the real hardware — vendored from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_fr3`
model (Apache-2.0, license kept under `src/deepussim/assets/franka_fr3/`). Genesis only
bundles the older Panda; FR3 differs in joint limits, dynamics, and meshes. The FR3
model has no gripper — it ends at the `fr3_link7` flange with an attachment site 0.107 m
out, where the US probe mesh mounts. IK drives `fr3_link7`, but the rosbag/hand-eye are
measured from the *flange*, so `SceneConfig.probe_offset = trans(0,0,0.107) ∘ T_EE_FROM_PROBE`
(link7→flange→transducer); the bare flange offset misses real poses by ~15 cm, the composed
one reaches them to ~7 mm.

## Setup

Genesis (`genesis-world`) supports **Python 3.10–3.12** (not 3.13). Use a conda env:

```bash
conda create -n deepussim python=3.11 -y
conda activate deepussim
pip install -e ".[dev]"
# Genesis does NOT bundle torch — install it explicitly. On Linux the default wheel
# is the CUDA build (verified: torch 2.12.0+cu130 on an RTX 4060):
pip install torch
# pin the resolved Genesis version afterwards:
pip freeze | grep genesis-world
```

Verify the sim end-to-end (needs a GPU; ~60s to compile kernels on first run):

```bash
python scripts/smoke_sim.py   # Franka presses the phantom; prints contact force + pose
```

The geometric core (`geometry`, `us.reslice`, `us.renderer`, `calib.registration`)
only needs numpy/scipy and runs without Genesis installed. The `sim` package imports
Genesis lazily, so the rest of the package is importable without it.

### HPC / Apptainer (NHR@FAU)

For reproducible GPU/sim runs on the cluster (Alex/Helma), the brittle
`genesis-world` + `torch` + CUDA stack is containerized in [`apptainer/`](apptainer/).
Build on a frontend, then run with the GPU:

```bash
export WORK=/path/to/your/work
bash apptainer/build.sh                                   # → $WORK/deepussim.sif
apptainer/run.sh python scripts/smoke_sim.py              # --nv + live src bind
sbatch apptainer/job.slurm                                # single-GPU batch run
```

The conda env above is still fine for the CPU-only core. See
[`apptainer/README.md`](apptainer/README.md) for build/run/Slurm details and the
one CUDA/driver caveat (container CUDA must match Alex's host driver).

## Reproduce

All commands assume the conda env above is active. The data files (CBCT, rosbags, phantom
mesh) are **not** in git — see [`docs/data_layout.md`](docs/data_layout.md) for the `data/`
tree and how to fetch it. The tests and the synthetic quickstart need neither data nor a GPU.

```bash
pytest -q          # geometric core + pipeline unit tests (no data, no GPU)
```

### Quickstart — synthetic (no data, no GPU)

An aligned dataset from a generated phantom, exercising the whole reslice→render→mask path:

```bash
python scripts/make_synthetic_phantom.py --out data/phantom
python scripts/run_scaleup.py --volume data/phantom/intensity.nii.gz \
    --labels data/phantom/labels.nii.gz --out data/synth_ds --n 64
```

### Stage 1 end-to-end (real data)

Run the calibrated generator from the raw inputs, in order. Steps 1–3 and 5 are CPU-only;
step 4b (the sim force channel) needs a CUDA GPU. `us_spacing = 0.166112957` mm/px is Feng's
measured US pixel size.

**1 · Assets** — place `data/cbct/` (intensity + label NRRD, surface STL) and `data/rosbags/`
per [`docs/data_layout.md`](docs/data_layout.md). The volume/mesh/label are exported once from
the CBCT DICOM in 3D Slicer.

**2 · Extract the real US sequences** (pure Python, no ROS install) — pose-synced, dark-tagged:

```bash
python scripts/extract_rosbags.py data/rosbags/phantom.bag data/rosbags/phantom1.bag \
    --out data/sequences --preview 8
```

**3 · Fit the US fan geometry** (`U_T_img`) from the real B-mode + `us_spacing`; prints the
`configs/renderer.yaml` geometry block and an outline overlay:

```bash
python scripts/fit_us_geometry.py --seq data/sequences/phantom.npz data/sequences/phantom1.npz \
    --us-spacing 0.166112957 --overlay data/sequences/fan_fit.png
```

**4 · Generate the geometry/force channels** — reslice the CBCT along probe poses, render a
US-like image, and read anatomy masks free from the label volume.

  *4a · no-sim* (geometric poses, CPU) — surface-constrained sweep over the phantom:

```bash
python scripts/run_scaleup.py --volume data/cbct/intensity.nrrd \
    --labels data/cbct/labels.nrrd --mesh data/cbct/phantom_surface.stl \
    --trajectory surface --config configs/renderer.yaml --out data/ds --n 64
```

  *4b · sim force channel* (GPU) — the FR3 follows the **real `replay`** trajectory onto the
  phantom (placed *belly-up* and seated on the contacts), force-servoing each pose to a realistic
  ~few-N contact, writing (US image + pose + anatomy mask + contact force):

```bash
python scripts/run_scaleup.py --volume data/cbct/intensity.nrrd \
    --labels data/cbct/labels.nrrd --mesh data/cbct/phantom_surface.stl \
    --config configs/renderer.yaml --out data/ds_sim --sim --headless --n 64
# drop --headless to watch live in the Genesis viewer; --force-n / --contact-timeconst tune the contact
```

  `--trajectory raster`/`surface`/`contact` drive a *generated* trajectory instead, but on this
  phantom they mis-orient (the mesh is not watertight — see the Trajectory-generation note); use
  `replay` until that is fixed.
```

**5 · LC2 registration** (`{US ↔ CBCT slice}` pairs) — refine each real frame's calibration
pose against the CBCT by image content. The belly-up lie-down + contact-seat placement is
applied automatically from `--mesh`:

```bash
python scripts/run_lc2.py --seq data/sequences/phantom.npz --volume data/cbct/intensity.nrrd \
    --mesh data/cbct/phantom_surface.stl --us-spacing 0.166112957 --n 32 --out data/lc2_phantom.npz
```

### Watch / verify the calibration

```bash
python scripts/view_sim.py        # interactive viewer: arm sweeping the lying phantom
python scripts/verify_replay.py   # geometric check of the probe-mount + placement calibration
python scripts/view_trajectory.py --mesh data/cbct/phantom_surface.stl --trajectory raster \
    --out data/trajectories/raster.png --save-trajectory data/trajectories/raster.npz  # render+save a trajectory
```

## Status

Stage 1's **geometry, calibration, trajectory, and sim loop are implemented and verified
end-to-end**; the **appearance (learned renderer)** is the remaining build.

**Implemented & verified.**
- **Geometric core** — SE(3) + quaternions, convex-fan reslice, first-pass acoustic renderer,
  rigid registration (numpy/scipy only, no GPU).
- **Sim** — `sim.scene` (FR3 presses the phantom on the GPU, Genesis 0.4.7); the real probe
  mesh is mounted on the flange (`probe_offset = trans(0,0,0.107) ∘ T_EE_FROM_PROBE`, IK reaches
  real poses to ~7 mm); `servo_to_force` holds a realistic ~few-N contact on a soft
  (`contact_timeconst`) surface (vs. the ~10²–10³ N of a rigid press).
- **Real-data ingestion** — the two ROS1 rosbags are extracted (pose-synced, dark-tagged) via
  `data.rosbag`; the real CBCT loads via `data.load_nrrd`; the measured hand-eye `E_T_U` and
  robot↔phantom `PhTR` live in `calib.transforms`.
- **Calibration resolved** — `verify_replay.py` fixed the probe mount + placement direction
  (contact frames ~1–3 cm on-surface, dark frames 13–21 cm off, both sequences); the CBCT
  scan-frame orientation is resolved to **belly-up** (`Rx90·Rz180`) and centralised in
  `calib.seat_phantom_placement` — validated in sim (14/14 real poses reachable from above, fan
  82% inside tissue).
- **US intrinsics** — the convex fan (`radius 52.5`, `fov 68.1°`, `depth 101.5 mm`) is fitted
  from the real B-mode + Feng's `us_spacing` (`calib.us_geometry`).
- **Scale-up loop** — `run_scaleup --sim` drives the arm along the real `replay` trajectory,
  presses, bridges each achieved pose into the CBCT frame, and reslices to (US image + anatomy
  mask + contact force). `--trajectory` also accepts generated sources mapped into the sim.
- **LC2 registration** — image-based `lc2_similarity` + constrained 6-DoF `register_frame_lc2`,
  initialised from the seated placement. 47 tests pass. (Absolute LC2 stays low on this
  low-texture phantom — it is *not* a reliable arbiter; see the placement note above.)

**Pending (Stage 1).**
- **Learned renderer (the appearance branch)** — only renderer-*parameter* fitting exists; train
  an image-translation renderer (pix2pix paired on the LC2 `{US ↔ slice}` pairs / CycleGAN
  unpaired at scale). This is the main remaining piece.
- **Generated trajectory on this phantom** — the surface mesh is not watertight, so the
  mesh-normal samplers (`surface_sweep`/`surface_raster`/`contact_raster`) mis-orient the probe;
  `replay` is the only reliable trajectory for now. Fix: a watertight mesh, or orient generated
  poses from the real EE poses (see the Trajectory-generation note above).
- **LC2 accuracy / placement** — LC2 removes the gross error but **saturates before mm precision
  on this low-texture phantom**, and the refinement is bound-limited (the placement still carries
  a ~cm offset, seated on the contact cloud rather than from fiducials). Tighten the seat (along
  the normal / per-frame) so LC2 polishes rather than hunts.
- **Contact-depth realism** — force magnitude is realistic, but on the soft contact the probe
  indents deeper than physical; mm-indent-at-target needs a deformable soft-body phantom or finer
  contact/servo tuning. Reslicing is still rigid (no tissue deformation).
- **CT scale to confirm** — `intensity.nrrd` reports 0.742822 mm isotropic vs Feng's `ct_spacing`
  0.810738 (~9% off, likely a resample on export); confirm before trusting the CBCT-side mm scale.

**Stage 2.** Encoder evaluation, segmentation, and navigation follow the proposal and begin
once Stage 1 produces data.
