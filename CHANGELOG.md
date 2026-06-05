# Changelog

This file tracks meaningful repository changes.

## [Unreleased]

### infra

- **Apptainer image for the GPU/sim path** (`apptainer/`). Containerizes the
  version-sensitive `genesis-world` + `torch` + Taichi/CUDA stack so batch runs on
  NHR@FAU (Alex/Helma) are reproducible; the CPU-only core still runs in the conda
  env. `deepussim.def` (Python 3.11, PyPI torch wheel = CUDA runtime, `pip install
  -e .[dev]`, headless GL libs, baked editable so a host `src/` bind = live code);
  `build.sh` (frontend build, caches→`$WORK`); `run.sh` (`--nv` + live src bind);
  `job.slurm` (single-GPU Alex example); `apptainer/README.md`. **Built & validated
  on `alex1`**: image 3.7 GB, pinned `torch==2.12.0+cu130` / `genesis-world==1.1.0`,
  `56 passed` in-container. **GPU sim verified** via `srun` on an A40 (driver
  595.71.05 / CUDA 13.2 — cu130 compatible): `smoke_sim.py` ran on the `gs.cuda`
  backend, probe contacted the phantom (`|f| = 33.6 N`), ~900 FPS after kernel
  compile. (Harmless `libEGL dri2` warnings from Genesis' offscreen visualizer;
  physics + the numpy US render are unaffected.)

## 2026-06-04 — Correct the placement to belly-up; sim trajectory generation

**Commits:** `8b45433`

Supersedes the orientation of the previous entry. That `Rx(90°)` lie-down tipped the phantom
standing→lying but left it **belly-down**, so the probe grazed / imaged *out* of the body (~10% of
the fan in tissue) and the LC2 ≈ 0.26 reported there was a weak textured-surface graze, not a deep
registration. The real rig has it lying **belly-up** — the probe presses straight down onto the
up-facing anterior surface (the real EE axial is world `-z`) and images into the body.

### calib

- **[API]** `calib.placement.seat_phantom_placement` lie-down is now `Rx(90°)·Rz(180°)` (Rx tips
  it off-end, Rz flips it belly-up). Validated in sim: **14/14** real poses reachable pressing from
  above, fan **82%** inside tissue (vs belly-down `Rx(90°)`: 0/18 reachable, ~10%). **LC2 is not the
  arbiter** here — the low-texture body makes it prefer the wrong belly-down graze, so the physical
  prior + in-tissue geometry decide the orientation. `view_sim.py` + the `transforms.py` note mirror it.
- **[API]** `calib.placement.mm_to_meters` — inverse of `meters_to_mm` (maps a CBCT-frame pose back
  to the sim world to drive the arm).

### pipeline / scripts

- **[API]** `pipeline.sampling.contact_raster` — raster the face the arm actually scanned (from the
  real contact cloud) rather than the mesh's geometric +z top.
- `run_scaleup --sim` follows a **generated** trajectory (mapped into the sim world) as well as the
  real `replay`; `--trajectory` adds `contact`; `--save-trajectory` dumps the poses (`T_cbct_from_probe`).
- **[API]** `UltrasoundScene` optional offscreen render camera (`SceneConfig.render_camera`, `.render()`).
- New `scripts/view_trajectory.py` — paper-quality trajectory figure / GIF (generate or load a `.npz`).

> ⚠️ **Known limitation.** The supplied surface mesh is **not watertight**, so its normals are
> unreliable (real-axial · mesh-normal ≈ −0.26): the mesh-normal samplers (`surface_sweep` /
> `surface_raster` / `contact_raster`) mis-orient the probe. `replay` is the only reliable trajectory
> for now; a robust generated trajectory needs a watertight mesh or orientation from the real EE
> poses. 47 tests pass.

## 2026-06-04 — Fix the 90° CBCT-frame roll in the LC2 init (lie-down placement)

**Commits:** `2769501`

Resolves the previous entry's open question — *"confirming the measured CBCT frame matches the
NRRD affine"*. It did not: the measured `T_WORLD_FROM_CBCT` is expressed in the CBCT scan/optical
frame `{c}`, rolled 90° from the DICOM-LPS frame of our exported `intensity.nrrd`. Applied as-is
it stands the phantom on end, so the LC2 init aimed the fan straight out the floor (into air) —
LC2 ≈ 0 and the constrained refinement could not climb (the ~90° gross error LC2 cannot grind).
`view_sim.py`/`run_scaleup.py` already tipped the phantom back down (`Rx(90°)` about its centre —
the real rig has it *lying*), but that correction never reached the LC2 path. No new matrix from
Feng is needed; the fix was already in our own code.

### calib

- **[API]** `calib.placement.seat_phantom_placement(mesh, contact_origins_m, base)` — one rigid
  `T_world_from_cbctm`: the lie-down roll about the phantom centre + a median contact-seat onto
  the real probe cloud. Reslicing derives `sim_to_cbct = meters_to_mm(invert(·))` from the same
  transform, so mesh, volume and probe poses stay consistent.
- `calib.transforms` — document the `{c}`↔DICOM-LPS 90° roll on `T_WORLD_FROM_CBCT` and point to
  the helper (use it, not the raw matrix, to bridge poses into the DICOM volume).

### scripts

- `scripts/run_lc2.py` — **the bug site**: the calibration init now goes through the seated
  bridge (new `--mesh` arg) instead of the raw matrix.
- `scripts/run_scaleup.py` — use the shared helper; drop the inline lie-down/contact-seat copy.

> **Real-data status.** Replaying the real EE poses into the volume, the fan now images *into*
> the tissue (inside-volume 49%→~100%, the resliced surface boundary lines up with the US
> near-field band — see the broken/fixed/refined overlay) and LC2 climbs on every contact frame
> (init ≈0.016→0.11, refined ≈0.26, 6/6 improved; `Rx+90` beats `Rx-90`, 0.10 vs 0.04). The
> gross 90° is gone. Absolute LC2 stays modest and the refinement is bound-limited (~cm residual):
> this phantom is low-texture in CBCT, so the metric saturates — mm-accuracy is not yet
> established here. Next: tighten the placement (seat along the normal / per-frame) so LC2 polishes
> rather than hunts, then the learned renderer. 46 tests pass.

## 2026-06-02 — LC2 multimodal registration (similarity + fan unwrap + constrained 6-DoF)

**Commits:** `8bb43b8`

The calibration-initialized, constrained-range LC2 rigid refinement (Wein 2013, Fuerst 2014;
the workflow in Li et al. "Robotic Ultrasound Makes CBCT Alive") that grinds the geometric
chain's ~cm pose residual toward mm and produces the `{US ↔ CBCT slice}` pairs the renderer
will be supervised on.

### calib

- **[API]** `calib.lc2.lc2_similarity` — the LC2 metric (`US ≈ α·CT + β·|∇CT| + γ` per local
  window, explained variance weighted by window US variance; predictors normalized so the
  per-window 2×2 solve stays well-scaled on smooth volumes). `lc2_map` exposes the per-pixel
  field; `gradient_magnitude` the |∇CT|.
- **[API]** `calib.lc2.register_frame_lc2` — bounded 6-DoF Powell search around the calibration
  pose (perturb in the probe frame, ±`max_trans_mm`/`max_rot_deg`) maximizing LC2.
- **[API]** `calib.us_geometry.unwrap_fan` — resample a B-mode frame onto the `(n_ax, n_lat)`
  reslice fan grid so the real US and resliced CBCT share a layout for LC2.

### scripts

- `scripts/run_lc2.py` — per-frame driver: calibration init
  `C_T_U = (world←cbct)⁻¹ · ee · E_T_U`, unwrap, register, report LC2 before/after, save poses.

### tests

- LC2 high for a true linear combination, low/dropping when independent/misaligned; fan unwrap
  maps depth→radius; 6-DoF register recovers a perturbed synthetic pose (4.1 mm/LC2 0.65 →
  1.7 mm/LC2 0.997). 46 tests pass.

> **Real-data status.** The driver runs end-to-end and LC2 improves on every frame (≈0.01 →
> 0.24), but absolute LC2 is low: the calibration pose starts the fan partly off-tissue (the
> known ~cm residual). Next: a registration-quality pass — wider/multi-start search and
> confirming the measured CBCT frame matches the NRRD affine (same family as the `ct_spacing`
> question).

## 2026-06-02 — Calibrate the convex probe geometry from the real US fan

**Commits:** `22d31bc`

With Feng's US pixel spacing (`0.166112957` mm/px), the real probe's imaging fan can be
measured instead of guessed — completing the `U_T_img` intrinsic of the geometric chain.

### calib

- **[API]** New `calib.us_geometry`: `fit_fan_geometry(image, us_spacing_mm)` fits the convex
  sector of a B-mode frame (largest-component mask → straight side-edge lines → virtual apex →
  central-column radii → fov) and scales pixel radii to mm, returning a physical
  `ProbeGeometry`. `contact_envelope` builds the robust max-projection over contact frames;
  `fit_fan_pixels` exposes the raw pixel fit (`FanFit`).

### scripts

- `scripts/fit_us_geometry.py` — fit the geometry from extracted sequence(s) + `--us-spacing`,
  print the `renderer.yaml` block, and save a fan-outline overlay.

### config

- `configs/renderer.yaml` geometry now holds the **fitted** convex params (both phantom
  sequences; apex (434,−270) px, r0=316 r1=927 px, residual 3.1 px): `radius_mm 52.5`,
  `fov_deg 68.1`, `depth_mm 101.5` (were placeholder 55/70/110).

> ⚠️ **CT spacing to confirm with Feng.** `data/cbct/intensity.nrrd` reports 0.742822 mm
> isotropic (raw NRRD `space directions`, confirmed by SimpleITK), but Feng's `ct_spacing` is
> 0.810738 (~9% off). Likely the NRRD was resampled on export; the true physical CT scale must
> be confirmed before the CBCT-side mm calibration is trusted. The US side (0.166) is unaffected.

## 2026-06-01 — Close the real-data sim loop (lying phantom, hand-eye, soft contact)

**Commits:** `889b6b6`

`run_scaleup --sim` now runs end-to-end on the real CBCT + rosbags: the FR3 replays the real
probe sweep on the real phantom and writes (US image + pose + anatomy mask + contact force)
with realistic ~few-N forces, non-blank images, and anatomy masks.

### sim

- **Probe-mount reconciliation.** `SceneConfig.probe_offset` is now
  `trans(0,0,0.107) ∘ T_EE_FROM_PROBE` (link7→flange→transducer). IK drives `fr3_link7`, but
  the rosbag pose and hand-eye are measured from the *flange* (link7 + 0.107 m); the bare
  placeholder `[0,0,0.287]` missed real recorded poses by ~15 cm, the composed offset reaches
  them to ~7 mm.
- **[API]** `SceneConfig.phantom_scale` (mesh unit scale; `0.001` loads a mm mesh as metres)
  and `SceneConfig.contact_timeconst` (rigid-contact compliance; soft tissue-like values give
  realistic few-N forces instead of a rigid press's 10²–10³ N). `servo_to_force` settle bumped
  for the compliant surface.

### pipeline

- **[API]** `generate_dataset(force_target_n=...)` routes the sim press through
  `servo_to_force` (holds a target contact force) instead of `servo_to_contact`.

### scripts

- `run_scaleup --sim` rewritten: places the phantom **lying** (`T_WORLD_FROM_CBCT ∘ Rx(90°)`,
  matching the rig and `view_sim.py`), seats it onto the real contact cloud, and **derives the
  reslice bridge from the same transform** (round-trip 0.006 mm). Trajectory **replays** the
  rosbag contact poses (reachable + on-surface by construction), subsampled to `--n`. New
  flags: `--force-n`, `--bags`, `--contact-timeconst`.

### docs

- README: ordered **Reproduce** section (synthetic → real no-sim → real sim force channel);
  updated probe-mount, force-from-physics, and status notes. Known gap recorded: on the soft
  contact the probe indents deeper than physical (mm-indent-at-target needs a deformable
  soft-body phantom or finer contact/servo tuning).

## 2026-06-01 — Real US probe in sim + contact-force servoing

**Commits:** `993dce3`

### sim

- Mounted the **real US probe** under `fr3_link7`: the convex hull of `convex_model_stl.stl`
  (scaled to metres, committed as `us_probe_hull.obj`) replaces the placeholder cylinder, so
  the probe is part of the articulated arm and its contact loads back into it (the flange no
  longer has to touch the phantom). MuJoCo collides mesh geoms via the convex hull, so the
  hull is collision-equivalent yet small enough to track. `SceneConfig.probe_offset` moves to
  the real tip (~0.287 m on link7).
- **[API]** Added `UltrasoundScene.servo_to_force`: a proportional depth-vs-force loop that
  holds a realistic contact force (~few N) instead of the fixed-depth press's 10²–10³ N.
- Verified on GPU with the real phantom placed via `T_WORLD_FROM_CBCT`: the probe reaches a
  mapped surface-trajectory pose and contacts; servoing pulls the force from ~63 N to ~7 N.

## 2026-06-01 — Multi-line raster trajectory for area-coverage scale-up

**Commits:** `51e873d`

### pipeline / scripts

- Added `pipeline.sampling.surface_raster`: covers a patch of the phantom's top surface with
  `n_lines` parallel `surface_sweep`s of `n_per_line` poses each, stepping across the
  perpendicular in-plane axis and alternating direction (serpentine) for a continuous
  lawnmower path; the KD-tree is built once and shared. Factored the per-guide pose logic out
  of `surface_sweep` so both share it. **[API]**
- Wired into `run_scaleup.py` as `--trajectory raster` (`--lines` / `--per-line` /
  `--cross-frac`), so one command generates an area-covering geometric dataset.
- Verified on the real phantom: 120 poses cover ~208×113 mm, all resting at the standoff with
  axial · inward-normal ≈ 1.0. Grows the geometry branch from a single line to a
  density-controlled patch; trajectories are reused once the learned renderer lands.

## 2026-06-01 — Curvilinear (convex) probe geometry; drop the linear model

**Commits:** `02c611f`

### us

- **[BREAKING][API]** `ProbeGeometry` now models the real **curvilinear (convex)** probe: a
  sector fan whose scan lines diverge from a virtual apex at `z = -radius_mm`, span
  `± fov_deg/2`, and sample from the face out to `radius_mm + depth_mm`. Fields changed from
  `width_mm` to `radius_mm` + `fov_deg`. The rectangular linear model was removed (a convex
  array with `radius → ∞` is its degenerate case, so one geometry suffices) — this matches
  the actual probe and unifies the geometry used by reslice, render, calibration, and
  scale-up.
- `reslice`/`render` keep the same interface (`n_ax`/`n_lat`/`axial_depths_mm`/`plane_grid`),
  so no call sites changed; `configs/renderer.yaml` and the reslice tests move to the convex
  fields. `radius_mm`/`fov_deg` are placeholders until estimated from the real US fan.

## 2026-05-31 — Surface-constrained trajectory + wire into scale-up

**Commits:** `0803d4f`

### pipeline / scripts

- Added `pipeline.sampling.surface_sweep` (+ `top_sweep_endpoints`): the real trajectory
  generator. It constrains poses to the phantom surface mesh — sample a point, estimate a
  PCA-smoothed normal (robust to the threshold mesh's staircase noise), rest the probe on the
  surface with the axial axis pointing inward, and order the points into a scan path. **[API]**
- Wired it into `run_scaleup.py` as `--trajectory surface --mesh <stl>`, and switched volume
  loading to `load_volume` so the real `.nrrd` assets load (the old `load_nifti` could not).
  **[BREAKING]** for the script's volume-format assumption.
- Verified on the real phantom: poses sit exactly at the standoff, axial · inward-normal
  ≥ 0.998, and reslicing produces anatomy-bearing slices with free aligned masks. This closes
  the pipeline's geometry branch (Assets → Trajectory → reslice).

## 2026-05-31 — Resolve hand-eye/placement by replay; drop H1/H2 naming

**Commits:** `0803d4f`

### calib / scripts

- Added `scripts/verify_replay.py`: a Genesis-free geometric check that drives the probe along
  the real rosbag poses and measures its distance to the phantom surface. It resolves the two
  ambiguous calibration directions — contact frames land ~1–3 cm on the surface and dark frames
  13–21 cm off (lift-off), on both sequences, vs 0.17–0.76 m for every alternative.
- **[API]** `calib.transforms` now encodes the resolved result with directionally accurate
  names instead of H1/H2 hypotheses: `T_PROBE_FROM_EE` (the delivered hand-eye matrix),
  `T_EE_FROM_PROBE` (its inverse — the probe mount the chain uses), and `T_WORLD_FROM_CBCT`
  (the delivered placement, applied directly as CBCT→world). Removed the `ETU`/`probe_offset`
  hypothesis API.
- Removed informal personal attributions from code and docs (neutral "as delivered" wording).

## 2026-05-31 — Rewrite README to the two-stage pipeline (paper-standard)

**Commits:** `00e8e50`

### docs

- Reframed `README.md` around the research question (is a pretrained US foundation-model
  encoder a more robust representation than from-scratch) and the two-stage real-to-sim /
  sim-to-real structure, replacing the earlier single-flow description.
- Added a rendered **Mermaid** pipeline figure, the formal US-pixel → CBCT-voxel transform
  chain with a per-segment provenance table, the design invariants (free masks, force from
  physics, mesh ≠ volume, the three-models distinction, dark-frame tagging), and the open
  `E_T_U` H1/H2 direction question. Updated Layout and Status to the current real-data state.

## 2026-05-31 — Document data layout & out-of-band sharing

**Commits:** `1f302bf`

### docs

- Added `docs/data_layout.md`: the `data/` tree, where each asset comes from (raw inputs +
  probe mesh on the shared Google Drive; intensity/label/surface derived in 3D Slicer), and
  how to reproduce every derived asset from raw inputs. Data stays out of git — the rosbags
  and the 122 MB intensity volume exceed GitHub's 100 MB per-file limit — so teammates
  assemble `data/` locally from the Drive folder.

## 2026-05-31 — Real-data ingestion: rosbags, calibration transforms, NRRD volumes

**Commits:** `7298d94`

### data — real US sequence extraction

- Added `data/rosbag.py` to extract the two real ROS1 rosbags (`phantom.bag`,
  `phantom1.bag`) in pure Python via the `rosbags` library — no ROS1/ROS2 install or
  format conversion needed. It reads the `/frame_grabber/us_img` (8UC3) and
  `/fr3/state/current_pose` (PoseStamped, `T_base_from_ee` = R T_E(t)) topics, matches
  each image to its nearest pose by header stamp, and converts ROS `xyzw` quaternions to
  the scalar-first convention used elsewhere. **[API]** `extract_sequence`, `UsFrame`,
  `UsSequence`.
- Dark / non-contact frames (probe lifted off) are **tagged** (`contact=False`) with a
  configurable intensity threshold and never silently dropped; each frame keeps its
  grayscale mean so the cut can be re-tuned without re-reading the bag.
- Added `scripts/extract_rosbags.py` to dump synced sequences to compressed `.npz`
  (images + poses + contact + stamps) plus preview PNGs. Verified: 648 / 447 frames,
  median image↔pose sync error 2.7 ms.

### calib — measured rig transforms

- Added `calib/transforms.py` holding the two real calibration matrices so they live in
  code rather than scattered in chat: `ETU` (end-effector ↔ probe, Rz(45°) + −0.183 m)
  and `T_PHANTOM_FROM_ROBOT` (PhTR, robot expressed in the phantom frame). **[API]**
- `probe_offset(hypothesis)` returns `T_ee_from_probe` for the unresolved ETU direction
  (H1 = use `ETU` directly, H2 = its inverse); the replay task will disambiguate which.

### data — NRRD volume loading

- Added `load_nrrd` and an extension-dispatching `load_volume` to `data/volume.py` so 3D
  Slicer's default NRRD exports (intensity + label) load directly via SimpleITK,
  factoring the SimpleITK→affine logic shared with `load_dicom_series`. **[API]**

### Repository hygiene

- **[BREAKING]** Fixed `.gitignore`: the unanchored `data/` pattern was also matching the
  `src/deepussim/data/` source package, so the entire data subpackage (`volume.py`,
  `record.py`, `rosbag.py`, …) had never been tracked. Anchored to `/data/` and
  `/datasets/`; added `/*.bag` and `/*.stl` to stop stray raw inputs at the repo root
  from being committed. The `src/deepussim/data/` package must now be added to git.

## 2026-05-28 — Update robot-object placement and import phantom mesh

**Commits:** `972d2fe`

### sim — phantom mesh

- Imported a real phantom surface mesh and updated the robot/object placement so the arm
  reaches a mesh phantom rather than only the box fallback, making the contact geometry in
  the viewer representative of the intended scene.

## 2026-05-24 — Expand data collection guide: pitfalls + deliverables

**Commits:** `d1ec1fc`

### docs

- Expanded `docs/data_collection.md` with on-site pitfalls and a format-flexible
  deliverables list, so the field collection produces inputs the pipeline can consume.

## 2026-05-24 — Revert the base/pedestal mount estimate

**Commits:** `f7db450`

### sim — probe mount

- Reverted the speculative base/pedestal mount parameters, keeping the flange-mounted
  probe as the single supported attachment to avoid an unverified offset entering the
  transform chain.

## 2026-05-24 — Add placeholder US probe on the FR3 flange

**Commits:** `4aadc7a`

### sim — probe mount

- Added a placeholder US probe on the FR3 `link7` flange (and exploratory base/pedestal
  mount params) so scale-up and the viewer have a probe geometry/offset to drive, pending
  the real hand-eye transform. The probe offset is an estimate, not a measurement.

## 2026-05-24 — Add on-site data collection checklist

**Commits:** `076d4b5`

### docs

- Added `docs/data_collection.md` as the on-site checklist for the real-robot acquisition.

## 2026-05-24 — Use the real Franka FR3 model (vendored from MuJoCo Menagerie)

**Commits:** `272b385`

### sim — robot model

- Vendored the Franka Research 3 (FR3) MJCF + meshes (Apache-2.0, license kept) and
  switched the scene to it, because Genesis bundles only the older Panda and FR3 differs
  in joint limits, dynamics, and meshes. The model is gripper-less, ending at the
  `fr3_link7` flange where the probe mounts.

## 2026-05-24 — Default the sim to viewer-on with a --headless toggle

**Commits:** `e10db57`

### sim — runtime

- **[API]** Made the Genesis viewer on by default (`SceneConfig.show_viewer=True`) with a
  `--headless` flag on `run_scaleup.py` for batch runs, so interactive inspection is the
  default and headless is opt-in.

## 2026-05-24 — Close the sim→reslice loop: placement bridge + servo_to_contact

**Commits:** `a972209`

### calib / sim / pipeline

- **[API]** Added `calib.placement` to bridge sim-world poses (metres) into the CBCT frame
  (millimetres), the sim analogue of the Step-3 calibration, so a pose read from Genesis
  can be turned into a reslice pose.
- Added `UltrasoundScene.servo_to_contact` (approach-then-press) so scale-up gets reachable
  poses that actually maintain plausible contact instead of arbitrary planes, and wired the
  sim path of `run_scaleup.py` to map achieved pose + contact force into the dataset.

## 2026-05-23 — Add interactive viewer + scene camera options

**Commits:** `057c2e9`

### sim — tooling

- Added `scripts/view_sim.py` and scene camera options for interactive inspection of the
  arm/phantom setup.

## 2026-05-23 — Implement sim.scene against Genesis 0.4.7; add quaternion helpers

**Commits:** `4ccd7f7`

### sim / geometry

- **[API]** Implemented `sim.scene` against the Genesis 0.4.7 API (init / scene / morphs /
  IK / contact-force readouts), localising the call sites so an API move is contained.
- Added scalar-first quaternion helpers (`quat_to_mat`, `mat_to_quat`,
  `pose_from_pos_quat`) to `geometry.py` to match Genesis' pose convention.

## 2026-05-23 — Scaffold deepussim: CBCT-reslice US simulation pipeline

**Commits:** `84ae61a`

### Initial scaffold

- Established the package: `geometry` (SE(3) primitives), `data` (volume/record IO), `us`
  (plane reslice + first-pass B-mode renderer), `calib` (Kabsch registration + renderer
  parameter fitting), `pipeline` (pose sampling + scale-up dataset generation), plus the
  `run_scaleup` / `run_real_collection` / `make_synthetic_phantom` scripts and the
  geometry / quaternion / reslice / renderer unit tests.
- Encoded two deliberate design facts: anatomy masks are free (reslice the label volume at
  the same pose as the intensity volume), and contact force comes from physics, not the
  CBCT.

---

## Contributor Notes

### When to add an entry

Add an entry for any commit that meaningfully changes behaviour, adds a feature, removes
something, or fixes a bug. Skip trivial commits such as typo fixes, comment-only edits, and
formatting-only changes.

### Where to add it

Always append new entries at the top, immediately below `## [Unreleased]`. Do not edit or
reshuffle existing entries.

### Format

```
## YYYY-MM-DD — <short title (≤ 72 chars)>

**Commits:** `24c956a`, `e73334c`

### <Module or area changed>

- What changed and why (not just "updated X" — explain the consequence)
- Breaking changes must be marked **[BREAKING]**
- New public APIs must be marked **[API]**
```

### Scope

Group bullets by module or behavior area. Skip files that only had mechanical renames,
import updates, or formatting-only changes. Focus on behavioral and interface impact.
