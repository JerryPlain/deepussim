# Changelog

This file tracks meaningful repository changes.

## [Unreleased]

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
