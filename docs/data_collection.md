# On-site data collection checklist

What to capture so the deepussim pipeline (CBCT → real US collection → registration →
renderer calibration → sim scale-up) can be built on *real* data. Ordered by how hard
each item is to redo — do the irreplaceable ones first.

## ⭐ Top priority (hard/impossible to redo later)

### 1. CBCT scan with fiducials (Step 1 + the key to Step 3)
- [ ] Attach **≥4 fiducial markers** to the phantom that are **both CT-visible**
      (radio-opaque beads / divot markers) **and locatable by the robot or US**
      (touchable with the probe/pointer tip, or visible in US). Spread them out,
      **non-coplanar**. These are the backbone of CBCT↔robot registration; without
      them you fall back to hard US↔CT image registration.
- [ ] Scan the phantom **with the markers attached**, in the same configuration you'll
      use for robot scanning.
- [ ] **Export the full DICOM series** (not screenshots/JPEGs) — the headers carry voxel
      spacing, orientation and origin that we need. NIfTI is also fine if it preserves them.
- [ ] **Do not move or deform** the phantom/markers between the CBCT scan and robot
      scanning, or fix them in a rigid jig so the transform stays valid.
- [ ] Record the **phantom identity**: commercial model + datasheet (e.g. CIRS / Kyoto
      Kagaku) or, if custom, the materials and internal geometry. Material acoustic
      properties feed the US renderer; known geometry makes segmentation trivial.
- [ ] Note CBCT **voxel spacing (mm)** and scan settings.

### 2. Probe mount / hand-eye (Step 3 → sets `SceneConfig.probe_offset`)
- [ ] Measure how the US probe is fixed to the **FR3 flange (fr3_link7)**: the
      bracket/fixture **dimensions and orientation** (CAD if available), and where the
      **US image-plane origin** sits relative to the probe face.
- [ ] If possible, run a **hand-eye calibration** (touch fiducials with the probe/pointer
      tip at several known robot poses, or image a calibration target).
- [ ] **Photos**, multiple angles: probe-on-flange mount, phantom + fiducial placement,
      whole setup. (These also let us match the sim visually.)

## High priority (Step 2 — the real US collection)

### 3. Synchronized (US frame ↔ robot pose ↔ force)
- [ ] **US B-mode frames**: clean frames without overlays/annotations if possible, with
      **per-frame timestamps** and the frame rate.
- [ ] **Robot pose per frame**: FR3 flange pose `T_base→flange` from forward kinematics,
      timestamped. Note the quaternion order and base-frame convention.
- [ ] **Contact force per frame**: FR3 measured external wrench (force/torque), timestamped.
- [ ] **Time synchronization** between the US stream and the robot logs — a common clock,
      a hardware trigger, or at minimum both logged with timestamps and a known offset.
      This is the hardest part; plan it before you start.

### 4. US imaging parameters (Step 4 — needed to match the renderer)
- [ ] **Probe type**: linear / curvilinear / phased. (The current renderer + reslice assume
      a **linear** probe with a rectangular field of view; curvilinear needs a fan geometry.)
- [ ] Per-acquisition: **imaging depth (mm)**, **field-of-view width (mm)** or
      **pixel spacing (mm/px)**, **centre frequency (MHz)**, **gain**, **TGC** curve,
      **focus**, **dynamic range**. These map onto `RendererParams` + `ProbeGeometry`.
- [ ] US machine model + probe model.

## Nice to have
- [ ] A few repeated sweeps of the same region (for consistency / noise characterisation).
- [ ] The phantom segmentation, if the scanner/vendor can provide it (else we run
      TotalSegmentator or use the known phantom geometry afterwards).

## What each item unblocks
| Capture | Unblocks |
|---|---|
| CBCT DICOM + spacing | Step 1; reslice source; anatomy labels |
| Fiducials (CT + robot/US locatable) | Step 3 registration `T_cbct_from_simworld` (replaces hand-tuned anchors) |
| Probe mount dims / hand-eye | `probe_offset` (FR3 flange → US image plane) |
| Synced US + pose + force | Step 2 real dataset; Step 4 calibration target |
| US imaging params | `RendererParams` / `ProbeGeometry`; renderer calibration |
| Phantom datasheet/materials | Label volume + renderer acoustic priors |
