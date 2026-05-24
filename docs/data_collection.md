# On-site data collection guide

You collect first; I'll write the ingestion loaders to match **whatever format your
machines export** — DICOM, CSV, rosbag, video, etc. Format is flexible. What is *not*
flexible is the list of fields below: some of them **cannot be recovered after the fact**,
and missing one means another trip. Read the pitfalls first.

---

## ⚠️ Pitfalls that waste a trip (read before you start)

1. **One phantom, scannable by both CT and US.** Use the *same physical phantom* for the
   CBCT and the robot/US session, and make sure it's tissue-mimicking (echogenic) **and**
   CT-visible. Don't use a CT phantom for the scan and a different US phantom for scanning.

2. **Fiducials must be on the phantom *during* the CBCT scan.** You can't add markers
   afterwards — they won't be in the CT. Attach ≥4, **non-coplanar** and in an
   **asymmetric** arrangement (symmetry → ambiguous registration). Prefer low-artifact
   markers (glass/ceramic beads, CT spheres); **avoid metal** (streak artifacts). They must
   also be locatable by the robot/US (touchable with the probe tip, or visible in US).

3. **Don't move or deform the phantom between CBCT and the whole robot session**, and don't
   shift the probe in its holder mid-session. If anything moves, the registration is void.

4. **Keep the robot base frame consistent.** No re-mounting, re-zeroing, or base re-homing
   between the fiducial touches and the US scanning — or you must re-register.

5. **Time sync is the #1 failure mode.** If US frames and robot logs can't be aligned in
   time, every (image, pose, force) tuple is wrong. Decide the method *before* collecting.
   If unsure, record a **shared sync event** (a sharp tap / quick lift) at the start and end
   of each sweep so the two streams can be aligned offline.

6. **Log robot joint angles q1..q7, not only the flange pose.** From joints I can recompute
   FK in any convention — this protects against quaternion-order (wxyz vs xyzw) and
   base-frame mistakes, which are easy to get wrong and impossible to fix blind.

7. **Export the full CBCT DICOM series** (the folder of `.dcm`), not screenshots or a single
   slice — voxel spacing and orientation live in the headers.

8. **Freeze US imaging settings during a sweep** (depth, gain, frequency, TGC, dynamic
   range). If they must change, log them per frame. Avoid burned-in overlays/text on the
   image if you can; otherwise note the image region and the pixel→mm scale.

9. **Write down units and frame conventions** (mm vs m, quaternion order, which frame the
   force is in, base-frame origin). One sentence each saves hours.

---

## ⭐ Must capture (mapped to the pipeline)

**CBCT (Step 1 + key to Step 3)**
- [ ] Full DICOM series of the phantom **with fiducials attached**, covering the whole
      phantom + markers. Note voxel spacing.
- [ ] Phantom identity: commercial model + datasheet, or (if custom) materials + geometry.

**Robot + US, synchronized (Step 2)** — per frame, time-aligned:
- [ ] US B-mode frame (clean if possible) + timestamp; frame rate.
- [ ] Robot **joint angles q1..q7** + flange pose `T_base→flange` + timestamp.
- [ ] **External wrench** (force/torque) + timestamp; note the frame and whether tool
      weight / gravity is compensated.
- [ ] Cover a **range** of poses/angles/regions, not a single sweep (dataset diversity).

**US imaging params (Step 4)**
- [ ] Probe **type** (linear / curvilinear / phased — current renderer assumes **linear**).
- [ ] Per-acquisition: depth (mm), field-of-view width (mm) or **pixel spacing (mm/px)**,
      centre frequency (MHz), gain, TGC, focus, dynamic range. US machine + probe model.

**Hand-eye / mount (Step 3 → `probe_offset`)**
- [ ] Probe-on-flange fixture **dimensions + orientation** (CAD if available), and the US
      image-plane origin relative to the probe face.
- [ ] If possible, a **hand-eye calibration** (touch fiducials with the probe tip at several
      known robot poses, or image a calibration target). If not, save the touch poses so I
      can solve it offline.
- [ ] **Photos** (with a ruler): mount, phantom + fiducials, whole setup.

---

## What to bring back (deliverables — format flexible)

One folder per session:

```
session_YYYYMMDD/
  cbct/              DICOM series (or NIfTI), fiducials visible
  robot_log.csv      one row/sample: t, q1..q7, flange pos+quat, Fx..Tz   (rosbag/parquet ok)
  us/                image sequence or video + per-frame timestamps
  us_params.txt      probe type, depth_mm, width_mm or mm/px, freq, gain, TGC, dyn range, models
  fiducials.csv      marker positions in robot base frame (touched); CBCT coords if you have them
  handeye/           calibration result OR touch-fiducial poses OR fixture CAD/dims
  photos/            mount / phantom+markers / setup (with a ruler)
  README.txt         phantom model, machine+probe models, units, frame conventions, sync method,
                     and which files belong to which sweep
```

Don't stress about exact schemas — I'll write the loader to whatever you export. Just make
sure the **must-capture** fields above are present, because those can't be reconstructed later.

---

## What each item unblocks
| Capture | Unblocks |
|---|---|
| CBCT DICOM + spacing | Step 1; reslice source; anatomy labels |
| Fiducials (CT + robot/US locatable) | Step 3 registration `T_cbct_from_simworld` (replaces hand-tuned anchors) |
| Joint angles q1..q7 / flange pose | robot FK; per-frame probe pose |
| External wrench | force labels; force-servoing target later |
| Probe fixture dims / hand-eye | `probe_offset` (FR3 flange → US image plane) + the sim probe geometry |
| Synced US + pose + force | Step 2 real dataset; Step 4 calibration target |
| US imaging params | `RendererParams` / `ProbeGeometry`; renderer calibration |
| Phantom datasheet/materials | label volume + renderer acoustic priors |
