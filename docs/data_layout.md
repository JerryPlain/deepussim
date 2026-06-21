# Data layout & how to get it

**Data files are not in git** — and cannot be. The rosbags (~1.1 GB / 779 MB) and the CBCT
intensity volume (122 MB) each exceed GitHub's 100 MB per-file hard limit, raw scans may be
sensitive, and large binaries bloat the history. So the repo tracks **code + the small
text artifacts** (the calibration matrices live in
[`calib/transforms.py`](../src/deepussim/calib/transforms.py), not as a file), and the
actual data is shared out-of-band.

> `.gitignore` anchors `/data/` to the repo root, so nothing under `data/` is committed.
> Assemble the tree below locally; the code reads it by relative path.

## Where the data lives

- **Raw inputs** (DICOM, rosbags, probe mesh): **shared Google Drive folder** —
  <https://drive.google.com/drive/folders/1_i2uUaQEUuji7q3dmmKmMw8uejgDA38m>
  (the real scan sequences plus the probe mesh).
- **Derived assets** (NIfTI/NRRD/STL produced in 3D Slicer): uploaded to the **same Drive
  folder** so teammates don't each have to re-run Slicer + TotalSegmentator. They can also
  be regenerated locally — see [Reproduce derived assets](#reproduce-derived-assets).

## Target `data/` tree

Download from the Drive and place files at these paths (create the folders as needed):

```
data/
  cbct_20260612/
    DICOM1/                   raw CBCT DICOM (folder, from Drive)        [raw]
    intensity.nrrd            CBCT intensity volume, 470x308x326 @0.74mm [derived ①]
    labels.nrrd               TotalSegmentator label volume, 53 organs  [derived ③]
    labels_colortable.csv     label id -> organ name + color            [derived ③]
    phantom_surface.stl       phantom outer surface, mm (CBCT frame)    [derived ②]
    phantom_surface_m.stl     same, scaled to metres for Genesis        [derived ②]
  probe/
    convex_model_stl.stl      US probe outer shell (supplied)           [raw]
  rosbags/
    phantom.bag               real sequence 1 (ROS1)                    [raw]
    phantom1.bag              real sequence 2 (ROS1)                    [raw]
  sequences/                  pose-synced, dark-tagged extracts (.npz)  [derived, optional]
    phantom.npz  phantom1.npz
```

Notes:
- The intensity / label / surface assets must all derive from the **same CBCT DICOM** so
  they share one affine (the masks are only "free" because the label volume reslices at the
  same pose as the intensity volume).
- `phantom_surface_m.stl` is the metres copy Genesis loads; its placement in the sim world
  is computed from `PhTR` at replay time, **not** by recentring the mesh.
- A maintainer may symlink these paths to wherever they downloaded the files instead of
  copying — the code only cares about the paths under `data/`.

## Quick check after assembling

```bash
conda activate deepussim
python -c "from deepussim.data.volume import load_volume; \
v=load_volume('data/cbct_20260612/intensity.nrrd'); print(v.shape, v.spacing.round(3))"
# expect: (470, 308, 326) [0.743 0.743 0.743]
```

## Reproduce derived assets

If you only have the **raw** inputs, regenerate the derived assets:

### From the CBCT DICOM (3D Slicer 5.x)
Load `data/cbct_20260612/DICOM1` (DICOM module → Import → Load), then:

1. **① intensity** — `File → Save Data`, save the volume as `intensity.nrrd` (or `.nii.gz`).
2. **② phantom surface** — `Segment Editor` → Threshold (lower ≈ −500) to capture the solid
   body → `Apply` → right-click the segmentation → *Export visible segments to models* →
   `File → Save Data` the model as `phantom_surface.stl`. Then make the metres copy:
   ```bash
   python -c "import trimesh; m=trimesh.load('data/cbct_20260612/phantom_surface.stl'); \
   m.apply_scale(0.001); m.export('data/cbct_20260612/phantom_surface_m.stl')"
   ```
3. **③ labels** — install the *TotalSegmentator* extension → run it on the volume
   (task `total`, speed `Fast`) → right-click the segmentation →
   *Export visible segments to binary labelmap* → save the labelmap as `labels.nrrd` and the
   `..._ColorTable.csv` as `labels_colortable.csv`.

### From the rosbags (no ROS install needed)
```bash
pip install rosbags
python scripts/extract_rosbags.py data/rosbags/phantom.bag data/rosbags/phantom1.bag \
    --out data/sequences --preview 8
```
Produces `data/sequences/<name>.npz` (images + `T_base_from_ee` poses + contact flags) plus
preview PNGs. See [`data/rosbag.py`](../src/deepussim/data/rosbag.py) for the topic names and
the dark-frame tagging.
