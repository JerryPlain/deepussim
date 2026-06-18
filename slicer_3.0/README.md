# DeepUSSim Slicer 3.0

This folder contains the current CBCT-to-US slicing workflow.

The default convention is:

- phantom local origin is `0 0 0` in DICOM-LPS millimetres;
- every DICOM/recon pair must first be rebuilt into a `physical_frame_report.json`;
- slicing uses the recon affine from that report, not a hard-coded old volume;
- the fan apex is the intersection between the probe beam line and the detected phantom surface edge.

## Step 1: Build The CBCT Frame

Run this once for each DICOM/recon pair:

```powershell
conda activate deepussim
python .\deepussim-main\slicer_3.0\01_build_dicom_physical_frame.py `
  --dicom-dir .\cbct_20260612\cbct_20260612\DICOM1 `
  --recon .\cbct_20260612\cbct_20260612\CBCT.mhd `
  --phantom-origin-lps-mm 0 0 0 `
  --out-dir .\deepussim-main\slicer_3.0\outputs\step01_physical_frame_origin000
```

To use another scan, change only `--dicom-dir`, `--recon`, and optionally `--out-dir`.

The output report contains both DICOM and recon voxel-to-phantom affine matrices:

```text
outputs/step01_physical_frame_origin000/physical_frame_report.json
outputs/step01_physical_frame_origin000/physical_frame_matrices.npz
```

## Step 2: Generate A Fan Slice

The current default example is 2026-06-12 `scan1`, frame `17`, using the saved
2026-06-12 phantom placement matrix:

```powershell
python .\deepussim-main\slicer_3.0\04_sector_slice_from_probe_pose.py
```

Equivalent explicit command:

```powershell
python .\deepussim-main\slicer_3.0\04_sector_slice_from_probe_pose.py `
  --step01-report .\deepussim-main\slicer_3.0\outputs\step01_physical_frame_origin000\physical_frame_report.json `
  --volume-path .\cbct_20260612\cbct_20260612\CBCT.mhd `
  --sequence .\deepussim-main\data\sequences_20260612\scan1.npz `
  --sequence-frame 17 `
  --world-from-phantom .\deepussim-main\slicer_3.0\outputs\yesterday_T_world_from_phantom_liedown.txt `
  --depth-mm 100 `
  --fov-deg 100 `
  --near-mm 20
```

Important outputs:

```text
cbct_rect_frame.png
cbct_sector_frame.png
cbct_sector_content_zoom.png
compare_cbct_sector_zoom_us.png
sector_report.json
```

## Coordinate Notes

- Raw volume frame is DICOM/LPS millimetres: `+X = Left`, `+Y = Posterior`, `+Z = Superior`.
- `phantom_centered_frame` keeps the same axis directions, but its origin is `[0,0,0]`.
- The recon MHD spacing/origin/direction are used when reslicing; do not reuse a report from another DICOM/recon scan.
- 3D Slicer displays in RAS internally, so display code converts `LPS -> RAS` with `diag(-1,-1,1)`.
- The saved `world-from-phantom` matrix passed to the slicer is already after the simulator lie-down rotation.

