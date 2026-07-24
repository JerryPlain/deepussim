# Validated CBCT slice baseline: bottom-tangent v1

This document freezes the CBCT slicing procedure accepted on 2026-07-22. Treat it as
the reference baseline before experimenting with registration, masks, rendering, or a
different image geometry. Its machine-readable lock file is
`configs/cbct_slice_bottom_tangent_v1.json`.

## Reference implementation

The canonical implementation is
`scripts.preview_us_normal_reslice_region.cbct_bottom_tangent_reslice`. The batch driver
is `scripts/sample_rosbag_reslices.py`.

The baseline deliberately does **not** crop and resize an already sampled CBCT image.
It samples the three-dimensional CBCT directly onto the native two-dimensional US pixel
grid at `0.166112957 mm/px` in both directions.

## Fixed procedure

1. Read `T_world_from_ee` for the selected US frame.
2. Apply the unchanged hand-eye transform:
   `T_world_from_probe = T_world_from_ee @ T_EE_FROM_PROBE`.
3. Apply the selected phantom placement and convert translation from metres to
   millimetres:
   `T_phantom_from_probe = inv(T_world_from_phantom) @ T_world_from_probe`.
4. Construct the ultrasound plane from the probe local X-Z plane. Probe X is lateral,
   probe Z is depth, and probe Y is the plane normal.
5. Sample the existing `360 x 300 mm` broad rectangular CBCT plane only to locate the
   validated CBCT surface intersection along the probe-depth line. This step preserves
   the accepted position logic.
6. Fit the real-US inner arc in display pixels. Its circle centre column is fixed to the
   image centre; centre row and radius are fitted from the bright upper boundary using
   threshold `30`.
7. Use the normals through the inner-arc/top-border intersections as the two sector side
   boundaries. Extend them beyond the screen as necessary.
8. Set the outer arc to the concentric circle tangent to the bottom image border.
9. Anchor the central inner-arc point at the detected CBCT surface plus `15 mm` along
   probe depth.
10. Build one Cartesian grid with the original US rows and columns and isotropic spacing
    `0.166112957 mm/px`. Transform every grid point through the inverse reconstruction
    affine and directly sample the source CBCT with linear interpolation.
11. Keep only points that are both inside the CBCT volume and inside the fitted US fan.
    Normalize intensities for display. Do not crop, resize, or stretch afterwards.

## Invariants

- The robot pose and slice-position chain is unchanged.
- The central inner-arc anchor retains the accepted surface-intersection logic.
- CBCT content is sampled once from the 3-D source volume; there is no image-space
  resampling after the slice is created.
- Rows and columns use the same physical pixel size.
- The output shape is exactly the source US frame shape.
- LC2 is an optional later pose refinement and is not part of this baseline slice
  construction.
- Liver or other label masks are optional later consumers and must reuse the exact same
  grid with nearest-neighbour interpolation.

## 0612 reference inputs

- Reference frame: `scan5`, frame `126`.
- CBCT: `../cbct_20260612/cbct_20260612/CBCT.mhd`.
- Reconstruction affine: `physical_frame_report.json`, key
  `phantom_centered_frame.recon_affine_centered_from_ijk_mm`.
- Phantom placement: `yesterday_T_world_from_phantom_liedown.txt`.

These input paths and their SHA-256 values are recorded in the lock file. The raw CBCT
payload is data, not implementation, and is not duplicated by this baseline.

## Verify that the baseline has not drifted

From the repository root, run:

```powershell
python scripts/verify_cbct_slice_baseline.py
```

Add `--include-reference-inputs` to check the 0612 CBCT header, affine report, and
phantom-placement matrix as well. A changed hash does not automatically mean a change is
wrong; it means the current result must no longer be described as this exact v1 baseline
until it has been reviewed and deliberately re-versioned.
