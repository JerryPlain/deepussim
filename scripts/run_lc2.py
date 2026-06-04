#!/usr/bin/env python
"""LC2 refinement: register each real US frame to the CBCT, refining the calibration pose.

For every (in-contact) US frame the calibration chain gives an initial pose in the CBCT frame
(``C_T_U = C_T_R · R_T_E(t) · E_T_U``); LC2 then maximises the multimodal similarity over a
*constrained* 6-DoF range to grind the ~cm residual to mm (Wein/Fuerst; Li et al.). The fan
geometry is fitted from the sequence envelope (so reslice samples the CBCT on the same fan as
the real probe) and each US frame is unwrapped onto that grid before scoring.

    python scripts/run_lc2.py --seq data/sequences/phantom.npz --volume data/cbct/intensity.nrrd \
        --us-spacing 0.166112957 --n 32 --out data/lc2_phantom.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from deepussim.data.volume import load_volume
from deepussim.calib import (
    T_WORLD_FROM_CBCT, T_EE_FROM_PROBE,
    contact_envelope, fit_fan_pixels, fit_fan_geometry, unwrap_fan,
    register_frame_lc2, lc2_similarity, seat_phantom_placement,
)
from deepussim.calib.placement import meters_to_mm
from deepussim.geometry import invert, compose
from deepussim.us.reslice import reslice_volume


def make_init_pose(sim_to_cbct: np.ndarray):
    """Build the per-frame calibration pose ``C_T_U`` (probe in CBCT mm) from the seated bridge.

    ``sim_to_cbct = meters_to_mm(inv(world<-cbct))`` already folds in the lie-down roll and the
    contact-seat (see :func:`calib.seat_phantom_placement`); each frame just composes the EE pose
    with the hand-eye and maps it through the bridge (m -> mm, matching the volume).
    """
    def init_pose_cbct_mm(ee_pose_world_m: np.ndarray) -> np.ndarray:
        return compose(sim_to_cbct, meters_to_mm(compose(ee_pose_world_m, T_EE_FROM_PROBE)))
    return init_pose_cbct_mm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", required=True, help="extracted .npz sequence (images, poses, contact)")
    ap.add_argument("--volume", required=True, help="CBCT intensity volume (.nrrd/.nii.gz)")
    ap.add_argument("--mesh", default="data/cbct/phantom_surface.stl",
                    help="phantom surface STL (CBCT mm) used to seat the lie-down placement")
    ap.add_argument("--us-spacing", type=float, required=True, help="US pixel size (mm/px)")
    ap.add_argument("--n", type=int, default=32, help="number of contact frames to register")
    ap.add_argument("--n-lat", type=int, default=256)
    ap.add_argument("--n-ax", type=int, default=512)
    ap.add_argument("--max-trans-mm", type=float, default=15.0)
    ap.add_argument("--max-rot-deg", type=float, default=15.0)
    ap.add_argument("--out", help="save refined poses + scores here (.npz)")
    args = ap.parse_args()

    import trimesh

    volume = load_volume(args.volume)
    z = np.load(args.seq, allow_pickle=True)
    images, poses, contact = z["images"], z["poses"], z["contact"]

    # fan geometry from the contact envelope -> pixel fit (for unwrap) + ProbeGeometry (for reslice)
    env = contact_envelope(images, contact)
    fan = fit_fan_pixels(env)
    geom = fit_fan_geometry(env, args.us_spacing, n_lat=args.n_lat, n_ax=args.n_ax)
    print(f"fan: r0={fan.r0_px:.0f} r1={fan.r1_px:.0f} fov={fan.fov_deg:.1f}  "
          f"geom: radius_mm={geom.radius_mm:.1f} depth_mm={geom.depth_mm:.1f}")

    # Seat the phantom (lie-down roll + contact-seat) and derive the probe-pose bridge, so the
    # calibration init images *into* the tissue instead of grazing the surface (the ~90 deg LC2
    # alone cannot grind). Same placement as scripts/view_sim.py and run_scaleup.
    contact_idx = np.where(contact)[0]
    origins = np.array([compose(np.asarray(poses[i], dtype=float), T_EE_FROM_PROBE)[:3, 3]
                        for i in contact_idx])
    T_world_from_cbctm = seat_phantom_placement(trimesh.load(args.mesh), origins, T_WORLD_FROM_CBCT)
    init_pose_cbct_mm = make_init_pose(meters_to_mm(invert(T_world_from_cbctm)))

    idx = contact_idx[np.linspace(0, len(contact_idx) - 1, min(args.n, len(contact_idx)), dtype=int)]
    refined, lc2_before, lc2_after = [], [], []
    for k, i in enumerate(idx):
        us_polar = unwrap_fan(images[i], fan, n_ax=args.n_ax, n_lat=args.n_lat)
        init = init_pose_cbct_mm(np.asarray(poses[i], dtype=float))
        before = lc2_similarity(us_polar, reslice_volume(volume, init, geom))
        pose, after = register_frame_lc2(us_polar, volume, geom, init,
                                         max_trans_mm=args.max_trans_mm, max_rot_deg=args.max_rot_deg)
        refined.append(pose); lc2_before.append(before); lc2_after.append(after)
        print(f"  frame {i:4d}: LC2 {before:.3f} -> {after:.3f}")

    lb, la = np.array(lc2_before), np.array(lc2_after)
    print(f"\nLC2 mean: {lb.mean():.3f} -> {la.mean():.3f}  (improved {int((la > lb).sum())}/{len(la)})")
    if args.out:
        np.savez_compressed(args.out, frame_index=idx, refined_pose=np.array(refined),
                            lc2_before=lb, lc2_after=la)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
