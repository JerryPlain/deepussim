#!/usr/bin/env python
"""Build the two CUT domain pools (fan layout) for the learned renderer (B1).

  target_us.npz   real US contact frames, unwrapped onto the (n_ax, n_lat) fan grid
  source_cbct.npz CBCT reslices at the real in-contact poses (the anatomy the renderer maps from)

Both are derived from the already-extracted sequences (poses + contact + US frames) and the CBCT
volume — no rosbags, no GPU. CUT is unpaired, so we only need the two pools; the cm-level
registration never enters. See docs/renderer.md.

    python scripts/prep_renderer_data.py \
        --sequences data/sequences/phantom.npz data/sequences/phantom1.npz \
        --volume data/cbct_20260612/intensity.nrrd --mesh data/cbct_20260612/phantom_surface.stl \
        --config configs/renderer.yaml --us-spacing 0.166112957 --out data/renderer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from deepussim.data.volume import load_volume
from deepussim.us.reslice import ProbeGeometry, reslice_volume
from deepussim.calib.us_geometry import contact_envelope, fit_fan_pixels, unwrap_fan
from deepussim.calib.transforms import T_EE_FROM_PROBE, T_WORLD_FROM_CBCT
from deepussim.calib.placement import seat_phantom_placement, meters_to_mm, sim_pose_to_cbct
from deepussim.geometry import invert, compose


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", nargs="+", required=True, help="extracted sequence .npz")
    ap.add_argument("--volume", required=True, help="CBCT intensity volume")
    ap.add_argument("--mesh", required=True, help="phantom surface STL (CBCT mm) for placement")
    ap.add_argument("--config", help="renderer/geometry YAML (for n_ax/n_lat + fan size)")
    ap.add_argument("--us-spacing", type=float, default=0.166112957, help="US pixel size mm/px")
    ap.add_argument("--out", default="data/renderer", help="output dir for the two pools")
    args = ap.parse_args()

    geom = ProbeGeometry()
    if args.config:
        geom = ProbeGeometry(**yaml.safe_load(Path(args.config).read_text()).get("geometry", {}))
    n_ax, n_lat = geom.n_ax, geom.n_lat

    # gather contact frames (US image + EE world pose) from the sequences
    us_frames, ee_world = [], []
    for s in args.sequences:
        d = np.load(s, allow_pickle=True)
        P, C, I = d["poses"], d["contact"].astype(bool), d["images"]
        for i in range(len(P)):
            if C[i]:
                ee_world.append(P[i]); us_frames.append(I[i])
    ee_world = np.asarray(ee_world, dtype=float)
    print(f"contact frames: {len(us_frames)}")

    # --- target pool: unwrap each real US frame onto the fan grid --------------------
    env = contact_envelope(np.asarray(us_frames))            # solid fan from many frames
    fan = fit_fan_pixels(env)
    target = np.stack([unwrap_fan(np.asarray(f, float), fan, n_ax=n_ax, n_lat=n_lat)
                       for f in us_frames]).astype(np.float32)
    print(f"target US pool: {target.shape}  (fan {n_ax}x{n_lat})")

    # --- source pool: reslice the CBCT at each real contact pose ---------------------
    import trimesh
    mesh = trimesh.load(args.mesh)
    origins = np.array([compose(p, T_EE_FROM_PROBE)[:3, 3] for p in ee_world])
    placement = seat_phantom_placement(mesh, origins, T_WORLD_FROM_CBCT)
    s2c = meters_to_mm(invert(placement))
    volume = load_volume(args.volume)
    source = np.stack([reslice_volume(volume, sim_pose_to_cbct(compose(p, T_EE_FROM_PROBE), s2c),
                                      geom, order=1) for p in ee_world]).astype(np.float32)
    print(f"source CBCT pool: {source.shape}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "target_us.npz", images=target)
    np.savez_compressed(out / "source_cbct.npz", images=source)
    print(f"wrote {out}/target_us.npz + source_cbct.npz")


if __name__ == "__main__":
    main()
