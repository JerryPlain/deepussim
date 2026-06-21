#!/usr/bin/env python
"""Replay verification (doc v2 §4.5): confirm the probe mount + placement geometrically.

Drive the probe along the real rosbag EE poses and measure where the probe origin lands
relative to the phantom surface mesh, comparing each probe-mount direction against each
placement direction. The resolved calibration places **contact** frames on the surface
(~cm) while **dark** / non-contact frames sit clearly off it (lift-off) — no Genesis/GPU
needed, this is a pure geometric check via trimesh proximity.

    python scripts/verify_replay.py            # full 2x2 sweep + contact/dark separation
    python scripts/verify_replay.py --bags data/rosbags/phantom.bag

The resolved directions are baked into calib.transforms: mount = T_EE_FROM_PROBE, placement
= T_WORLD_FROM_CBCT (the measured matrix used directly as CBCT->world). This reproduces it.
"""
from __future__ import annotations

import argparse

import numpy as np

from deepussim.calib.transforms import T_EE_FROM_PROBE, T_PROBE_FROM_EE, T_ROBOT_FROM_PHANTOM
from deepussim.geometry import invert, compose


def _load_sequences(paths):
    """Frame lists from pre-extracted .npz (keys: poses, contact) — no rosbags dep."""
    from types import SimpleNamespace
    seqs = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        P, C = d["poses"], d["contact"].astype(bool)
        seqs[p] = SimpleNamespace(frames=[SimpleNamespace(pose=P[i], contact=bool(C[i]))
                                          for i in range(len(P))])
    return seqs


def _probe_origins(frames, T_ee_from_probe):
    return np.array([compose(f.pose, T_ee_from_probe)[:3, 3] for f in frames])


def _sample(frames, n):
    if len(frames) <= n:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, n, dtype=int)]


def main() -> None:
    import trimesh
    import warnings
    warnings.filterwarnings("ignore")  # trimesh proximity emits a benign divide warning

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags", nargs="+",
                    default=["data/rosbags/phantom.bag", "data/rosbags/phantom1.bag"])
    ap.add_argument("--sequences", nargs="+",
                    help="pre-extracted .npz (keys: poses, contact) INSTEAD of --bags (no rosbags dep)")
    ap.add_argument("--mesh", default="data/cbct_20260612/phantom_surface_m.stl",
                    help="phantom surface mesh in CBCT frame, metres")
    ap.add_argument("--n", type=int, default=60, help="frames sampled per class for the sweep")
    args = ap.parse_args()

    base_mesh = trimesh.load(args.mesh)
    if args.sequences:
        seqs = _load_sequences(args.sequences)
    else:
        from deepussim.data.rosbag import extract_sequence   # needs the optional 'rosbags' lib
        seqs = {b: extract_sequence(b) for b in args.bags}
    names = list(seqs.keys())

    placements = {
        "place=T_WORLD_FROM_CBCT": T_ROBOT_FROM_PHANTOM,
        "place=inverse         ": invert(T_ROBOT_FROM_PHANTOM),
    }
    mounts = {"mount=T_EE_FROM_PROBE": T_EE_FROM_PROBE, "mount=inverse       ": T_PROBE_FROM_EE}

    # --- 2x2 sweep on contact frames of the first sequence ---
    bag0 = names[0]
    contact0 = _sample([f for f in seqs[bag0].frames if f.contact], args.n)
    print(f"== sweep on {bag0} ({len(contact0)} contact frames) ==")
    best = None
    for pname, T_wc in placements.items():
        m = base_mesh.copy(); m.apply_transform(T_wc)
        pq = trimesh.proximity.ProximityQuery(m)
        for hname, T_ep in mounts.items():
            d = np.abs(-pq.signed_distance(_probe_origins(contact0, T_ep)))
            med = float(np.median(d))
            on = float((d < 0.02).mean())
            print(f"  {pname} | {hname:21s}: median {med * 100:6.2f} cm   on-surface(<2cm) {on * 100:3.0f}%")
            if best is None or med < best[0]:
                best = (med, pname, hname, T_wc, T_ep)

    _, bp, bh, T_wc, T_ep = best
    print(f"\n>> winner: {bp.strip()} + {bh}  (median {best[0] * 100:.2f} cm on contact frames)")

    # --- contact vs dark separation for the winner, every bag ---
    pq = trimesh.proximity.ProximityQuery(base_mesh.copy().apply_transform(T_wc))
    print("\n== contact vs dark separation (winning combo) ==")
    for b, seq in seqs.items():
        row = []
        for label, sel in (("contact", True), ("dark", False)):
            fr = _sample([f for f in seq.frames if f.contact == sel], args.n)
            if not fr:
                row.append(f"{label} n/a (0 frames)")
                continue
            d = np.abs(-pq.signed_distance(_probe_origins(fr, T_ep)))
            row.append(f"{label} median {np.median(d) * 100:5.2f}cm (<2cm {int((d < 0.02).mean() * 100):3d}%)")
        print(f"  {b}:  " + "   ".join(row))
    print("\nExpected: contact ~cm on-surface, dark clearly off (lift-off) -> chain validated.")


if __name__ == "__main__":
    main()
