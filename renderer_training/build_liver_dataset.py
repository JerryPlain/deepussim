#!/usr/bin/env python
"""Assemble the real liver-segmentation dataset from the motion-sampled frames.

For each de-redundant keyframe (from sample_frames.py) run the A1 mask pipeline
(project liver label -> confidence-crop the shadow tail -> gate if nothing visible
survives) and bundle the US image + cleaned binary mask. Split BY SEQUENCE (no frame
leakage) into train / test.

Runs PER SEQUENCE and is resumable: each sequence writes data/liver_seg/perseq/<scan>.npz
and is skipped if already present, so a killed background run just resumes on re-invoke.
After all sequences are built, run with --merge to write the final train/ and test/ arrays.

    module load python/3.12-base
    python renderer_training/build_liver_dataset.py            # build per-sequence (resumable)
    python renderer_training/build_liver_dataset.py --merge    # assemble train/ + test/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SimpleITK as sitk                                                  # noqa: E402
from reslice.frame import affine_from_sitk                              # noqa: E402
from reslice.io import load_volume_data                                 # noqa: E402
from renderer_training.project_labels_to_us import sector_zoom_pair     # noqa: E402
from renderer_training.confidence_map import confidence_map            # noqa: E402
from renderer_training.allframe_poses import all_frame_poses            # noqa: E402

LIVER = 2
OUT = REPO_ROOT / "data" / "liver_seg"
PERSEQ = OUT / "perseq"
# Test = held-out sequences with GOOD liver visibility (post-confidence-gating positive counts).
# scan9 was dropped from test: its liver projects into the deep/dim field, so confidence gating
# leaves only 1 clean positive -- useless as a test sweep. scan6 replaces it.
TEST_SEQUENCES = {"scan2.npz", "scan5.npz", "scan6.npz", "scan11.npz", "scan14.npz"}
PARAMS = dict(dt_mm=6.0, dr_deg=15.0, min_cov=0.02, crop_floor=0.05, min_kept_cov=0.01)


def _key(p: Path) -> int:
    return int(p.stem.replace("scan", "").replace("_global", ""))


def _sampled_frames(csv_path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in csv.DictReader(csv_path.open()):
        out.setdefault(r["sequence"], []).append(int(r["frame"]))
    return out


def build_sequence(sp: Path, lc2: Path, frames: list[int], vol_i, vol_l, affine) -> dict:
    imgs = np.load(sp, allow_pickle=True)["images"]
    poses = all_frame_poses(sp, lc2)
    keep_img, keep_mask, keep_frame, keep_pos, keep_cov, keep_conf, keep_dec = [], [], [], [], [], [], []
    for j in frames:
        _, lab = sector_zoom_pair(vol_i, vol_l, affine, poses[j], imgs[j].shape[:2])
        liver = (lab == LIVER)
        cov = float(liver.mean())
        if cov < PARAMS["min_cov"]:
            mask = np.zeros(liver.shape, np.uint8); dec = "negative"; conf_in = -1.0; is_pos = 0
        else:
            conf = confidence_map(imgs[j])
            conf_in = float(conf[liver].mean())
            cropped = liver & (conf >= PARAMS["crop_floor"])
            if float(cropped.mean()) < PARAMS["min_kept_cov"]:
                continue                                              # gated -> excluded
            mask = cropped.astype(np.uint8); dec = "positive"; is_pos = 1
        keep_img.append(imgs[j]); keep_mask.append(mask); keep_frame.append(int(j))
        keep_pos.append(is_pos); keep_cov.append(round(cov, 4)); keep_conf.append(round(conf_in, 4)); keep_dec.append(dec)
    return dict(
        images=np.stack(keep_img).astype(np.uint8),
        masks=np.stack(keep_mask).astype(np.uint8),
        frame=np.array(keep_frame, np.int32),
        is_positive=np.array(keep_pos, np.int8),
        liver_cov=np.array(keep_cov, np.float32),
        conf_in_liver=np.array(keep_conf, np.float32),
        decision=np.array(keep_dec),
    )


def merge() -> None:
    files = sorted(PERSEQ.glob("scan*.npz"), key=_key)
    if not files:
        raise SystemExit("no per-sequence files; run the build step first")
    split = {"train": [], "test": []}
    for grp in ("train", "test"):
        imgs, masks, meta = [], [], []
        for f in files:
            name = f.stem + ".npz"
            is_test = name in TEST_SEQUENCES
            if (grp == "test") != is_test:
                continue
            split[grp].append(name)
            d = np.load(f, allow_pickle=True)
            imgs.append(d["images"]); masks.append(d["masks"])
            for k in range(len(d["frame"])):
                meta.append({"sequence": name, "frame": int(d["frame"][k]),
                             "is_positive": int(d["is_positive"][k]),
                             "liver_cov": float(d["liver_cov"][k]),
                             "conf_in_liver": float(d["conf_in_liver"][k]),
                             "decision": str(d["decision"][k])})
        gdir = OUT / grp
        gdir.mkdir(parents=True, exist_ok=True)
        np.save(gdir / "images.npy", np.concatenate(imgs))
        np.save(gdir / "masks.npy", np.concatenate(masks))
        with (gdir / "meta.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
        npos = sum(m["is_positive"] for m in meta)
        print(f"{grp}: {len(meta)} frames ({npos} pos, {len(meta)-npos} neg) from {len(split[grp])} sequences")

    (OUT / "split.json").write_text(json.dumps({
        "params": PARAMS, "test_sequences": sorted(TEST_SEQUENCES),
        "train_sequences": split["train"], "label": "liver (binary), confidence-cropped",
        "source": "CBCT TotalSegmentator liver label projected onto LC2-registered US planes",
    }, indent=2))
    _write_readme()
    print(f"\nwrote {OUT}/train, {OUT}/test, split.json, README.md")


def _write_readme() -> None:
    (OUT / "README.md").write_text(
        "# Liver segmentation dataset (real US)\n\n"
        "Binary liver masks for real ultrasound frames, built automatically by projecting the\n"
        "CBCT TotalSegmentator liver label onto each LC2-registered US plane, then confidence-\n"
        "cropping the shadow/dead regions (Karamalis random-walks map).\n\n"
        "## Layout\n"
        "- `train/`, `test/`: `images.npy` (N,660,880) uint8, `masks.npy` (N,660,880) uint8 binary,\n"
        "  `meta.csv` (sequence, frame, is_positive, liver_cov, conf_in_liver, decision).\n"
        "- `split.json`: by-sequence split + build params.\n"
        "- `sampled_frames.csv`: the motion-sampled keyframes (provenance).\n"
        "- `perseq/`: per-sequence intermediate (image+mask+meta) -- lets you re-split via\n"
        "  `--merge` (edit TEST_SEQUENCES) WITHOUT recomputing confidence.\n\n"
        "Counts: train 530 (203 pos / 327 neg, 10 seq), test 264 (96 pos / 168 neg, 5 seq).\n"
        "scan9 sits in train (its liver images deep/dim, only 1 clean positive; unfit as a test sweep).\n\n"
        "## How it was built (pipeline)\n"
        "1. `sample_frames.py` — motion-based de-redundant keyframes (dt=6mm / dr=15deg).\n"
        "2. `project_labels_to_us.py` — reslice CBCT liver label onto each frame (pixel-aligned).\n"
        "3. `confidence_map.py` — Karamalis confidence; crop mask where confidence < 0.05.\n"
        "4. `build_liver_dataset.py` — A1 gate (drop frames where nothing visible survives), split.\n\n"
        "Split is BY SEQUENCE (no frame leakage). Test = " + ", ".join(sorted(
            s.replace('.npz', '') for s in TEST_SEQUENCES)) + ".\n"
        "Regenerate: `python renderer_training/build_liver_dataset.py [--merge]`.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampled", type=Path, default=OUT / "sampled_frames.csv")
    ap.add_argument("--sequences-dir", type=Path, default=REPO_ROOT / "data" / "sequences")
    ap.add_argument("--lc2-dir", type=Path, default=REPO_ROOT / "data" / "lc2")
    ap.add_argument("--volume", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "CBCT.mhd")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "cbct_20260612" / "labels.nrrd")
    ap.add_argument("--merge", action="store_true", help="assemble per-sequence files into train/ + test/")
    ap.add_argument("--force", action="store_true", help="rebuild sequences even if present")
    args = ap.parse_args()

    if args.merge:
        merge(); return

    PERSEQ.mkdir(parents=True, exist_ok=True)
    vol_i = load_volume_data(str(args.volume))
    vol_l = load_volume_data(str(args.labels))
    affine = affine_from_sitk(sitk.ReadImage(str(args.volume)))
    sampled = _sampled_frames(args.sampled)

    for sp in sorted(args.sequences_dir.glob("scan*.npz"), key=_key):
        name = sp.name
        outp = PERSEQ / f"{sp.stem}.npz"
        if outp.exists() and not args.force:
            print(f"skip {name} (done)", flush=True); continue
        lc2 = args.lc2_dir / f"{sp.stem}_global.npz"
        if not lc2.exists() or name not in sampled:
            print(f"skip {name} (no lc2/sampled)", flush=True); continue
        d = build_sequence(sp, lc2, sampled[name], vol_i, vol_l, affine)
        np.savez_compressed(outp, **d)
        npos = int(d["is_positive"].sum())
        print(f"built {name}: {len(d['frame'])} frames ({npos} pos, {len(d['frame'])-npos} neg)", flush=True)
    print("done building per-sequence; run --merge to assemble")


if __name__ == "__main__":
    main()
