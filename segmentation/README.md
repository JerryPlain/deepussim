# SAM2 liver segmentation

Frozen SAM2 Hiera image encoder + a small trainable conv decoder head, on the real US
liver dataset (`data/liver_seg/`, built by `renderer_training/build_liver_dataset.py`).

**Group A (this):** LIMITED real US only — get the pipeline working, get a baseline Dice.
**Group B (later):** limited real + generated US — same code, add the synthetic frames to train.

## Run (Alex / SLURM)

```bash
# 1. once, on the LOGIN node (needs internet): venv + torch + sam2 + checkpoint
bash segmentation/setup_env.sh

# 2. submit the GPU job
sbatch segmentation/train.slurm
# watch: tail -f segmentation/logs/sam2-liver-A_<jobid>.out
```

Defaults: SAM2.1 Hiera-small, 40 positive + 40 negative train frames, 60 epochs, 1× A40.
Override the venv/checkpoint paths via `SAM2_VENV` / `SAM2_CKPT` / `SAM2_CFG` env vars.

## What it does

- Freezes the SAM2 image encoder; trains only the decoder head (`SegHead`).
- "Limited real" = subsample `train/` to `--n-pos` + `--n-neg` frames (seeded, reproducible).
- Loss = BCE + soft Dice at 256×256; eval = Dice / IoU on the full `test/` split (upsampled to 660×880).
- Reports overall Dice and Dice over positive (liver-present) test frames; writes
  `runs/seg_sam2_A_limited/{head_best.pt,metrics.json}`.

## Notes

- Test set = held-out sequences (scan2/5/6/11/14); no frame leakage. See `data/liver_seg/split.json`.
- Mask GT is confidence-cropped CBCT-projected liver — coarse boundaries. Report tolerance-aware
  metrics too if tight Dice looks pessimistic.
- Smoke-test locally first if a GPU is free: `python segmentation/train_sam2_head.py --sam2-ckpt ... --epochs 2`.
