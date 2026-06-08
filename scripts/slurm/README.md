# SLURM jobs (NHR@FAU Alex)

Batch scripts for the GPU/sim pipeline. Each runs inside the Apptainer image
(`apptainer/run.sh` wraps `--nv` + the live repo bind); the CPU-only core does **not**
need SLURM — the conda env is fine for reslice/render/tests.

Submit from the **repo root** on Alex (paths resolve against `$SLURM_SUBMIT_DIR`):

```bash
sbatch scripts/slurm/<job>.slurm
```

## Jobs

| Script | Stage | What it does | Key overrides (env) |
|---|---|---|---|
| `scaleup_sim.slurm` | 1 | Genesis `--sim` scale-up: drive the FR3 along the trajectory, force-servo onto the seated phantom, write (US image, pose, anatomy mask, contact force) per reached+contacting pose. | `TRAJ` (contact\|replay), `N`, `FORCE_N`, `OUT`, `VOLUME/LABELS/MESH` |
| `renderer_train.slurm` | 1 (B1) | Train the learned US renderer (CUT) on the two fan-layout pools → believable US appearance. Pure torch (no container rebuild). **Build pools first** on CPU/conda: `scripts/prep_renderer_data.py`. | `EPOCHS`, `BATCH`, `DATA`, `OUT` |
| `renderer_eval.slurm` | 1 (B1) | Evaluate a trained renderer: generate fakes + structure-consistency / surface-localization (anti-hallucination) on GPU, cache fakes. FID/KID run after on the login node (`--from-cache`; container has no torchvision). | `RUN`, `DATA`, `N` |

> **`scaleup_sim.slurm` adds force/contact/reachability only** — the US *appearance* is the
> placeholder renderer there. `renderer_train.slurm` is what makes the image look real (see
> [`docs/renderer.md`](../../docs/renderer.md)).

## Naming convention

`<stage-or-component>_<task>.slurm`, lower_snake_case — e.g. `scaleup_sim.slurm`,
`renderer_train.slurm` (planned). One job per file; parameterise with env-var overrides
(see each script's header) rather than forking near-duplicate scripts.

## Prerequisites

- Image built once on a frontend: `bash apptainer/build.sh` → `$WORK/deepussim.sif`
  (see [`apptainer/README.md`](../../apptainer/README.md)).
- Inputs under the repo's `data/` (bind-mounted) or `$WORK` (auto-mounted); override the
  `VOLUME/LABELS/MESH/OUT` env vars to point elsewhere.
