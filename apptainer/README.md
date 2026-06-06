# Apptainer (NHR@FAU) — deepussim GPU/sim image

Containerizes the **brittle GPU path** of deepussim — `genesis-world` + `torch` +
Taichi/CUDA (kernels JIT-compiled at run time). Baking it into a `.sif` pins the
stack and makes batch runs on **Alex/Helma** reproducible.

The **CPU-only** parts (reslice / render / `pytest`) do **not** need this — the conda
env in the top-level [README](../README.md) is fine for those.

Reference: <https://doc.nhr.fau.de/environment/apptainer/>

## Files

| File | What it is |
|---|---|
| `deepussim.def` | Image definition (Python 3.11, torch+CUDA, `pip install -e .[dev]`) |
| `build.sh` | Build `deepussim.sif` on a frontend, caches → `$WORK` |
| `run.sh` | `apptainer run --nv` + live `src/` bind + repo bind + `--pwd` repo |

SLURM batch jobs live in [`scripts/slurm/`](../scripts/slurm/) (e.g. `scaleup_sim.slurm`).

## 1. Build (on a frontend node)

Build on an **AlmaLinux** frontend — Alex, Helma, Fritz, Woody, or Meggie
(Ubuntu nodes can't build). From the repo root:

```bash
export WORK=/path/to/your/work        # NHR@FAU $WORK filesystem
bash apptainer/build.sh               # → $WORK/deepussim.sif
```

`build.sh` points the image **and** all build/pull caches at `$WORK` (never `$HOME`,
which has a tiny quota). Permission error on the frontend? It auto-falls back to
`/apps/singularity/apptainer-wrapper.sh build`.

First build compiles nothing GPU-side; it just installs wheels (a few minutes).

## 2. Run with the GPU

```bash
apptainer/run.sh python scripts/smoke_sim.py            # quick Genesis smoke test
apptainer/run.sh python scripts/run_scaleup.py --sim ...
```

`run.sh` adds `--nv` (host driver passthrough), bind-mounts your **live** `src/`
over the baked copy (edit code, no rebuild — the image installed it editable), and
sets `--pwd` to the repo. `$WORK`/`$HOME` mount by default, so data paths there work.

For a **frozen** run off the baked code (true reproducibility), drop the `src` bind:
`apptainer run --nv $WORK/deepussim.sif python ...`.

## 3. Batch (Slurm)

```bash
sbatch scripts/slurm/scaleup_sim.slurm
```

Defaults read inputs from the repo's `data/`; override with env vars
(`TRAJ`/`N`/`FORCE_N`/`OUT`/…) — see [`scripts/slurm/README.md`](../scripts/slurm/README.md).
It's a **single-node** GPU job — NHR@FAU advises against MPI-in-container, and
deepussim's sim is single-GPU anyway.

## CUDA / driver compatibility — the one thing to watch

`--nv` injects Alex's **host driver**; the container brings torch's bundled CUDA
runtime. The container CUDA must be **≤** what the host driver supports. The default
`deepussim.def` uses the plain PyPI torch wheel (the path the top-level README
verified). If a `--nv` run fails with *"driver too old / CUDA X not supported"*:

1. Check the driver on a GPU node: `nvidia-smi` (top-right CUDA version).
2. In `deepussim.def`, pin a matching older wheel, e.g.
   `TORCH_PIP="torch --index-url https://download.pytorch.org/whl/cu124"`, rebuild.

The resolved torch/Genesis versions are recorded in `/opt/deepussim/PINNED.txt`
inside the image (`apptainer/run.sh cat /opt/deepussim/PINNED.txt`).
