# B1 — Learned US renderer (CBCT slice → US-like image)

**Status:** scaffolded on branch `feat/learned-renderer`; first run is a **feasibility probe**, not
a final model. This doc records *what* we build and *why*, and the decisions for review.

## What it replaces

| Role | Before (placeholder) | B1 (learned) |
|---|---|---|
| **renderer** — produces the image | physics B-mode model ([`us/renderer.py`](../src/deepussim/us/renderer.py)): 7 hand-set params | **CUT generator** (ResNet), CBCT-reslice → US ([`renderer/networks.py`](../src/deepussim/renderer/networks.py)) |
| **appearance supervision** — how real US constrains it | Nelder-Mead fit of the 7 params to `1−NCC` (optimisation, not learning) | **adversarial + PatchNCE** training against real US ([`renderer/cut.py`](../src/deepussim/renderer/cut.py)) |

The physics renderer stays as a structural baseline/sanity check; the *appearance* now comes from
the learned generator. The two are kept separate so "is the synthesis good" never gets conflated
with "is the encoder good" (the project's eval red line).

## Method: CUT (Contrastive Unpaired Translation)

Generator `G: CBCT-reslice → US`, trained with
`L = L_GAN(G,D) + λ_NCE · PatchNCE(x, G(x)) [+ PatchNCE(y, G(y))]`:

- **L_GAN** (LSGAN, PatchGAN discriminator on real US) — learns US *appearance*.
- **PatchNCE** — pulls an output patch toward the input patch at the *same location*, pushes from
  others → **structure preservation without pixel-aligned pairs**.

Both domains live in the `(n_ax, n_lat)` fan layout that `reslice` produces and
`calib.us_geometry.unwrap_fan` maps real US onto.

### Why CUT (and not the alternatives)

Selection criterion is **"which method improves as data grows, given three lasting facts"**:
registration stays cm-level, real US keeps being collected, and the renderer must not pollute the
encoder eval.

| | pix2pix | **CUT** | diffusion |
|---|---|---|---|
| tolerates cm / no pairing | ✗ needs pixel pairs | ✓ unpaired | ✓ |
| cheap to grow (add US, fine-tune) | ✗ needs more *paired* data | ✓ add to pool | ✓ |
| works on today's small data | ~ | ✓ | ✗ needs scale |
| structure-preserving (no eval pollution) | ✓ | ✓ (PatchNCE) | ⚠ most prone to hallucinate |

→ **CUT now**; re-evaluate **diffusion later**, once the real-US pool is large/diverse, using the
same metrics. "We'll collect more data" *strengthens* the unpaired choice (no precise pairs needed).

### Why not first push LC2 to mm

LC2 is stuck at ~cm because the **phantom is low-texture** → the registration objective is flat near
the optimum (no fine features to pin the last mm). Getting to mm is a **data/phantom change** (add
fiducials/texture — a collection task), not a code tweak. We deliberately chose an **unpaired**
method so mm precision isn't required. mm registration matters for the *pose/mask ground-truth
precision* (downstream nav), not for B1 — so it's tracked separately, not a prerequisite.

## Data

Two pools, built once on the conda env (CPU) by
[`scripts/prep_renderer_data.py`](../scripts/prep_renderer_data.py):

- `data/renderer/source_cbct.npz` — CBCT reslices at the real in-contact poses.
- `data/renderer/target_us.npz` — real US contact frames, `unwrap_fan`-ed to the fan grid.

Current pools: **771 source + 771 target** at 512×256 (the two real sequences). Highly correlated,
single orientation → first model is a probe; it grows with Friday's collection (esp. tilt diversity).

## Train

```bash
# 1) build pools once (CPU/conda)
python scripts/prep_renderer_data.py --sequences data/sequences/*.npz \
    --volume data/cbct/intensity.nrrd --mesh data/cbct/phantom_surface.stl \
    --config configs/renderer.yaml --out data/renderer
# 2) train on the GPU (existing sif — training is pure torch, no rebuild)
sbatch scripts/slurm/renderer_train.slurm
```

Outputs `runs/renderer_cut/generator.pt` + `samples_ep*.npz` (src/fake/tgt stacks) for offline
inspection.

## Go/no-go for the probe

The probe answers: **can this low-texture CBCT be translated into believable US at all?**
- visual: `samples_ep*.npz` — does `fake` look like real US while keeping the CBCT structure?
- quantitative (to add): FID/KID vs real US **+** a structure-consistency check (the generated
  anatomy must still match the reslice mask — guards against hallucination).
If it can't, the lever is the **source representation** or **more/▲-diverse real data**, not bigger nets.

## Open decisions (defaults chosen; flag to change)

1. **Source domain** = CBCT reslice (not the near-black physics render). ✅ default
2. **Unpaired** (no LC2 pairs); optional weak structure term left off. ✅ default
3. **In-repo compact CUT** (pure torch, no torchvision) rather than vendoring the upstream repo —
   testable + light container (no rebuild). ✅ default
4. **GPU** = a100 (a40 fallback). ✅ default
5. **Probe now on 771 frames**, scale after collection. ✅ default
6. Metrics: FID + structure-consistency — **to wire up**.

## Integration (later)

Wrap the trained `G` as a `NeuralRenderer` exposing the same `render(intensity, geom)` signature as
the physics renderer, so `pipeline.scaleup.generate_dataset` can switch physics↔learned by a flag —
keeping the two appearance models cleanly swappable.
