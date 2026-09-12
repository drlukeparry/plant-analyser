# Running on an HPC cluster via SLURM

See [ADA_CLUSTER_REFERENCE.md](ADA_CLUSTER_REFERENCE.md) for the official
Ada SLURM commands, partitions, qos rules, and resource limits distilled
for quick reference.

Configured for the **University of Nottingham Ada cluster**
(`hpclogin01.ada.nottingham.ac.uk`), verified against the login node
directly (`sinfo`, `module avail`) and against a working prior job script
on the same cluster (`ezzly2`'s `stablediffusion3Drandomcellconditional`
project — coincidentally doing something very similar: a VAE autoencoder
+ 3D latent diffusion UNet). No code changes were needed in the Python
scripts: `pick_device()` (in `proto_diffusion2d.py`, used by every script
here) already prefers `cuda` over `mps` over `cpu`, so the exact same
scripts run unmodified on this cluster's GPU nodes.

## Confirmed cluster details

- **Partition:** `ampereq` (A100-full x8 per node, 7-day limit; there's
  also `ampere-devq` for quick <1hr test jobs, and `q4bioq` has H200s if
  you need more memory per GPU).
- **No `--account` needed** on this cluster.
- **Modules:** `cuda-uoneasy/12.6.0` and `anaconda-uoneasy/2023.09-0`
  (current names — an older job script on this cluster used `cuda/local/11.0`
  and `anaconda-uon/3`, which no longer exist; `module avail` is the source
  of truth if these ever change again).
- Ada also has a pre-built `pytorch-uoneasy/2.1.2-foss-2023a-CUDA-12.1.1`
  module as a simpler alternative to `env_setup.sh` if you don't need a
  specific torch version — see the note in that script.

If you move this to a *different* cluster, everything below still applies
in spirit, but re-run the discovery commands (`sinfo -o "%P %G %l"`,
`module avail`, check whether `--account` is required) and update the
`#SBATCH` lines and module names accordingly.

## One-time setup

```bash
bash slurm/env_setup.sh
```

Creates a conda env named `solidtexturenets` with everything from
`requirements.txt` plus a CUDA-enabled PyTorch build.

## Running a single training job

```bash
mkdir -p slurm_logs
sbatch slurm/train_ldm.slurm
```

Defaults to the width=128/n_blocks=16 capacity bump benchmarked locally
tonight (~2.6 it/s on Apple Silicon MPS -- expect substantially faster on
a real GPU). Override any hyperparameter via environment variables at
submit time, e.g.:

```bash
WIDTH=64 N_BLOCKS=8 ITERATIONS=50000 sbatch slurm/train_ldm.slurm
```

To resume from an existing checkpoint (same `--load-checkpoint` mechanism
used all night, including the periodic `--checkpoint-every` saves):

```bash
LOAD_CHECKPOINT=Trained_diffusion_proto3d/ldm_overnight_continued/denoiser3d_ldm.pytorch \
  sbatch slurm/train_ldm.slurm
```

## Running a capacity sweep (the HPC-native version of tonight's diagnostics)

Tonight's width/n_blocks comparisons ran one at a time, each taking
30 minutes to several hours on a single laptop GPU. On a cluster, run them
all as parallel array tasks instead:

```bash
mkdir -p slurm_logs
VAE_CHECKPOINT=Trained_diffusion_proto3d/vae2d_real/vae2d.pytorch \
  sbatch slurm/sweep_capacity.slurm
```

Edit the `CONFIGS` array in `sweep_capacity.slurm` to add/remove
`width,n_blocks` pairs, and update `#SBATCH --array=0-N` to match (0-indexed,
inclusive -- 5 configs needs `--array=0-4`).

## Running the boundary-conditioning DDIM experiment (ROADMAP2/PLAN.md Phase 5/6)

Three scripts cover `src/models/ddim`'s SDF-crop conditioning experiment —
does giving the patch-level DDIM a `boundary_sdf` crop as clean spatial
conditioning change what it generates, versus a baseline with no positional
input and a GVAE-conditioning ablation. Unlike the texture LDM above, these
scripts don't take CLI hyperparameters (they're one-off experiment scripts,
not a general trainer), so the `.slurm` wrappers call their `train()`
functions directly rather than passing `--flag` args.

**Prerequisite:** `outputs/phase1`, `outputs/phase2/graphs_norm`, and
`outputs/phase3` (Phase 1–3 of `PLAN.md` — segmentation, graph extraction,
GVAE) must already exist on the cluster filesystem, computed from
`datasets/DO/EH/VM`. These are gitignored (per-project convention) and too
large to commit, unlike `Textures/multi_test/` above. Full chain, computed
on Ada end-to-end (`fetch_datasets.sh` → Phase 1 → Phase 2 → Phase 3 →
Phase 5 build → training → validation):

```bash
mkdir -p slurm_logs

# 0. Raw datasets (published INBD/MiSCS dataset -- Ada has internet access)
bash slurm/fetch_datasets.sh                     # DO, EH, VM

# 1. Phase 1: classical-watershed segmentation, one array task per species
PHASE1_JOB=$(sbatch --parsable slurm/phase1_segment.slurm)

# 2. Phase 2: graph extraction (raw + norm variants), all species together
PHASE2_JOB=$(sbatch --parsable \
    --dependency=afterok:${PHASE1_JOB}_0:${PHASE1_JOB}_1:${PHASE1_JOB}_2 \
    slurm/phase2_build_graphs.slurm)

# 3. Phase 3: GVAE, trained on all three species combined (tag="norm")
PHASE3_JOB=$(sbatch --parsable --dependency=afterok:$PHASE2_JOB slurm/phase3_train_gvae.slurm)

# 4. Phase 5: build the boundary_sdf cache + DDIM patch sets (DO-only, what
#    the boundary-conditioning experiment below actually trains on)
DATA_JOB=$(sbatch --parsable --dependency=afterok:$PHASE3_JOB slurm/build_boundary_data.slurm)

# 5. Train all three arms (baseline/boundary/nocond) as a 3-task array job,
#    resumable in TOTAL_STEPS/CHUNK_STEPS chunks
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB slurm/train_boundary_ddim.slurm)

# 6. Once baseline (array task 0) and boundary (array task 1) have a checkpoint,
#    run the actual correlation check the experiment exists to answer
sbatch --dependency=afterok:${TRAIN_JOB}_0:${TRAIN_JOB}_1 slurm/validate_boundary.slurm
```

Each of Phase 1/2/3's scripts skips work already done (segmented images,
cached graphs), so re-submitting after a partial failure resumes rather than
restarting. See each `.slurm` file's header for partition/GPU rationale —
Phase 1/2 are CPU-only, Phase 3/5/6 need a GPU.

**Already have Phase 1–3 computed elsewhere (e.g. run locally)?** Skip
straight to step 4 and transfer those outputs instead of recomputing them
on Ada:

```bash
ADA_USER=yourusername bash slurm/sync_data_to_ada.sh   # DO-only subset, rsync from your local machine
```

Re-submitting `train_boundary_ddim.slurm` later resumes every arm from its
last checkpoint (each arm's `train()` defaults to `resume=True`), so a
7-day-max job doesn't need to fit the whole run in one submission.

**Fixed as part of deploying this to Ada:** `device()` in
`src/models/gvae/train.py` (used by every `src/models/ddim` script) only
checked `mps > cpu` — it would have silently trained on CPU on this
cluster's GPU nodes. Now checks `cuda > mps > cpu`.

## What ships as-is, no changes needed

- `Textures/multi_test/` (160KB, 10 synthetic textures) — small enough to
  commit/copy directly, no separate data-transfer step needed.
- `vae2d.py`, `proto_diffusion3d.py`, `proto_diffusion3d_multi.py`,
  `proto_diffusion3d_ldm.py` — all device-agnostic already.
- `requirements.txt` — used as-is by `env_setup.sh`.

## What this package does not (yet) handle

- **Multi-GPU / distributed training** — every script here trains on a
  single device. These models are small (0.3-15M params); a single GPU is
  almost certainly enough, but if you need to scale to a much larger
  dataset or model, the training loops would need `DistributedDataParallel`
  or similar, which isn't implemented in any script here.
- **Job dependencies** — if you want the capacity sweep to auto-launch
  only after `vae2d.py` finishes training a fresh VAE (rather than pointing
  at an already-trained one), chain the jobs with `sbatch --dependency=afterok:<jobid>`
  rather than editing these scripts.
