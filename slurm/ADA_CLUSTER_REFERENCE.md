# Ada (University of Nottingham HPC) — SLURM reference

Distilled from the official UoN Ada documentation, kept here so the
`.slurm` scripts in this directory can be understood/adjusted without
needing to look it up again. Login node: `hpclogin01.ada.nottingham.ac.uk`.

## Commands

| Command | Purpose |
|---|---|
| `sbatch` | Submit a job |
| `squeue` | List jobs / show status (`-u $USER` to filter to your own) |
| `scancel <job_id>` | Delete a job |
| `sinfo` | Show node/partition status |
| `sacct` | Resource usage/history for jobs (`-S`/`-E` to set a date range) |

`man <command>` for full option lists.

## Partitions relevant to this repo

| Partition | GPUs | Walltime max | Notes |
|---|---|---|---|
| `ampereq` | 8x A100-full (80GB) per node, 3 nodes | 7 days | main GPU partition — what `train_ldm.slurm`/`sweep_capacity.slurm` use |
| `ampere-devq` | same nodes as `ampereq` | **1 hour** | shares hardware with `ampereq`; capped to 12 cores/1 GPU per user; needs `--qos=gpu-dev`; use for quick interactive tests |
| `ampere-mq` / `ampere-m-devq` | A100s split into 7x MIG instances (~10GB each) | 7 days / 1 hour | not used by this repo — full A100s are more than enough for these model sizes |
| `shortq` | none (CPU only) | 12 hours | general CPU partition — what `phase1_segment.slurm`/`phase2_build_graphs.slurm` use for the (non-GPU) segmentation/graph-extraction steps |
| `devq` | none (CPU only) | **1 hour** | CPU equivalent of `ampere-devq`; needs `--qos=dev` (see below) |

No `--account` needed for `ampereq`/`ampere-devq` (only required for large MIG
allocations on `ampere-mq`, via `--account=uon-ampere-m`).

## GPU request syntax — cluster-specific gotcha

Ada does **not** accept a bare `--gres=gpu:1`. The GPU type must be named:

```
--gres=gpu:A100-full:1      # ampereq / ampere-devq
--gres=gpu:A100-mig:1       # ampere-mq / ampere-m-devq
```

(Both `.slurm` scripts in this directory already use the correct
`A100-full` form.)

## `--qos` — required for the `dev` partitions only

| Partition | Required qos |
|---|---|
| `devq` (CPU) | `--qos=dev` |
| `ampere-devq` / `ampere-m-devq` | `--qos=gpu-dev` |
| `ampereq`, `shortq` | none |

Forgetting `--qos=gpu-dev` on `ampere-devq` is a real failure mode (job
held with `QOSNotAllowed`) — this bit an earlier version of the quick
interactive test command suggested for this repo.

### Correct interactive test session command

```bash
srun --partition=ampere-devq --qos=gpu-dev --gres=gpu:A100-full:1 \
     --cpus-per-task=8 --mem=16g --time=00:15:00 --pty bash
```

## Resource limits (dev qos)

| Resource | `--qos=dev` (CPU) | `--qos=gpu-dev` |
|---|---|---|
| Jobs | 4 | 4 |
| CPU cores | 96 | 12 |
| GPUs | n/a | 1 |
| Job time | 1 hour | 1 hour |

Production/exploration tier limits (non-dev) are much higher (600 or 96
total CPU cores, 6 or 1 A100-full GPUs respectively) — see the full UoN
docs if a long/large run needs more than the dev qos allows.

## `--mem` vs `--mem-per-cpu`

Official examples mostly use `--mem=<total per node>` (e.g. `--mem=16g`).
This repo's `.slurm` scripts use `--mem-per-cpu=4G` instead (matching a
previously-working script on this same cluster) — both are valid SLURM
options; `--mem-per-cpu` × `--cpus-per-task` gives the equivalent total.
