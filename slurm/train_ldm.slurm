#!/usr/bin/env bash
#SBATCH --job-name=ldm_texture
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampereq          # Ada cluster: A100 partition (confirmed via sinfo)
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00              # walltime -- adjust to your job's size
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# slurm/train_ldm.slurm
#
# Single-job SLURM batch script for the VAE+LDM multi-texture training
# (proto_diffusion3d_ldm.py). Configured for the University of Nottingham
# Ada cluster (hpclogin01.ada.nottingham.ac.uk) -- confirmed via `sinfo`
# (ampereq = A100-full x8 per node) and `module avail` on the login node.
# No --account needed on this cluster. Submit from the repo root:
#
#   mkdir -p slurm_logs
#   sbatch slurm/train_ldm.slurm
#
# Override any of the training hyperparameters via environment variables
# at submit time, e.g. to run the width=128/n_blocks=16 capacity bump
# benchmarked locally at ~2.6 it/s (expect much faster on an A100):
#
#   WIDTH=128 N_BLOCKS=16 ITERATIONS=200000 sbatch slurm/train_ldm.slurm
#
# pick_device() in proto_diffusion2d.py already prefers cuda > mps > cpu,
# so no code changes were needed to run on this GPU node.

set -euo pipefail

# module load / conda's activation hook both reference some unset
# variables internally -- under `set -u` this fails with an unbound-
# variable error. Relax -u just for this block.
set +u
module load cuda-uoneasy/12.6.0
module load anaconda-uoneasy/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

# Defaults match the capacity bump discussed for the next LDM run;
# override any of these via environment variables at submit time.
VAE_CHECKPOINT="${VAE_CHECKPOINT:-Trained_diffusion_proto3d/vae2d_real/vae2d.pytorch}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-}"          # set to resume from an existing checkpoint
WIDTH="${WIDTH:-128}"
N_BLOCKS="${N_BLOCKS:-16}"
ITERATIONS="${ITERATIONS:-200000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
# bfloat16 measured ~22x faster than float32 at width=128/n_blocks=16 on an
# A100 (~58 it/s vs ~2.6 it/s on Apple Silicon MPS for reference) -- the
# tiny VAE showed no such benefit (too small to be compute-bound), but this
# larger diffusion network clearly benefits from tensor-core bf16.
DTYPE="${DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-Trained_diffusion_proto3d/slurm_${SLURM_JOB_ID}}"

RESUME_FLAG=()
if [[ -n "$LOAD_CHECKPOINT" ]]; then
    RESUME_FLAG=(--load-checkpoint "$LOAD_CHECKPOINT")
fi

python3 proto_diffusion3d_ldm.py \
    --multi-texture-dir Textures/multi_test \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --width "$WIDTH" --n-blocks "$N_BLOCKS" \
    --iterations "$ITERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --preview-texture-ids 3 5 8 \
    --dtype "$DTYPE" --device cuda \
    --out-dir "$OUT_DIR" \
    "${RESUME_FLAG[@]}"

echo "Done. Outputs in $OUT_DIR"
