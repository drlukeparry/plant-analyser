#!/usr/bin/env bash
#SBATCH --job-name=ldm_capacity_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampereq          # Ada cluster: A100 partition (confirmed via sinfo)
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# slurm/sweep_capacity.slurm
#
# Runs several (width, n_blocks) capacity configurations as parallel array
# tasks instead of sequentially -- this is the HPC version of the
# width/depth/capacity diagnostics run one at a time overnight on a single
# laptop GPU: on a cluster they can all launch at once. Configured for the
# University of Nottingham Ada cluster (ampereq = A100-full x8 per node);
# each array task requests 1 GPU, so up to 8 tasks can run concurrently
# on one node as SLURM schedules them.
#
# Submit from the repo root (VAE_CHECKPOINT must already exist -- train it
# first with a plain vae2d.py run, or point at an already-trained one):
#
#   mkdir -p slurm_logs
#   VAE_CHECKPOINT=Trained_diffusion_proto3d/vae2d_real/vae2d.pytorch \
#     sbatch slurm/sweep_capacity.slurm
#
# Add/remove entries in CONFIGS below and update the #SBATCH --array range
# to match (0-indexed, inclusive) -- e.g. 5 configs needs --array=0-4.

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

# One "width,n_blocks" pair per array index -- edit freely, matching
# tonight's manual sweep: default, capacity-bumped, and beyond.
CONFIGS=(
    "32,4"
    "64,8"
    "128,16"
    "96,12"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
WIDTH="${CONFIG%,*}"
N_BLOCKS="${CONFIG#*,}"

VAE_CHECKPOINT="${VAE_CHECKPOINT:?Set VAE_CHECKPOINT to a trained vae2d.pytorch path}"
ITERATIONS="${ITERATIONS:-100000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
OUT_DIR="Trained_diffusion_proto3d/sweep_${SLURM_ARRAY_JOB_ID}_w${WIDTH}_b${N_BLOCKS}"

echo "Array task $SLURM_ARRAY_TASK_ID: width=$WIDTH n_blocks=$N_BLOCKS -> $OUT_DIR"

python3 proto_diffusion3d_ldm.py \
    --multi-texture-dir Textures/multi_test \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --width "$WIDTH" --n-blocks "$N_BLOCKS" \
    --iterations "$ITERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --preview-texture-ids 3 5 8 \
    --dtype float32 --device cuda \
    --out-dir "$OUT_DIR"

echo "Done. Outputs in $OUT_DIR"
