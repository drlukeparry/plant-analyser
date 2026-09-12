#!/usr/bin/env bash
#SBATCH --job-name=ddim_boundary_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampere-devq      # quick job: 40 generations at 50 DDIM steps each
#SBATCH --qos=gpu-dev
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# slurm/validate_boundary.slurm
#
# Runs src/models/ddim/validate_boundary_conditioning.py -- the correlation
# check (boundary_sdf vs. generated cell size/density/orientation) that
# actually answers the experiment's question, once both the "baseline" and
# "boundary" arms of train_boundary_ddim.slurm have a saved checkpoint
# (ddim_unet_baseline_do256.pt, ddim_unet_boundary.pt in outputs/phase5/).
# Uses ampere-devq/gpu-dev (1hr cap) since this is 40 generations at 50 DDIM
# steps each, not a training run -- see ADA_CLUSTER_REFERENCE.md.
#
# Submit from the repo root, chained after the two training array tasks it
# needs (baseline=0, boundary=1; the nocond arm isn't required here):
#
#   TRAIN_JOB=<jobid from sbatch train_boundary_ddim.slurm>
#   sbatch --dependency=afterok:${TRAIN_JOB}_0:${TRAIN_JOB}_1 \
#     slurm/validate_boundary.slurm
#
# N_SAMPLES and MISMATCH default to validate_boundary_conditioning.py's own
# CLI defaults; override at submit time, e.g. MISMATCH=mismatch for the
# confound-control rerun (see that script's main() docstring):
#
#   N_SAMPLES=80 MISMATCH=mismatch sbatch ... slurm/validate_boundary.slurm

set -euo pipefail

set +u
module load cuda-uoneasy/12.6.0
module load anaconda-uoneasy/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

N_SAMPLES="${N_SAMPLES:-40}"
MISMATCH="${MISMATCH:-}"

python3 -m src.models.ddim.validate_boundary_conditioning "$N_SAMPLES" "$MISMATCH"

echo "Done. Results/preview in outputs/phase5/validate_boundary_conditioning_*"
