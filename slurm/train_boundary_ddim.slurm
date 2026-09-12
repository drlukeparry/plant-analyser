#!/usr/bin/env bash
#SBATCH --job-name=ddim_boundary_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampereq          # Ada cluster: A100 partition (confirmed via sinfo)
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00              # walltime -- adjust to TOTAL_STEPS/CHUNK_STEPS below
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# slurm/train_boundary_ddim.slurm
#
# Runs the three arms of the SDF-crop boundary-conditioning experiment
# (ROADMAP2/PLAN.md: does concatenating a boundary_sdf crop as clean spatial
# conditioning change what the DDIM generates, vs. a baseline with no
# positional input at all, vs. an ablation with the GVAE composition latent
# dropped) as parallel SLURM array tasks on the University of Nottingham Ada
# cluster:
#
#   array task 0 -> baseline   (src/models/ddim/train_baseline_do256.py)
#   array task 1 -> boundary   (src/models/ddim/train_and_plot_chunk.py)
#   array task 2 -> nocond     (src/models/ddim/train_and_plot_chunk_nocond.py)
#
# Requires slurm/build_boundary_data.slurm to have completed first (needs
# outputs/phase5/patch_fields_boundary.npy and patch_fields_baseline_do256.npy
# to exist). Submit from the repo root, chained after the data job:
#
#   mkdir -p slurm_logs
#   DATA_JOB=$(sbatch --parsable slurm/build_boundary_data.slurm)
#   sbatch --dependency=afterok:$DATA_JOB slurm/train_boundary_ddim.slurm
#
# Each array task trains in resumable TOTAL_STEPS/CHUNK_STEPS chunks (one
# short-lived `python3` process per chunk, same rationale as train_boundary.py's
# train() docstring: natural checkpoints, no long-lived-process memory
# creep) and re-submitting this same script later resumes every arm from its
# last checkpoint automatically (train()'s resume=True default). Override
# via environment variables at submit time, e.g.:
#
#   TOTAL_STEPS=50000 CHUNK_STEPS=2000 sbatch --dependency=afterok:$DATA_JOB \
#     slurm/train_boundary_ddim.slurm
#
# pick_device() equivalent here is src/models/gvae/train.py's device(),
# fixed to prefer cuda > mps > cpu (it previously only checked mps > cpu,
# which would have silently trained on CPU on this cluster).

set -euo pipefail

set +u
module load cuda-uoneasy/12.6.0
module load anaconda-uoneasy/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

TOTAL_STEPS="${TOTAL_STEPS:-20000}"
CHUNK_STEPS="${CHUNK_STEPS:-2000}"
N_CHUNKS=$(( (TOTAL_STEPS + CHUNK_STEPS - 1) / CHUNK_STEPS ))

ARMS=(baseline boundary nocond)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID: arm=$ARM, $N_CHUNKS chunks of $CHUNK_STEPS steps (total $TOTAL_STEPS)"

for ((i = 1; i <= N_CHUNKS; i++)); do
    echo "--- $ARM: chunk $i/$N_CHUNKS ---"
    case "$ARM" in
        baseline)
            python3 -c "from src.models.ddim.train_baseline_do256 import train; train(steps=$CHUNK_STEPS)"
            ;;
        boundary)
            python3 -m src.models.ddim.train_and_plot_chunk "$CHUNK_STEPS"
            ;;
        nocond)
            python3 -m src.models.ddim.train_and_plot_chunk_nocond "$CHUNK_STEPS"
            ;;
        *)
            echo "unknown arm: $ARM" >&2
            exit 1
            ;;
    esac
done

echo "Done. $ARM checkpoint/history in outputs/phase5/ (ddim_ckpt_*, ddim_train_history_*, progress_step*.png)"
