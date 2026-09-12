#!/usr/bin/env bash
#SBATCH --job-name=phase3_train_gvae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampereq          # Ada cluster: A100 partition (confirmed via sinfo)
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# slurm/phase3_train_gvae.slurm
#
# Runs Phase 3 (src/models/gvae/train.py's train_multi, via the module's
# `norm` CLI mode -- no species prefix, so it trains on every cached graph
# in outputs/phase2/graphs_norm/ combined: DO+EH+VM, matching the existing
# local gvae_norm.pt/feature_scaler_norm.npz convention that
# src/models/ddim's build_patches_*.py scripts already expect) on Ada.
#
# Needs a GPU: train_multi runs on `device()` (src/models/gvae/train.py,
# now cuda > mps > cpu). STEPS defaults to train_multi's own default (1500,
# matching the existing local gvae_norm.pt) -- override at submit time:
#
#   STEPS=3000 sbatch --dependency=afterok:<phase2 jobid> slurm/phase3_train_gvae.slurm
#
# Submit from the repo root, after Phase 2:
#
#   mkdir -p slurm_logs
#   sbatch --dependency=afterok:<phase2 jobid> slurm/phase3_train_gvae.slurm

set -euo pipefail

set +u
module load cuda-uoneasy/12.6.0
module load anaconda-uoneasy/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

STEPS="${STEPS:-}"

if [[ -n "$STEPS" ]]; then
    python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.graph.build_all_graphs import out_dir_for
from src.models.gvae.train import train_multi
graph_paths = sorted(out_dir_for('norm').glob('*.pt'))
train_multi(graph_paths, steps=$STEPS, tag='norm')
"
else
    python3 -m src.models.gvae.train norm
fi

echo "Done. Checkpoint in outputs/phase3/gvae_norm.pt (+ feature_scaler_norm.npz, train_history_norm.npy)"
