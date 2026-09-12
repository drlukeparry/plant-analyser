#!/usr/bin/env bash
#SBATCH --job-name=phase2_build_graphs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shortq           # Ada cluster: CPU partition, 12hr max
#SBATCH --cpus-per-task=96           # shortq's per-job core cap -- images processed in parallel, see below
#SBATCH --mem-per-cpu=4G             # matches this repo's other .slurm scripts
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# slurm/phase2_build_graphs.slurm
#
# Runs Phase 2 (src/graph/build_all_graphs.py -- RAG construction + node
# feature extraction, cached per-image as torch_geometric Data) over every
# label map in outputs/phase1/, for both variants Phase 3 needs:
#   raw  -> outputs/phase2/graphs/       (pixel-unit features)
#   norm -> outputs/phase2/graphs_norm/  (size-normalized features -- what
#           gvae_norm.pt/phase3_train_gvae.slurm actually trains on)
#
# CPU-only (no torch GPU ops) -- no --gres line, same `shortq` partition as
# phase1_segment.slurm and the same up-to-96-cores budget. build_all_graphs.py
# processes one image at a time in a single process, so this drives
# src/graph/build_one_graph.py (same build_graph call, same skip-if-exists
# output convention) over every label map via `xargs -P` instead of its
# sequential loop -- same rationale as phase1_segment.slurm.
#
# Not species-specific (globs every *_full_labels.npy in outputs/phase1/),
# so this is a single job, not an array -- run after ALL THREE
# phase1_segment.slurm array tasks finish. Skips any image already cached,
# so re-running is safe.
#
# Submit from the repo root, after Phase 1:
#
#   mkdir -p slurm_logs
#   PHASE1_JOB=<jobid from sbatch slurm/phase1_segment.slurm>
#   sbatch --dependency=afterok:${PHASE1_JOB}_0:${PHASE1_JOB}_1:${PHASE1_JOB}_2 \
#     slurm/phase2_build_graphs.slurm

set -euo pipefail

set +u
module load anaconda-uoneasy/2023.09-0     # no cuda-uoneasy module -- this step is CPU-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

PARALLEL="${PARALLEL:-$SLURM_CPUS_PER_TASK}"
LABEL_PATHS=$(find outputs/phase1 -name "*_full_labels.npy")

echo "=== 1/2: raw variant (outputs/phase2/graphs/), up to $PARALLEL parallel workers ==="
echo "$LABEL_PATHS" | xargs -P "$PARALLEL" -I{} python3 -m src.graph.build_one_graph {} raw

echo "=== 2/2: norm variant (outputs/phase2/graphs_norm/), up to $PARALLEL parallel workers ==="
echo "$LABEL_PATHS" | xargs -P "$PARALLEL" -I{} python3 -m src.graph.build_one_graph {} norm

echo "Done. Cached graphs in outputs/phase2/graphs/ and outputs/phase2/graphs_norm/"
