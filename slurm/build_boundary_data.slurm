#!/usr/bin/env bash
#SBATCH --job-name=ddim_boundary_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ampereq          # Ada cluster: A100 partition (confirmed via sinfo)
#SBATCH --gres=gpu:A100-full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# slurm/build_boundary_data.slurm
#
# One-off data-prep job for the SDF-crop boundary-conditioning experiment
# (src/models/ddim/{build_boundary_sdf,build_patches_boundary,
# build_patches_baseline_do256}.py). Run this once, before
# train_boundary_ddim.slurm, on the University of Nottingham Ada cluster:
#
#   mkdir -p slurm_logs
#   sbatch slurm/build_boundary_data.slurm
#
# Prerequisites this script does NOT create -- must already be on the
# cluster filesystem, not part of this repo's git history:
#   outputs/phase1/DO_*_full_labels.npy      (Phase 1 segmentation labels)
#   outputs/phase2/graphs_norm/DO_*.pt       (Phase 2 graph cache, tag="norm")
#   outputs/phase3/feature_scaler_norm.npz, outputs/phase3/gvae_norm.pt
#   datasets/DO/inputimages/<name>.jpg       (raw DO images)
#
# The first three are this project's own derived artifacts (not published
# anywhere) -- transfer them from your local machine with
# slurm/sync_data_to_ada.sh (run locally) rather than regenerating them here.
# datasets/DO is published (INBD/MiSCS dataset) -- fetch it directly on Ada
# with `bash slurm/fetch_datasets.sh` instead, no local transfer needed.
#
# A GPU is requested only because build_patches_*.py load the Phase 3 GVAE
# encoder onto `device()` (src/models/gvae/train.py, now cuda > mps > cpu)
# to compute per-patch conditioning latents -- the forward passes themselves
# are small, so this step is I/O/rasterization bound, not compute bound.

set -euo pipefail

set +u
module load cuda-uoneasy/12.6.0
module load anaconda-uoneasy/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

SPECIES="${SPECIES:-DO}"

echo "=== 1/3: build_boundary_sdf.py (whole-silhouette SDF cache) ==="
python3 -m src.models.ddim.build_boundary_sdf "$SPECIES"

echo "=== 2/3: build_patches_boundary.py (256px patches + boundary_sdf crop) ==="
python3 -m src.models.ddim.build_patches_boundary "$SPECIES"

echo "=== 3/3: build_patches_baseline_do256.py (control: same patches, no boundary channel) ==="
python3 -m src.models.ddim.build_patches_baseline_do256 "$SPECIES"

echo "Done. Outputs in outputs/phase5/ (patch_fields_boundary.npy, patch_fields_baseline_do256.npy, etc.)"
