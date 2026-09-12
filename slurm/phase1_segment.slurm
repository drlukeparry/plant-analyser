#!/usr/bin/env bash
#SBATCH --job-name=phase1_segment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shortq           # Ada cluster: CPU partition, 12hr max
#SBATCH --cpus-per-task=96           # shortq's per-job core cap -- see below, images run in parallel
#SBATCH --mem-per-cpu=4G             # matches this repo's other .slurm scripts; ~4G/worker is enough
                                      # headroom for a single 100MP VM image (see below)
#SBATCH --time=12:00:00              # walltime -- shortq's max; classical_watershed.py is
#SBATCH --array=0-2                  # per-image, several minutes each on the largest VM images
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# slurm/phase1_segment.slurm
#
# Runs Phase 1 (src/segmentation/segment_all.py -- classical rolling-ball +
# adaptive-threshold + watershed segmentation, see PLAN.md Phase 1) on Ada
# for all three species as parallel array tasks:
#
#   array task 0 -> DO (64 images)
#   array task 1 -> EH (82 images)
#   array task 2 -> VM (67 images, includes >100MP images -- segment_all.py
#                   already disables PIL's decompression-bomb guard for this)
#
# CPU-only (skimage/scipy watershed, no torch/GPU use) -- no --gres line,
# runs on `shortq` (Ada's general CPU partition, 12hr max) rather than a GPU
# partition. Each job can request up to 96 cores on this partition; since
# segment_all.py itself processes one image at a time in a single process,
# this drives src/segmentation/segment_one.py (same segment_classical call,
# same skip-if-exists output convention) over every image in the species via
# `xargs -P`, one process per image, instead of segment_all.py's sequential
# loop -- the real way to use that core budget on an embarrassingly
# parallel per-image workload.
#
# Prerequisite: datasets/DO, EH, VM must exist (fetch with
# `bash slurm/fetch_datasets.sh` first, on Ada -- see slurm/README.md).
# segment_one.py skips any image that already has a saved label map, so
# re-submitting after a partial/failed run (e.g. VM hitting the 12hr cap
# before finishing all 67 images) resumes safely rather than restarting.
#
# Submit from the repo root:
#
#   mkdir -p slurm_logs
#   sbatch slurm/phase1_segment.slurm

set -euo pipefail

set +u
module load anaconda-uoneasy/2023.09-0     # no cuda-uoneasy module -- this step is CPU-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate solidtexturenets
set -u

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm_logs

SPECIES_LIST=(DO EH VM)
SPECIES="${SPECIES_LIST[$SLURM_ARRAY_TASK_ID]}"
# Override down from the cpu allocation if a species OOMs -- VM's images run
# up to ~100MP, so its watershed peak memory per worker is the largest of
# the three; e.g. `PARALLEL=32 sbatch ...` for a safer VM-only resubmission.
PARALLEL="${PARALLEL:-$SLURM_CPUS_PER_TASK}"

echo "Array task $SLURM_ARRAY_TASK_ID: segmenting $SPECIES with up to $PARALLEL parallel workers"
find "datasets/$SPECIES/inputimages" -name "${SPECIES}_*.jpg" | \
    xargs -P "$PARALLEL" -I{} python3 -m src.segmentation.segment_one {}

echo "Done. Labels in outputs/phase1/${SPECIES}_*_full_labels.npy"
