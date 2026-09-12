#!/usr/bin/env bash
# slurm/sync_data_to_ada.sh
#
# Transfers the outputs/phase1|2|3 artifacts the boundary-conditioning DDIM
# pipeline needs onto the Ada cluster -- these are this project's OWN derived
# results (classical-watershed segmentation, graph extraction, a trained
# GVAE), not published anywhere, so re-running that pipeline on Ada from
# scratch is real, avoidable compute cost. (The raw datasets/DO images
# themselves ARE published -- fetch those on Ada directly with
# slurm/fetch_datasets.sh instead of transferring them from here.)
#
# Run this LOCALLY (on the machine that has the real outputs/ checked out --
# gitignored, per-project-convention large files, not something `sbatch` can
# pull from anywhere), not on Ada itself:
#
#   ADA_USER=yourusername bash slurm/sync_data_to_ada.sh
#
# Only rsyncs the DO-only subset build_boundary_sdf.py/build_patches_*.py
# actually read (species="DO" default) -- not the full outputs/ tree, which
# also holds EH/VM data unused by this pipeline (~28GB vs. ~7GB DO-only):
#
#   outputs/phase1/DO_*_full_labels.npy   (~6.2GB, Phase 1 segmentation)
#   outputs/phase2/graphs_norm/DO_*.pt    (~0.7GB, Phase 2 graph cache, tag="norm")
#   outputs/phase3/feature_scaler_norm.npz, outputs/phase3/gvae_norm.pt (~a few MB)
#
# rsync so re-running after this is safe/incremental (only changed files
# transfer). Requires SSH access to the login node already set up (the same
# access `sbatch`/`sinfo` commands assume) -- this script does not configure
# SSH keys or accounts.

set -euo pipefail

ADA_USER="${ADA_USER:?Set ADA_USER to your Ada username, e.g. ADA_USER=abc12de bash $0}"
ADA_HOST="${ADA_HOST:-hpclogin01.ada.nottingham.ac.uk}"
REMOTE_DIR="${REMOTE_DIR:-~/plant-analyser}"   # must match where the repo is checked out on Ada

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RSYNC_OPTS=(-avz --progress --prune-empty-dirs)

echo "Syncing DO-only data from $LOCAL_ROOT to $ADA_USER@$ADA_HOST:$REMOTE_DIR ..."

ssh "$ADA_USER@$ADA_HOST" "mkdir -p $REMOTE_DIR/outputs/phase1 $REMOTE_DIR/outputs/phase2/graphs_norm $REMOTE_DIR/outputs/phase3"

echo "--- outputs/phase1 (DO_*_full_labels.npy) ---"
rsync "${RSYNC_OPTS[@]}" --include="DO_*_full_labels.npy" --exclude="*" \
    "$LOCAL_ROOT/outputs/phase1/" "$ADA_USER@$ADA_HOST:$REMOTE_DIR/outputs/phase1/"

echo "--- outputs/phase2/graphs_norm (DO_*.pt) ---"
rsync "${RSYNC_OPTS[@]}" --include="DO_*.pt" --exclude="*" \
    "$LOCAL_ROOT/outputs/phase2/graphs_norm/" "$ADA_USER@$ADA_HOST:$REMOTE_DIR/outputs/phase2/graphs_norm/"

echo "--- outputs/phase3 (feature_scaler_norm.npz, gvae_norm.pt) ---"
rsync "${RSYNC_OPTS[@]}" \
    "$LOCAL_ROOT/outputs/phase3/feature_scaler_norm.npz" \
    "$LOCAL_ROOT/outputs/phase3/gvae_norm.pt" \
    "$ADA_USER@$ADA_HOST:$REMOTE_DIR/outputs/phase3/"

echo "Done. On Ada, outputs/phase1, outputs/phase2/graphs_norm, and outputs/phase3 under" \
     "$REMOTE_DIR should now match the source needed for slurm/build_boundary_data.slurm." \
     "Still need datasets/DO on Ada? Run slurm/fetch_datasets.sh there (or here first, then" \
     "let this script's next run pick it up -- but fetching directly on Ada is faster)."
