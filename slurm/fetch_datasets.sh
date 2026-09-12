#!/usr/bin/env bash
# slurm/fetch_datasets.sh
#
# Downloads the raw DO/EH/VM shrub cross-section datasets (MiSCS, Anadon-
# Rosell et al., used by the INBD paper: github.com/alexander-g/INBD) directly
# from their GitHub release, instead of transferring datasets/DO/inputimages
# from a local checkout. Mirrors github.com/alexander-g/INBD/blob/master/
# fetch_datasets.py's URLs/extraction (confirmed in PLAN.md: the locally-held
# datasets/DO|EH|VM/ already matches these released zips exactly), adapted to
# this repo's `datasets/` (plural) layout instead of INBD's `dataset/`.
#
# Run this wherever the repo is checked out and needs datasets/<SPECIES> --
# typically on the Ada login node itself (it has outbound internet access),
# so no local-machine upload is needed for this part:
#
#   bash slurm/fetch_datasets.sh          # all three: DO, EH, VM
#   bash slurm/fetch_datasets.sh DO       # just one species
#
# Skips any species whose datasets/<SPECIES>/inputimages already exists, so
# re-running is a no-op once fetched -- doesn't re-download or overwrite.
#
# This only gets the raw inputimages/annotations. It does NOT regenerate
# outputs/phase1 (segmentation)/phase2 (graphs)/phase3 (GVAE) -- those are
# this project's own derived artifacts, not published anywhere, and
# re-running that pipeline on Ada from scratch is exactly what
# sync_data_to_ada.sh exists to avoid. Use that script to transfer the
# already-computed outputs/phase1|2|3 from your local machine instead.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS_DIR="$REPO_ROOT/datasets"
SPECIES_LIST=("$@")
[[ ${#SPECIES_LIST[@]} -eq 0 ]] && SPECIES_LIST=(DO EH VM)

declare -A URLS=(
    [DO]="https://github.com/alexander-g/INBD/releases/download/dataset_v1/DO_v1.zip"
    [EH]="https://github.com/alexander-g/INBD/releases/download/dataset_v1/EH_v1.zip"
    [VM]="https://github.com/alexander-g/INBD/releases/download/dataset_v1/VM_v1.zip"
)

mkdir -p "$DATASETS_DIR"

for species in "${SPECIES_LIST[@]}"; do
    url="${URLS[$species]:-}"
    if [[ -z "$url" ]]; then
        echo "unknown species '$species' -- expected one of: ${!URLS[*]}" >&2
        exit 1
    fi
    if [[ -d "$DATASETS_DIR/$species/inputimages" ]]; then
        echo "datasets/$species/inputimages already exists, skipping"
        continue
    fi

    tmp_zip="$(mktemp -t "${species}_v1.XXXXXX.zip")"
    echo "Downloading $url ..."
    curl -fL --progress-bar -o "$tmp_zip" "$url"

    echo "Extracting into $DATASETS_DIR ..."
    unzip -q -o "$tmp_zip" -d "$DATASETS_DIR"
    rm -f "$tmp_zip"

    if [[ ! -d "$DATASETS_DIR/$species/inputimages" ]]; then
        echo "warning: expected $DATASETS_DIR/$species/inputimages after extraction, not found -- check the zip's internal layout" >&2
    fi
done

echo "Done. datasets/ now has: $(ls "$DATASETS_DIR")"
