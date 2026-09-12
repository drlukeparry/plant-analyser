#!/usr/bin/env bash
# slurm/env_setup.sh
#
# One-time environment setup on the HPC login node. Run this once before
# submitting any job with sbatch:
#
#   bash slurm/env_setup.sh
#
# Configured for the University of Nottingham Ada cluster
# (hpclogin01.ada.nottingham.ac.uk), confirmed via `module avail` on the
# login node: cuda-uoneasy/12.6.0 is the default CUDA module there.
#
# Note: Ada also has a pre-built `pytorch-uoneasy/2.1.2-foss-2023a-CUDA-12.1.1`
# module -- `module load pytorch-uoneasy` is a simpler alternative to this
# whole script if you don't need a specific torch version, since it comes
# with a matching CUDA already wired up (don't load cuda-uoneasy alongside
# it in that case, it brings its own).

set -euo pipefail

module load anaconda-uoneasy/2023.09-0
module load cuda-uoneasy/12.6.0

ENV_NAME="solidtexturenets"

if ! command -v conda &> /dev/null; then
    echo "conda not found on PATH -- module load failed or conda isn't on PATH after loading." >&2
    exit 1
fi

conda create -y -n "$ENV_NAME" python=3.11

# conda's activation hook references some unset variables internally --
# under `set -u` (active via set -euo pipefail above) this fails with an
# unbound-variable error. Relax -u just for the activation itself.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# Matches the cluster's cuda-uoneasy/12.6.0 module exactly -- PyTorch
# publishes a cu126 wheel index (confirmed: torch up to 2.14.0, torchvision
# up to 0.29.0, both built for CUDA 12.6, python 3.10-3.13).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -e "$(dirname "$0")/.."

echo "Environment '$ENV_NAME' ready. Activate with: conda activate $ENV_NAME"
