#!/bin/bash
# Unit tests + toy demo (a few minutes; also runs on CPU).
#SBATCH --job-name=tests
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/tests.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
python -m pytest -q tests
python examples/toy2d.py --out assets/toy2d_pipeline.png
