#!/bin/bash
# Paper headline, MNIST (K=2000, r=50, 200k steps). FID50k 0.75 @ 51 NFE.
#SBATCH --job-name=mnist
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/mnist.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
cd experiments
python prepare_mnist.py
python mnist_pixel_ccvfm.py            # add --smoke for a 3-minute sanity pass
# -> mnist_pixel_ccvfm_outputs/planC_results.csv
