#!/bin/bash
# Review-period lightweight MNIST grid (rank sweep, training seeds, KDE source) as a job array.
#SBATCH --job-name=review_mnist_light
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/review_mnist_light.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd experiments
[ -f .data/mnist.npz ] || python prepare_mnist.py
# 18 cells: rank sweep (15), extra r=30 seeds (2), KDE h=0.2 (1); KDE h=0.05 seeds below
CELLS=(
  "ccvfm 10 0" "ccvfm 10 1" "ccvfm 10 2" "ccvfm 30 0" "ccvfm 30 1" "ccvfm 30 2"
  "ccvfm 50 0" "ccvfm 50 1" "ccvfm 50 2" "ccvfm 200 0" "ccvfm 200 1" "ccvfm 200 2"
  "ccvfm 784 0" "ccvfm 784 1" "ccvfm 784 2" "ccvfm 30 3" "ccvfm 30 4" "kde 30 0 0.2"
)
read -r SRC R S H <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
python rebuttal/mnist_light.py --source $SRC --rank $R --seed $S ${H:+--kde-h $H} \
    --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r$R.pt \
    --out-dir rebuttal_outputs/mnist_light
# KDE h=0.05 (seeds 0, 1) and the memorisation check: see experiments/rebuttal/README.md
