#!/bin/bash
# Paper headline, ImageNet-32 (K=5000, r=80, 400k steps, ~25 h). Downloads ~4 GB on first run.
#SBATCH --job-name=imagenet32
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/imagenet32.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
cd experiments
python imagenet32_pixel_hrf2_ccvfm.py --tag t_hrf2 \
    --K 5000 --rank 80 --cov-nit 1500 \
    --train-steps 400000 --bs 256 \
    --base-lr 1.4e-4 --warmup-steps 5000 \
    --n-gen 50000 \
    --coupling-mode general_t
# -> imagenet32_planT_outputs_t_hrf2/planT_results_t_hrf2.csv  (FID50k 8.76 @ L=50)
