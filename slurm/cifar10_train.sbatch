#!/bin/bash
# Paper headline, CIFAR-10 part 1/2: Stage I (K=10000, r=80) + 400k Stage III steps (~24 h).
#SBATCH --job-name=cifar10_train
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/cifar10_train.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
cd experiments
python cifar10_pixel_hrf2_ccvfm.py --tag pp \
    --K 10000 --rank 80 --cov-nit 1500 \
    --train-steps 400000 --bs 128 \
    --unet-base 128 --unet-attention "16" --unet-res-blocks 2 \
    --coupling-mode general_t
# -> cifar10_planS_outputs_pp/{gmm_k10000.pt, unet.pt, unet_ema.pt}
