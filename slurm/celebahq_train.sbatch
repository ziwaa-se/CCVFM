#!/bin/bash
# Paper headline, CelebA-HQ 256 (DC-AE f32c32 latents, K=10000, r=128, DiT-L, 400k steps, ~48 h).
#SBATCH --job-name=celebahq_train
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
# Submit from the repository root:  mkdir -p logs && sbatch slurm/celebahq_train.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd experiments
python celebahq_dcae_dit_cfg.py --tag qcfg \
    --n-data 28000 --K 10000 --rank 128 --cov-nit 2000 \
    --train-steps 400000 --bs 128 \
    --hidden-size 1024 --depth 24 --num-heads 16 \
    --base-lr 2e-4 --min-lr 2e-6 --warmup-steps 8000 \
    --null-prob 0.10 --tau-L 50 --tau-power 2.0 --n-gen 10000
# -> celebahq_dcae_planO_outputs_qcfg/{gmm_k10000.pt, dit_corr_ema.pt}
