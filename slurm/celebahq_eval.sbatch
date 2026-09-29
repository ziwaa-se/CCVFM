#!/bin/bash
# CelebA-HQ 256: streamed 30k-sample FID sweep over guidance w in {0,1,1.5,2}. The paper reports w=0 (no guidance).
#SBATCH --job-name=celebahq_eval
#SBATCH --account=YOUR_ACCOUNT          # <- edit
#SBATCH --partition=YOUR_GPU_PARTITION  # <- edit
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Submit from the repository root:  mkdir -p logs && sbatch slurm/celebahq_eval.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
cd experiments
python eval_celebahq_dcae_dit_cfg.py
