#!/bin/bash
# Paper headline, CIFAR-10 part 2/2: continue Stage III 400k -> 720k, then 50k-sample FID sweep.
#SBATCH --job-name=cifar10_resume
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
# Submit from the repository root:  mkdir -p logs && sbatch slurm/cifar10_resume.sbatch
set -euo pipefail
# source your environment here, e.g.  conda activate ccvfm
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${SLURM_SUBMIT_DIR:-.}"
cd experiments
python resume_cifar10_pixel_hrf2_ccvfm.py \
    --start-step 400000 --train-steps 720000 --bs 128 \
    --unet-base 128 --unet-attention "16" --unet-res-blocks 2 \
    --coupling-mode general_t
# -> cifar10_planS_outputs_pp/planSpp_results_step720000.csv  (FID50k 6.35 @ L=50)
# Re-evaluate any milestone later with:
#   CCVFM_CKPT_TAG=step720000 python eval_cifar10_checkpoint.py
