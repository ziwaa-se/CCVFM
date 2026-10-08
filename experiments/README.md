# Paper experiments

These are the scripts that produced the numbers in the paper, kept as they were
run: same hyperparameters, same seeds, same FID code. Relative to the original
research code we only renamed modules, removed machine-specific paths, and added
`fid_utils.py` and `prepare_mnist.py`. The cleaner, general-purpose
implementation lives in [`../ccvfm`](../ccvfm).

Run everything **from this directory**: the scripts import each other by module
name and write outputs and caches to relative paths (`.data/`, `*_outputs*/`).
Every trainer accepts `--smoke` for a few-minute sanity pass. SLURM templates for
each command are in [`../slurm`](../slurm).

## Environment

```bash
pip install -r ../requirements-experiments.txt   # torch, torchvision, diffusers, datasets, ...
```

We ran all experiments on a single 96 GB GPU with PyTorch 2.10 / CUDA 12.9.
Datasets download on first use: MNIST and CIFAR-10 from torchvision,
ImageNet-32 from the Hugging Face dataset `ChocolateDave/imagenet-32` (~4 GB,
cached to `.data/imagenet32_train.npy`), CelebA-HQ 256 from `korexyz/celeba-hq-256x256`,
and the DC-AE f32c32 autoencoder from `mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers`.
Set `HF_HOME` to a disk with at least 50 GB free.

## Headline numbers: commands and provenance

### MNIST: FID<sub>50k</sub> 0.75 at 51 NFE (Table 1)

```bash
python prepare_mnist.py            # writes .data/mnist.npz
python mnist_pixel_ccvfm.py        # Stage I (K=2000, r=50) + 200k Stage III steps + FID sweep
```
Output: `mnist_pixel_ccvfm_outputs/planC_results.csv`, where row `StageIII_EMA_50step`
is the headline. The job takes about 50 minutes on one GH200 (44 of them Stage III).
It reproduces exactly: a re-run gives 0.761 and a re-training with
`--train-seed 1` gives 0.747 (see `../results/review/`).

### CIFAR-10: FID<sub>50k</sub> 6.35 at 51 NFE (Table 3a)

The headline checkpoint is the EMA network after **720k** Stage III steps. It was
trained in two jobs: 400k steps, then a continuation.
```bash
python cifar10_pixel_hrf2_ccvfm.py --tag pp --K 10000 --rank 80 --cov-nit 1500 \
    --train-steps 400000 --bs 128 --unet-base 128 --unet-attention "16" \
    --unet-res-blocks 2 --coupling-mode general_t
python resume_cifar10_pixel_hrf2_ccvfm.py --start-step 400000 --train-steps 720000 \
    --bs 128 --unet-base 128 --unet-attention "16" --unet-res-blocks 2 --coupling-mode general_t
```
Output: `cifar10_planS_outputs_pp/planSpp_results_step720000.csv`. The same
network at 400k steps gives 7.04 (`../results/cifar10_k10000_step400000.csv`).
`CCVFM_CKPT_TAG=<tag> python eval_cifar10_checkpoint.py` re-evaluates any saved milestone.

### ImageNet-32: FID<sub>50k</sub> 8.76 at 51 NFE (Table 3b)

```bash
python imagenet32_pixel_hrf2_ccvfm.py --tag t_hrf2 --K 5000 --rank 80 --cov-nit 1500 \
    --train-steps 400000 --bs 256 --base-lr 1.4e-4 --warmup-steps 5000 \
    --n-gen 50000 --coupling-mode general_t
```
Takes about 25 h. Our run crashed after its 300k checkpoint and was finished with
`resume_imagenet32_pixel_hrf2_ccvfm.py --start_step 300000 --end_step 400000`,
which restarts Adam's moments. An uninterrupted run follows the same recipe.

### CelebA-HQ 256: FID 4.17 at 51 NFE, no guidance (Table 3c)

```bash
python celebahq_dcae_dit_cfg.py --tag qcfg --n-data 28000 --K 10000 --rank 128 \
    --cov-nit 2000 --train-steps 400000 --bs 128 --hidden-size 1024 --depth 24 \
    --num-heads 16 --base-lr 2e-4 --min-lr 2e-6 --warmup-steps 8000 \
    --null-prob 0.10 --tau-L 50 --tau-power 2.0 --n-gen 10000
python eval_celebahq_dcae_dit_cfg.py   # streamed 30k-sample sweep, guidance w in {0, 1, 1.5, 2}
```
The paper reports the `w = 0` row, i.e. no classifier-free guidance. The
generated pool is compared against all ~28k CelebA-HQ images. The correction net
is a DiT-L (~458M parameters) on the 2048-d DC-AE latent. Our training run hit a
node failure at step ~73k and was continued from the 60k checkpoint with
`resume_celebahq_dcae_dit_cfg.py --start-step 60000 --extra-steps 340000`, which
uses a fresh AdamW and a cosine schedule with peak 1e-4. The full 400k steps take
~52 h, so on a 48 h queue use the same resume script to split the run.

## Review-period experiments

The additional experiments from our public responses to the reviewers (Theorem 3 checks,
rank and KDE ablations, training- and sampling-seed spreads, compression scaling and
Stage-I cost) are in [`rebuttal/`](rebuttal/), with their own README.

## File map

| file | role |
|---|---|
| `mnist_pixel_gmm_core.py` | Stage I (`ems_coreset_gpu`, `learn_lowrank_cov_fast`), `GMMState`, Stage II sampler, data-anchored coupling at t = 0, MNIST FID |
| `mnist_pixel_ccvfm.py` | MNIST headline trainer and evaluator |
| `cifar10_sdvae_latent_ccvfm.py` | general-t closed-form sampler and coupling (`sample_velocity_general_t`, `coupled_sample_general_t`), EMA and LR schedule helpers |
| `cifar10_pixel_hrf2_ccvfm.py`, `resume_cifar10_pixel_hrf2_ccvfm.py`, `eval_cifar10_checkpoint.py` | CIFAR-10 |
| `imagenet32_pixel_hrf2_ccvfm.py`, `resume_imagenet32_pixel_hrf2_ccvfm.py` | ImageNet-32 |
| `celebahq_dcae_unet.py` | CelebA-HQ loading and DC-AE encode/decode (also a U-Net variant) |
| `celebahq_dcae_dit.py` | DiT-style correction transformer |
| `celebahq_dcae_dit_cfg.py`, `resume_celebahq_dcae_dit_cfg.py`, `eval_celebahq_dcae_dit_cfg.py` | CelebA-HQ headline (DiT-L) |
| `hrf_models/` | HRF2 dual-branch U-Net (third-party, see `THIRD_PARTY_NOTICES.md`) |
| `fid_utils.py` | eigenvalue-based FID used by the CIFAR-10 evaluator |

**Implementation notes.**

- In these scripts the per-component covariance $L_kL_k^\top+\sigma^2 I$ is initialised by randomized SVD and refined by a few hundred Adam steps on the responsibility-weighted Gaussian likelihood (`learn_lowrank_cov_fast`). The `ccvfm` package uses the closed-form PPCA estimator, which is the same model without the refinement.
- The Stage III source $V_0$ is drawn from the surrogate component chosen by the posterior of the training image $x_1$ (`coupled_sample_gpu` / `coupled_sample_general_t`). Its marginal is the surrogate law $\tilde\pi(\cdot\mid x_t,t)$ used at inference.
- The CIFAR-10 and ImageNet-32 CSVs label NFE as $J\cdot L$ (network evaluations). The paper adds one for the closed-form Stage II draw, so L = 50 is reported as 51 NFE.
