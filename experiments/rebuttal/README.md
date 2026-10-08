# Review-period experiments

Code for the additional experiments reported in our public responses to the
NeurIPS 2026 reviewers. Each script reproduces one group of results; the raw
numbers we reported are in [`../../results/review/`](../../results/review/).

Run everything **from `experiments/`** (the scripts import the paper modules
next to them and read `.data/mnist.npz`; create it once with
`python prepare_mnist.py`). Outputs go to `--out-dir`; Stage-I fits are cached
under `rebuttal_outputs/gmm_cache/`. These runs use deliberately lightweight
budgets, so their absolute numbers are not comparable to the paper's tables.

| Result (reviewer question) | Script | Result file | Cost (1 GPU) |
|---|---|---|---|
| Sampling-seed FID error bars on the MNIST headline (scTg W1) | `mnist_sampling_seeds.py` | `mnist_k2000_sampling_seeds.csv` | ~1 h after `mnist_pixel_ccvfm.py` |
| Headline retrained with another training seed (scTg W1) | `../mnist_pixel_ccvfm.py --train-seed 1` | `mnist_k2000_retrain_seed1.csv` | ~1 h |
| Training-seed spread, lightweight MNIST (scTg W1) | `mnist_light.py --source ccvfm --rank 30 --seed {0..4}` | `mnist_light_rank_and_seeds.csv` | ~20 min per run |
| Rank ablation r ∈ {10, 30, 50, 200, 784} (scTg W2) | `mnist_light.py --source ccvfm --rank R --seed {0,1,2}` | `mnist_light_rank_and_seeds.csv` | ~20 min per run |
| KDE source baseline, h ∈ {0.05, 0.2} (VtwU Q1) | `mnist_light.py --source kde --kde-h H` | `mnist_light_kde.csv` | ~20 min per run |
| Equal-reference 1-NN memorisation check (VtwU Q1) | `kde_memorization.py` | `kde_memorization_equalref.csv` | minutes |
| Monte-Carlo validation of both identities of Theorem 3 (HaW3 Q1) | `theorem3_moments.py` | `theorem3_moments.csv` | ~15 min |
| Compression scaling in K: 2D, helix, image proxy and d_eff (HaW3 Q3, VtwU W3) | `compression_scaling.py` | `compression_slopes.csv`, `sw2_vs_k_lowdim.csv`, `transport_cost_vs_k_images.csv` | ~1 h |
| Stage-I cost on MNIST / CIFAR-10 (HaW3 Q2) | `compression_scaling.py` (timing columns) | `stage1_cost_mnist_cifar10.csv` | included above |
| Stage-I cost on ImageNet-32 (HaW3 Q2) | `stage1_cost_imagenet32.py` | `stage1_cost_imagenet32.csv` | ~10 min + download |

## Commands

```bash
cd experiments
python prepare_mnist.py

# scTg W1: sampling seeds on the headline model (needs mnist_pixel_ccvfm_outputs/)
python mnist_pixel_ccvfm.py
python rebuttal/mnist_sampling_seeds.py --model-dir mnist_pixel_ccvfm_outputs \
    --out-dir rebuttal_outputs/sampling_seeds
python mnist_pixel_ccvfm.py --train-seed 1 --out-dir mnist_pixel_ccvfm_outputs_seed1

# scTg W1-W2: lightweight MNIST, training seeds and rank sweep
for R in 10 30 50 200 784; do for S in 0 1 2; do
  python rebuttal/mnist_light.py --source ccvfm --rank $R --seed $S \
      --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r$R.pt \
      --out-dir rebuttal_outputs/mnist_light
done; done
for S in 3 4; do   # r = 30 uses five seeds
  python rebuttal/mnist_light.py --source ccvfm --rank 30 --seed $S \
      --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt --out-dir rebuttal_outputs/mnist_light
done

# VtwU Q1: KDE source and the memorisation check
python rebuttal/mnist_light.py --source kde --kde-h 0.05 --seed 0 \
    --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt --out-dir rebuttal_outputs/mnist_light
python rebuttal/mnist_light.py --source kde --kde-h 0.05 --seed 1 \
    --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt --out-dir rebuttal_outputs/mnist_light
python rebuttal/mnist_light.py --source kde --kde-h 0.2 --seed 0 \
    --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt --out-dir rebuttal_outputs/mnist_light
python rebuttal/kde_memorization.py --run-dir rebuttal_outputs/mnist_light \
    --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt

# HaW3 Q1-Q3, VtwU W3: theory checks, scaling, Stage-I cost
python rebuttal/theorem3_moments.py --out-dir rebuttal_outputs/theorem3
python rebuttal/compression_scaling.py --out-dir rebuttal_outputs/compression
python rebuttal/stage1_cost_imagenet32.py --out-dir rebuttal_outputs/stage1_imagenet32
```

`../../slurm/review_mnist_light.sbatch` runs the lightweight MNIST grid as a SLURM job array.

## Notes

- **Theorem 3.** The surrogate column draws V<sub>0</sub> from the closed-form law
  π̃(· | x<sub>0</sub>, 0) conditionally independently of V<sub>1</sub>, which is the
  sampling rule the theorem analyses. Both identities agree with the Monte-Carlo
  estimates to 0.03–0.3%, and the mean-gap term is 0 to four decimals because the
  Stage-I fixed point preserves the global mean.
- **Compression proxy.** On images the script measures the Sinkhorn transport cost,
  an entropic quantisation-cost proxy rather than W<sub>2</sub><sup>2</sup> itself;
  d<sub>eff</sub> = −2 / slope.
- **Stage-I timing** covers the coreset iterations and covariance fits, excluding
  data loading.
