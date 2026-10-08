# Result files

Raw CSVs written by the scripts in `experiments/` for the runs reported in the paper
(single GPU, InceptionV3 pool features; `nfe` for the paper-script CSVs counts
correction-network evaluations, the paper adds one for the closed-form Stage II draw).

| file | what |
|---|---|
| `paper_headline.csv` | every number in the paper's main tables, with baselines and protocol |
| `mnist_k2000_planC_seed0.csv` | MNIST headline run (Table 1) |
| `cifar10_k10000_step{400000,720000}.csv` | CIFAR-10, K=10000; the headline is the 720k-step EMA |
| `imagenet32_k5000_step400000.csv` | ImageNet-32, K=5000, 400k steps (Table 3b) |
| `mnist_source_ablation.csv` | MNIST, coreset surrogate source (5 seeds) vs N(0, I) source (2 seeds), same network and budget |

## `review/`: experiments from the public review responses

Produced by `experiments/rebuttal/` (see its README for commands). Lightweight budgets;
absolute values are not comparable to the main tables.

| file | what |
|---|---|
| `mnist_k2000_sampling_seeds.csv` | 5 sampling seeds on the MNIST headline checkpoint (mean ± std) |
| `mnist_k2000_retrain_seed1.csv` | MNIST headline re-trained end to end with Stage III seed 1 |
| `mnist_light_rank_and_seeds.csv` | lightweight MNIST (K=1000, 80k steps): rank r ∈ {10, 30, 50, 200, 784}, 3–5 training seeds |
| `mnist_light_kde.csv` | KDE source, h ∈ {0.05, 0.2} |
| `kde_memorization_equalref.csv` | equal-reference 1-NN distances to train vs test (KDE source vs coreset surrogate) |
| `theorem3_moments.csv` | Monte-Carlo E‖V1 − V0‖² at t = 0 vs both closed forms of Theorem 3, 7 datasets |
| `compression_slopes.csv` | log-log slopes of the surrogate gap in K, implied effective dimensions |
| `sw2_vs_k_lowdim.csv` | sliced W2 vs K on the thin circle and two moons (3 seeds) |
| `transport_cost_vs_k_images.csv` | transport-cost proxy and Stage-I timing vs K on MNIST / CIFAR-10 (3 seeds) |
| `stage1_cost_mnist_cifar10.csv`, `stage1_cost_imagenet32.csv` | Stage-I wall-clock and peak memory (excludes data loading) |
