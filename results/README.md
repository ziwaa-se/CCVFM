# Result files

Raw CSVs written by the scripts in `experiments/` for the runs reported in the paper
(single GPU, InceptionV3 pool features; `nfe` for the paper-script CSVs counts
correction-network evaluations, the paper adds one for the closed-form Stage II draw).

| file | what |
|---|---|
| `paper_headline.csv` | every number in the paper's main tables, with baselines and protocol |
| `mnist_k2000_planC_seed0.csv` | MNIST headline run (Table 1) |
| `mnist_k2000_planC_seed1.csv` | independent re-training with Stage III seed 1 |
| `mnist_k2000_sampling_seeds.csv` | 5 sampling seeds on the headline checkpoint (mean ± std) |
| `cifar10_k10000_step{400000,720000}.csv` | CIFAR-10, K=10000; the headline is the 720k-step EMA |
| `imagenet32_k5000_step400000.csv` | ImageNet-32, K=5000, 400k steps (Table 3b) |
| `stage1_cost_*.csv` | wall-clock / memory of Stage I on one GPU (excludes data loading) |
| `mnist_source_ablation.csv` | matched-budget MNIST ablation: coreset surrogate source (5 seeds) vs N(0, I) source (2 seeds) |
| `training_target_second_moment.csv` | measured E‖V1 − V0‖² at t = 0 for each source, with the closed-form check |
