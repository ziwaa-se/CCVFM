#!/usr/bin/env python3
"""Stage-I cost on ImageNet-32 (review response to HaW3, Q2).

Runs the Stage-I EMS-Sinkhorn coreset + closed-form PPCA lift on all 1.28M
ImageNet-32 images (K=5000, r=60) and records wall-clock and peak GPU memory.
The dataset (~4 GB) downloads on first use, as in imagenet32_pixel_hrf2_ccvfm.py.

Run from experiments/:
    python rebuttal/stage1_cost_imagenet32.py --out-dir rebuttal_outputs/stage1_imagenet32
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # experiments/
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from imagenet32_pixel_hrf2_ccvfm import load_imagenet32_pixels  # noqa: E402
from mnist_pixel_gmm_core import (  # noqa: E402
    ems_coreset_gpu, learn_lowrank_cov_ppca, LowRankGMM, save_gmm,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=5000)
    p.add_argument("--rank", type=int, default=60)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    x_pm1, _ = load_imagenet32_pixels()
    print(f"data loaded in {time.time()-t0:.0f}s  shape={x_pm1.shape}", flush=True)

    rg = np.random.default_rng(1234)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(x_pm1, args.K, lam=args.lam,
                                             nit=100, rg=rg)
    t_ems = time.time() - t0
    t1 = time.time()
    L_all, sigma2 = learn_lowrank_cov_ppca(x_pm1, centers, resp, rank=args.rank)
    t_cov = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30

    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(args.out_dir, f"gmm_imagenet32_k{args.K}_r{args.rank}.pt"))

    csv_path = os.path.join(args.out_dir, "stage1_cost_imagenet32.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "d", "K", "rank", "ems_s", "cov_s",
                    "total_s", "peak_mem_gb"])
        w.writerow(["imagenet32", len(x_pm1), x_pm1.shape[1], args.K, args.rank,
                    f"{t_ems:.0f}", f"{t_cov:.0f}", f"{t_ems + t_cov:.0f}",
                    f"{peak:.2f}"])
    print(f"[+] Stage I: EMS {t_ems:.0f}s + cov {t_cov:.0f}s "
          f"(total {(t_ems+t_cov)/60:.1f} min), peak {peak:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
