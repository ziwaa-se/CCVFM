#!/usr/bin/env python3
"""Compression scaling in K and Stage-I cost (review responses to HaW3 Q2-Q3 and VtwU W3).

Part 1  low-dimensional benchmarks (thin circle, two moons): sliced-W2 between
        surrogate draws and held-out data across K; log-log slope vs the K^{-1/2}
        benchmark.
Part 2  the submission's helix (a noise-thickened curve in R^3): same diagnostic
        with a small entropic bandwidth; slope vs the nominal -1/3.
Part 3  MNIST / CIFAR-10: Sinkhorn transport cost (1/n) sum T_ib ||x_i - mu_b||^2
        across a 16x range of K (an entropic quantisation-cost proxy, not W2^2),
        implied effective dimension d_eff = -2 / slope, and the Stage-I wall-clock
        and peak GPU memory of every fit.

Run from experiments/:
    python rebuttal/compression_scaling.py --out-dir rebuttal_outputs/compression
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from toy_core import (coreset_to_gmm, ems_coreset, rng, sample_gmm,  # noqa: E402
                      sample_helix3d, sample_thin_circle, sample_two_moons, sliced_w2)

# name: (sampler, lam, cov_floor, rank, d)
LOWDIM = {
    "ThinCircle": (lambda n, rg_: sample_thin_circle(n, rg_), 0.05, 0.010, 1, 2),
    "TwoMoons": (lambda n, rg_: sample_two_moons(n, rg_), 0.05, 0.012, 1, 2),
}
K_LOW = [2, 4, 8, 16, 32, 64, 128, 256]
K_HELIX = [4, 8, 16, 32, 64, 128, 256]


def fit_slope(Ks, vals):
    return float(np.polyfit(np.log(np.asarray(Ks)), np.log(np.asarray(vals)), 1)[0])


def lowdim(seeds, rows, slopes):
    for name, (sampler, lam, floor, rank, d) in LOWDIM.items():
        per_seed = []
        for seed in seeds:
            rg_ = rng(seed)
            x = sampler(20000, rg_)
            x_ref = sampler(20000, rng(seed + 77))
            sw2s = []
            for K in K_LOW:
                centers, weights, _, tmat = ems_coreset(x, K, lam, 45, rg_)
                gmm = coreset_to_gmm(x, centers, weights, tmat, cov_floor=floor,
                                     rank=min(rank, d - 1) or 1)
                sw2 = sliced_w2(sample_gmm(gmm, 20000, rg_), x_ref, n_proj=200, rg=rng(seed + 5))
                sw2s.append(sw2)
                rows.append([name, d, seed, K, f"{sw2:.6f}"])
            per_seed.append(fit_slope(K_LOW, sw2s))
            print(f"[{name}] seed={seed} slope={per_seed[-1]:.3f}", flush=True)
        slopes.append([name, d, f"{np.mean(per_seed):.3f}", f"{np.std(per_seed):.3f}",
                       f"{-1.0 / d:.3f}", ""])


def helix(seeds, rows, slopes):
    per_seed = []
    for seed in seeds:
        rg_ = rng(seed)
        x = sample_helix3d(20000, rg_)
        ref = sample_helix3d(20000, rng(seed + 77))
        sw2s = []
        for K in K_HELIX:
            centers, weights, _, tmat = ems_coreset(x, K, 0.02, 60, rg_)
            g = coreset_to_gmm(x, centers, weights, tmat, cov_floor=0.002)
            sw2s.append(sliced_w2(sample_gmm(g, 20000, rg_), ref, n_proj=200, rg=rng(seed + 5)))
            rows.append(["helix", 3, seed, K, f"{sw2s[-1]:.6f}"])
        per_seed.append(fit_slope(K_HELIX, sw2s))
        print(f"[helix] seed={seed} slope={per_seed[-1]:.3f}", flush=True)
    m = float(np.mean(per_seed))
    slopes.append(["helix", 3, f"{m:.3f}", f"{np.std(per_seed):.3f}", f"{-1.0 / 3:.3f}",
                   f"{-1.0 / m:.2f}"])


def images(image_seeds, mnist_data, img_rows, slopes):
    import torch
    from mnist_pixel_gmm_core import (DEVICE, ems_coreset_gpu, learn_lowrank_cov_ppca,
                                      load_mnist, to_flat)
    sweeps = [("mnist", [250, 500, 1000, 2000, 4000], 30, 1.5),
              ("cifar", [625, 1250, 2500, 5000, 10000], 80, 0.5)]
    for ds, Ks, rank, lam in sweeps:
        if ds == "mnist":
            x = to_flat(load_mnist(mnist_data)[0])
        else:
            from cifar10_pixel_hrf2_ccvfm import load_cifar10_pixels
            x = load_cifar10_pixels(".data")[0]
        per_seed = []
        for sd in image_seeds:
            rg = np.random.default_rng(sd)
            costs = []
            for K in Ks:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                centers, weights, resp = ems_coreset_gpu(x, K, lam=lam, nit=100, rg=rg)
                t_ems = time.time() - t0
                t1 = time.time()
                learn_lowrank_cov_ppca(x, centers, resp, rank=rank)
                t_cov = time.time() - t1
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                xc = torch.tensor(x, dtype=torch.float32, device=DEVICE)
                cc = torch.tensor(centers, dtype=torch.float32, device=DEVICE)
                rr = torch.tensor(resp, dtype=torch.float32, device=DEVICE)
                tot = 0.0
                for i in range(0, len(xc), 4096):
                    tot += float((rr[i:i + 4096] * torch.cdist(xc[i:i + 4096], cc) ** 2).sum())
                cost = tot / len(xc)
                costs.append(cost)
                img_rows.append([ds, len(x), x.shape[1], K, rank, sd, f"{cost:.4f}",
                                 f"{t_ems:.0f}", f"{t_cov:.0f}", f"{t_ems + t_cov:.0f}",
                                 f"{peak:.2f}"])
                print(f"[{ds}] seed={sd} K={K}: cost={cost:.4f} "
                      f"stage I {t_ems + t_cov:.0f}s, peak {peak:.2f} GB", flush=True)
                del xc, cc, rr
                torch.cuda.empty_cache()
            per_seed.append(fit_slope(Ks, costs))
        m = float(np.mean(per_seed))
        slopes.append([ds, x.shape[1], f"{m:.3f}", f"{np.std(per_seed):.3f}",
                       f"{-1.0 / x.shape[1]:.5f}", f"{-2.0 / m:.1f}"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="11,28,45")
    p.add_argument("--image-seeds", default="1234,5678,9012")
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--mnist-data", default=".data/mnist.npz")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    low_rows, img_rows, slopes = [], [], []
    lowdim(seeds, low_rows, slopes)
    helix(seeds, low_rows, slopes)
    if not args.skip_images:
        images([int(s) for s in args.image_seeds.split(",")], args.mnist_data, img_rows, slopes)
    with open(os.path.join(args.out_dir, "sw2_vs_k_lowdim.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["target", "d", "seed", "K", "sliced_w2"]] + low_rows)
    with open(os.path.join(args.out_dir, "transport_cost_vs_k_images.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["dataset", "n", "d", "K", "rank", "seed", "transport_cost",
                                  "ems_s", "cov_s", "stage1_s", "peak_mem_gb"]] + img_rows)
    with open(os.path.join(args.out_dir, "slopes.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["target", "d", "slope_mean", "slope_std", "nominal_-1/d",
                                  "implied_d_eff"]] + slopes)
    print("[+] wrote slopes.csv", flush=True)


if __name__ == "__main__":
    main()
