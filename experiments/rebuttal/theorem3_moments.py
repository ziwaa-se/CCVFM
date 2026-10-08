#!/usr/bin/env python3
"""Monte-Carlo validation of both identities of Theorem 3 (review response to HaW3, Q1).

At t = 0 the training-target second moment E||V1 - V0||^2 is measured under
  (i)  the surrogate source analysed in Theorem 3, V0 ~ pi_tilde(. | x0, 0) drawn
       conditionally independently of V1, and compared with
       tr Cov(rho1) + tr Cov(rho1_tilde) + ||E X1 - E X1_tilde||^2;
  (ii) the independent Gaussian source V0 ~ N(0, I), compared with E||V1||^2 + d.

Datasets: the submission's five toy targets (ring-6, moons, pinwheel, checkerboard,
helix) and the image benchmarks MNIST (K=2000, r=50) and CIFAR-10 (K=10000, r=80).

Run from experiments/:
    python rebuttal/theorem3_moments.py --out-dir rebuttal_outputs/theorem3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from toy_core import TARGETS, build_gmm, rng, sample_gmm  # noqa: E402


def toy_rows():
    rows = []
    for name, sampler, dim, K, lam, floor in TARGETS:
        rg_ = rng(11)
        x = sampler(200000, rg_)
        gmm = build_gmm(x[:20000], K, lam, floor, rng(11))
        x0 = rg_.standard_normal((200000, dim))
        x1 = x[rg_.choice(200000, 200000, replace=True)]
        x1t = sample_gmm(gmm, 200000, rg_)
        v1 = x1 - x0
        m_sur = float(((x1 - x1t) ** 2).sum(1).mean())
        m_gau = float(((v1 - rg_.standard_normal((200000, dim))) ** 2).sum(1).mean())
        mean_x = x.mean(0)
        tr_data = float(((x - mean_x) ** 2).sum(1).mean())
        w, mu, covs = gmm.weights, gmm.means, gmm.covs
        mu_bar = (w[:, None] * mu).sum(0)
        th_i = (tr_data + float((w * np.trace(covs, axis1=1, axis2=2)).sum())
                + float((w * ((mu - mu_bar) ** 2).sum(1)).sum())
                + float(((mean_x - mu_bar) ** 2).sum()))
        th_ii = float((v1 ** 2).sum(1).mean()) + dim
        rows.append([name, dim, m_sur, th_i, m_gau, th_ii])
        print(f"[{name}] surrogate {m_sur:.3f} (theory {th_i:.3f}) | "
              f"gaussian {m_gau:.3f} (theory {th_ii:.3f})", flush=True)
    return rows


def image_rows(datasets, cache_dir, mnist_data):
    import torch
    from mnist_pixel_gmm_core import (DEVICE, GMMState, LowRankGMM, ems_coreset_gpu,
                                      learn_lowrank_cov_ppca, load_mnist, sample_velocity_gpu,
                                      save_gmm, to_flat)
    rows = []
    for ds in datasets:
        if ds == "mnist":
            x = to_flat(load_mnist(mnist_data)[0])
            K, rank, lam = 2000, 50, 1.5
        else:
            from cifar10_pixel_hrf2_ccvfm import load_cifar10_pixels
            x = load_cifar10_pixels(".data")[0]
            K, rank, lam = 10000, 80, 0.5
        d = x.shape[1]
        cache = os.path.join(cache_dir, f"gmm_{ds}_k{K}_r{rank}.pt")
        if os.path.exists(cache):
            sd = torch.load(cache, map_location="cpu", weights_only=False)
            gmm = LowRankGMM(sd["weights"], sd["means"], sd["factors"], sd["noise_var"])
        else:
            centers, weights, resp = ems_coreset_gpu(x, K, lam=lam, nit=100,
                                                     rg=np.random.default_rng(1234))
            L_all, sigma2 = learn_lowrank_cov_ppca(x, centers, resp, rank=rank)
            gmm = LowRankGMM(weights, centers, L_all, sigma2)
            os.makedirs(cache_dir, exist_ok=True)
            save_gmm(gmm, cache)
        state = GMMState(gmm)

        # Monte-Carlo estimates: 50 batches x 1024 pairs, shared (x0, x1) across columns
        with torch.no_grad():
            g = torch.Generator(device=DEVICE).manual_seed(7)
            x_all = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            m_sur = m_gau = v1_sq = 0.0
            for _ in range(50):
                idx = torch.randint(0, len(x_all), (1024,), device=DEVICE, generator=g)
                x1 = x_all[idx]
                x0 = torch.randn(1024, d, device=DEVICE, generator=g)
                v1 = x1 - x0
                v0_sur = sample_velocity_gpu(x0, state, generator=g)
                v0_gau = torch.randn(1024, d, device=DEVICE, generator=g)
                m_sur += ((v1 - v0_sur) ** 2).sum(-1).mean().item() / 50
                m_gau += ((v1 - v0_gau) ** 2).sum(-1).mean().item() / 50
                v1_sq += (v1 ** 2).sum(-1).mean().item() / 50

        # Closed form of Theorem 3(i) from the data and the GMM parameters
        xd = x.astype(np.float64)
        w = np.asarray(gmm.weights, dtype=np.float64)
        mu = np.asarray(gmm.means, dtype=np.float64)
        L = np.asarray(gmm.factors, dtype=np.float64)
        s2 = float(gmm.noise_var)
        mean_x = xd.mean(0)
        tr_data = float(((xd - mean_x) ** 2).sum(1).mean())
        mu_bar = (w[:, None] * mu).sum(0)
        tr_within = float((w * ((L ** 2).sum(axis=(1, 2)) + d * s2)).sum())
        tr_between = float((w * ((mu - mu_bar) ** 2).sum(1)).sum())
        mean_gap = float(((mean_x - mu_bar) ** 2).sum())
        th_i = tr_data + tr_within + tr_between + mean_gap
        rows.append([ds, d, m_sur, th_i, m_gau, v1_sq + d])
        print(f"[{ds}] surrogate {m_sur:.1f} (theory {th_i:.1f}; mean gap {mean_gap:.4f}) | "
              f"gaussian {m_gau:.1f} (theory {v1_sq + d:.1f})", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--images", default="mnist,cifar", help="comma list, or '' to skip")
    p.add_argument("--cache-dir", default="rebuttal_outputs/gmm_cache")
    p.add_argument("--mnist-data", default=".data/mnist.npz")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = toy_rows()
    if args.images:
        rows += image_rows(args.images.split(","), args.cache_dir, args.mnist_data)
    with open(os.path.join(args.out_dir, "theorem3_moments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "d", "surrogate_measured", "surrogate_theory_3i",
                    "gaussian_measured", "gaussian_theory_3ii", "ratio_gauss_over_surrogate"])
        for name, d, ms, ti, mg, tii in rows:
            w.writerow([name, d, f"{ms:.4f}", f"{ti:.4f}", f"{mg:.4f}", f"{tii:.4f}",
                        f"{mg / ms:.2f}"])
    print("[+] wrote theorem3_moments.csv", flush=True)


if __name__ == "__main__":
    main()
