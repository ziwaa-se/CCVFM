#!/usr/bin/env python3
"""Equal-reference 1-NN memorisation diagnostic for the KDE source (review response to VtwU, Q1).

10k generated samples (L = 10) are compared against five random 10k subsets of the
training set and against the 10k held-out test set, for the KDE-source model and the
coreset-GMM (CCVFM) control. A ratio below 1 means samples sit closer to training
than to held-out data. Uses the networks written by mnist_light.py.

Run from experiments/:
    python rebuttal/kde_memorization.py --run-dir rebuttal_outputs/mnist_light \
        --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r30.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from mnist_light import generate, nn_dist_gpu  # noqa: E402
from mnist_pixel_gmm_core import (DEVICE, CorrectionUNet, GMMState, LowRankGMM,  # noqa: E402
                                  load_mnist, to_flat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="--out-dir used for mnist_light.py")
    p.add_argument("--gmm-cache", required=True)
    p.add_argument("--kde-h", type=float, default=0.05)
    p.add_argument("--data", default=".data/mnist.npz")
    args = p.parse_args()
    x2d, _, xt2d, _ = load_mnist(args.data)
    x_train, x_test = to_flat(x2d), to_flat(xt2d)
    d = torch.load(args.gmm_cache, map_location="cpu", weights_only=False)
    state = GMMState(LowRankGMM(d["weights"], d["means"], d["factors"], d["noise_var"]))
    x_all = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    rows = []
    for source, tag in (("kde", f"kde_K1000_r30_seed0_h{args.kde_h:g}"),
                        ("ccvfm", "ccvfm_K1000_r30_seed0")):
        net = CorrectionUNet().to(DEVICE)
        net.load_state_dict(torch.load(os.path.join(args.run_dir, f"net_{tag}.pt"),
                                       map_location=DEVICE, weights_only=False))
        net.eval()
        gen = generate(source, net, state, x_all, args.kde_h, 10000, 10, seed=990).reshape(10000, -1)
        rg = np.random.default_rng(7)
        d_tr = [nn_dist_gpu(gen, x_train[rg.choice(60000, 10000, replace=False)]).mean()
                for _ in range(5)]
        d_te = nn_dist_gpu(gen, x_test).mean()
        rows.append([source, f"{np.mean(d_tr):.4f}", f"{np.std(d_tr):.4f}", f"{d_te:.4f}",
                     f"{np.mean(d_tr) / d_te:.3f}"])
        print(f"[{source}] NN->train (10k subsets) {np.mean(d_tr):.4f} +- {np.std(d_tr):.4f} | "
              f"NN->test (10k) {d_te:.4f} | ratio {np.mean(d_tr) / d_te:.3f}", flush=True)
    with open(os.path.join(args.run_dir, "memorization_equalref.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "nn_train_mean", "nn_train_std", "nn_test", "ratio_train_over_test"])
        w.writerows(rows)
    print("[+] wrote memorization_equalref.csv", flush=True)


if __name__ == "__main__":
    main()
