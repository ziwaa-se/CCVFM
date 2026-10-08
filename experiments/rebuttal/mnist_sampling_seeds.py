#!/usr/bin/env python3
"""Sampling-seed FID error bars on the MNIST headline model (review response to scTg, W1).

Loads the Stage-I GMM and EMA correction U-Net written by ../mnist_pixel_ccvfm.py and
regenerates the 50k-sample FID pool under several independent sampling seeds.

Run from experiments/ (after python mnist_pixel_ccvfm.py):
    python rebuttal/mnist_sampling_seeds.py --model-dir mnist_pixel_ccvfm_outputs \
        --out-dir rebuttal_outputs/sampling_seeds
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from mnist_pixel_gmm_core import (  # noqa: E402
    load_mnist, LowRankGMM, GMMState, IncFeat, get_feats, compute_fid, DEVICE,
)
from mnist_pixel_ccvfm import CorrectionUNetC, generate_corrected_planC  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True,
                   help="output directory of mnist_pixel_ccvfm.py")
    p.add_argument("--K", type=int, default=2000)
    p.add_argument("--unet-base", type=int, default=128)
    p.add_argument("--seeds", type=str, default="101,202,303,404,505")
    p.add_argument("--L-list", type=str, default="10,20,50")
    p.add_argument("--n-gen", type=int, default=50000)
    p.add_argument("--data", type=str, default=".data/mnist.npz")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(os.path.join(args.model_dir, f"gmm_k{args.K}.pt"),
                   map_location="cpu", weights_only=False)
    state = GMMState(LowRankGMM(d["weights"], d["means"], d["factors"],
                                d["noise_var"]))
    model = CorrectionUNetC(base=args.unet_base).to(DEVICE)
    sd = torch.load(os.path.join(args.model_dir, "corr_unet_ema.pt"),
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    x_train_2d, _, _, _ = load_mnist(args.data)
    inc = IncFeat().to(DEVICE)
    f_real = get_feats(inc, x_train_2d[:50000])

    rows = []
    for L in [int(s) for s in args.L_list.split(",")]:
        fids5, fids50 = [], []
        for seed in [int(s) for s in args.seeds.split(",")]:
            gen = generate_corrected_planC(state, model, args.n_gen,
                                           corr_steps=L, seed=seed)
            f_gen = get_feats(inc, gen)
            f5 = compute_fid(f_gen[:5000], f_real[:5000])
            f50 = compute_fid(f_gen, f_real)
            fids5.append(f5); fids50.append(f50)
            rows.append([L, L + 1, seed, f"{f5:.3f}", f"{f50:.3f}"])
            print(f"L={L} seed={seed}: FID5k={f5:.3f} FID50k={f50:.3f}", flush=True)
        print(f"== L={L}: FID50k = {np.mean(fids50):.3f} +- {np.std(fids50):.3f} "
              f"(5k: {np.mean(fids5):.3f} +- {np.std(fids5):.3f})", flush=True)
        rows.append([L, L + 1, "mean+-std",
                     f"{np.mean(fids5):.3f}+-{np.std(fids5):.3f}",
                     f"{np.mean(fids50):.3f}+-{np.std(fids50):.3f}"])

    with open(os.path.join(args.out_dir, "mnist_fid_sampling_seeds.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "NFE", "seed", "fid_5k", "fid_50k"])
        w.writerows(rows)
    print("[+] done", flush=True)


if __name__ == "__main__":
    main()
