#!/usr/bin/env python3
"""Lightweight MNIST runs used in the review responses (scTg W1-W2, VtwU Q1).

One run = one (source, seed, K, rank) cell of the lightweight recipe
(80k Stage III iterations, base-64 U-Net, FID50k at L in {5, 10, 20}):

  ccvfm   the CCVFM correction flow (same Stage III as ../mnist_pixel_ccvfm.py).
          Used for the 5-training-seed spread and the rank sweep r in {10,...,784}.
  kde     KDE source: x_train[j] + h * eps (the K = n, equal-weight, isotropic
          endpoint of the surrogate family), plus 1-NN memorisation diagnostics.

Stage I (EMS coreset + closed-form PPCA) is shared through --gmm-cache, built once
per (K, rank). The full-rank arm (r = 784) uses sigma^2 -> 0, L = U Lambda^{1/2}.

Run from experiments/ (after python prepare_mnist.py), e.g.
    python rebuttal/mnist_light.py --source ccvfm --seed 0 --rank 50 \
        --gmm-cache rebuttal_outputs/gmm_cache/gmm_mnist_k1000_r50.pt \
        --out-dir rebuttal_outputs/mnist_light
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
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from mnist_pixel_gmm_core import (  # noqa: E402
    D, DEVICE, CorrectionUNet, GMMState, IncFeat, LowRankGMM, compute_fid, coupled_sample_gpu,
    ems_coreset_gpu, get_feats, learn_lowrank_cov_ppca, load_mnist, save_gmm, save_grid, to_flat,
)


@torch.no_grad()
def sample_gmm_gpu(state: GMMState, n, generator=None):
    """X ~ rho1_tilde (categorical + low-rank Gaussian): the inference-time source."""
    comp = torch.multinomial(state.weights, n, replacement=True, generator=generator)
    z_r = torch.randn(n, state.r, device=DEVICE, generator=generator)
    z_d = torch.randn(n, state.d, device=DEVICE, generator=generator)
    return (state.means[comp] + torch.einsum("bdr,br->bd", state.L[comp], z_r)
            + state.sigma * z_d)


@torch.no_grad()
def training_source(source, x1, x0, x_all, state, kde_h, g):
    """Source endpoint x1_tilde = x0 + V0 for a training batch."""
    bs = x1.shape[0]
    if source == "kde":
        j = torch.randint(0, len(x_all), (bs,), device=DEVICE, generator=g)
        return x_all[j] + kde_h * torch.randn(bs, D, device=DEVICE, generator=g)
    return x0 + coupled_sample_gpu(x1 - x0, x0, state, generator=g)


def train(source, x_train_flat, state, args, loss_csv):
    net = CorrectionUNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    x_all = torch.tensor(x_train_flat, dtype=torch.float32, device=DEVICE)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(args.seed)
    # The original runs estimated E||V1 - V0||^2 here (20 x 1024 pairs) before training;
    # replaying those draws keeps the random stream identical to the reported runs.
    for _ in range(20):
        idx = torch.randint(0, len(x_all), (1024,), device=DEVICE, generator=g)
        training_source(source, x_all[idx], torch.randn(1024, D, device=DEVICE, generator=g),
                        x_all, state, args.kde_h, g)
    with open(loss_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "loss"])
        run, t0 = 0.0, time.time()
        net.train()
        for step in range(args.iters):
            idx = torch.randint(0, len(x_all), (args.bs,), device=DEVICE, generator=g)
            x1 = x_all[idx]
            x0 = torch.randn(args.bs, D, device=DEVICE, generator=g)
            torch.randn(args.bs, D, device=DEVICE, generator=g)  # keeps the original RNG stream
            xt1 = training_source(source, x1, x0, x_all, state, args.kde_h, g)
            tau = torch.rand(args.bs, device=DEVICE, generator=g).view(-1, 1)
            v_tau = ((1 - tau) * (xt1 - x0) + tau * (x1 - x0)).reshape(-1, 1, 28, 28)
            target = (x1 - xt1).reshape(-1, 1, 28, 28)
            pred = net(v_tau, x0.reshape(-1, 1, 28, 28), tau.view(-1),
                       torch.zeros(args.bs, device=DEVICE))
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            if (step + 1) % 100 == 0:
                wr.writerow([step + 1, f"{run / 100:.6f}"])
                run = 0.0
            if (step + 1) % 2000 == 0:
                print(f"    step {step + 1}/{args.iters} loss={loss.item():.5f} "
                      f"wall={time.time() - t0:.0f}s", flush=True)
    net.eval()
    return net, time.time() - t0


@torch.no_grad()
def generate(source, net, state, x_all, kde_h, n, L, seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    out = []
    for start in range(0, n, 512):
        nb = min(512, n - start)
        x0 = torch.randn(nb, D, device=DEVICE, generator=g)
        torch.randn(nb, D, device=DEVICE, generator=g)  # keeps the original RNG stream
        if source == "kde":
            j = torch.randint(0, len(x_all), (nb,), device=DEVICE, generator=g)
            xt1 = x_all[j] + kde_h * torch.randn(nb, D, device=DEVICE, generator=g)
        else:
            xt1 = sample_gmm_gpu(state, nb, generator=g)
        v = (xt1 - x0).reshape(nb, 1, 28, 28)
        x0_img = x0.reshape(nb, 1, 28, 28)
        t_cond = torch.zeros(nb, device=DEVICE)
        for i in range(L):
            v = v + net(v, x0_img, torch.full((nb,), i / L, device=DEVICE), t_cond) / L
        out.append((v + x0_img).clamp(0, 1).reshape(nb, 28, 28).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def nn_dist_gpu(gen_flat, ref_flat, bs=256):
    ref = torch.tensor(ref_flat, dtype=torch.float32, device=DEVICE)
    out = []
    for i in range(0, len(gen_flat), bs):
        gb = torch.tensor(gen_flat[i:i + bs], dtype=torch.float32, device=DEVICE)
        out.append(torch.cdist(gb, ref).min(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def load_or_build_gmm(path, x_train, K, rank):
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return LowRankGMM(d["weights"], d["means"], d["factors"], d["noise_var"])
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(x_train, K, lam=1.5, nit=100,
                                             rg=np.random.default_rng(1234))
    L_all, sigma2 = learn_lowrank_cov_ppca(x_train, centers, resp, rank=rank)
    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_gmm(gmm, path)
    print(f"Stage I built in {time.time() - t0:.0f}s", flush=True)
    return gmm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["ccvfm", "kde"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--rank", type=int, default=30)
    p.add_argument("--iters", type=int, default=80000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--kde-h", type=float, default=0.05)
    p.add_argument("--fid-L", type=str, default="5,10,20")
    p.add_argument("--n-gen", type=int, default=50000)
    p.add_argument("--gmm-cache", type=str, required=True, help="Stage-I cache; built if missing")
    p.add_argument("--data", type=str, default=".data/mnist.npz")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.source}_K{args.K}_r{args.rank}_seed{args.seed}"
    if args.source == "kde":
        tag += f"_h{args.kde_h:g}"

    x_train_2d, _, x_test_2d, _ = load_mnist(args.data)
    x_train, x_test = to_flat(x_train_2d), to_flat(x_test_2d)
    state = GMMState(load_or_build_gmm(args.gmm_cache, x_train, args.K, args.rank))

    net, train_s = train(args.source, x_train, state, args,
                         os.path.join(args.out_dir, f"loss_{tag}.csv"))
    torch.save(net.state_dict(), os.path.join(args.out_dir, f"net_{tag}.pt"))

    x_all = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    inc = IncFeat().to(DEVICE)
    f_real = get_feats(inc, x_train_2d[:50000])
    rows = []
    for L in [int(s) for s in args.fid_L.split(",")]:
        gen = generate(args.source, net, state, x_all, args.kde_h, args.n_gen, L,
                       seed=990 + args.seed)
        f_gen = get_feats(inc, gen)
        fid50 = compute_fid(f_gen, f_real)
        rows.append([args.source, args.seed, args.K, args.rank,
                     args.kde_h if args.source == "kde" else "", L, L + 1,
                     f"{fid50:.3f}", f"{train_s:.0f}"])
        print(f"  [{tag}] L={L}: FID50k={fid50:.3f}", flush=True)
        if L == 10:
            save_grid(gen[:100], os.path.join(args.out_dir, f"grid_{tag}_L10.png"),
                      f"{tag} L=10 (FID50k={fid50:.2f})")
            n_mem = min(10000, len(gen))
            g10 = gen[:n_mem].reshape(n_mem, -1)
            d_tr, d_te = nn_dist_gpu(g10, x_train[:50000]), nn_dist_gpu(g10, x_test)
            with open(os.path.join(args.out_dir, f"memorization_{tag}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["quantile", "nn_dist_train", "nn_dist_test"])
                for q in (5, 25, 50, 75, 95):
                    w.writerow([q, f"{np.percentile(d_tr, q):.4f}", f"{np.percentile(d_te, q):.4f}"])
                w.writerow(["mean", f"{d_tr.mean():.4f}", f"{d_te.mean():.4f}"])
    with open(os.path.join(args.out_dir, f"summary_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "seed", "K", "rank", "kde_h", "L", "NFE", "fid_50k", "train_s"])
        w.writerows(rows)
    print(f"[+] wrote summary_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
