#!/usr/bin/env python3
"""MNIST with the ccvfm package: Stage I -> II -> III and FID, on one GPU.

This is the light "K=1000, r=30, 80k steps" recipe of the paper's MNIST scaling
study, written against the small ccvfm API instead of the paper scripts
(experiments/mnist_pixel_ccvfm.py reproduces the headline K=2000 run).

    python examples/mnist_quickstart.py --out-dir outputs/mnist_quickstart
    python examples/mnist_quickstart.py --iters 2000 --fid-n 5000   # quick look
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ccvfm import EMA, UNetCorrection, ccvfm_loss, fit_coreset_gmm, sample  # noqa: E402
from ccvfm.metrics import InceptionFeatures, frechet_distance  # noqa: E402


def load_mnist(root):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    return tr.data.float().div(255.0).reshape(-1, 784)


def save_grid(x, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = x[:100].clamp(0, 1).reshape(10, 10, 28, 28).permute(0, 2, 1, 3).reshape(280, 280)
    plt.figure(figsize=(5, 5.3))
    plt.imshow(imgs.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title, fontsize=10)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/mnist_quickstart")
    p.add_argument("--data", default=".data")
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--rank", type=int, default=30)
    p.add_argument("--lam", type=float, default=1.5, help="Sinkhorn bandwidth (paper: 1.5)")
    p.add_argument("--iters", type=int, default=80000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--unet-base", type=int, default=64)
    p.add_argument("--fid-n", type=int, default=50000, help="0 disables FID")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    x = load_mnist(args.data).to(dev)
    print(f"MNIST train {tuple(x.shape)} on {dev}", flush=True)

    # ---------------- Stage I ----------------
    t0 = time.time()
    gmm = fit_coreset_gmm(x, K=args.K, rank=args.rank, lam=args.lam, n_iter=100,
                          seed=args.seed, verbose=True)
    gmm.save(os.path.join(args.out_dir, f"gmm_k{args.K}_r{args.rank}.pt"))
    print(f"Stage I: {time.time() - t0:.0f}s", flush=True)

    # ---------------- Stage III training ----------------
    net = UNetCorrection((1, 28, 28), base=args.unet_base).to(dev)
    print(f"correction U-Net: {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M params")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ema = EMA(net, 0.9999)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.iters):
        idx = torch.randint(0, len(x), (args.bs,), device=dev, generator=g)
        loss = ccvfm_loss(net, gmm, x[idx], t=0.0, source="posterior", generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(net)
        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1}/{args.iters} loss {loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    ema.copy_to(net)
    net.eval()
    torch.save(net.state_dict(), os.path.join(args.out_dir, "correction_ema.pt"))

    # ---------------- sampling + FID ----------------
    inc = ref = None
    if args.fid_n > 0:
        inc = InceptionFeatures().to(dev)
        ref = inc.features(x[:args.fid_n].reshape(-1, 1, 28, 28).cpu())
    rows = []
    for L in [0, 1, 5, 10, 20]:
        sg = torch.Generator(device=dev).manual_seed(99)
        n = max(args.fid_n, 100)
        imgs = sample(net, gmm, n, L=L, generator=sg, batch_size=1000).clamp(0, 1)
        label = "Stage II (no network)" if L == 0 else f"CCVFM L={L}"
        fid = float("nan")
        if inc is not None:
            fid = frechet_distance(ref, inc.features(imgs.reshape(-1, 1, 28, 28).cpu()))
        nfe = L + 1
        rows.append([label, L, nfe, f"{fid:.3f}"])
        print(f"{label:<24} NFE={nfe:<3} FID{args.fid_n // 1000}k={fid:.3f}", flush=True)
        save_grid(imgs, os.path.join(args.out_dir, f"samples_L{L}.png"),
                  f"{label}  (NFE {nfe}, FID {fid:.2f})")
    with open(os.path.join(args.out_dir, "results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["method", "L", "nfe", f"fid_{args.fid_n}"]] + rows)


if __name__ == "__main__":
    main()
