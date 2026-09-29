#!/usr/bin/env python3
"""CCVFM on 2D toy targets, end to end, in a few minutes on a CPU.

For each target it
  1. fits a K-atom Sinkhorn coreset + low-rank GMM (Stage I),
  2. samples the closed-form conditional velocity law with no network (Stage II),
  3. trains a small correction MLP from that surrogate source (Stage III, CCVFM),
  4. trains the *same* MLP with the same budget from an N(0, I) source (the
     hierarchical-rectified-flow / HRF2 baseline),
and plots samples plus sliced-W2 versus the number of correction steps L.

    python examples/toy2d.py --out assets/toy2d_pipeline.png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ccvfm import MLPCorrection, ccvfm_loss, fit_coreset_gmm, sample  # noqa: E402
from ccvfm.metrics import sliced_wasserstein  # noqa: E402


# ----------------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------------
def make_moons(n, g):
    t = torch.rand(n, generator=g) * math.pi
    upper = torch.rand(n, generator=g) < 0.5
    x = torch.where(upper, torch.cos(t), 1 - torch.cos(t))
    y = torch.where(upper, torch.sin(t), 0.5 - torch.sin(t))
    pts = torch.stack([x - 0.5, y - 0.25], 1) * 2.0
    return pts + 0.08 * torch.randn(n, 2, generator=g)


def make_spiral(n, g):
    t = torch.sqrt(torch.rand(n, generator=g)) * 3.0 * math.pi
    r = t / (3.0 * math.pi) * 3.0
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return pts + 0.07 * torch.randn(n, 2, generator=g)


def make_checkerboard(n, g):
    x1 = torch.rand(n, generator=g) * 4 - 2
    x2 = torch.rand(n, generator=g) - torch.randint(0, 2, (n,), generator=g) * 2.0
    x2 = x2 + (torch.floor(x1) % 2)
    return torch.stack([x1, x2], 1) * 1.4


TARGETS = {"two moons": make_moons, "spiral": make_spiral, "checkerboard": make_checkerboard}


# ----------------------------------------------------------------------------
def train(gmm, x, source, steps, bs, lr, seed, device):
    torch.manual_seed(seed)
    net = MLPCorrection(2, hidden=256, depth=4).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (bs,), device=device, generator=g)
        loss = ccvfm_loss(net, gmm, x[idx], source=source, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/toy2d_pipeline.png")
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ls = [1, 2, 4, 8, 16]
    L_show = 2
    rows = []
    fig, axes = plt.subplots(len(TARGETS), 5, figsize=(19, 3.9 * len(TARGETS)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for r, (name, fn) in enumerate(TARGETS.items()):
        t0 = time.time()
        g = torch.Generator().manual_seed(args.seed)
        x = fn(args.n, g).float().to(device)
        x_ref = fn(5000, torch.Generator().manual_seed(args.seed + 1)).float().to(device)

        gmm = fit_coreset_gmm(x, K=args.K, rank=args.rank, seed=args.seed)       # Stage I
        sg = torch.Generator(device=device).manual_seed(123)
        stage2 = sample(None, gmm, 5000, generator=sg)                            # Stage II
        sw_stage2 = sliced_wasserstein(stage2, x_ref)

        nets = {src: train(gmm, x, src, args.steps, args.bs, args.lr, args.seed, device)
                for src in ("posterior", "gaussian")}                             # Stage III
        curves, shown = {}, {}
        for src, net in nets.items():
            smp_src = "gaussian" if src == "gaussian" else "surrogate"
            curves[src] = []
            for L in Ls:
                sg = torch.Generator(device=device).manual_seed(123)
                smp = sample(net, gmm, 5000, L=L, source=smp_src, generator=sg)
                sw = sliced_wasserstein(smp, x_ref)
                curves[src].append(sw)
                rows.append([name, "CCVFM" if src == "posterior" else "Gaussian source (HRF2)",
                             L, f"{sw:.4f}"])
                if L == L_show:
                    shown[src] = smp
        rows.append([name, "Stage II (no network)", 0, f"{sw_stage2:.4f}"])
        print(f"[{name}] Stage II SW2={sw_stage2:.3f} | CCVFM "
              + " ".join(f"L{L}:{s:.3f}" for L, s in zip(Ls, curves["posterior"]))
              + " | Gaussian " + " ".join(f"L{L}:{s:.3f}" for L, s in zip(Ls, curves["gaussian"]))
              + f"  ({time.time() - t0:.0f}s)", flush=True)

        # ---------------- plotting ----------------
        xr = x_ref.cpu().numpy()
        lim = np.abs(xr).max() * 1.15
        ax = axes[r, 0]
        ax.scatter(xr[:, 0], xr[:, 1], s=1.5, c="#9aa5b1", alpha=0.5)
        mu, w = gmm.means.cpu().numpy(), gmm.weights.cpu().numpy()
        ax.scatter(mu[:, 0], mu[:, 1], s=w / w.max() * 120, c="#111", marker="o",
                   edgecolors="white", linewidths=0.6)
        ax.set_title(f"{name}\ntarget + K={args.K} coreset atoms", fontsize=11)
        panels = [(stage2, f"Stage II: closed form, no network\nSW$_2$ = {sw_stage2:.3f}", "#1baf7a"),
                  (shown["posterior"], f"CCVFM, L = {L_show} steps\nSW$_2$ = "
                   f"{curves['posterior'][Ls.index(L_show)]:.3f}", "#2a78d6"),
                  (shown["gaussian"], f"Gaussian source (HRF2), L = {L_show}\nSW$_2$ = "
                   f"{curves['gaussian'][Ls.index(L_show)]:.3f}", "#eb6834")]
        for c, (pts, title, col) in enumerate(panels, start=1):
            pts = pts.cpu().numpy()
            axes[r, c].scatter(pts[:, 0], pts[:, 1], s=1.5, c=col, alpha=0.5)
            axes[r, c].set_title(title, fontsize=11)
        for c in range(4):
            axes[r, c].set_xlim(-lim, lim)
            axes[r, c].set_ylim(-lim, lim)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_aspect("equal")
        ax = axes[r, 4]
        ax.plot(Ls, curves["posterior"], "-o", c="#2a78d6", label="CCVFM (surrogate source)")
        ax.plot(Ls, curves["gaussian"], "-o", c="#eb6834", label="Gaussian source (HRF2)")
        ax.axhline(sw_stage2, ls="--", c="#1baf7a", label="Stage II (0 network evals)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(L) for L in Ls])
        ax.set_xlabel("correction steps L (network evaluations)")
        ax.set_ylabel("sliced W$_2$ to target  (lower is better)")
        ax.grid(alpha=0.3, which="both")
        if r == 0:
            ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([["target", "method", "L", "sliced_w2"]] + rows)
    print(f"saved {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
