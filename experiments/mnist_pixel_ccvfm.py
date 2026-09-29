#!/usr/bin/env python3
"""MNIST 3-stage CoresetFM — Plan C.

Push MNIST pixel FID_50k under 1.0 by direct scaling of the Plan B winners:
  - K = 2000 (up from 1000)
  - rank = 50 (up from 30)
  - Stage III training iters = 200k (up from 80k)
  - Correction U-Net base channels = 128 (up from 64)
  - EMA decay = 0.9999 (kept; was neutral but harmless)
  - Inference ablation adds L = 50 (free; best-reachable row)

Use --smoke for a tiny sanity-check pass.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist_pixel_gmm_core import (  # noqa: E402
    D,
    DEVICE,
    GMMState,
    IncFeat,
    LowRankGMM,
    compute_fid,
    coupled_sample_gpu,
    ems_coreset_gpu,
    generate_stage2_fast,
    get_feats,
    learn_lowrank_cov_fast,
    load_mnist,
    sample_velocity_gpu,
    save_gmm,
    save_grid,
    to_flat,
)

OUT_DIR = "mnist_pixel_ccvfm_outputs"


# ===================================================================
# Parametrized correction U-Net (base channels configurable)
# ===================================================================


class SinEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        h = self.d // 2
        f = torch.exp(
            -math.log(10000) * torch.arange(h, device=t.device).float() / h
        )
        e = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)
        return self.net(e)


class RB(nn.Module):
    def __init__(self, ci, co, td=128):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.tp = nn.Linear(td, co)
        self.n1 = nn.GroupNorm(min(8, co), co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.tp(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h))) + self.sk(x)


class CorrectionUNetC(nn.Module):
    """Same topology as Plan B's CorrectionUNet but with configurable
    base channel width. base=128 gives ~24M params (vs ~6M at base=64)."""

    def __init__(self, base=128, td=128):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.tau_emb = SinEmb(td)
        self.t_emb = SinEmb(td)
        self.comb = nn.Linear(2 * td, td)
        self.e1 = RB(2, c1, td)
        self.d1 = nn.Conv2d(c1, c1, 3, 2, 1)
        self.e2 = RB(c1, c2, td)
        self.d2 = nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = RB(c2, c3, td)
        self.u2 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.de2 = RB(c3, c2, td)
        self.u1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.de1 = RB(c2, c1, td)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, v_tau_img, x0_img, tau, t):
        te = self.comb(torch.cat([self.tau_emb(tau), self.t_emb(t)], 1))
        inp = torch.cat([v_tau_img, x0_img], 1)
        h1 = self.e1(inp, te)
        h2 = self.e2(self.d1(h1), te)
        h = self.mid(self.d2(h2), te)
        h = self.de2(torch.cat([self.u2(h), h2], 1), te)
        h = self.de1(torch.cat([self.u1(h), h1], 1), te)
        return self.out(h)


# ===================================================================
# EMA with atomic save
# ===================================================================


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

    def save(self, path: str):
        tmp = path + ".tmp"
        torch.save(self.shadow, tmp)
        os.replace(tmp, path)


# ===================================================================
# Stage III training — parameterized
# ===================================================================


def train_correction_planC(
    x_train_flat,
    state: GMMState,
    out_dir: str,
    n_iter: int = 200000,
    bs: int = 64,
    lr: float = 2e-4,
    ema_decay: float = 0.9999,
    unet_base: int = 128,
    checkpoint_every: int = 20000,
    train_seed: int = 0,
    log_every: int = 2000,
):
    model = CorrectionUNetC(base=unet_base).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet base={unet_base} params={n_params / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)

    x_all = torch.tensor(x_train_flat, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(train_seed)

    raw_ckpt = os.path.join(out_dir, "corr_unet.pt")
    ema_ckpt = os.path.join(out_dir, "corr_unet_ema.pt")

    t0 = time.time()
    model.train()
    for step in range(n_iter):
        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D, device=DEVICE, generator=g)
        v_true = x1 - x0

        with torch.no_grad():
            v0 = coupled_sample_gpu(v_true, x0, state, generator=g)

        v_true_img = v_true.reshape(bs, 1, 28, 28)
        v0_img = v0.reshape(bs, 1, 28, 28)
        x0_img = x0.reshape(bs, 1, 28, 28)

        tau = torch.rand(bs, device=DEVICE, generator=g)
        tau_b = tau.view(-1, 1, 1, 1)
        v_tau = (1 - tau_b) * v0_img + tau_b * v_true_img
        target = v_true_img - v0_img

        t_cond = torch.zeros(bs, device=DEVICE)
        pred = model(v_tau, x0_img, tau, t_cond)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)

        if (step + 1) % log_every == 0:
            wall = time.time() - t0
            print(
                f"    step {step+1}/{n_iter}  loss={loss.item():.6f}  "
                f"wall={wall:.0f}s",
                flush=True,
            )

        if (step + 1) % checkpoint_every == 0 or (step + 1) == n_iter:
            tmp = raw_ckpt + ".tmp"
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, raw_ckpt)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema, time.time() - t0


# ===================================================================
# Generation — mirrors mnist_pixel_gmm_core.generate_corrected_fast
# but uses our parametrized UNet
# ===================================================================


@torch.no_grad()
def generate_corrected_planC(
    state: GMMState, model: nn.Module, n: int, corr_steps: int = 10,
    batch_size: int = 512, seed: int = 99,
):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    model.eval()
    out = []
    h = 1.0 / corr_steps
    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D, device=DEVICE, generator=g)
        v = sample_velocity_gpu(x0, state, generator=g)
        v_img = v.reshape(nb, 1, 28, 28)
        x0_img = x0.reshape(nb, 1, 28, 28)
        t_cond = torch.zeros(nb, device=DEVICE)
        for i in range(corr_steps):
            tau = torch.full((nb,), i / corr_steps, device=DEVICE)
            dv = model(v_img, x0_img, tau, t_cond)
            v_img = v_img + h * dv
        x1 = (x0_img + v_img).clamp(0, 1).reshape(nb, 28, 28)
        out.append(x1.cpu().numpy())
    return np.concatenate(out)


# ===================================================================
# Main
# ===================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="tiny sanity-check run (~3 min on 1 GH200)")
    p.add_argument("--K", type=int, default=None,
                   help="override coreset size (default 2000 real, 100 smoke)")
    p.add_argument("--iters", type=int, default=None,
                   help="override Stage III iters (default 200000 real, 400 smoke)")
    p.add_argument("--rank", type=int, default=None,
                   help="override GMM low-rank (default 50 real, 30 smoke)")
    p.add_argument("--unet-base", type=int, default=None,
                   help="override U-Net base channels (default 128 real, 64 smoke)")
    p.add_argument("--train-seed", type=int, default=0,
                   help="Stage III training RNG seed")
    p.add_argument("--out-dir", type=str, default=None,
                   help="override output directory (default mnist_pixel_ccvfm_outputs)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        cfg = dict(
            K=100, rank=30, ems_nit=30, cov_nit=100,
            train_iters=400, ema_decay=0.99,
            unet_base=64,
            N_GEN=2000, N_REAL_TRAIN=2000,
            feat_5k=1000, feat_10k=1500, feat_50k=2000,
            checkpoint_every=200,
        )
    else:
        cfg = dict(
            K=2000, rank=50, ems_nit=100, cov_nit=800,
            train_iters=200000, ema_decay=0.9999,
            unet_base=128,
            N_GEN=50000, N_REAL_TRAIN=50000,
            feat_5k=5000, feat_10k=10000, feat_50k=50000,
            checkpoint_every=20000,
        )

    # CLI overrides
    for key, val in [
        ("K", args.K), ("train_iters", args.iters),
        ("rank", args.rank), ("unet_base", args.unet_base),
    ]:
        if val is not None:
            cfg[key] = val

    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    rg = np.random.default_rng(42)
    torch.manual_seed(42)

    x_train_2d, _, x_test_2d, _ = load_mnist(
        os.path.join(os.getcwd(), ".data", "mnist.npz")
    )
    x_train = to_flat(x_train_2d)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Mode: {'SMOKE' if args.smoke else 'REAL'}  cfg={cfg}", flush=True)
    print(f"Train: {x_train.shape}  Test: {x_test_2d.shape}", flush=True)

    # Inception + real features
    print("\nInception V3 features (pool sizes "
          f"{cfg['feat_5k']}/{cfg['feat_10k']}/{cfg['feat_50k']})...",
          flush=True)
    t = time.time()
    inc = IncFeat().to(DEVICE)
    feat_real_5k = get_feats(inc, x_test_2d[:cfg["feat_5k"]])
    feat_real_10k = get_feats(inc, x_test_2d[:cfg["feat_10k"]])
    feat_real_50k = get_feats(inc, x_train_2d[:cfg["feat_50k"]])
    print(f"  real features in {time.time()-t:.0f}s", flush=True)

    def fid_three(imgs_pool):
        f_pool = get_feats(inc, imgs_pool)
        return (
            compute_fid(feat_real_5k, f_pool[:cfg["feat_5k"]]),
            compute_fid(feat_real_10k, f_pool[:cfg["feat_10k"]]),
            compute_fid(feat_real_50k, f_pool[:cfg["feat_50k"]]),
        )

    results = []

    # ==== STAGE I ====
    print("\n" + "=" * 72, flush=True)
    print(f"STAGE I  K={cfg['K']}  rank={cfg['rank']}", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(
        x_train, cfg["K"], lam=1.5, nit=cfg["ems_nit"], rg=rg
    )
    print(f"  EMS in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    L_all, sigma2 = learn_lowrank_cov_fast(
        x_train, centers, resp, rank=cfg["rank"], nit=cfg["cov_nit"]
    )
    print(f"  Low-rank cov in {time.time()-t1:.0f}s  sigma2={sigma2:.6f}",
          flush=True)

    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(OUT_DIR, f"gmm_k{cfg['K']}.pt"))
    state = GMMState(gmm)
    print(f"  Stage I total {time.time()-t0:.0f}s", flush=True)

    # ==== STAGE II ====
    print("\n" + "=" * 72, flush=True)
    print(f"STAGE II  (1-step, {cfg['N_GEN']} samples)", flush=True)
    print("=" * 72, flush=True)
    t0 = time.time()
    imgs_s2 = generate_stage2_fast(state, cfg["N_GEN"], batch_size=4096, seed=99)
    print(f"  generated in {time.time()-t0:.0f}s", flush=True)
    f5, f10, f50 = fid_three(imgs_s2)
    print(f"  Stage II  FID 5k={f5:.2f}  10k={f10:.2f}  50k={f50:.2f}",
          flush=True)
    results.append(("StageII_1step", 1, f5, f10, f50))
    save_grid(
        imgs_s2[:100], os.path.join(OUT_DIR, "stage2_planC.png"),
        f"Stage II 1-step  (FID50k={f50:.2f})",
    )

    # ==== STAGE III ====
    print("\n" + "=" * 72, flush=True)
    print(f"STAGE III  iters={cfg['train_iters']}  "
          f"unet_base={cfg['unet_base']}  ema_decay={cfg['ema_decay']}",
          flush=True)
    print("=" * 72, flush=True)
    raw_model, ema, corr_time = train_correction_planC(
        x_train, state, out_dir=OUT_DIR,
        n_iter=cfg["train_iters"], bs=64, lr=2e-4,
        ema_decay=cfg["ema_decay"], unet_base=cfg["unet_base"],
        checkpoint_every=cfg["checkpoint_every"],
        train_seed=args.train_seed,
    )
    print(f"  Stage III training {corr_time:.0f}s", flush=True)

    ema_model = CorrectionUNetC(base=cfg["unet_base"]).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    # Inference sweep including L=50 (new for Plan C)
    for steps in [1, 5, 10, 20, 50]:
        t0 = time.time()
        imgs = generate_corrected_planC(
            state, ema_model, cfg["N_GEN"],
            corr_steps=steps, batch_size=512, seed=99,
        )
        print(f"  [EMA] L={steps:<3d} gen {cfg['N_GEN']} "
              f"in {time.time()-t0:.0f}s", flush=True)
        f5, f10, f50 = fid_three(imgs)
        print(f"  Stage III EMA L={steps:<3d}  "
              f"FID 5k={f5:.2f}  10k={f10:.2f}  50k={f50:.2f}",
              flush=True)
        results.append((f"StageIII_EMA_{steps}step", steps + 1, f5, f10, f50))
        save_grid(
            imgs[:100],
            os.path.join(OUT_DIR, f"stage3_ema_{steps}step_planC.png"),
            f"EMA {steps}-step  (FID50k={f50:.2f})",
        )

    # Raw @ L=10 for completeness (matches Plan B row)
    t0 = time.time()
    imgs_raw10 = generate_corrected_planC(
        state, raw_model, cfg["N_GEN"],
        corr_steps=10, batch_size=512, seed=99,
    )
    print(f"  [RAW] L=10  gen {cfg['N_GEN']} in {time.time()-t0:.0f}s",
          flush=True)
    f5, f10, f50 = fid_three(imgs_raw10)
    print(f"  Stage III RAW L=10  "
          f"FID 5k={f5:.2f}  10k={f10:.2f}  50k={f50:.2f}", flush=True)
    results.append(("StageIII_raw_10step", 11, f5, f10, f50))
    save_grid(
        imgs_raw10[:100],
        os.path.join(OUT_DIR, "stage3_raw_10step_planC.png"),
        f"RAW 10-step  (FID50k={f50:.2f})",
    )

    # ==== CSV ====
    csv_path = os.path.join(OUT_DIR, "planC_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nfe", "fid_5k", "fid_10k", "fid_50k"])
        for row in results:
            w.writerow(row)
    print(f"\nWrote {csv_path}", flush=True)

    # ==== SUMMARY ====
    print("\n" + "=" * 76, flush=True)
    print(f"{'Method':<24}{'NFE':>6}{'FID 5k':>12}{'FID 10k':>12}{'FID 50k':>14}",
          flush=True)
    print("-" * 76, flush=True)
    for m, n, f5, f10, f50 in results:
        print(f"{m:<24}{n:>6d}{f5:>12.3f}{f10:>12.3f}{f50:>14.3f}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
