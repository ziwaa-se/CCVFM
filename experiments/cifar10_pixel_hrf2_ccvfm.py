#!/usr/bin/env python3
"""CIFAR-10 Plan S: HRF2's dual-branch UNet + our GMM posterior coupling.

This is exactly HRF2's published CIFAR-10 recipe, with ONE change:
  - HRF2: v_0 ~ N(0, I) independent noise
  - Ours: v_0 ~ π̃(v|x_t, t; x_1) GMM posterior coupling

Everything else (architecture, optimizer, training loop, inference) matches
HRF2 exactly so the result directly tests whether data-informed v_0
initialization helps a strong flow-matching baseline.

HRF2 reference: FID 3.706 on CIFAR-10 at 500 NFE (their Table 7).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import site
import sys
import time

import numpy as np

sys.path.insert(0, site.getusersitepackages())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import Inception_V3_Weights, inception_v3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# HRF2's dual-branch UNet
from hrf_models.unet_2_unet import UNetModelWrapper as DualBranchUNet

# Our GMM machinery (shape-agnostic — works in pixel space)
from mnist_pixel_gmm_core import (
    GMMState,
    LowRankGMM,
    compute_fid,
    coupled_sample_gpu,
    ems_coreset_gpu,
    learn_lowrank_cov_fast,
    sample_velocity_gpu,
    save_gmm,
)
import mnist_pixel_gmm_core as _mplf

# General-t samplers (shape-agnostic)
import cifar10_sdvae_latent_ccvfm as pg
from cifar10_sdvae_latent_ccvfm import (
    sample_velocity_general_t,
    coupled_sample_general_t,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mplf.DEVICE = DEVICE
pg.DEVICE = DEVICE

C, H, W = 3, 32, 32
D_PIX = C * H * W  # 3072


# ===================================================================
# Inception FID
# ===================================================================

class IncFeatRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.m.eval()
        self.m.fc = nn.Identity()

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, 299, mode="bilinear", align_corners=False)
        mu = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.m((x - mu) / sd)


@torch.no_grad()
def feats_from_hwc(inc, imgs_hwc, bs=64):
    chw = imgs_hwc.transpose(0, 3, 1, 2)
    fs = []
    for i in range(0, len(chw), bs):
        batch = torch.tensor(chw[i:i+bs], dtype=torch.float32, device=DEVICE)
        fs.append(inc(batch).cpu().numpy())
    return np.concatenate(fs)


# ===================================================================
# Data
# ===================================================================

def load_cifar10_pixels(data_dir: str = ".data"):
    os.makedirs(data_dir, exist_ok=True)
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True)
    x_hwc_01 = np.array(train_ds.data, dtype=np.float32) / 255.0
    x_chw = x_hwc_01.transpose(0, 3, 1, 2)
    # Normalize to [-1, 1] for training
    x_pm1 = (x_chw * 2.0 - 1.0).reshape(len(x_hwc_01), -1)
    return x_pm1, x_hwc_01


def latents_to_hwc_01(latents_flat: np.ndarray) -> np.ndarray:
    """(n, 3072) [-1,1] → (n, 32, 32, 3) [0,1]."""
    x_chw = latents_flat.reshape(-1, 3, 32, 32)
    x_chw = (x_chw + 1.0) / 2.0
    return np.clip(x_chw, 0, 1).transpose(0, 2, 3, 1)


# ===================================================================
# EMA
# ===================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def save(self, path):
        torch.save(self.shadow, path, pickle_protocol=5)


# ===================================================================
# Stage III training: HRF2 recipe + GMM coupling
# ===================================================================

def train_stage3(
    x_train_pm1, state, out_dir,
    n_iter=400000, bs=128, base_lr=2e-4,
    warmup_steps=5000, grad_clip=1.0, ema_decay=0.9999,
    log_every=500, ckpt_every=10000, milestone_every=100000,
    coupling_mode="general_t",
    unet_base=128, unet_attention="16", unet_res_blocks=2,
    start_step=0,
):
    """coupling_mode: 'general_t' uses multi-t posterior coupling,
       't0' uses t=0 coupling (simpler, Plan I recipe).
    start_step > 0: resume training.  The caller must have loaded
       model.state_dict() from unet.pt and ema.shadow from unet_ema.pt
       BEFORE calling this function; this routine will fast-forward the
       LR scheduler and open train_loss.csv in append mode.
    """
    model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=unet_res_blocks,
        num_channels=unet_base,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=unet_attention,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Dual-branch UNet params: {n_params/1e6:.2f}M", flush=True)
    print(f"  Coupling mode: {coupling_mode}", flush=True)
    if start_step > 0:
        print(f"  RESUMING from step {start_step}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    # HRF2's warmup schedule: linear warmup, then constant
    def warmup_lr(step):
        return min(step, warmup_steps) / warmup_steps

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_lr)
    ema = EMA(model, decay=ema_decay)

    raw_ckpt = os.path.join(out_dir, "unet.pt")
    ema_ckpt = os.path.join(out_dir, "unet_ema.pt")
    loss_csv = os.path.join(out_dir, "train_loss.csv")

    # Resume logic: load checkpoints, fast-forward scheduler, open CSV in append mode.
    if start_step > 0:
        print(f"  loading raw weights from {raw_ckpt} ...", flush=True)
        model.load_state_dict(
            torch.load(raw_ckpt, weights_only=False, map_location=DEVICE),
            strict=True)
        print(f"  loading EMA weights from {ema_ckpt} ...", flush=True)
        ema_shadow = torch.load(ema_ckpt, weights_only=False,
                                 map_location=DEVICE)
        for n, p in ema_shadow.items():
            ema.shadow[n].copy_(p)
        # Fast-forward scheduler so LR matches the warmup/constant schedule
        # at the resume step.
        for _ in range(start_step):
            sched.step()
        csv_mode = "a"
    else:
        csv_mode = "w"

    with open(loss_csv, csv_mode, newline="") as f:
        w = csv.writer(f)
        if csv_mode == "w":
            w.writerow(["step", "loss", "lr", "wall_s"])
        else:
            w.writerow(["# resumed from step", start_step, "", ""])

    x_all = torch.tensor(x_train_pm1, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    g = torch.Generator(device=DEVICE)
    # Different RNG seed on resume so we don't replay the same minibatches.
    g.manual_seed(1337 + start_step)
    t_start = time.time()
    model.train()

    # Loop index: iterations remaining.  Global "step" = start_step + i.
    for i in range(n_iter - start_step):
        step = start_step + i
        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D_PIX, device=DEVICE, generator=g)
        v_true = x1 - x0

        # Sample flow-matching time t ~ U[0,1] (HRF2 recipe)
        t_scalar = torch.rand(1, generator=g, device=DEVICE).item() * 0.999
        t_cond = torch.full((bs,), t_scalar, device=DEVICE)
        x_t = (1.0 - t_scalar) * x0 + t_scalar * x1

        # Sample v_0 using our coupling (key difference from HRF2)
        with torch.no_grad():
            if coupling_mode == "t0" or t_scalar < 1e-6:
                v_0 = coupled_sample_gpu(v_true, x0, state, generator=g)
            else:
                v_0 = coupled_sample_general_t(
                    v_true, x0, x1, x_t, t_scalar, state, generator=g)

        # Sample τ ~ U[0,1] and construct v_τ
        tau = torch.rand(bs, device=DEVICE, generator=g)
        tau_b = tau.view(-1, 1)
        v_tau = (1 - tau_b) * v_0 + tau_b * v_true
        target = v_true - v_0  # acceleration target

        # Reshape to image format for UNet
        v_tau_img = v_tau.reshape(bs, C, H, W)
        x_t_img = x_t.reshape(bs, C, H, W)
        target_img = target.reshape(bs, C, H, W)

        # HRF2 UNet signature: forward(t_v, v, t, xt)
        pred = model(tau, v_tau_img, t_cond, x_t_img)
        loss = ((pred - target_img) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(model)

        if (step + 1) % log_every == 0 or i == 0:
            wall = time.time() - t_start
            cur_lr = sched.get_last_lr()[0]
            print(f"    step {step+1}/{n_iter}  loss={loss.item():.5f}  "
                  f"lr={cur_lr:.2e}  wall={wall:.0f}s", flush=True)
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, f"{loss.item():.6f}", f"{cur_lr:.6e}", f"{wall:.1f}"])

        if (step + 1) % ckpt_every == 0 or (step + 1) == n_iter:
            tmp_raw = raw_ckpt + ".tmp"
            tmp_ema = ema_ckpt + ".tmp"
            torch.save(model.state_dict(), tmp_raw, pickle_protocol=5)
            ema.save(tmp_ema)
            os.replace(tmp_raw, raw_ckpt)
            os.replace(tmp_ema, ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)
            if (step + 1) % milestone_every == 0:
                tag = f"step{step+1}"
                ms_raw = os.path.join(out_dir, f"unet_{tag}.pt")
                ms_ema = os.path.join(out_dir, f"unet_ema_{tag}.pt")
                torch.save(model.state_dict(), ms_raw, pickle_protocol=5)
                ema.save(ms_ema)
                print(f"    milestone checkpoint saved: {tag}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema


# ===================================================================
# Inference: HRF2's nested ODE integration
# ===================================================================

@torch.no_grad()
def generate_hrf2(state, model, n, flow_steps, corr_steps,
                   batch_size=128, seed=99):
    """HRF2-style nested integration:
      for j = 0..J-1:
        t_j = j/J
        v ~ π̃(v|x, t_j)            ← GMM marginal at current x
        for i = 0..L-1:
          τ_i = i/L
          v ← v + (1/L) · model(τ_i, v, t_j, x)
        x ← x + (1/J) · v
    """
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    model.eval()
    out = np.zeros((n, 32, 32, 3), dtype=np.float32)

    dt = 1.0 / flow_steps
    ds = 1.0 / corr_steps

    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x = torch.randn(nb, D_PIX, device=DEVICE, generator=g)

        for j in range(flow_steps):
            t_val = j * dt
            t_cond = torch.full((nb,), t_val, device=DEVICE)

            if t_val < 1e-6:
                v = sample_velocity_gpu(x, state, generator=g)
            else:
                v = sample_velocity_general_t(x, t_val, state, generator=g)

            v_img = v.reshape(nb, C, H, W)
            x_img = x.reshape(nb, C, H, W)

            for i in range(corr_steps):
                tau = torch.full((nb,), i * ds, device=DEVICE)
                v_img = v_img + ds * model(tau, v_img, t_cond, x_img)

            v_final = v_img.reshape(nb, D_PIX)
            x = x + dt * v_final

        x_hwc = latents_to_hwc_01(x.cpu().numpy())
        out[start:start + nb] = x_hwc

    return out


def save_grid(imgs_hwc, path, title=""):
    n = min(100, len(imgs_hwc))
    nr = 10
    fig, ax = plt.subplots(nr, nr, figsize=(10, 10))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    for i in range(nr * nr):
        a = ax[i // nr, i % nr]
        if i < n:
            a.imshow(np.clip(imgs_hwc[i], 0, 1))
        a.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", type=str, default="s_hrf2_coupling")
    p.add_argument("--K", type=int, default=5000)
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--cov-nit", type=int, default=1500)
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--n-gen", type=int, default=10000)
    p.add_argument("--unet-base", type=int, default=128,
                   help="UNet base channels (must be multiple of 128 for GroupNorm)")
    p.add_argument("--unet-attention", type=str, default="16",
                   help="attention resolutions, e.g. '16' or '16,8'")
    p.add_argument("--unet-res-blocks", type=int, default=2)
    p.add_argument("--coupling-mode", choices=["t0", "general_t"],
                   default="general_t")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.K = 100
        args.rank = 30
        args.cov_nit = 100
        args.train_steps = 500
        args.bs = 32
        args.n_gen = 500
        args.warmup_steps = 50

    out_dir = f"cifar10_planS_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  cfg: K={args.K} r={args.rank} steps={args.train_steps} "
          f"bs={args.bs} warmup={args.warmup_steps} "
          f"coupling={args.coupling_mode}", flush=True)

    # Data
    x_pm1, x_hwc_01 = load_cifar10_pixels()
    print(f"CIFAR: {x_pm1.shape} (pm1), {x_hwc_01.shape} (hwc)", flush=True)

    # Inception real features
    print("\nInception features (50k real pixels)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    real_feats = feats_from_hwc(inc, x_hwc_01[:50000])
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats[:10000]
    feat_real_50k = real_feats
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # Stage I
    print(f"\n===== STAGE I: EMS K={args.K} + rank-{args.rank} cov =====",
          flush=True)
    rg = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(
        x_pm1, args.K, lam=0.5, nit=100, rg=rg)
    print(f"  EMS done in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    L_all, sigma2 = learn_lowrank_cov_fast(
        x_pm1, centers, resp, rank=args.rank, nit=args.cov_nit,
        data_batch=1024, s2_floor=0.001)
    print(f"  cov done in {time.time()-t1:.0f}s, sigma2={sigma2:.6f}",
          flush=True)
    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(out_dir, f"gmm_k{args.K}.pt"))
    gmm_state = GMMState(gmm)

    # Stage II reference (1-step generation with GMM only)
    print("\n===== STAGE II: 1-step GMM generation =====", flush=True)
    t0 = time.time()
    g_gen = torch.Generator(device=DEVICE)
    g_gen.manual_seed(99)
    n_s2 = min(args.n_gen, 5000)
    stage2_imgs = np.zeros((n_s2, 32, 32, 3), dtype=np.float32)
    for start in range(0, n_s2, 256):
        nb = min(256, n_s2 - start)
        x0 = torch.randn(nb, D_PIX, device=DEVICE, generator=g_gen)
        v = sample_velocity_gpu(x0, gmm_state, generator=g_gen)
        x1_flat = (x0 + v).cpu().numpy()
        stage2_imgs[start:start+nb] = latents_to_hwc_01(x1_flat)
    print(f"  Stage II {n_s2} in {time.time()-t0:.0f}s", flush=True)
    stage2_feats = feats_from_hwc(inc, stage2_imgs)
    stage2_fid_5k = compute_fid(feat_real_5k, stage2_feats[:5000])
    print(f"  Stage II FID_5k = {stage2_fid_5k:.3f}", flush=True)
    save_grid(stage2_imgs[:100],
              os.path.join(out_dir, f"stage2_{args.tag}.png"),
              f"Stage II FID_5k={stage2_fid_5k:.1f}")

    # Stage III training
    print(f"\n===== STAGE III: HRF2 dual-branch UNet training "
          f"({args.train_steps} steps) =====", flush=True)
    model, ema = train_stage3(
        x_pm1, gmm_state, out_dir,
        n_iter=args.train_steps, bs=args.bs,
        base_lr=args.base_lr,
        warmup_steps=args.warmup_steps,
        coupling_mode=args.coupling_mode,
        unet_base=args.unet_base,
        unet_attention=args.unet_attention,
        unet_res_blocks=args.unet_res_blocks)

    print("\n===== INFERENCE + FID =====", flush=True)
    ema_model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=args.unet_res_blocks,
        num_channels=args.unet_base,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=args.unet_attention,
        dropout=0.1,
    ).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    results_csv = os.path.join(out_dir, f"planS_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["J", "L", "nfe", "fid_5k", "fid_10k", "fid_50k"])
        csv.writer(f).writerow(
            ["stage2", "1step", 1, f"{stage2_fid_5k:.4f}", "NA", "NA"])

    # HRF2 paper's recommended sampling configs (from Table 7)
    configs = [(1, 10), (1, 20), (1, 50), (2, 50), (2, 250)] \
        if not args.smoke else [(1, 5), (2, 2)]
    results = []
    for J, L in configs:
        nfe = J * L
        t0 = time.time()
        imgs = generate_hrf2(gmm_state, ema_model, n=args.n_gen,
                              flow_steps=J, corr_steps=L,
                              batch_size=128, seed=99)
        print(f"  [EMA] J={J} L={L} (NFE={nfe}) gen {args.n_gen} in "
              f"{time.time()-t0:.0f}s", flush=True)
        g_feats = feats_from_hwc(inc, imgs)
        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        f50 = compute_fid(feat_real_50k, g_feats[:50000]) \
            if args.n_gen >= 50000 else float("nan")
        print(f"    FID 5k={f5:.3f} 10k={f10:.3f} 50k={f50:.3f}", flush=True)
        results.append((J, L, nfe, f5, f10, f50))
        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [J, L, nfe, f"{f5:.4f}", f"{f10:.4f}",
                 (f"{f50:.4f}" if np.isfinite(f50) else "NA")])
        save_grid(imgs[:100],
                  os.path.join(out_dir, f"stage3_J{J}_L{L}_{args.tag}.png"),
                  f"J={J} L={L} FID10k={f10:.1f}")

    print("\n" + "=" * 70)
    print(f"  Plan S (HRF2 dual-branch + GMM coupling) — {args.tag}")
    print(f"  K={args.K} r={args.rank} coupling={args.coupling_mode}")
    for J, L, nfe, f5, f10, f50 in results:
        print(f"  J={J} L={L} (NFE={nfe}):  FID 5k={f5:.3f}  10k={f10:.3f}  "
              f"50k={f50:.3f}")
    print(f"  [HRF2 reference: FID 3.706 at NFE=500]")
    print(f"  [Plan I reference (SD-VAE latent): FID_50k 14.38 at NFE=501]")
    print("=" * 70)


if __name__ == "__main__":
    main()
