#!/usr/bin/env python3
"""ImageNet-32 Plan T: HRF2's dual-branch UNet + our GMM posterior coupling.

Same idea as Plan S (CIFAR-10), but on ImageNet-32:
  - HRF2: v_0 ~ N(0, I) independent noise
  - Ours: v_0 ~ π̃(v|x_t, t; x_1) GMM posterior coupling

Architecture matches HRF2's published ImageNet-32 configuration (Appendix F.2):
  attention_resolutions="16,8" (vs CIFAR's just "16"), 46.2M params.

HRF2 reference ImageNet-32 FIDs (Table 7):
  NFE 5:   48.93   NFE 10:  20.29   NFE 20:  12.49
  NFE 50:  9.02    NFE 100: 7.68    NFE 500: 6.503  ← target to beat

Data source: HuggingFace `ChocolateDave/imagenet-32` (parquet, no auth).
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
# Data — ImageNet-32 via HuggingFace (ChocolateDave/imagenet-32)
# ===================================================================

def load_imagenet32_pixels(data_dir: str = ".data", n_max: int | None = None):
    """Load ImageNet-32 training set, cached as .npy for fast reload.

    First run: downloads via HF datasets, decodes 1.28M PIL images to a
    (N, 32, 32, 3) uint8 array, saves to .data/imagenet32_train.npy (~3.9 GB).
    Subsequent runs: loads the .npy directly.

    Returns:
      x_pm1   : (N, 3072) float32 in [-1, 1]
      x_hwc_01: (N, 32, 32, 3) float32 in [0, 1]
    """
    os.makedirs(data_dir, exist_ok=True)
    cached = os.path.join(data_dir, "imagenet32_train.npy")

    if os.path.exists(cached):
        print(f"  Loading cached ImageNet-32 from {cached}...", flush=True)
        t0 = time.time()
        arr = np.load(cached, mmap_mode="r")
        if n_max is not None and n_max < len(arr):
            arr = np.array(arr[:n_max])
        else:
            arr = np.array(arr)
        print(f"  Loaded {arr.shape} in {time.time()-t0:.0f}s", flush=True)
    else:
        print("  Downloading ImageNet-32 via HF (ChocolateDave/imagenet-32)...",
              flush=True)
        from datasets import load_dataset
        t0 = time.time()
        ds = load_dataset("ChocolateDave/imagenet-32", split="train")
        print(f"  HF dataset ready ({len(ds)} rows) in {time.time()-t0:.0f}s",
              flush=True)

        n = len(ds)
        arr = np.zeros((n, 32, 32, 3), dtype=np.uint8)
        t0 = time.time()
        img_key = None
        for i in range(n):
            row = ds[i]
            if img_key is None:
                for k in ("image", "img", "pixel_values"):
                    if k in row:
                        img_key = k
                        break
                if img_key is None:
                    raise KeyError(f"no image field in row keys: {list(row.keys())}")
            img = row[img_key]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                arr[i] = np.asarray(img, dtype=np.uint8)
            else:
                arr[i] = np.asarray(img, dtype=np.uint8)
            if (i + 1) % 100000 == 0:
                print(f"    decoded {i+1}/{n} in {time.time()-t0:.0f}s",
                      flush=True)
        print(f"  decoded all {n} rows in {time.time()-t0:.0f}s", flush=True)
        np.save(cached, arr)
        print(f"  cached to {cached}", flush=True)
        if n_max is not None and n_max < len(arr):
            arr = arr[:n_max]

    x_hwc_01 = arr.astype(np.float32) / 255.0
    x_chw = x_hwc_01.transpose(0, 3, 1, 2)
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
    n_iter=400000, bs=256, base_lr=1.4e-4,
    warmup_steps=5000, grad_clip=1.0, ema_decay=0.9999,
    log_every=500, ckpt_every=20000, coupling_mode="general_t",
):
    model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16,8",  # ImageNet-32 specific (CIFAR uses "16")
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Dual-branch UNet params: {n_params/1e6:.2f}M", flush=True)
    print(f"  attention_resolutions=16,8 (ImageNet-32 config)", flush=True)
    print(f"  Coupling mode: {coupling_mode}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)

    def warmup_lr(step):
        return min(step, warmup_steps) / warmup_steps

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_lr)
    ema = EMA(model, decay=ema_decay)

    x_all = torch.tensor(x_train_pm1, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    raw_ckpt = os.path.join(out_dir, "unet.pt")
    ema_ckpt = os.path.join(out_dir, "unet_ema.pt")
    loss_csv = os.path.join(out_dir, "train_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "lr", "wall_s"])

    g = torch.Generator(device=DEVICE)
    g.manual_seed(1337)
    t_start = time.time()
    model.train()

    for step in range(n_iter):
        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D_PIX, device=DEVICE, generator=g)
        v_true = x1 - x0

        t_scalar = torch.rand(1, generator=g, device=DEVICE).item() * 0.999
        t_cond = torch.full((bs,), t_scalar, device=DEVICE)
        x_t = (1.0 - t_scalar) * x0 + t_scalar * x1

        with torch.no_grad():
            if coupling_mode == "t0" or t_scalar < 1e-6:
                v_0 = coupled_sample_gpu(v_true, x0, state, generator=g)
            else:
                v_0 = coupled_sample_general_t(
                    v_true, x0, x1, x_t, t_scalar, state, generator=g)

        tau = torch.rand(bs, device=DEVICE, generator=g)
        tau_b = tau.view(-1, 1)
        v_tau = (1 - tau_b) * v_0 + tau_b * v_true
        target = v_true - v_0

        v_tau_img = v_tau.reshape(bs, C, H, W)
        x_t_img = x_t.reshape(bs, C, H, W)
        target_img = target.reshape(bs, C, H, W)

        pred = model(tau, v_tau_img, t_cond, x_t_img)
        loss = ((pred - target_img) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(model)

        if (step + 1) % log_every == 0 or step == 0:
            wall = time.time() - t_start
            cur_lr = sched.get_last_lr()[0]
            print(f"    step {step+1}/{n_iter}  loss={loss.item():.5f}  "
                  f"lr={cur_lr:.2e}  wall={wall:.0f}s", flush=True)
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, f"{loss.item():.6f}", f"{cur_lr:.6e}", f"{wall:.1f}"])

        if (step + 1) % ckpt_every == 0 or (step + 1) == n_iter:
            torch.save(model.state_dict(), raw_ckpt, pickle_protocol=5)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema


# ===================================================================
# Inference: HRF2's nested ODE integration
# ===================================================================

@torch.no_grad()
def generate_hrf2(state, model, n, flow_steps, corr_steps,
                   batch_size=128, seed=99):
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
    p.add_argument("--tag", type=str, default="t_hrf2")
    p.add_argument("--n-data", type=int, default=0,
                   help="0 = use all of ImageNet-32 train (~1.28M)")
    p.add_argument("--K", type=int, default=10000)
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--cov-nit", type=int, default=1500)
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--base-lr", type=float, default=1.4e-4)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--n-gen", type=int, default=10000)
    p.add_argument("--coupling-mode", choices=["t0", "general_t"],
                   default="general_t")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_data = 2000
        args.K = 200
        args.rank = 30
        args.cov_nit = 100
        args.train_steps = 500
        args.bs = 32
        args.n_gen = 500
        args.warmup_steps = 50

    out_dir = f"imagenet32_planT_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  cfg: K={args.K} r={args.rank} steps={args.train_steps} "
          f"bs={args.bs} lr={args.base_lr} warmup={args.warmup_steps} "
          f"coupling={args.coupling_mode}", flush=True)

    # Data
    n_max = args.n_data if args.n_data > 0 else None
    x_pm1, x_hwc_01 = load_imagenet32_pixels(n_max=n_max)
    print(f"ImageNet-32: {x_pm1.shape} (pm1), {x_hwc_01.shape} (hwc)",
          flush=True)

    # Inception real features (use first 50k training images)
    print("\nInception features (50k real pixels)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    n_real = min(50000, len(x_hwc_01))
    real_feats = feats_from_hwc(inc, x_hwc_01[:n_real])
    feat_real_5k = real_feats[:min(5000, n_real)]
    feat_real_10k = real_feats[:min(10000, n_real)]
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
    stage2_fid_5k = compute_fid(feat_real_5k, stage2_feats[:len(feat_real_5k)])
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
        coupling_mode=args.coupling_mode)

    print("\n===== INFERENCE + FID =====", flush=True)
    ema_model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16,8",
        dropout=0.1,
    ).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    results_csv = os.path.join(out_dir, f"planT_results_{args.tag}.csv")
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
        f5 = compute_fid(feat_real_5k, g_feats[:len(feat_real_5k)])
        f10 = compute_fid(feat_real_10k, g_feats[:min(10000, len(g_feats))])
        f50 = compute_fid(feat_real_50k, g_feats[:min(50000, len(g_feats))]) \
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
    print(f"  Plan T (HRF2 dual-branch + GMM coupling) on ImageNet-32 — {args.tag}")
    print(f"  K={args.K} r={args.rank} coupling={args.coupling_mode}")
    for J, L, nfe, f5, f10, f50 in results:
        print(f"  J={J} L={L} (NFE={nfe}):  FID 5k={f5:.3f}  10k={f10:.3f}  "
              f"50k={f50:.3f}")
    print(f"  [HRF2 reference ImageNet-32 (Table 7):]")
    print(f"    NFE 5=48.93  10=20.29  20=12.49  50=9.02  100=7.68  500=6.503")
    print("=" * 70)


if __name__ == "__main__":
    main()
