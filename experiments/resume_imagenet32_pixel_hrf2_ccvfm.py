#!/usr/bin/env python3
"""Resume Plan T (HRF2 on ImageNet-32) Stage III training from checkpoint.

Loads unet.pt + unet_ema.pt from step 300k and trains to 400k with the
same hyperparameters as the original job. Skips Stage I/II
because gmm_k5000.pt is unchanged.

Adam optimizer momentum was not checkpointed, so momentum restarts from
zero — this is fine at this point in training (past warmup, well into
the flat-LR regime).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from imagenet32_pixel_hrf2_ccvfm import (
    D_PIX,
    C, H, W,
    DEVICE,
    DualBranchUNet,
    EMA,
    IncFeatRGB,
    compute_fid,
    feats_from_hwc,
    generate_hrf2,
    load_imagenet32_pixels,
    save_grid,
)
from mnist_pixel_gmm_core import GMMState, LowRankGMM
from cifar10_sdvae_latent_ccvfm import (
    coupled_sample_general_t,
    sample_velocity_general_t,
)
from mnist_pixel_gmm_core import coupled_sample_gpu

OUT_DIR = "imagenet32_planT_outputs_t_hrf2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start_step", type=int, default=300000)
    p.add_argument("--end_step", type=int, default=400000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--base_lr", type=float, default=1.4e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=20000)
    p.add_argument("--coupling_mode", default="general_t",
                   choices=["t0", "general_t"])
    p.add_argument("--n_gen", type=int, default=50000)
    p.add_argument("--tag", default="t_hrf2_resumed")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    # ---- Data ----
    print("\nLoading ImageNet-32...", flush=True)
    x_pm1, x_hwc_01 = load_imagenet32_pixels(n_max=None)
    print(f"  train n={len(x_pm1)}", flush=True)

    # ---- Inception features for FID ----
    inc = IncFeatRGB().to(DEVICE)
    feat_real_5k = feats_from_hwc(inc, x_hwc_01[:5000])
    feat_real_10k = feats_from_hwc(inc, x_hwc_01[:10000])
    feat_real_50k = feats_from_hwc(inc, x_hwc_01[:50000])

    # ---- Load GMM (unchanged since original run) ----
    print("\nLoading GMM (5 GB)...", flush=True)
    gmm_path = os.path.join(OUT_DIR, "gmm_k5000.pt")
    gmm_data = torch.load(gmm_path, weights_only=False)
    if isinstance(gmm_data, dict) and "weights" in gmm_data:
        gmm = LowRankGMM(gmm_data["weights"], gmm_data["means"],
                         gmm_data["factors"], float(gmm_data["noise_var"]))
    else:
        gmm = gmm_data
    gmm_state = GMMState(gmm)

    # ---- Build model + load checkpoints ----
    print("\nBuilding model and loading checkpoints...", flush=True)
    model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16,8",
        dropout=0.1,
    ).to(DEVICE)
    raw_ckpt = os.path.join(OUT_DIR, "unet.pt")
    ema_ckpt = os.path.join(OUT_DIR, "unet_ema.pt")
    model.load_state_dict(
        torch.load(raw_ckpt, weights_only=False), strict=True)
    print(f"  loaded raw model from {raw_ckpt}", flush=True)

    ema = EMA(model, decay=args.ema_decay)
    ema_shadow = torch.load(ema_ckpt, weights_only=False)
    # EMA shadow is a state-dict-like mapping
    for k, v in ema_shadow.items():
        ema.shadow[k].copy_(v)
    print(f"  loaded EMA shadow from {ema_ckpt}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M", flush=True)

    # ---- Optimizer (constant LR, no warmup since we're past it) ----
    opt = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    x_all = torch.tensor(x_pm1, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    # ---- Training ----
    loss_csv = os.path.join(OUT_DIR, "train_loss_resume.csv")
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "lr", "wall_s"])

    g = torch.Generator(device=DEVICE)
    g.manual_seed(1338)  # different from original to decorrelate RNG
    t_start = time.time()
    model.train()

    total_steps = args.end_step - args.start_step
    print(f"\nTraining from step {args.start_step} to {args.end_step} "
          f"({total_steps} steps)", flush=True)

    for local_step in range(total_steps):
        step = args.start_step + local_step
        idx = torch.randint(0, n_data, (args.bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(args.bs, D_PIX, device=DEVICE, generator=g)
        v_true = x1 - x0

        t_scalar = torch.rand(1, generator=g, device=DEVICE).item() * 0.999
        t_cond = torch.full((args.bs,), t_scalar, device=DEVICE)
        x_t = (1.0 - t_scalar) * x0 + t_scalar * x1

        with torch.no_grad():
            if args.coupling_mode == "t0" or t_scalar < 1e-6:
                v_0 = coupled_sample_gpu(v_true, x0, gmm_state, generator=g)
            else:
                v_0 = coupled_sample_general_t(
                    v_true, x0, x1, x_t, t_scalar, gmm_state, generator=g)

        tau = torch.rand(args.bs, device=DEVICE, generator=g)
        tau_b = tau.view(-1, 1)
        v_tau = (1 - tau_b) * v_0 + tau_b * v_true
        target = v_true - v_0

        v_tau_img = v_tau.reshape(args.bs, C, H, W)
        x_t_img = x_t.reshape(args.bs, C, H, W)
        target_img = target.reshape(args.bs, C, H, W)

        pred = model(tau, v_tau_img, t_cond, x_t_img)
        loss = ((pred - target_img) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)

        if (local_step + 1) % args.log_every == 0 or local_step == 0:
            wall = time.time() - t_start
            print(f"    step {step+1}/{args.end_step}  "
                  f"loss={loss.item():.5f}  lr={args.base_lr:.2e}  "
                  f"wall={wall:.0f}s", flush=True)
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, f"{loss.item():.6f}",
                     f"{args.base_lr:.6e}", f"{wall:.1f}"])

        if ((local_step + 1) % args.ckpt_every == 0
                or (step + 1) == args.end_step):
            torch.save(model.state_dict(), raw_ckpt, pickle_protocol=5)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()

    # ---- Inference sweep ----
    print("\n===== INFERENCE + FID (final) =====", flush=True)
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

    results_csv = os.path.join(OUT_DIR, f"planT_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["J", "L", "nfe", "fid_5k", "fid_10k", "fid_50k"])

    configs = [(1, 10), (1, 20), (1, 50), (2, 50), (2, 250)]
    for J, L in configs:
        nfe = J * L
        t0 = time.time()
        imgs = generate_hrf2(gmm_state, ema_model, n=args.n_gen,
                              flow_steps=J, corr_steps=L,
                              batch_size=128, seed=99)
        wall = time.time() - t0
        g_feats = feats_from_hwc(inc, imgs)
        if not np.isfinite(g_feats).all():
            g_feats = np.nan_to_num(g_feats)
        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        f50 = compute_fid(feat_real_50k, g_feats[:50000])
        print(f"  [EMA] J={J} L={L} NFE={nfe}  "
              f"FID 5k={f5:.3f}  10k={f10:.3f}  50k={f50:.3f}  "
              f"({wall:.0f}s)", flush=True)
        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [J, L, nfe, f"{f5:.4f}", f"{f10:.4f}", f"{f50:.4f}"])
        save_grid(imgs[:100],
                  os.path.join(OUT_DIR,
                               f"stage3_J{J}_L{L}_{args.tag}.png"),
                  f"J={J} L={L} NFE={nfe}  FID50k={f50:.2f}")

    print("\n" + "=" * 70)
    print(f"  Plan T resumed and completed to step {args.end_step}")
    print(f"  {results_csv}")


if __name__ == "__main__":
    main()
