#!/usr/bin/env python3
"""Resume Plan Q-CFG Stage III training from the step-60k checkpoint.

In our run the original training job was killed by a node failure at step ~73k.
The latest atomic checkpoint on disk corresponds to step 60000. This
script:
  1. Reloads dit_corr.pt + dit_corr_ema.pt from celebahq_dcae_planO_outputs_qcfg.
  2. Reloads gmm_k10000.pt (the K=10000 r=128 surrogate from Stage I).
  3. Re-encodes CelebA-HQ -> 2048-d DC-AE latents (latents are not persisted).
  4. Continues training with the same DiTCorrectionCFG architecture and the
     same posterior-coupling + 10%-null-dropout recipe.
  5. Uses a fresh AdamW + fresh cosine LR over the resumed segment (peak
     halved to 1e-4 so the previously-converged weights are not knocked
     around by an aggressive restart).
  6. Atomic-checkpoints every 20k steps; saves step-tagged milestones
     every 100k steps.
  7. Runs the in-job CFG sweep (w in {0, 1.0, 1.5, 2.0}) at the end.

Wall budget: 48h. At ~130 steps/min, 340k extra steps need ~43.6h, plus
~30 min eval. Comfortably inside walltime if no further node failure.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celebahq_dcae_unet import (  # noqa: E402
    DCAE_PATH,
    DEVICE,
    D_LATENT,
    LATENT_C,
    LATENT_SPATIAL,
    IncFeatRGB,
    encode_to_latents,
    feats_from_bchw,
    load_celebahq,
    make_tau_grid,
    save_grid,
)
from celebahq_dcae_dit_cfg import (  # noqa: E402
    DiTCorrectionCFG,
    _coupled_sample_with_cluster,
    generate_corrected_dit_cfg,
)
from cifar10_sdvae_latent_ccvfm import EMA, cosine_lr  # noqa: E402
from mnist_pixel_gmm_core import (  # noqa: E402
    GMMState,
    LowRankGMM,
    compute_fid,
)
from diffusers import AutoencoderDC  # noqa: E402

OUT_DIR = "celebahq_dcae_planO_outputs_qcfg"
DCAE_SCALE = 0.3189

# DiT-L architecture (matches Plan Q-CFG)
HIDDEN_SIZE = 1024
DEPTH = 24
NUM_HEADS = 16
NUM_CLUSTERS = 10000
BS = 128

# Resumed-segment LR schedule (halved peak vs from-scratch run)
BASE_LR = 1e-4
MIN_LR = 2e-6
WARMUP_STEPS = 2000
EMA_DECAY = 0.9999
NULL_PROB = 0.10

CKPT_EVERY = 20000
MILESTONE_EVERY = 100000
LOG_EVERY = 500

TAU_L = 50
TAU_POWER = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-step", type=int, default=60000,
                   help="Step count of the checkpoint being resumed from.")
    p.add_argument("--extra-steps", type=int, default=340000,
                   help="Additional training steps. start+extra=400k by default.")
    p.add_argument("--n-data", type=int, default=28000)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--n-gen", type=int, default=10000,
                   help="Samples per guidance scale for the in-job eval.")
    return p.parse_args()


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp, pickle_protocol=5)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    total_steps = args.start_step + args.extra_steps
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {OUT_DIR}", flush=True)
    print(f"  resume from step {args.start_step} -> {total_steps} "
          f"(+{args.extra_steps})", flush=True)

    print(f"\nLoading DC-AE...", flush=True)
    vae = AutoencoderDC.from_pretrained(DCAE_PATH).to(DEVICE)
    vae.eval()

    print(f"\nLoading CelebA-HQ (n={args.n_data})...", flush=True)
    imgs_bchw = load_celebahq(args.n_data, target_res=256)
    print(f"  shape: {tuple(imgs_bchw.shape)}", flush=True)

    print("\n===== STAGE 0: encode CelebA-HQ -> 2048-d latents =====", flush=True)
    latents = encode_to_latents(vae, imgs_bchw, bs=8)

    if not args.no_flip:
        print("\nFlip aug...", flush=True)
        n_flip = len(imgs_bchw)
        latents_flip = np.zeros((n_flip, D_LATENT), dtype=np.float32)
        bs_flip = 8
        for i in range(0, n_flip, bs_flip):
            batch = imgs_bchw[i:i + bs_flip].to(DEVICE)
            batch = torch.flip(batch, dims=[3])
            batch = batch * 2.0 - 1.0
            with torch.no_grad():
                lat = vae.encode(batch).latent * DCAE_SCALE
            latents_flip[i:i + bs_flip] = lat.reshape(
                lat.shape[0], -1).cpu().numpy()
        latents = np.concatenate([latents, latents_flip], axis=0)
        print(f"  latents after flip aug: {latents.shape}", flush=True)

    print(f"\nInception features for final eval ({len(imgs_bchw)} real)...",
          flush=True)
    inc = IncFeatRGB().to(DEVICE)
    real_feats = feats_from_bchw(inc, imgs_bchw, bs=32)
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats[:10000] if len(real_feats) >= 10000 else real_feats
    feat_real_full = real_feats
    n_full = len(real_feats)
    print(f"  feat shape: {real_feats.shape}", flush=True)
    del imgs_bchw
    torch.cuda.empty_cache()

    gmm_path = os.path.join(OUT_DIR, "gmm_k10000.pt")
    print(f"\nLoading GMM from {gmm_path}...", flush=True)
    gmm_data = torch.load(gmm_path, weights_only=False)
    if isinstance(gmm_data, dict) and "weights" in gmm_data:
        gmm = LowRankGMM(gmm_data["weights"], gmm_data["means"],
                         gmm_data["factors"], float(gmm_data["noise_var"]))
    else:
        gmm = gmm_data
    gmm_state = GMMState(gmm)
    K = gmm.weights.shape[0]
    print(f"  K={K}, d={gmm.means.shape[1]}, r={gmm.factors.shape[2]}",
          flush=True)

    raw_ckpt = os.path.join(OUT_DIR, "dit_corr.pt")
    ema_ckpt = os.path.join(OUT_DIR, "dit_corr_ema.pt")
    model = DiTCorrectionCFG(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=NUM_HEADS,
        patch_size=1, input_size=LATENT_SPATIAL, num_clusters=K,
    ).to(DEVICE)

    print(f"\nLoading raw weights from {raw_ckpt}...", flush=True)
    raw_sd = torch.load(raw_ckpt, weights_only=False)
    model.load_state_dict(raw_sd, strict=True)

    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    ema = EMA(model, decay=EMA_DECAY)
    print(f"Loading EMA shadow from {ema_ckpt}...", flush=True)
    ema_sd = torch.load(ema_ckpt, weights_only=False)
    for k, v in ema_sd.items():
        if k in ema.shadow:
            ema.shadow[k] = v.clone().to(ema.shadow[k].device)
    print("  EMA shadow restored.", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.0)

    x_all = torch.tensor(latents, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    loss_csv = os.path.join(OUT_DIR, "train_loss_resume.csv")
    write_header = not os.path.exists(loss_csv)
    if write_header:
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr", "wall_s"])

    g = torch.Generator(device=DEVICE)
    g.manual_seed(7777 + args.start_step)
    t_start = time.time()
    model.train()

    print(f"\n===== STAGE III RESUME: train steps {args.start_step} -> {total_steps} =====",
          flush=True)

    extra = args.extra_steps
    for local_step in range(extra):
        global_step = args.start_step + local_step
        lr = cosine_lr(local_step, extra, BASE_LR, MIN_LR, WARMUP_STEPS)
        for pg_opt in opt.param_groups:
            pg_opt["lr"] = lr

        idx = torch.randint(0, n_data, (BS,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(BS, D_LATENT, device=DEVICE, generator=g)
        v_true = x1 - x0

        with torch.no_grad():
            v_0, b_star = _coupled_sample_with_cluster(
                v_true, x0, gmm_state, generator=g)

        null_mask = torch.rand(BS, device=DEVICE, generator=g) < NULL_PROB
        cluster_idx = torch.where(null_mask, K, b_star.long())

        tau = torch.rand(BS, device=DEVICE, generator=g)
        v_tau = (1.0 - tau.view(-1, 1)) * v_0 + tau.view(-1, 1) * v_true
        target = v_true - v_0
        t_cond = torch.zeros(BS, device=DEVICE)

        v_tau_img = v_tau.reshape(BS, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        x0_img = x0.reshape(BS, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        target_img = target.reshape(BS, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)

        pred = model(v_tau_img, x0_img, tau, t_cond, cluster_idx)
        loss = ((pred - target_img) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if (local_step + 1) % LOG_EVERY == 0 or local_step == 0:
            wall = time.time() - t_start
            print(f"    step {global_step+1}/{total_steps}  "
                  f"loss={loss.item():.5f}  lr={lr:.2e}  wall={wall:.0f}s",
                  flush=True)
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [global_step + 1, f"{loss.item():.6f}",
                     f"{lr:.6e}", f"{wall:.1f}"])

        if (local_step + 1) % CKPT_EVERY == 0 or (local_step + 1) == extra:
            _atomic_save(model.state_dict(), raw_ckpt)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {global_step+1}",
                  flush=True)

        if (local_step + 1) % MILESTONE_EVERY == 0 or (local_step + 1) == extra:
            tag = f"step{global_step+1}"
            _atomic_save(model.state_dict(),
                         os.path.join(OUT_DIR, f"dit_corr_{tag}.pt"))
            tmp_ema = os.path.join(OUT_DIR, f"dit_corr_ema_{tag}.pt.tmp")
            final_ema = os.path.join(OUT_DIR, f"dit_corr_ema_{tag}.pt")
            torch.save(ema.shadow, tmp_ema, pickle_protocol=5)
            os.replace(tmp_ema, final_ema)
            print(f"    milestone saved: {tag}", flush=True)

    torch.cuda.synchronize()
    model.eval()

    print(f"\n===== FINAL EVAL @ step {total_steps} (CFG sweep) =====",
          flush=True)
    ema_model = DiTCorrectionCFG(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=HIDDEN_SIZE, depth=DEPTH, num_heads=NUM_HEADS,
        patch_size=1, input_size=LATENT_SPATIAL, num_clusters=K,
    ).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    tau_grid = make_tau_grid(L=TAU_L, power=TAU_POWER).to(DEVICE)
    results_csv = os.path.join(
        OUT_DIR, f"planQcfg_results_step{total_steps}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["guidance_scale", "nfe", "fid_5k", "fid_10k",
             f"fid_{n_full//1000}k", "wall_seconds"])

    for w in [0.0, 1.0, 1.5, 2.0]:
        nfe = TAU_L * (2 if w > 0 else 1)
        t0 = time.time()
        imgs = generate_corrected_dit_cfg(
            gmm_state, ema_model, vae, n=args.n_gen, tau_grid=tau_grid,
            guidance_scale=w, batch_size=128, seed=99,
        )
        wall = time.time() - t0
        g_bchw = torch.tensor(imgs.transpose(0, 3, 1, 2))
        g_feats = feats_from_bchw(inc, g_bchw, bs=32)
        del g_bchw
        torch.cuda.empty_cache()
        if not np.isfinite(g_feats).all():
            g_feats = np.nan_to_num(g_feats)

        n_eval_full = min(n_full, len(g_feats))
        f5 = compute_fid(feat_real_5k, g_feats[:5000]) \
            if len(g_feats) >= 5000 else float("nan")
        f10 = compute_fid(feat_real_10k, g_feats[:10000]) \
            if len(g_feats) >= 10000 else float("nan")
        f_full = compute_fid(feat_real_full[:n_eval_full],
                             g_feats[:n_eval_full]) \
            if n_eval_full > 0 else float("nan")
        print(f"  [w={w}] gen {args.n_gen} in {wall:.0f}s  "
              f"FID 5k={f5:.3f} 10k={f10:.3f} {n_full//1000}k={f_full:.3f}",
              flush=True)
        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [w, nfe, f"{f5:.4f}", f"{f10:.4f}",
                 f"{f_full:.4f}", f"{wall:.0f}"])
        save_grid(
            imgs[:100],
            os.path.join(OUT_DIR, f"stage3_cfg_w{w}_step{total_steps}.png"),
            f"Plan Q-CFG step{total_steps} w={w} NFE={nfe} "
            f"FID{n_full//1000}k={f_full:.2f}")

    print("\n" + "=" * 70)
    print(f"  Plan Q-CFG resume complete. Step {total_steps}.")
    print(f"  results: {results_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
