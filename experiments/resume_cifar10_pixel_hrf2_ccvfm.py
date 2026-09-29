#!/usr/bin/env python3
"""Resume Plan S++ Stage III from step 400000 onward.

Loads cifar10_planS_outputs_pp/{gmm_k10000.pt, unet.pt, unet_ema.pt} and
continues Stage III training for another --train-steps (default 800000)
with --start-step=400000.  Uses Plan S++'s architecture (base=128,
attention="16" — same simple arch as original Plan S).

Launch via slurm/cifar10_resume.sbatch.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cifar10_pixel_hrf2_ccvfm import (  # noqa: E402
    DEVICE, DualBranchUNet, IncFeatRGB, feats_from_hwc,
    generate_hrf2, load_cifar10_pixels, save_grid,
    train_stage3, compute_fid,
)
from mnist_pixel_gmm_core import GMMState, LowRankGMM  # noqa: E402

OUT_DIR = "cifar10_planS_outputs_pp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=800000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--unet-base", type=int, default=128)
    ap.add_argument("--unet-attention", type=str, default="16")
    ap.add_argument("--unet-res-blocks", type=int, default=2)
    ap.add_argument("--coupling-mode", default="general_t")
    ap.add_argument("--start-step", type=int, default=400000,
                    help="resume from this step (requires existing "
                    "unet.pt and unet_ema.pt in out_dir)")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--milestone-every", type=int, default=100000)
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    print("Loading CIFAR pixels ...", flush=True)
    x_pm1, x_hwc_01 = load_cifar10_pixels()
    inc = IncFeatRGB().to(DEVICE)
    print("Real Inception features (50k pool) ...", flush=True)
    feat_real_5k = feats_from_hwc(inc, x_hwc_01[:5000])
    feat_real_10k = feats_from_hwc(inc, x_hwc_01[:10000])
    feat_real_50k = feats_from_hwc(inc, x_hwc_01[:50000])

    print(f"Loading saved GMM from {OUT_DIR}/gmm_k10000.pt ...", flush=True)
    gmm_data = torch.load(os.path.join(OUT_DIR, "gmm_k10000.pt"),
                          weights_only=False)
    if isinstance(gmm_data, dict) and "weights" in gmm_data:
        gmm = LowRankGMM(gmm_data["weights"], gmm_data["means"],
                         gmm_data["factors"], float(gmm_data["noise_var"]))
    else:
        gmm = gmm_data
    gmm_state = GMMState(gmm)
    print("  GMM loaded", flush=True)

    print(f"\nProbe-building UNet base={args.unet_base} "
          f"attention={args.unet_attention} ...", flush=True)
    try:
        probe = DualBranchUNet(
            dim=(3, 32, 32),
            num_res_blocks=args.unet_res_blocks,
            num_channels=args.unet_base,
            channel_mult=(1, 2, 2, 2),
            num_heads=4, num_head_channels=64,
            attention_resolutions=args.unet_attention,
            dropout=0.1).to(DEVICE)
        print(f"  probe params: "
              f"{sum(p.numel() for p in probe.parameters())/1e6:.2f}M",
              flush=True)
        del probe
        torch.cuda.empty_cache()
    except Exception as e:
        traceback.print_exc()
        print(f"\nUNET CONSTRUCTION FAILED: {e}", flush=True)
        sys.exit(2)

    print(f"\n===== Stage III training "
          f"(from step {args.start_step} to {args.train_steps}) =====",
          flush=True)
    model, ema = train_stage3(
        x_pm1, gmm_state, OUT_DIR,
        n_iter=args.train_steps, bs=args.bs,
        coupling_mode=args.coupling_mode,
        unet_base=args.unet_base,
        unet_attention=args.unet_attention,
        unet_res_blocks=args.unet_res_blocks,
        start_step=args.start_step,
        ckpt_every=args.ckpt_every,
        milestone_every=args.milestone_every)

    print("\n===== INFERENCE + FID @ 50k =====", flush=True)
    ema_model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=args.unet_res_blocks,
        num_channels=args.unet_base,
        channel_mult=(1, 2, 2, 2),
        num_heads=4, num_head_channels=64,
        attention_resolutions=args.unet_attention,
        dropout=0.1).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    csv_path = os.path.join(OUT_DIR, f"planSpp_results_step{args.train_steps}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["J", "L", "nfe", "fid_5k", "fid_10k", "fid_50k"])

    for J, L in [(1, 10), (1, 20), (1, 50), (2, 50)]:
        nfe = J * L
        t0 = time.time()
        imgs = generate_hrf2(gmm_state, ema_model, n=50000,
                             flow_steps=J, corr_steps=L,
                             batch_size=128, seed=99)
        wall = time.time() - t0
        g_feats = feats_from_hwc(inc, imgs)
        if not np.isfinite(g_feats).all():
            g_feats = np.nan_to_num(g_feats)
        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        f50 = compute_fid(feat_real_50k, g_feats[:50000])
        print(f"  [EMA step{args.train_steps}] J={J} L={L} NFE={nfe}  "
              f"FID 5k={f5:.3f} 10k={f10:.3f} 50k={f50:.3f} "
              f"({wall:.0f}s)", flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [J, L, nfe, f"{f5:.4f}", f"{f10:.4f}", f"{f50:.4f}"])
        save_grid(imgs[:100],
                  os.path.join(OUT_DIR,
                               f"stage3_J{J}_L{L}_step{args.train_steps}.png"),
                  f"J={J} L={L} NFE={nfe}  FID50k={f50:.2f}")


if __name__ == "__main__":
    main()
