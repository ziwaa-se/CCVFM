#!/usr/bin/env python3
"""50k-sample FID sweep of a CIFAR-10 CCVFM checkpoint (the headline evaluation).

Loads cifar10_planS_outputs_pp/{gmm_k10000.pt, unet_ema_<tag>.pt} and sweeps
(J, L) in {(1,10), (1,20), (1,50), (2,50)}. The paper headline (FID50k 6.35 at
L=50) is the EMA checkpoint after 720k Stage III steps: set CCVFM_CKPT_TAG to
another step tag (e.g. step400000) to evaluate a different milestone. If the
tagged file is missing the live unet_ema.pt is used instead.
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cifar10_pixel_hrf2_ccvfm import (  # noqa: E402
    DEVICE,
    DualBranchUNet,
    IncFeatRGB,
    feats_from_hwc,
    generate_hrf2,
    load_cifar10_pixels,
    save_grid,
)
from mnist_pixel_gmm_core import GMMState, LowRankGMM  # noqa: E402
from fid_utils import compute_fid_robust  # noqa: E402

OUT_DIR = "cifar10_planS_outputs_pp"
CKPT_TAG = os.environ.get("CCVFM_CKPT_TAG", "step720000")
RESULTS_CSV = os.path.join(OUT_DIR, f"planSpp_results_{CKPT_TAG}.csv")


def _pick_ema_ckpt() -> str:
    """Prefer the step-tagged milestone; fall back to the live checkpoint."""
    tagged = os.path.join(OUT_DIR, f"unet_ema_{CKPT_TAG}.pt")
    if os.path.isfile(tagged):
        return tagged
    live = os.path.join(OUT_DIR, "unet_ema.pt")
    print(f"[warn] {tagged} not found; falling back to {live}", flush=True)
    return live


def main():
    print(f"Device: {DEVICE}", flush=True)

    print("Loading CIFAR-10 + Inception features (50k pool)...", flush=True)
    _, x_hwc_01 = load_cifar10_pixels()
    inc = IncFeatRGB().to(DEVICE)
    feat_real_5k = feats_from_hwc(inc, x_hwc_01[:5000])
    feat_real_10k = feats_from_hwc(inc, x_hwc_01[:10000])
    feat_real_50k = feats_from_hwc(inc, x_hwc_01[:50000])
    print("  real features extracted", flush=True)

    print(f"Loading GMM from {OUT_DIR}/gmm_k10000.pt (~9.9 GB)...", flush=True)
    gmm_data = torch.load(os.path.join(OUT_DIR, "gmm_k10000.pt"),
                          weights_only=False)
    if isinstance(gmm_data, dict) and "weights" in gmm_data:
        gmm = LowRankGMM(gmm_data["weights"], gmm_data["means"],
                         gmm_data["factors"], float(gmm_data["noise_var"]))
    else:
        gmm = gmm_data
    gmm_state = GMMState(gmm)

    ckpt_path = _pick_ema_ckpt()
    print(f"Loading EMA UNet from {ckpt_path} ...", flush=True)
    ema_model = DualBranchUNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(DEVICE)
    ema_sd = torch.load(ckpt_path, weights_only=False)
    ema_model.load_state_dict(ema_sd, strict=True)
    ema_model.eval()
    print(f"  params: "
          f"{sum(p.numel() for p in ema_model.parameters())/1e6:.2f}M",
          flush=True)

    configs = [(1, 10), (1, 20), (1, 50), (2, 50)]
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["J", "L", "nfe", "fid_5k", "fid_10k", "fid_50k"])

    print(f"\n===== Inference @ 50k samples (ckpt={CKPT_TAG}) =====",
          flush=True)
    n_gen = 50000
    for J, L in configs:
        nfe = J * L
        t0 = time.time()
        imgs = generate_hrf2(gmm_state, ema_model, n=n_gen,
                             flow_steps=J, corr_steps=L,
                             batch_size=128, seed=99)
        wall = time.time() - t0
        g_feats = feats_from_hwc(inc, imgs)
        if not np.isfinite(g_feats).all():
            g_feats = np.nan_to_num(g_feats)
        f5 = compute_fid_robust(feat_real_5k, g_feats[:5000])
        f10 = compute_fid_robust(feat_real_10k, g_feats[:10000])
        f50 = compute_fid_robust(feat_real_50k, g_feats[:50000])
        print(f"  [EMA {CKPT_TAG}] J={J} L={L} NFE={nfe}  "
              f"FID 5k={f5:.3f}  10k={f10:.3f}  50k={f50:.3f}  "
              f"({wall:.0f}s)", flush=True)
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow(
                [J, L, nfe, f"{f5:.4f}", f"{f10:.4f}", f"{f50:.4f}"])
        save_grid(
            imgs[:100],
            os.path.join(OUT_DIR, f"stage3_J{J}_L{L}_{CKPT_TAG}.png"),
            f"Step {CKPT_TAG}  J={J} L={L} NFE={nfe}  FID50k={f50:.2f}",
        )

    print("\n" + "=" * 60)
    print(f"  Plan S++ @ {CKPT_TAG} 50k FID sweep done")
    print(f"  results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
