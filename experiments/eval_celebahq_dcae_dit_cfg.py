#!/usr/bin/env python3
"""Streaming 30k-protocol CFG eval for Plan Q-CFG.

Drops the original eval's all-at-once 30k generation (which OOM'd at 128 GB)
in favor of chunked generation: 5k samples per chunk,
features extracted per chunk, full image tensors freed before the next.

Otherwise identical protocol: 30k generated vs 28k CelebA-HQ real,
InceptionV3 features, sweep over guidance_scale in {0.0, 1.0, 1.5, 2.0}.
"""
from __future__ import annotations

import csv
import gc
import os
import site
import sys
import time

sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from celebahq_dcae_unet import (  # noqa: E402
    DCAE_PATH,
    DEVICE,
    LATENT_C,
    LATENT_SPATIAL,
    IncFeatRGB,
    feats_from_bchw,
    load_celebahq,
    make_tau_grid,
    save_grid,
)
from celebahq_dcae_dit_cfg import (  # noqa: E402
    DiTCorrectionCFG,
    generate_corrected_dit_cfg,
)
from mnist_pixel_gmm_core import (  # noqa: E402
    GMMState,
    LowRankGMM,
    compute_fid,
)
from diffusers import AutoencoderDC  # noqa: E402

OUT_DIR = "celebahq_dcae_planO_outputs_qcfg"
GMM_PATH = os.path.join(OUT_DIR, "gmm_k10000.pt")
EMA_PATH = os.path.join(OUT_DIR, "dit_corr_ema.pt")

# DiT-L (matches planQ-CFG)
HIDDEN_SIZE = 1024
DEPTH = 24
NUM_HEADS = 16
TAU_L = 50
TAU_POWER = 2.0
NUM_CLUSTERS = 10000

N_REAL = 30000          # but capped to len(CelebA-HQ) ~28k
N_GEN = 30000
CHUNK = 5000            # streaming chunk size; 6 chunks => 30k total
GUIDANCE_SCALES = [0.0, 1.0, 1.5, 2.0]


def _generate_features_streamed(gmm_state, ema_model, vae, inc, tau_grid,
                                guidance_scale, n_total=N_GEN, chunk=CHUNK,
                                base_seed=99, save_first_n=100):
    """Generate `n_total` samples in `chunk`-sized blocks; return
    (features [n_total, 2048] np.float32, first_chunk_imgs [save_first_n, ...]).

    Each chunk uses seed = base_seed + chunk_index so chunks are deterministic
    but distinct (same RNG path the original full-N call would have produced
    is *not* preserved; that path is intractable in chunked mode).
    """
    feats_chunks = []
    saved_imgs = None
    n_done = 0
    chunk_idx = 0
    while n_done < n_total:
        nb = min(chunk, n_total - n_done)
        t0 = time.time()
        imgs = generate_corrected_dit_cfg(
            gmm_state, ema_model, vae, n=nb, tau_grid=tau_grid,
            guidance_scale=guidance_scale, batch_size=128,
            seed=base_seed + chunk_idx,
        )
        wall_gen = time.time() - t0

        if saved_imgs is None and save_first_n > 0:
            saved_imgs = imgs[:save_first_n].copy()

        # imgs: (nb, 256, 256, 3) np.float32 in [0,1]
        g_bchw = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).contiguous()
        del imgs

        t1 = time.time()
        chunk_feats = feats_from_bchw(inc, g_bchw, bs=32)
        wall_feat = time.time() - t1
        del g_bchw
        gc.collect()
        torch.cuda.empty_cache()

        if not np.isfinite(chunk_feats).all():
            chunk_feats = np.nan_to_num(chunk_feats)
        feats_chunks.append(chunk_feats.astype(np.float32))

        n_done += nb
        chunk_idx += 1
        print(f"    chunk {chunk_idx}: gen {wall_gen:.0f}s + "
              f"feat {wall_feat:.0f}s  ({n_done}/{n_total} samples)",
              flush=True)

    feats = np.concatenate(feats_chunks, axis=0)
    del feats_chunks
    gc.collect()
    return feats, saved_imgs


def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {OUT_DIR}", flush=True)
    print(f"  ckpt:    {EMA_PATH}", flush=True)
    print(f"  N_GEN={N_GEN}, CHUNK={CHUNK}", flush=True)

    if not os.path.exists(EMA_PATH):
        print(f"[fatal] EMA checkpoint not found at {EMA_PATH}", flush=True)
        return
    if not os.path.exists(GMM_PATH):
        print(f"[fatal] GMM not found at {GMM_PATH}", flush=True)
        return

    print(f"\nLoading DC-AE...", flush=True)
    vae = AutoencoderDC.from_pretrained(DCAE_PATH).to(DEVICE)
    vae.eval()

    print(f"\nLoading CelebA-HQ ({N_REAL} images)...", flush=True)
    imgs_bchw = load_celebahq(N_REAL, target_res=256)
    n_real = len(imgs_bchw)
    print(f"  CelebA-HQ: {tuple(imgs_bchw.shape)}", flush=True)

    print(f"\nInception features on {n_real} real images...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    feat_real = feats_from_bchw(inc, imgs_bchw, bs=32)
    print(f"  {time.time()-t0:.0f}s, shape={feat_real.shape}", flush=True)
    del imgs_bchw
    gc.collect()
    torch.cuda.empty_cache()

    feat_real_5k = feat_real[:5000]
    feat_real_10k = feat_real[:10000]
    n_full = min(len(feat_real), N_GEN)
    feat_real_full = feat_real[:n_full]

    print(f"\nLoading GMM from {GMM_PATH}...", flush=True)
    gmm_data = torch.load(GMM_PATH, weights_only=False)
    if isinstance(gmm_data, dict) and "weights" in gmm_data:
        gmm = LowRankGMM(gmm_data["weights"], gmm_data["means"],
                         gmm_data["factors"], float(gmm_data["noise_var"]))
    else:
        gmm = gmm_data
    gmm_state = GMMState(gmm)
    print(f"  K={gmm.weights.shape[0]}, d={gmm.means.shape[1]}, "
          f"r={gmm.factors.shape[2]}", flush=True)

    print(f"\nLoading EMA DiT-L/CFG from {EMA_PATH}...", flush=True)
    ema_model = DiTCorrectionCFG(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=HIDDEN_SIZE, depth=DEPTH,
        num_heads=NUM_HEADS, patch_size=1, input_size=LATENT_SPATIAL,
        num_clusters=NUM_CLUSTERS,
    ).to(DEVICE)
    ema_sd = torch.load(EMA_PATH, weights_only=False)
    ema_model.load_state_dict(ema_sd, strict=True)
    ema_model.eval()
    print(f"  params: "
          f"{sum(p.numel() for p in ema_model.parameters())/1e6:.2f}M",
          flush=True)

    tau_grid = make_tau_grid(L=TAU_L, power=TAU_POWER).to(DEVICE)

    results_csv = os.path.join(OUT_DIR, "planQcfg_results_30k.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["guidance_scale", "nfe",
             "fid_5k", "fid_10k", f"fid_{n_full//1000}k",
             "wall_seconds"])

    for w in GUIDANCE_SCALES:
        nfe = TAU_L * (2 if w > 0 else 1)
        print(f"\n===== guidance w={w}, NFE={nfe}, n_gen={N_GEN} "
              f"(streamed in {N_GEN // CHUNK} chunks of {CHUNK}) =====",
              flush=True)
        t0 = time.time()
        g_feats, saved_imgs = _generate_features_streamed(
            gmm_state, ema_model, vae, inc, tau_grid,
            guidance_scale=w, n_total=N_GEN, chunk=CHUNK,
            base_seed=99, save_first_n=100,
        )
        wall = time.time() - t0
        print(f"  total {wall:.0f}s ({wall/60:.1f} min) for {N_GEN} samples",
              flush=True)

        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        f_full = compute_fid(feat_real_full[:n_full], g_feats[:n_full])
        print(f"  FID_5k={f5:.3f}  FID_10k={f10:.3f}  "
              f"FID_{n_full//1000}k={f_full:.3f}", flush=True)

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [w, nfe, f"{f5:.4f}", f"{f10:.4f}", f"{f_full:.4f}",
                 f"{wall:.0f}"])

        if saved_imgs is not None:
            save_grid(
                saved_imgs,
                os.path.join(OUT_DIR, f"stage3_cfg_w{w}_30k.png"),
                f"Plan Q-CFG  w={w}  NFE={nfe}  "
                f"FID{n_full//1000}k={f_full:.2f}",
            )

        del g_feats, saved_imgs
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"  Plan Q-CFG 30k-protocol streaming eval complete")
    print(f"  results: {results_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
