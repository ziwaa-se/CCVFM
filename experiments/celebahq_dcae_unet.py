#!/usr/bin/env python3
"""CelebA-HQ 256×256 Plan N — DC-AE latents + CoresetFM.

Pipeline:
  Stage 0: CelebA-HQ 28k images at 256×256 → DC-AE f32c32 encode → 32×8×8
           = 2048-d latent  (floor FID = 2.49, measured)
  Stage I: EMS coreset (K=5000) + low-rank covariance (rank=80)
  Stage II: closed-form sampler
  Stage III: LatentCorrectionUNet (45M params, base=200) trained with
             pre-specified τ grid (20 points, quadratic density near 0)
  Inference: Euler over the fixed τ grid

Launch:
  (see slurm/ for job templates)
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
from diffusers import AutoencoderDC
from torchvision.models import Inception_V3_Weights, inception_v3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override planG latent shape constants via monkey-patch at import time
import cifar10_sdvae_latent_ccvfm as pg
pg.LATENT_C = 32
pg.LATENT_SPATIAL = 8
pg.D_LATENT = 32 * 8 * 8  # 2048

from cifar10_sdvae_latent_ccvfm import (  # noqa: E402
    EMA,
    LatentCorrectionUNet,
    cosine_lr,
)

from mnist_pixel_gmm_core import (  # noqa: E402
    GMMState,
    LowRankGMM,
    compute_fid,
    coupled_sample_gpu,
    ems_coreset_gpu,
    learn_lowrank_cov_fast,
    sample_velocity_gpu,
    save_gmm,
)

# Prefer local cached DC-AE weights if present; otherwise fall back to the
# HuggingFace repo ID (huggingface_hub will download into HF_HOME on first use).
import os as _os
_DCAE_LOCAL = _os.environ.get("CCVFM_DCAE_DIR", _os.path.expanduser("~/dcae_weights_f32"))
DCAE_PATH = _DCAE_LOCAL if (_os.path.isdir(_DCAE_LOCAL) and _os.listdir(_DCAE_LOCAL)) \
    else "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-patch DEVICE in helper modules to match current script's device
import mnist_pixel_gmm_core as _mplf
_mplf.DEVICE = DEVICE
pg.DEVICE = DEVICE

DCAE_SCALE = 0.3189
LATENT_RES = 256                                # native CelebA-HQ resolution
LATENT_C = 32
LATENT_SPATIAL = LATENT_RES // 32               # 8
D_LATENT = LATENT_C * LATENT_SPATIAL * LATENT_SPATIAL  # 2048


# ===================================================================
# Pre-specified τ grid (same as Plan M1c)
# ===================================================================

def make_tau_grid(L: int = 20, power: float = 2.0) -> torch.Tensor:
    t = torch.arange(L, dtype=torch.float32) / L
    return t ** power


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
def feats_from_bchw(inc, imgs_bchw, bs=32):
    fs = []
    for i in range(0, len(imgs_bchw), bs):
        batch = imgs_bchw[i:i+bs].to(DEVICE)
        fs.append(inc(batch).cpu().numpy())
    return np.concatenate(fs)


# ===================================================================
# CelebA-HQ loader
# ===================================================================

def load_celebahq(n: int, target_res: int = 256):
    """Load CelebA-HQ via HF datasets. Returns (n, 3, 256, 256) in [0,1]."""
    from datasets import load_dataset
    import torchvision.transforms.functional as TF
    print(f"  Loading CelebA-HQ from HF (up to {n} images)...", flush=True)
    t0 = time.time()
    ds = load_dataset("korexyz/celeba-hq-256x256", split="train", streaming=False)
    total = len(ds)
    n = min(n, total)
    print(f"  Loaded {total} images total; using {n}", flush=True)

    out = torch.zeros(n, 3, target_res, target_res)
    for i in range(n):
        img = ds[i]["image"].convert("RGB")
        t = TF.to_tensor(img)  # (3, H, W) in [0,1]
        if t.shape[-1] != target_res or t.shape[-2] != target_res:
            t = F.interpolate(t.unsqueeze(0), size=(target_res, target_res),
                              mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
        out[i] = t
    print(f"  Loaded in {time.time()-t0:.0f}s", flush=True)
    return out


# ===================================================================
# DC-AE encode / decode
# ===================================================================

@torch.no_grad()
def encode_to_latents(vae, imgs_bchw, bs=8):
    n = len(imgs_bchw)
    out = np.zeros((n, D_LATENT), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, bs):
        x = imgs_bchw[i:i+bs].to(DEVICE)
        x = x * 2.0 - 1.0
        lat = vae.encode(x).latent * DCAE_SCALE
        out[i:i+bs] = lat.reshape(lat.shape[0], -1).cpu().numpy()
    print(f"  encoded {n} latents in {time.time()-t0:.0f}s", flush=True)
    return out


@torch.no_grad()
def decode_latents(vae, latents_flat, bs=8):
    """Decode latents back to 256×256 RGB in [0,1]. Returns (n, 256, 256, 3)."""
    n = latents_flat.shape[0]
    out = np.zeros((n, 256, 256, 3), dtype=np.float32)
    for i in range(0, n, bs):
        lat = latents_flat[i:i+bs].to(DEVICE).reshape(
            -1, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        lat = lat / DCAE_SCALE
        x = vae.decode(lat).sample
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        out[i:i+bs] = x.permute(0, 2, 3, 1).cpu().numpy()
    return out


# ===================================================================
# Stage III training (τ-grid)
# ===================================================================

def train_stage3(
    latents_np, state, tau_grid, out_dir,
    n_iter=800000, bs=128, base_lr=2e-4, min_lr=2e-6,
    warmup_steps=4000, ema_decay=0.9999,
    log_every=500, ckpt_every=20000, unet_base=200,
):
    model = LatentCorrectionUNet(base=unet_base).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  LatentCorrectionUNet params: {n_params/1e6:.2f}M", flush=True)
    print(f"  τ grid: {tau_grid.cpu().tolist()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
    ema = EMA(model, decay=ema_decay)

    x_all = torch.tensor(latents_np, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    raw_ckpt = os.path.join(out_dir, "corr_unet.pt")
    ema_ckpt = os.path.join(out_dir, "corr_unet_ema.pt")
    loss_csv = os.path.join(out_dir, "train_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "lr", "wall_s"])

    g = torch.Generator(device=DEVICE)
    g.manual_seed(1337)
    t_start = time.time()
    model.train()

    L_tau = len(tau_grid)

    for step in range(n_iter):
        lr = cosine_lr(step, n_iter, base_lr, min_lr, warmup_steps)
        for pg_opt in opt.param_groups:
            pg_opt["lr"] = lr

        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D_LATENT, device=DEVICE, generator=g)
        v_true = x1 - x0

        with torch.no_grad():
            v_0 = coupled_sample_gpu(v_true, x0, state, generator=g)

        tau_idx = torch.randint(0, L_tau, (bs,), device=DEVICE, generator=g)
        tau = tau_grid[tau_idx]

        v_tau = (1.0 - tau).view(-1, 1) * v_0 + tau.view(-1, 1) * v_true
        target = v_true - v_0
        t_cond = torch.zeros(bs, device=DEVICE)

        v_tau_img = v_tau.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        x0_img = x0.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        target_img = target.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)

        pred = model(v_tau_img, x0_img, tau, t_cond)
        loss = ((pred - target_img) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if (step + 1) % log_every == 0 or step == 0:
            wall = time.time() - t_start
            print(f"    step {step+1}/{n_iter}  loss={loss.item():.5f}  "
                  f"lr={lr:.2e}  wall={wall:.0f}s", flush=True)
            with open(loss_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step + 1, f"{loss.item():.6f}", f"{lr:.6e}", f"{wall:.1f}"])

        if (step + 1) % ckpt_every == 0 or (step + 1) == n_iter:
            torch.save(model.state_dict(), raw_ckpt, pickle_protocol=5)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema


# ===================================================================
# Inference over τ grid
# ===================================================================

@torch.no_grad()
def generate_corrected(state, model, vae, n, tau_grid, batch_size=128, seed=99):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    model.eval()
    out = np.zeros((n, 256, 256, 3), dtype=np.float32)

    tau_pts = tau_grid.cpu().tolist() + [1.0]
    dtaus = [tau_pts[i + 1] - tau_pts[i] for i in range(len(tau_grid))]

    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D_LATENT, device=DEVICE, generator=g)
        v = sample_velocity_gpu(x0, state, generator=g)
        v_img = v.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        x0_img = x0.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)

        t_cond = torch.zeros(nb, device=DEVICE)
        for i, tau_val in enumerate(tau_grid.cpu().tolist()):
            tau = torch.full((nb,), tau_val, device=DEVICE)
            v_img = v_img + dtaus[i] * model(v_img, x0_img, tau, t_cond)

        x1_latent_flat = (x0_img + v_img).reshape(nb, D_LATENT).cpu()
        dec = decode_latents(vae, x1_latent_flat, bs=8)
        out[start:start + nb] = dec

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
    p.add_argument("--tag", type=str, default="n")
    p.add_argument("--n-data", type=int, default=28000)
    p.add_argument("--K", type=int, default=5000)
    p.add_argument("--rank", type=int, default=80)
    p.add_argument("--cov-nit", type=int, default=1500)
    p.add_argument("--train-steps", type=int, default=800000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--unet-base", type=int, default=200)
    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--tau-L", type=int, default=20)
    p.add_argument("--tau-power", type=float, default=2.0)
    p.add_argument("--n-gen", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_data = 500
        args.K = 200
        args.rank = 20
        args.cov_nit = 100
        args.train_steps = 500
        args.bs = 32
        args.n_gen = 500
        args.unet_base = 96
        args.warmup_steps = 50

    out_dir = f"celebahq_dcae_planN_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  cfg: K={args.K}, rank={args.rank}, train_steps={args.train_steps}, "
          f"bs={args.bs}, n_gen={args.n_gen}, unet_base={args.unet_base}",
          flush=True)
    print(f"  τ: L={args.tau_L}, power={args.tau_power}", flush=True)

    tau_grid = make_tau_grid(L=args.tau_L, power=args.tau_power).to(DEVICE)

    print(f"\nLoading DC-AE from {DCAE_PATH}...", flush=True)
    vae = AutoencoderDC.from_pretrained(DCAE_PATH).to(DEVICE)
    vae.eval()

    print(f"\nLoading CelebA-HQ (n={args.n_data})...", flush=True)
    imgs_bchw = load_celebahq(args.n_data, target_res=256)
    print(f"  CelebA-HQ: {tuple(imgs_bchw.shape)}", flush=True)

    print("\n===== STAGE 0: encode CelebA-HQ -> 2048-d latents =====", flush=True)
    latents = encode_to_latents(vae, imgs_bchw, bs=8)
    print(f"  latents: {latents.shape}, mean={latents.mean():.3f}, "
          f"std={latents.std():.3f}", flush=True)

    print("\nInception features (10k real)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    real_feats = feats_from_bchw(inc, imgs_bchw[:10000], bs=32)
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # Free pixel data after features
    del imgs_bchw
    torch.cuda.empty_cache()

    # ---- Stage I ----
    print(f"\n===== STAGE I: EMS K={args.K} + low-rank r={args.rank} =====", flush=True)
    rg = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(latents, args.K, lam=0.5, nit=100, rg=rg)
    print(f"  EMS done in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    L_all, sigma2 = learn_lowrank_cov_fast(
        latents, centers, resp, rank=args.rank, nit=args.cov_nit,
        data_batch=1024, s2_floor=0.001)
    print(f"  cov done in {time.time()-t1:.0f}s, sigma2={sigma2:.6f}", flush=True)

    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(out_dir, f"gmm_k{args.K}.pt"))
    gmm_state = GMMState(gmm)

    # ---- Stage II reference ----
    print("\n===== STAGE II: closed-form -> decode -> FID =====", flush=True)
    t0 = time.time()
    g_gen = torch.Generator(device=DEVICE)
    g_gen.manual_seed(99)
    n_s2 = min(args.n_gen, 5000)
    stage2_imgs = np.zeros((n_s2, 256, 256, 3), dtype=np.float32)
    gen_bs = 128
    for start in range(0, n_s2, gen_bs):
        nb = min(gen_bs, n_s2 - start)
        x0 = torch.randn(nb, D_LATENT, device=DEVICE, generator=g_gen)
        v = sample_velocity_gpu(x0, gmm_state, generator=g_gen)
        x1_lat_flat = (x0 + v).cpu()
        stage2_imgs[start:start + nb] = decode_latents(vae, x1_lat_flat, bs=8)
    print(f"  Stage II {n_s2} in {time.time()-t0:.0f}s", flush=True)
    s2_bchw = torch.tensor(stage2_imgs.transpose(0, 3, 1, 2))
    stage2_feats = feats_from_bchw(inc, s2_bchw)
    stage2_fid_5k = compute_fid(feat_real_5k, stage2_feats[:5000])
    print(f"  Stage II FID_5k={stage2_fid_5k:.3f}", flush=True)
    save_grid(
        stage2_imgs[:100],
        os.path.join(out_dir, f"stage2_latent_{args.tag}.png"),
        f"Stage II FID5k={stage2_fid_5k:.1f}")
    del s2_bchw, stage2_imgs

    # ---- Stage III ----
    print(f"\n===== STAGE III: train correction net "
          f"({args.train_steps} steps) =====", flush=True)
    model, ema = train_stage3(
        latents, gmm_state, tau_grid, out_dir,
        n_iter=args.train_steps, bs=args.bs,
        base_lr=args.base_lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, unet_base=args.unet_base)

    print("\n===== INFERENCE + FID =====", flush=True)
    ema_model = LatentCorrectionUNet(base=args.unet_base).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    results_csv = os.path.join(out_dir, f"planN_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["method", "nfe", "fid_5k", "fid_10k"])
        csv.writer(f).writerow(
            ["StageII_1step", 0, f"{stage2_fid_5k:.4f}", "NA"])

    t0 = time.time()
    imgs = generate_corrected(
        gmm_state, ema_model, vae, n=args.n_gen, tau_grid=tau_grid,
        batch_size=128, seed=99)
    print(f"  [EMA] L={args.tau_L} gen {args.n_gen} in "
          f"{time.time()-t0:.0f}s", flush=True)
    g_bchw = torch.tensor(imgs.transpose(0, 3, 1, 2))
    g_feats = feats_from_bchw(inc, g_bchw)
    f5 = compute_fid(feat_real_5k, g_feats[:5000])
    f10 = compute_fid(feat_real_10k, g_feats[:10000])
    print(f"    FID 5k={f5:.3f}  10k={f10:.3f}", flush=True)
    with open(results_csv, "a", newline="") as f:
        csv.writer(f).writerow(
            [f"StageIII_EMA_L{args.tau_L}", args.tau_L,
             f"{f5:.4f}", f"{f10:.4f}"])
    save_grid(
        imgs[:100],
        os.path.join(out_dir, f"stage3_ema_L{args.tau_L}_{args.tag}.png"),
        f"EMA L={args.tau_L}  FID10k={f10:.1f}")

    print("\n" + "=" * 70)
    print(f"  Plan N ({args.tag}) — CelebA-HQ DC-AE")
    print(f"  DC-AE 256 recon floor: 2.49")
    print(f"  Stage II: FID_5k={stage2_fid_5k:.3f}")
    print(f"  Stage III EMA (NFE={args.tau_L}): FID_5k={f5:.3f}  FID_10k={f10:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
