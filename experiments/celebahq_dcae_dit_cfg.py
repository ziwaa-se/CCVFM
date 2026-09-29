#!/usr/bin/env python3
"""Plan Q-CFG: Plan Q-big retrain with classifier-free guidance.

Architecture change vs Plan Q-big:
  - DiTCorrectionCFG subclass adds a learnable cluster embedding
    (K+1 entries; index K is the "null" / unconditional sentinel).
  - The cluster embedding is added to the existing (tau, t) AdaLN conditioning.

Training change:
  - For each batch, the posterior-coupling step already picks a dominant
    GMM component b* in [0, K). With probability `null_prob = 0.1`, replace
    b* with the null index K so the same network learns both the
    cluster-conditional and unconditional velocity fields.

Inference change:
  - generate_corrected_dit_cfg performs two forward passes per inner step
    (cluster=b* and cluster=null) and combines them via the standard CFG
    formula: v_cfg = (1 + w) * v_cond - w * v_uncond.
  - The in-job final eval sweeps guidance scales w in {0, 1.0, 1.5, 2.0}
    at L=50 and reports FID at 5k/10k/30k against CelebA-HQ.

Everything else (Stage I EMS coreset, GMM lift, Stage II sampler, Stage III
training schedule, U-Net dimensions) is identical to Plan Q-big.
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

import celebahq_dcae_unet as pN  # noqa: F401
from celebahq_dcae_unet import (  # noqa: E402
    DCAE_PATH,
    DEVICE,
    D_LATENT,
    LATENT_C,
    LATENT_SPATIAL,
    IncFeatRGB,
    decode_latents,
    encode_to_latents,
    feats_from_bchw,
    load_celebahq,
    make_tau_grid,
    save_grid,
)
from celebahq_dcae_dit import DiTCorrection  # noqa: E402
from cifar10_sdvae_latent_ccvfm import EMA, cosine_lr  # noqa: E402
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
from diffusers import AutoencoderDC  # noqa: E402

DCAE_SCALE = 0.3189


# ===================================================================
# DiT-L correction with a cluster embedding (CFG-ready)
# ===================================================================

class DiTCorrectionCFG(DiTCorrection):
    """DiTCorrection + cluster-index conditioning for classifier-free guidance.

    The cluster embedding is summed into the AdaLN conditioning vector c
    alongside tau_embed(tau) and t_embed(t). Index `num_clusters` is the
    "null" / unconditional sentinel and is initialized to zero so the model
    starts in a state equivalent to the base unconditional DiTCorrection.
    """

    def __init__(self, *args, num_clusters: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        hidden_size = self.tau_embed.mlp[0].in_features \
            if hasattr(self.tau_embed, "mlp") else self.tau_embed.linear_1.out_features
        # Try a few common attribute names for hidden_size
        try:
            hidden_size = self.tau_embed.mlp[-1].out_features
        except Exception:
            hidden_size = self.tau_embed.linear_2.out_features
        self.cluster_embed = nn.Embedding(num_clusters + 1, hidden_size)
        nn.init.normal_(self.cluster_embed.weight, std=0.02)
        # Null index = exactly zero so the model starts at the unconditional baseline.
        with torch.no_grad():
            self.cluster_embed.weight[num_clusters].zero_()

    def forward(
        self, v_tau: torch.Tensor, x0: torch.Tensor,
        tau: torch.Tensor, t: torch.Tensor,
        cluster_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([v_tau, x0], dim=1)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        c = (self.tau_embed(tau)
             + self.t_embed(t)
             + self.cluster_embed(cluster_idx))
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final(x, c)
        return self.unpatchify(x)


# ===================================================================
# Stage III training with cluster conditioning + null dropout
# ===================================================================

def train_stage3_dit_cfg(
    latents_np, state, out_dir, num_clusters,
    n_iter=400000, bs=128, base_lr=2e-4, min_lr=2e-6,
    warmup_steps=8000, ema_decay=0.9999, null_prob=0.10,
    log_every=500, ckpt_every=20000,
    hidden_size=1024, depth=24, num_heads=16,
):
    model = DiTCorrectionCFG(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=hidden_size, depth=depth, num_heads=num_heads,
        patch_size=1, input_size=LATENT_SPATIAL,
        num_clusters=num_clusters,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DiT-L/CFG params: {n_params/1e6:.2f}M (incl. cluster embedding)",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0)
    ema = EMA(model, decay=ema_decay)

    x_all = torch.tensor(latents_np, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    raw_ckpt = os.path.join(out_dir, "dit_corr.pt")
    ema_ckpt = os.path.join(out_dir, "dit_corr_ema.pt")
    loss_csv = os.path.join(out_dir, "train_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "lr", "wall_s"])

    g = torch.Generator(device=DEVICE)
    g.manual_seed(1337)
    t_start = time.time()
    model.train()

    for step in range(n_iter):
        lr = cosine_lr(step, n_iter, base_lr, min_lr, warmup_steps)
        for pg_opt in opt.param_groups:
            pg_opt["lr"] = lr

        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D_LATENT, device=DEVICE, generator=g)
        v_true = x1 - x0

        with torch.no_grad():
            # coupled_sample_gpu draws v_0 ~ pisur via posterior coupling
            # given v_true and x0; we also need the dominant component b*.
            v_0, b_star = _coupled_sample_with_cluster(v_true, x0, state, generator=g)

        # 10% null dropout: replace cluster index with the null sentinel.
        null_mask = torch.rand(bs, device=DEVICE, generator=g) < null_prob
        cluster_idx = torch.where(null_mask, num_clusters, b_star.long())

        tau = torch.rand(bs, device=DEVICE, generator=g)
        v_tau = (1.0 - tau.view(-1, 1)) * v_0 + tau.view(-1, 1) * v_true
        target = v_true - v_0
        t_cond = torch.zeros(bs, device=DEVICE)

        v_tau_img = v_tau.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        x0_img = x0.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        target_img = target.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)

        pred = model(v_tau_img, x0_img, tau, t_cond, cluster_idx)
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
            tmp_raw = raw_ckpt + ".tmp"
            torch.save(model.state_dict(), tmp_raw, pickle_protocol=5)
            os.replace(tmp_raw, raw_ckpt)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema


def _coupled_sample_with_cluster(v_true, x0, state, generator):
    """Reproduce coupled_sample_gpu's posterior coupling but also return b*.

    coupled_sample_gpu returns only v_0; we need the cluster index too. We
    re-implement the same logic locally using state's GMM responsibilities.
    """
    # Compute posterior responsibilities r_b(v_true, x0, t=0).
    # At t=0, the surrogate pi-tilde(v|x0,0) has component k centered at
    # mu_k - x0 with covariance Sigma_k and weight w_k.
    bs = v_true.shape[0]
    K = state.K
    d = state.d
    # Means: m_k = mu_k - x0 -> shape (B, K, d)
    mu = state.means  # (K, d)
    x0_exp = x0.unsqueeze(1)  # (B, 1, d)
    mk = mu.unsqueeze(0) - x0_exp  # (B, K, d)
    diff = v_true.unsqueeze(1) - mk  # (B, K, d)
    L = state.L  # (K, d, r)
    s2 = state.s2

    # log N(diff; 0, L L^T + s2 I) using Woodbury identity:
    #   logdet = logdet(M) + d log(s2),  M = I_r + L^T L / s2
    # quadratic = ||diff||^2/s2 - diff^T L M^{-1} L^T diff / s2^2
    eye_r = torch.eye(L.shape[2], device=DEVICE)
    # Compute per-component log-likelihood
    LtL = torch.einsum("kdr,kds->krs", L, L)  # (K, r, r)
    M = eye_r.unsqueeze(0) + LtL / s2  # (K, r, r)
    M_inv = torch.linalg.inv(M)  # (K, r, r)
    ld = torch.logdet(M) + d * math.log(s2)  # (K,)
    diff_L = torch.einsum("bkd,kdr->bkr", diff, L)  # (B, K, r)
    diff_L_Mi = torch.einsum("bkr,krs->bks", diff_L, M_inv)  # (B, K, r)
    q1 = (diff * diff).sum(-1) / s2  # (B, K)
    q2 = (diff_L_Mi * diff_L).sum(-1) / s2 / s2  # (B, K)
    log_lik = -0.5 * (ld.unsqueeze(0) + q1 - q2)  # (B, K)
    log_post = log_lik + state.log_weights.unsqueeze(0)  # (B, K)

    # Sample b* ~ Cat(softmax(log_post))
    log_post = log_post - log_post.logsumexp(-1, keepdim=True)
    gumbel = -torch.log(-torch.log(
        torch.rand(bs, K, device=DEVICE, generator=generator) + 1e-30) + 1e-30)
    b_star = (log_post + gumbel).argmax(-1)  # (B,)

    # Sample v_0 ~ N(m_{b*}, L_{b*} L_{b*}^T + s2 I)
    L_b = L[b_star]  # (B, d, r)
    eps_r = torch.randn(bs, L.shape[2], device=DEVICE, generator=generator)
    eps_d = torch.randn(bs, d, device=DEVICE, generator=generator)
    v_0 = mk[torch.arange(bs), b_star] + torch.einsum("bdr,br->bd", L_b, eps_r) \
        + math.sqrt(s2) * eps_d

    return v_0, b_star


# ===================================================================
# CFG inference
# ===================================================================

@torch.no_grad()
def generate_corrected_dit_cfg(state, model, vae, n, tau_grid,
                               guidance_scale=1.5, batch_size=128, seed=99):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    out = np.zeros((n, 256, 256, 3), dtype=np.float32)
    K = state.K
    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D_LATENT, device=DEVICE, generator=g)
        # Stage II sample: pick b* by Cat(weights) then sample from component b*.
        b_star = torch.multinomial(
            state.weights.expand(nb, -1) if state.weights.dim() == 1
            else state.weights, 1, generator=g).squeeze(-1)
        # Build initial v from component b*
        L_b = state.L[b_star]
        eps_r = torch.randn(nb, state.L.shape[2], device=DEVICE, generator=g)
        eps_d = torch.randn(nb, state.d, device=DEVICE, generator=g)
        v = (state.means[b_star] - x0
             + torch.einsum("bdr,br->bd", L_b, eps_r)
             + math.sqrt(state.s2) * eps_d)
        x0_img = x0.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        cluster_cond = b_star.long()
        cluster_null = torch.full_like(cluster_cond, K)
        L_steps = len(tau_grid) - 1
        for li in range(L_steps):
            tau = tau_grid[li].expand(nb)
            dtau = tau_grid[li + 1] - tau_grid[li]
            v_img = v.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
            t_z = torch.zeros(nb, device=DEVICE)
            v_cond = model(v_img, x0_img, tau, t_z, cluster_cond)
            if guidance_scale > 0:
                v_unc = model(v_img, x0_img, tau, t_z, cluster_null)
                v_pred = (1.0 + guidance_scale) * v_cond - guidance_scale * v_unc
            else:
                v_pred = v_cond
            v = v + dtau * v_pred.reshape(nb, -1)
        x1_lat_flat = (x0 + v).cpu()
        out[start:start + nb] = decode_latents(vae, x1_lat_flat, bs=8)
    return out


# ===================================================================
# Entry point
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="qcfg")
    p.add_argument("--n-data", type=int, default=28000)
    p.add_argument("--K", type=int, default=10000)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--cov-nit", type=int, default=2000)
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=8000)
    p.add_argument("--null-prob", type=float, default=0.10)
    p.add_argument("--tau-L", type=int, default=50)
    p.add_argument("--tau-power", type=float, default=2.0)
    p.add_argument("--n-gen", type=int, default=10000)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_data = 500; args.K = 200; args.rank = 20; args.cov_nit = 100
        args.train_steps = 500; args.bs = 16; args.n_gen = 100
        args.hidden_size = 192; args.depth = 4; args.num_heads = 6
        args.warmup_steps = 50

    out_dir = f"celebahq_dcae_planO_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  cfg: K={args.K}, rank={args.rank}, train_steps={args.train_steps}, "
          f"bs={args.bs}, null_prob={args.null_prob}", flush=True)
    print(f"  DiT-L: hidden={args.hidden_size}, depth={args.depth}, "
          f"heads={args.num_heads}", flush=True)

    tau_grid = make_tau_grid(L=args.tau_L, power=args.tau_power).to(DEVICE)

    print(f"\nLoading DC-AE from {DCAE_PATH}...", flush=True)
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
            latents_flip[i:i + bs_flip] = lat.reshape(lat.shape[0], -1).cpu().numpy()
        latents = np.concatenate([latents, latents_flip], axis=0)
        print(f"  latents after flip aug: {latents.shape}", flush=True)

    print("\nInception features (10k real)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    real_feats = feats_from_bchw(inc, imgs_bchw[:10000], bs=32)
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats
    del imgs_bchw
    torch.cuda.empty_cache()

    # ---- Stage I ----
    print(f"\n===== STAGE I: EMS K={args.K} + low-rank r={args.rank} =====",
          flush=True)
    rg = np.random.default_rng(42); torch.manual_seed(42)
    centers, weights, resp = ems_coreset_gpu(
        latents, args.K, lam=0.5, nit=100, rg=rg)
    L_all, sigma2 = learn_lowrank_cov_fast(
        latents, centers, resp, rank=args.rank, nit=args.cov_nit,
        data_batch=256, s2_floor=0.001)
    print(f"  cov done, sigma2={sigma2:.6f}", flush=True)

    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(out_dir, f"gmm_k{args.K}.pt"))
    gmm_state = GMMState(gmm)

    # ---- Stage III ----
    print(f"\n===== STAGE III: train DiT-L/CFG ({args.train_steps} steps) =====",
          flush=True)
    model, ema = train_stage3_dit_cfg(
        latents, gmm_state, out_dir, args.K,
        n_iter=args.train_steps, bs=args.bs,
        base_lr=args.base_lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, null_prob=args.null_prob,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads)

    print("\n===== INFERENCE + FID (CFG sweep) =====", flush=True)
    ema_model = DiTCorrectionCFG(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads, patch_size=1, input_size=LATENT_SPATIAL,
        num_clusters=args.K,
    ).to(DEVICE)
    ema.copy_to(ema_model); ema_model.eval()

    results_csv = os.path.join(out_dir, f"planQcfg_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["method", "guidance_scale", "nfe",
                                "fid_5k", "fid_10k"])

    for w in [0.0, 1.0, 1.5, 2.0]:
        nfe = args.tau_L * (2 if w > 0 else 1)
        t0 = time.time()
        imgs = generate_corrected_dit_cfg(
            gmm_state, ema_model, vae, n=args.n_gen, tau_grid=tau_grid,
            guidance_scale=w, batch_size=128, seed=99)
        wall = time.time() - t0
        g_bchw = torch.tensor(imgs.transpose(0, 3, 1, 2))
        g_feats = feats_from_bchw(inc, g_bchw)
        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        print(f"  [w={w}] gen {args.n_gen} in {wall:.0f}s, "
              f"FID 5k={f5:.3f} 10k={f10:.3f} (NFE={nfe})", flush=True)
        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [f"StageIII_CFG", w, nfe, f"{f5:.4f}", f"{f10:.4f}"])
        save_grid(imgs[:100],
                  os.path.join(out_dir, f"stage3_cfg_w{w}_{args.tag}.png"),
                  f"CFG w={w} NFE={nfe} FID10k={f10:.1f}")

    print("\n" + "=" * 70)
    print(f"  Plan Q-CFG ({args.tag}) done. Results: {results_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
