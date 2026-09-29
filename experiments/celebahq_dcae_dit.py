#!/usr/bin/env python3
"""CelebA-HQ 256×256 Plan O — DC-AE latents + DiT-B/1 correction.

Differences vs Plan N-v2:
  - Correction net: DiT-B/1 (hidden=768, depth=12, heads=12, patch=1 on 8×8)
    with AdaLN-Zero conditioning on (τ, t). ~130M params.
  - Training τ signal: random τ ~ U[0, 1] per sample (continuous),
    instead of discrete draws from a fixed 20-pt grid.
  - Coreset: K=20,000, rank=160 (2× Plan N-v2).
  - Data: horizontal flip augmentation (28k → 56k latents).
  - Everything else (Stage I EMS, closed-form sampler, decoder, FID protocol,
    bs=128, AdamW cosine LR, 1.5M steps) matches Plan N-v2.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing planN triggers the latent-shape monkey-patch (32ch × 8×8 = 2048-d)
# on cifar10_sd_latent_planG, and exposes helpers we reuse.
import celebahq_dcae_unet as pN
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


# ===================================================================
# DiT-B/1 correction transformer
# ===================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Standard DiT 2D sincos positional embedding. Returns (N, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)  # (2, H, W)

    def sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
        omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = sincos_1d(embed_dim // 2, grid[0])
    emb_w = sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embed → 2-layer MLP, per DiT."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t, self.freq_dim))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SDPAAttention(nn.Module):
    """Multi-head self-attention using torch's fused SDPA (Flash Attention)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SDPAAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.adaLN(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiTCorrection(nn.Module):
    """DiT-style correction network for the latent velocity field.

    Inputs are two (B, C, H, W) tensors (v_tau and x0) concatenated on the
    channel dim before patch embedding, plus (τ, t) scalars per sample.
    Output is the predicted ∂v/∂τ field, same shape as v_tau.
    """

    def __init__(
        self,
        in_channels: int = 32,
        cond_channels: int = 32,
        out_channels: int = 32,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        input_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.out_channels = out_channels
        self.num_patches = (input_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels + cond_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        pos_np = get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_np).float().unsqueeze(0))

        self.tau_embed = TimestepEmbedder(hidden_size)
        self.t_embed = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final = FinalLayer(hidden_size, out_channels, patch_size)

        self.apply(self._init_weights)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.patch_size
        H = W = self.input_size // p
        C = self.out_channels
        x = x.reshape(B, H, W, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, C, H * p, W * p)

    def forward(
        self, v_tau: torch.Tensor, x0: torch.Tensor,
        tau: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([v_tau, x0], dim=1)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        c = self.tau_embed(tau) + self.t_embed(t)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final(x, c)
        return self.unpatchify(x)


# ===================================================================
# Stage III training (random τ ~ U[0,1])
# ===================================================================

def train_stage3_dit(
    latents_np, state, out_dir,
    n_iter=1500000, bs=128, base_lr=2e-4, min_lr=2e-6,
    warmup_steps=8000, ema_decay=0.9999,
    log_every=500, ckpt_every=20000,
    hidden_size=768, depth=12, num_heads=12,
):
    model = DiTCorrection(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=hidden_size, depth=depth, num_heads=num_heads,
        patch_size=1, input_size=LATENT_SPATIAL,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DiT-B/1 params: {n_params/1e6:.2f}M", flush=True)

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
            v_0 = coupled_sample_gpu(v_true, x0, state, generator=g)

        tau = torch.rand(bs, device=DEVICE, generator=g)

        v_tau = (1.0 - tau.view(-1, 1)) * v_0 + tau.view(-1, 1) * v_true
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
# Inference over fixed τ grid
# ===================================================================

@torch.no_grad()
def generate_corrected_dit(state, model, vae, n, tau_grid, batch_size=128, seed=99):
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


# ===================================================================
# Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", type=str, default="o")
    p.add_argument("--n-data", type=int, default=28000)
    p.add_argument("--K", type=int, default=20000)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--cov-nit", type=int, default=2000)
    p.add_argument("--train-steps", type=int, default=1500000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=8000)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--tau-L", type=int, default=20)
    p.add_argument("--tau-power", type=float, default=2.0)
    p.add_argument("--n-gen", type=int, default=10000)
    p.add_argument("--no-flip", action="store_true",
                   help="Disable horizontal-flip augmentation (Plan N parity)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_data = 500
        args.K = 200
        args.rank = 20
        args.cov_nit = 100
        args.train_steps = 500
        args.bs = 16
        args.n_gen = 500
        args.hidden_size = 192
        args.depth = 4
        args.num_heads = 6
        args.warmup_steps = 50

    out_dir = f"celebahq_dcae_planO_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  cfg: K={args.K}, rank={args.rank}, "
          f"train_steps={args.train_steps}, bs={args.bs}, n_gen={args.n_gen}",
          flush=True)
    print(f"  DiT: hidden={args.hidden_size}, depth={args.depth}, "
          f"heads={args.num_heads}, patch=1", flush=True)
    print(f"  τ: training random U[0,1], inference L={args.tau_L} "
          f"quadratic power={args.tau_power}", flush=True)
    print(f"  flip aug: {not args.no_flip}", flush=True)

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

    if not args.no_flip:
        print("\nEncoding horizontally-flipped copies for augmentation...",
              flush=True)
        # Streaming flip: never materialize a full flipped copy on CPU.
        n_flip = len(imgs_bchw)
        latents_flip = np.zeros((n_flip, D_LATENT), dtype=np.float32)
        bs_flip = 8
        t0 = time.time()
        for i in range(0, n_flip, bs_flip):
            batch = imgs_bchw[i:i + bs_flip].to(DEVICE)
            batch = torch.flip(batch, dims=[3])
            batch = batch * 2.0 - 1.0
            with torch.no_grad():
                lat = vae.encode(batch).latent * 0.3189
            latents_flip[i:i + bs_flip] = lat.reshape(lat.shape[0], -1).cpu().numpy()
        print(f"  encoded {n_flip} flipped latents in {time.time()-t0:.0f}s",
              flush=True)
        latents = np.concatenate([latents, latents_flip], axis=0)
        del latents_flip
        print(f"  latents after flip aug: {latents.shape}", flush=True)

    print("\nInception features (10k real)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    real_feats = feats_from_bchw(inc, imgs_bchw[:10000], bs=32)
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats
    print(f"  {time.time()-t0:.0f}s", flush=True)

    del imgs_bchw
    torch.cuda.empty_cache()

    # ---- Stage I ----
    print(f"\n===== STAGE I: EMS K={args.K} + low-rank r={args.rank} =====",
          flush=True)
    rg = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(
        latents, args.K, lam=0.5, nit=100, rg=rg)
    print(f"  EMS done in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    # Scale data_batch down as K grows to keep the (B,K,r) activations in
    # the low-rank cov fit within GPU memory. Heuristic: 1024 works fine at
    # K=5000 r=80 (Plan P); K=10000 r=160 requires dropping to 256 or the
    # backward's grad tensor (~13 GB for L_param) cannot allocate.
    d_batch_cov = 256 if args.K * args.rank >= 1_000_000 else 1024
    L_all, sigma2 = learn_lowrank_cov_fast(
        latents, centers, resp, rank=args.rank, nit=args.cov_nit,
        data_batch=d_batch_cov, s2_floor=0.001)
    print(f"  cov done in {time.time()-t1:.0f}s, sigma2={sigma2:.6f}",
          flush=True)

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
    print(f"\n===== STAGE III: train DiT correction "
          f"({args.train_steps} steps, random τ) =====", flush=True)
    model, ema = train_stage3_dit(
        latents, gmm_state, out_dir,
        n_iter=args.train_steps, bs=args.bs,
        base_lr=args.base_lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads)

    print("\n===== INFERENCE + FID =====", flush=True)
    ema_model = DiTCorrection(
        in_channels=LATENT_C, cond_channels=LATENT_C, out_channels=LATENT_C,
        hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads, patch_size=1, input_size=LATENT_SPATIAL,
    ).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    results_csv = os.path.join(out_dir, f"planO_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["method", "nfe", "fid_5k", "fid_10k"])
        csv.writer(f).writerow(
            ["StageII_1step", 0, f"{stage2_fid_5k:.4f}", "NA"])

    t0 = time.time()
    imgs = generate_corrected_dit(
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
    print(f"  Plan O ({args.tag}) — CelebA-HQ DC-AE + DiT-B/1")
    print(f"  DC-AE 256 recon floor: 2.49")
    print(f"  Stage II: FID_5k={stage2_fid_5k:.3f}")
    print(f"  Stage III EMA (NFE={args.tau_L}): "
          f"FID_5k={f5:.3f}  FID_10k={f10:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
