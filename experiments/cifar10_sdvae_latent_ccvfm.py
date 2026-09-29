#!/usr/bin/env python3
"""CIFAR-10 Plan G — HRF2-scale CoresetFM in SD VAE latent space.

Differences from Plan F:
  * LatentCorrectionUNet scaled to ~44M params (base=216, matching HRF2's 44.8M)
  * 400k training steps, cosine LR schedule (2e-4 → 2e-6)
  * Two surrogate modes selectable via --source:
      - "gmm"    : K=2000 low-rank r=50 GMM (same as Plan F)
      - "atomic" : near-delta mixture (rank=0, small fixed sigma²)
  * Output dir suffix from --tag so G1 / G2 don't overwrite each other

Launch:
  (see slurm/ for job templates)
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
from diffusers import AutoencoderKL
from torchvision import datasets
from torchvision.models import Inception_V3_Weights, inception_v3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

VAE_PATH = os.environ.get("CCVFM_SD_VAE_DIR", "stabilityai/sd-vae-ft-mse")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAE_SCALE = 0.18215
LATENT_RES = 128
LATENT_SPATIAL = LATENT_RES // 8   # 16
LATENT_C = 4
D_LATENT = LATENT_C * LATENT_SPATIAL * LATENT_SPATIAL  # 1024


# ===================================================================
# Inception FID helper
# ===================================================================

class IncFeatRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.m.eval()
        self.m.fc = nn.Identity()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, 299, mode="bilinear", align_corners=False)
        mu = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return self.m((x - mu) / sd)


# ===================================================================
# Data / encode / decode
# ===================================================================

def load_cifar10(data_dir: str = ".data") -> np.ndarray:
    os.makedirs(data_dir, exist_ok=True)
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True)
    return np.array(train_ds.data, dtype=np.float32) / 255.0


@torch.no_grad()
def encode_cifar_to_latents(vae, imgs_hwc, bs=64):
    vae.eval()
    n = len(imgs_hwc)
    out = np.zeros((n, D_LATENT), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, bs):
        x = torch.tensor(imgs_hwc[i:i+bs].transpose(0, 3, 1, 2),
                          dtype=torch.float32, device=DEVICE)
        x_up = F.interpolate(x, size=(LATENT_RES, LATENT_RES), mode="bicubic",
                              align_corners=False)
        x_up = x_up * 2.0 - 1.0
        lat = vae.encode(x_up).latent_dist.mean * VAE_SCALE
        out[i:i+bs] = lat.reshape(lat.shape[0], -1).cpu().numpy()
    print(f"  encoded {n} latents in {time.time()-t0:.0f}s", flush=True)
    return out


@torch.no_grad()
def decode_latents_to_32(vae, latents_flat, bs=64):
    vae.eval()
    n = latents_flat.shape[0]
    out = np.zeros((n, 32, 32, 3), dtype=np.float32)
    for i in range(0, n, bs):
        lat = latents_flat[i:i+bs].to(DEVICE).reshape(
            -1, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        lat = lat / VAE_SCALE
        x = vae.decode(lat).sample
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(32, 32), mode="bicubic",
                           align_corners=False).clamp(0, 1)
        out[i:i+bs] = x.permute(0, 2, 3, 1).cpu().numpy()
    return out


@torch.no_grad()
def feats_from_hwc(inc, imgs_hwc, bs=64):
    chw = imgs_hwc.transpose(0, 3, 1, 2)
    fs = []
    for i in range(0, len(chw), bs):
        batch = torch.tensor(chw[i:i+bs], dtype=torch.float32, device=DEVICE)
        fs.append(inc(batch).cpu().numpy())
    return np.concatenate(fs)


# ===================================================================
# Local kNN-PCA surrogate (Plan H: tangent-local LowRankGMM)
# ===================================================================

def learn_local_knn_cov(
    x_np: np.ndarray,
    mu: np.ndarray,
    rank: int,
    k_neighbors: int = 100,
    sigma2: float = 0.005,
    atom_batch: int = 64,
):
    """Hard-kNN local PCA per atom — one-shot replacement for learn_lowrank_cov_fast.

    For each atom k:
      1. find its k_neighbors nearest data points (E-step, hard assignment)
      2. weighted PCA on those centered neighbors -> rank-r tangent basis  (M-step)
      3. L_k = V_r · diag(S_r) / sqrt(k_neighbors)

    This is the hard-assignment limit of locally-weighted EM: responsibilities
    are clamped to 1 for the k_neighbors closest points and 0 elsewhere, so each
    component's covariance is constrained to a small tangent neighborhood.
    """
    n, d = x_np.shape
    K = mu.shape[0]

    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)        # (n, d)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)       # (K, d)

    L_all = torch.zeros(K, d, rank, device=DEVICE)
    x_sq = (x * x).sum(-1)                                             # (n,)

    t0 = time.time()
    for k_start in range(0, K, atom_batch):
        k_end = min(k_start + atom_batch, K)
        mu_batch = mu_t[k_start:k_end]                                 # (kb, d)
        m_sq = (mu_batch * mu_batch).sum(-1, keepdim=True)             # (kb, 1)
        xm = mu_batch @ x.T                                            # (kb, n)
        d2 = x_sq.unsqueeze(0) + m_sq - 2 * xm                         # (kb, n)

        _, nn_idx = torch.topk(d2, k_neighbors, largest=False, dim=1)  # (kb, k_nbrs)

        for i, k in enumerate(range(k_start, k_end)):
            nb = x[nn_idx[i]]                                          # (k_nbrs, d)
            diff = nb - mu_t[k]                                        # (k_nbrs, d)
            _, S_lr, V_lr = torch.svd_lowrank(diff, q=rank, niter=2)
            L_all[k] = V_lr * S_lr.unsqueeze(0) / math.sqrt(k_neighbors)

    print(f"  local kNN-PCA done in {time.time()-t0:.1f}s  "
          f"(K={K}, k_nbrs={k_neighbors}, rank={rank})", flush=True)
    return L_all.cpu().numpy(), sigma2


# ===================================================================
# Atomic surrogate (rank-0 near-delta limit of Corollary 1)
# ===================================================================

def build_atomic_gmm(centers: np.ndarray, weights: np.ndarray,
                     sigma2: float = 0.005) -> LowRankGMM:
    """Build a degenerate LowRankGMM with r=1 near-zero factors + small isotropic
    noise. This is the σ² → 0 limit of the Corollary 1 velocity distribution,
    which becomes a discrete mixture of atomic velocities m_k - x_0 with tiny
    smoothing. Closed-form sampling via the existing GMMState path.

    Use rank=1 with zeros (not rank=0) because GMMState requires r≥1 for the
    einsum shapes; a tiny L contributes 0 cov up to numerics.
    """
    K, d = centers.shape
    L = np.zeros((K, d, 1), dtype=np.float32)
    return LowRankGMM(weights, centers, L, sigma2)


# ===================================================================
# General-t velocity samplers (Prop 4.2: π̃(v|x_t, t) is a GMM at any t)
# ===================================================================

@torch.no_grad()
def sample_velocity_general_t(x_t, t_scalar, state: GMMState, generator=None):
    """Sample v ~ π̃(v|x_t, t) at general t ∈ (0, 1).

    At general t, the marginal x_t|k ~ N(t·μ_k, Σ_{t,k}) where
    Σ_{t,k} = s²(t)I + t²L_kL_k^T  with  s²(t) = (1-t)² + t²σ².

    Steps:
      1. Posterior weights p(k|x_t,t) ∝ w_k · N(x_t; t·μ_k, Σ_{t,k})
      2. Sample component k*
      3. Sample x₁ ~ p(x₁|x_t,t,k*) via Gaussian regression
      4. Return v = (x₁ - x_t) / (1-t)
    """
    if t_scalar < 1e-6:
        return sample_velocity_gpu(x_t, state, generator=generator)

    B, d = x_t.shape
    K, r = state.K, state.r
    t = t_scalar
    s2_t = (1 - t) ** 2 + t ** 2 * state.s2

    # --- Posterior weights: log N(x_t; t·μ_k, s²_t I + t²LL^T) ---
    diff = x_t.unsqueeze(1) - t * state.means.unsqueeze(0)  # (B, K, d)

    q1 = (diff * diff).sum(-1) / s2_t  # (B, K)

    dL = torch.einsum("bkd,kdr->bkr", diff, state.L)  # (B, K, r)
    eye_r = torch.eye(r, device=x_t.device)
    t2_s2t = t ** 2 / s2_t
    M_t = eye_r.unsqueeze(0) + t2_s2t * torch.einsum("kdr,kds->krs", state.L, state.L)
    M_t_inv = torch.linalg.inv(M_t)  # (K, r, r)

    dL_Mi = torch.einsum("bkr,krs->bks", dL, M_t_inv)
    q2 = (dL_Mi * dL).sum(-1) * t2_s2t / s2_t  # (B, K)

    log_det = torch.logdet(M_t) + d * math.log(s2_t)  # (K,)

    log_resp = state.log_weights.unsqueeze(0) - 0.5 * (log_det.unsqueeze(0) + q1 - q2)
    log_resp = log_resp - log_resp.max(dim=-1, keepdim=True).values
    resp = torch.softmax(log_resp, dim=-1)  # (B, K)

    comp = torch.multinomial(resp, 1, generator=generator).squeeze(-1)  # (B,)

    # --- Sample x₁ from p(x₁|x_t, t, k*) ---
    # Precision of x₁|x_t,t,k:  Λ = Σ_k⁻¹ + t²/(1-t)² I
    # For low-rank Σ_k = LL^T + σ²I, use Woodbury to sample.
    # Instead of inverting Λ, sample via:
    #   x₁ = μ_k + Σ_k·(Σ_{t,k})⁻¹ · t · (x_t - t·μ_k)  +  noise
    # where noise ~ N(0, Σ_k - t²·Σ_k·(Σ_{t,k})⁻¹·Σ_k)

    sel_mean = state.means[comp]  # (B, d)
    sel_L = state.L[comp]  # (B, d, r)
    sel_diff = x_t - t * sel_mean  # (B, d)

    # Compute Σ_{t,k}⁻¹ · sel_diff for the selected components
    # Σ_{t,k}⁻¹ = (1/s²_t)(I - t²L M_t⁻¹ L^T/s²_t)
    sel_M_t_inv = M_t_inv[comp]  # (B, r, r)
    Ld = torch.einsum("bdr,bd->br", sel_L, sel_diff)  # (B, r)
    MLd = torch.einsum("brs,bs->br", sel_M_t_inv, Ld)  # (B, r)
    Sigma_t_inv_diff = sel_diff / s2_t - t2_s2t / s2_t * torch.einsum("bdr,br->bd", sel_L, MLd)

    # Conditional mean: μ₁ = μ_k + t · Σ_k · Σ_{t,k}⁻¹ · (x_t - t·μ_k)
    # Σ_k · Σ_{t,k}⁻¹ · diff = (LL^T + σ²I) · Sigma_t_inv_diff
    # = L·(L^T · Sigma_t_inv_diff) + σ² · Sigma_t_inv_diff
    Lt_Sinv_d = torch.einsum("bdr,bd->br", sel_L, Sigma_t_inv_diff)  # (B, r)
    Sigma_k_Sinv_diff = torch.einsum("bdr,br->bd", sel_L, Lt_Sinv_d) + state.s2 * Sigma_t_inv_diff
    mu1_cond = sel_mean + t * Sigma_k_Sinv_diff  # (B, d)

    # Conditional covariance: Σ₁ = Σ_k - t²·Σ_k·Σ_{t,k}⁻¹·Σ_k
    # Sampling: x₁ = μ₁ + Σ₁^{1/2} · z
    # For the noise, use the identity:
    #   Σ₁ = (1-t)² · Σ_k · Σ_{t,k}⁻¹ · I  ... not quite.
    # Simpler: Σ₁ = Σ_k - t²·Σ_k·Σ_{t,k}⁻¹·Σ_k
    # Factor as: Σ₁ = L₁L₁^T + σ₁²I  (still low-rank + diagonal)
    # σ₁² = σ² - t²σ²·(σ²/s²_t - σ²t²/(s²_t)² · ...) → messy
    #
    # Practical shortcut: sample z ~ N(0, I), compute:
    #   noise_d = σ_cond · z_d
    #   noise_r = L_cond · z_r
    # where σ_cond² = σ²(1 - t²σ²/s²_t) = σ²(1-t)²/s²_t (diagonal part)
    # and L_cond captures the remaining low-rank part.

    sigma2_cond_diag = state.s2 * (1 - t) ** 2 / s2_t  # scalar

    # Low-rank conditional factor (approximate but correct for diagonal part):
    # The full Σ₁ low-rank part is complex; for the diagonal contribution,
    # σ²_cond is exact. The remaining low-rank noise contribution:
    # Σ₁ - σ²_cond·I = (LL^T)(1 - t²/s²_t·(LL^T + σ²I)·??) → very messy
    #
    # Clean approach: use the fact that Var(x₁|x_t,t,k) can be decomposed as
    # x₁ = μ₁_cond + (1-t)/√(s²_t) · [σ·ε_d + contribution from L cancelled]
    #
    # Simplest correct sampling: note that x₁ = μ_k + L·ε_r + σ·ε_d  (prior)
    # and x_t = t·(μ_k + L·ε_r + σ·ε_d) + (1-t)·ε₀ (with ε₀ ~ N(0,I))
    # So (ε₀, ε_r, ε_d) are jointly standard normal, constrained by x_t.
    #
    # ε₀ = (x_t - t·μ_k - t·L·ε_r - t·σ·ε_d) / (1-t)
    # Given x_t, the conditional on (ε_r, ε_d) is Gaussian.
    # Then x₁ = μ_k + L·ε_r + σ·ε_d.
    #
    # Let w = (ε_r, ε_d) ∈ R^{r+d}, w ~ N(0, I).
    # x_t = (1-t)·ε₀ + t·μ_k + A·w  where A = [t·L | t·σ·I_d] (d × (r+d))
    # ε₀ = (x_t - t·μ_k - A·w)/(1-t)
    # p(ε₀) ∝ exp(-||ε₀||²/2) = exp(-||(x_t-t·μ_k-A·w)/(1-t)||²/2)
    #
    # So w | x_t ~ N(μ_w, Σ_w) where:
    # Σ_w⁻¹ = I + A^T A / (1-t)²
    # μ_w = Σ_w · A^T · (x_t - t·μ_k) / (1-t)²
    #
    # Then x₁ = μ_k + [L | σI] · w = μ_k + B·w  where B = [L | σI] (d × (r+d))
    # x₁ | x_t = μ_k + B·μ_w + B·Σ_w^{1/2}·z
    #
    # A^T A = [t²L^TL, t²σL^T; t²σL, t²σ²I_d]
    # Σ_w⁻¹ = I_{r+d} + A^T A/(1-t)²

    # This is clean but (r+d) × (r+d) matrix inversion. For d=1024, r=120,
    # that's 1144×1144 per sample. Expensive but doable.
    #
    # ALTERNATIVELY: sample ε_r from its marginal, then ε_d from its conditional.
    # Marginal of ε_r | x_t:
    #   marginalize ε_d in the constraint. This gives a Gaussian in ε_r.
    #
    # After marginalizing ε_d:
    # x_t - t·μ_k = (1-t)ε₀ + t·L·ε_r + t·σ·ε_d
    # (1-t)ε₀ + t·σ·ε_d ~ N(0, ((1-t)² + t²σ²)I) = N(0, s²_t I)
    # So marginally: x_t - t·μ_k ~ N(t·L·ε_r, s²_t I)
    # → ε_r | x_t ~ N(μ_r, Σ_r) where
    # Σ_r⁻¹ = I_r + t²L^TL/s²_t = M_t  (already computed!)
    # μ_r = M_t⁻¹ · t·L^T·(x_t - t·μ_k)/s²_t

    # Sample ε_r
    mu_r = torch.einsum("brs,bs->br", sel_M_t_inv,
                        torch.einsum("bdr,bd->br", sel_L, sel_diff) * t / s2_t)
    # Σ_r = M_t⁻¹ → need Cholesky of M_t⁻¹
    # For r ≤ 120, Cholesky of (B, r, r) is fine
    Sigma_r = sel_M_t_inv  # (B, r, r)
    L_r = torch.linalg.cholesky(Sigma_r + 1e-6 * torch.eye(r, device=x_t.device))
    z_r = torch.randn(B, r, device=x_t.device, generator=generator)
    eps_r = mu_r + torch.einsum("brs,bs->br", L_r, z_r)  # (B, r)

    # Now sample ε_d | ε_r, x_t:
    # residual = x_t - t·μ_k - t·L·ε_r = (1-t)ε₀ + t·σ·ε_d
    # (1-t)ε₀ + tσ·ε_d ~ N(0, ((1-t)²+t²σ²)I) conditioned on the residual
    # Actually: residual is GIVEN, and (1-t)ε₀ + tσε_d = residual.
    # There are infinitely many (ε₀, ε_d) satisfying this.
    # But we don't need ε_d separately — we need x₁ = μ_k + L·ε_r + σ·ε_d.
    #
    # From the constraint: ε₀ = (residual - tσε_d)/(1-t)
    # p(ε₀)·p(ε_d) ∝ exp(-||(res - tσε_d)/(1-t)||²/2 - ||ε_d||²/2)
    # This gives ε_d | residual ~ N(μ_d, σ²_d I) where:
    # σ²_d = 1/(1 + t²σ²/(1-t)²) = (1-t)²/s²_t
    # μ_d = tσ/(s²_t) · residual ... wait:
    # Precision: 1 + t²σ²/(1-t)² = s²_t/(1-t)²
    # σ²_d = (1-t)²/s²_t
    # μ_d = σ²_d · tσ/(1-t)² · residual = tσ/s²_t · residual

    residual = sel_diff - t * torch.einsum("bdr,br->bd", sel_L, eps_r)  # (B, d)
    sigma2_d = (1 - t) ** 2 / s2_t
    mu_d = t * state.sigma / s2_t * residual  # using sigma = sqrt(s2)... wait
    # Actually: tσ/s²_t · residual. σ here is the GMM noise std = sqrt(state.s2)
    mu_d = t * math.sqrt(state.s2) / s2_t * residual
    z_d = torch.randn(B, d, device=x_t.device, generator=generator)
    eps_d = mu_d + math.sqrt(sigma2_d) * z_d  # (B, d)

    # Reconstruct x₁ = μ_k + L·ε_r + σ·ε_d
    x1 = sel_mean + torch.einsum("bdr,br->bd", sel_L, eps_r) + math.sqrt(state.s2) * eps_d

    v = (x1 - x_t) / (1 - t)
    return v


@torch.no_grad()
def coupled_sample_general_t(v_true, x0, x1, x_t, t_scalar, state: GMMState, generator=None):
    """Posterior coupling at general t.

    The posterior over components given (x_t, x₁) is:
      p(k|x₁) ∝ w_k · N(x₁; μ_k, Σ_k)
    (same as t=0 because x₀ doesn't depend on k).

    Then sample v₀ from π̃(v|x_t, t, k*) — the GMM conditional at the
    selected component, evaluated at general t.
    """
    if t_scalar < 1e-6:
        return coupled_sample_gpu(v_true, x0, state, generator=generator)

    B, d = x0.shape
    K, r = state.K, state.r
    t = t_scalar

    # Posterior over k given x₁ (same as t=0):
    # p(k|x₁) ∝ w_k · N(x₁; μ_k, Σ_k)
    w_vec = x1  # x₁ directly

    w_sq = (w_vec * w_vec).sum(-1, keepdim=True)
    m_sq = (state.means * state.means).sum(-1).unsqueeze(0)
    wm = w_vec @ state.means.T
    q1 = (w_sq + m_sq - 2 * wm) / state.s2  # (B, K)

    wL = torch.einsum("bd,kdr->bkr", w_vec, state.L)
    dL = wL - state.mL.unsqueeze(0)
    dL_Mi = torch.einsum("bkr,krs->bks", dL, state.M_inv)
    q2 = (dL_Mi * dL).sum(-1) / state.s2 ** 2

    log_resp = state.log_weights.unsqueeze(0) - 0.5 * (
        state.log_det.unsqueeze(0) + q1 - q2)
    log_resp = log_resp - log_resp.max(dim=-1, keepdim=True).values
    resp = torch.softmax(log_resp, dim=-1)

    comp = torch.multinomial(resp, 1, generator=generator).squeeze(-1)

    # Sample v₀ from π̃(v|x_t, t, k*) using the same auxiliary-variable approach
    sel_mean = state.means[comp]
    sel_L = state.L[comp]
    sel_diff = x_t - t * sel_mean

    s2_t = (1 - t) ** 2 + t ** 2 * state.s2
    t2_s2t = t ** 2 / s2_t

    eye_r = torch.eye(r, device=x_t.device)
    M_t = eye_r.unsqueeze(0) + t2_s2t * torch.einsum("kdr,kds->krs", state.L, state.L)
    sel_M_t_inv = torch.linalg.inv(M_t)[comp]

    # Sample ε_r | x_t, k*
    mu_r = torch.einsum("brs,bs->br", sel_M_t_inv,
                        torch.einsum("bdr,bd->br", sel_L, sel_diff) * t / s2_t)
    L_r = torch.linalg.cholesky(sel_M_t_inv + 1e-6 * eye_r)
    z_r = torch.randn(B, r, device=x_t.device, generator=generator)
    eps_r = mu_r + torch.einsum("brs,bs->br", L_r, z_r)

    # Sample ε_d | ε_r, x_t, k*
    residual = sel_diff - t * torch.einsum("bdr,br->bd", sel_L, eps_r)
    sigma2_d = (1 - t) ** 2 / s2_t
    mu_d = t * math.sqrt(state.s2) / s2_t * residual
    z_d = torch.randn(B, d, device=x_t.device, generator=generator)
    eps_d = mu_d + math.sqrt(sigma2_d) * z_d

    x1_sampled = sel_mean + torch.einsum("bdr,br->bd", sel_L, eps_r) + math.sqrt(state.s2) * eps_d
    v0 = (x1_sampled - x_t) / (1 - t)
    return v0


# ===================================================================
# Stage III: HRF2-scale LatentCorrectionUNet
# ===================================================================

class SinEmb(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        h = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device).float() / h)
        e = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], 1)
        return self.net(e)


class RB(nn.Module):
    def __init__(self, ci, co, td=128, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ci), ci)
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.tp = nn.Linear(td, co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.tp(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.sk(x)


class LatentCorrectionUNet(nn.Module):
    """Large U-Net for latent residual prediction (Plan G).

    base=216 → ~44M params (matching HRF2's 44.8M).
    Two RB blocks per resolution level for extra depth.
    """

    def __init__(self, td: int = 128, base: int = 200):
        super().__init__()
        self.tau_emb = SinEmb(td)
        self.t_emb = SinEmb(td)
        self.comb = nn.Linear(2 * td, td)

        # Encoder: 16x16 -> 8x8 -> 4x4
        self.e1a = RB(2 * LATENT_C, base, td)
        self.e1b = RB(base, base, td)
        self.d1 = nn.Conv2d(base, base, 3, stride=2, padding=1)

        self.e2a = RB(base, 2 * base, td)
        self.e2b = RB(2 * base, 2 * base, td)
        self.d2 = nn.Conv2d(2 * base, 2 * base, 3, stride=2, padding=1)

        # Mid: 4x4 with 4x base channels
        self.mid1 = RB(2 * base, 4 * base, td)
        self.mid2 = RB(4 * base, 4 * base, td)

        # Decoder with skip
        self.u2 = nn.ConvTranspose2d(4 * base, 2 * base, 4, stride=2, padding=1)
        self.de2a = RB(4 * base, 2 * base, td)           # cat skip e2b
        self.de2b = RB(2 * base, 2 * base, td)
        self.u1 = nn.ConvTranspose2d(2 * base, base, 4, stride=2, padding=1)
        self.de1a = RB(2 * base, base, td)               # cat skip e1b
        self.de1b = RB(base, base, td)

        self.out_norm = nn.GroupNorm(min(8, base), base)
        self.out_conv = nn.Conv2d(base, LATENT_C, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, v_tau_img, x0_img, tau, t):
        temb = self.comb(torch.cat([self.tau_emb(tau), self.t_emb(t)], 1))
        inp = torch.cat([v_tau_img, x0_img], 1)

        h1 = self.e1a(inp, temb)
        h1 = self.e1b(h1, temb)
        h2 = self.e2a(self.d1(h1), temb)
        h2 = self.e2b(h2, temb)

        h = self.mid1(self.d2(h2), temb)
        h = self.mid2(h, temb)

        h = self.de2a(torch.cat([self.u2(h), h2], 1), temb)
        h = self.de2b(h, temb)
        h = self.de1a(torch.cat([self.u1(h), h1], 1), temb)
        h = self.de1b(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))


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
        torch.save(self.shadow, path)


# ===================================================================
# LR schedule: linear warmup + cosine
# ===================================================================

def cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


# ===================================================================
# Stage III training
# ===================================================================

def train_stage3(
    latents_np, state, out_dir,
    n_iter=400000, bs=128, base_lr=2e-4, min_lr=2e-6,
    warmup_steps=2000, ema_decay=0.9999,
    log_every=500, ckpt_every=20000, unet_base=200,
    multi_t=False,
):
    model = LatentCorrectionUNet(base=unet_base).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  LatentCorrectionUNet params: {n_params/1e6:.2f}M", flush=True)
    print(f"  multi_t: {multi_t}", flush=True)

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

    for step in range(n_iter):
        lr = cosine_lr(step, n_iter, base_lr, min_lr, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]
        x0 = torch.randn(bs, D_LATENT, device=DEVICE, generator=g)
        v_true = x1 - x0

        if multi_t:
            # Sample flow-matching time t ~ U[0, 0.999] and compute x_t
            t_fm = torch.rand(bs, device=DEVICE, generator=g) * 0.999
            x_t = (1.0 - t_fm).view(-1, 1) * x0 + t_fm.view(-1, 1) * x1

            # Coupled sample from GMM at general (x_t, t_fm)
            # Use per-sample t: for efficiency, batch with a single scalar t
            # (approximate: use mean t for the batch, or loop)
            # For simplicity, use a single t for the whole batch:
            t_scalar = t_fm[0].item()
            with torch.no_grad():
                v_0 = coupled_sample_general_t(
                    v_true, x0, x1, x_t, t_scalar, state, generator=g)

            ctx_img = x_t.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
            t_cond = t_fm
        else:
            # Original: t=0 only
            with torch.no_grad():
                v_0 = coupled_sample_gpu(v_true, x0, state, generator=g)
            ctx_img = x0.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
            t_cond = torch.zeros(bs, device=DEVICE)

        # Correction flow: τ ~ U[0,1], interpolate in velocity space
        tau = torch.rand(bs, device=DEVICE, generator=g)
        v_tau = (1.0 - tau).view(-1, 1) * v_0 + tau.view(-1, 1) * v_true
        target = v_true - v_0

        v_tau_img = v_tau.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
        target_img = target.reshape(bs, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)

        pred = model(v_tau_img, ctx_img, tau, t_cond)
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
            torch.save(model.state_dict(), raw_ckpt)
            ema.save(ema_ckpt)
            print(f"    checkpointed raw+EMA at step {step+1}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, ema


# ===================================================================
# Inference
# ===================================================================

@torch.no_grad()
def generate_corrected(state, model, vae, n, corr_steps, batch_size=256, seed=99,
                       multi_t=False, flow_steps=1):
    """Generate samples with Stage II + Stage III correction.

    If multi_t=False (legacy): single correction at t=0, corr_steps Euler steps
      in τ space. Total NFE = corr_steps.

    If multi_t=True: nested integration —
      Outer: flow_steps steps in t-space (position)
      Inner: corr_steps steps in τ-space (velocity correction) at each t
      Total NFE = flow_steps * corr_steps.
    """
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    model.eval()
    out = np.zeros((n, 32, 32, 3), dtype=np.float32)

    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D_LATENT, device=DEVICE, generator=g)

        if not multi_t:
            # Legacy: single correction at t=0
            v = sample_velocity_gpu(x0, state, generator=g)
            v_img = v.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
            x0_img = x0.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
            ds = 1.0 / corr_steps
            t_cond = torch.zeros(nb, device=DEVICE)
            for i in range(corr_steps):
                tau = torch.full((nb,), i * ds, device=DEVICE)
                v_img = v_img + ds * model(v_img, x0_img, tau, t_cond)
            x1_latent_flat = (x0_img + v_img).reshape(nb, D_LATENT).cpu()
        else:
            # Multi-t: nested integration over (t, τ)
            x_t = x0.clone()  # start at t=0
            dt = 1.0 / flow_steps
            ds = 1.0 / corr_steps

            for t_step in range(flow_steps):
                t_val = t_step * dt
                # Get GMM velocity at (x_t, t)
                if t_val < 1e-6:
                    v0 = sample_velocity_gpu(x_t, state, generator=g)
                else:
                    v0 = sample_velocity_general_t(x_t, t_val, state, generator=g)

                # Correct v0 via τ-flow (inner loop)
                v = v0.clone()
                v_img = v.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
                ctx_img = x_t.reshape(nb, LATENT_C, LATENT_SPATIAL, LATENT_SPATIAL)
                t_cond = torch.full((nb,), t_val, device=DEVICE)
                for j in range(corr_steps):
                    tau = torch.full((nb,), j * ds, device=DEVICE)
                    v_img = v_img + ds * model(v_img, ctx_img, tau, t_cond)
                v_corrected = v_img.reshape(nb, D_LATENT)

                # Advance position
                x_t = x_t + dt * v_corrected

            x1_latent_flat = x_t.cpu()

        dec = decode_latents_to_32(vae, x1_latent_flat, bs=64)
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
    p.add_argument("--source", choices=["gmm", "atomic", "local_knn"], default="gmm")
    p.add_argument("--tag", type=str, default="gmm",
                   help="output dir suffix")
    p.add_argument("--K", type=int, default=2000)
    p.add_argument("--rank", type=int, default=50)
    p.add_argument("--cov-nit", type=int, default=1200)
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--unet-base", type=int, default=200)
    p.add_argument("--base-lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--atomic-sigma2", type=float, default=0.005)
    p.add_argument("--local-k-neighbors", type=int, default=100)
    p.add_argument("--local-sigma2", type=float, default=0.005)
    p.add_argument("--n-gen", type=int, default=10000)
    p.add_argument("--multi-t", action="store_true",
                   help="Train at random t~U[0,1] instead of t=0 only")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.K = 500
        args.rank = 20
        args.cov_nit = 200
        args.train_steps = 500
        args.bs = 64
        args.n_gen = 2000
        args.unet_base = 96
        args.warmup_steps = 50

    out_dir = f"cifar10_sd_latent_planG_outputs_{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  source: {args.source}", flush=True)
    print(f"  cfg: K={args.K}, rank={args.rank}, cov_nit={args.cov_nit}, "
          f"train_steps={args.train_steps}, bs={args.bs}, n_gen={args.n_gen}, "
          f"unet_base={args.unet_base}", flush=True)

    print(f"Loading SD VAE from {VAE_PATH}...", flush=True)
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
    vae.eval()

    x_train_hwc = load_cifar10()
    print(f"CIFAR train: {x_train_hwc.shape}", flush=True)

    print("\n===== STAGE 0: encode CIFAR -> 1024-d latents =====", flush=True)
    latents = encode_cifar_to_latents(vae, x_train_hwc, bs=64)
    print(f"  latents: {latents.shape}, mean={latents.mean():.3f}, "
          f"std={latents.std():.3f}", flush=True)

    print("\nInception features (50k real pixels)...", flush=True)
    inc = IncFeatRGB().to(DEVICE)
    t0 = time.time()
    real_feats = feats_from_hwc(inc, x_train_hwc[:50000])
    feat_real_5k = real_feats[:5000]
    feat_real_10k = real_feats[:10000]
    feat_real_50k = real_feats
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # -------- Stage I --------
    print(f"\n===== STAGE I: EMS K={args.K} =====", flush=True)
    rg = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(latents, args.K, lam=0.5, nit=100, rg=rg)
    print(f"  EMS done in {time.time()-t0:.0f}s", flush=True)

    if args.source == "gmm":
        t1 = time.time()
        L_all, sigma2 = learn_lowrank_cov_fast(
            latents, centers, resp, rank=args.rank, nit=args.cov_nit,
            data_batch=1024, s2_floor=0.001)
        print(f"  low-rank cov done in {time.time()-t1:.0f}s, "
              f"sigma2={sigma2:.6f}", flush=True)
        gmm = LowRankGMM(weights, centers, L_all, sigma2)
    elif args.source == "local_knn":
        print(f"  LOCAL kNN-PCA surrogate: k_nbrs={args.local_k_neighbors}, "
              f"rank={args.rank}, sigma2={args.local_sigma2}", flush=True)
        L_all, sigma2 = learn_local_knn_cov(
            latents, centers, rank=args.rank,
            k_neighbors=args.local_k_neighbors, sigma2=args.local_sigma2)
        gmm = LowRankGMM(weights, centers, L_all, sigma2)
    else:
        print(f"  ATOMIC surrogate: rank=1 near-zero factors, "
              f"sigma2={args.atomic_sigma2}", flush=True)
        gmm = build_atomic_gmm(centers, weights, sigma2=args.atomic_sigma2)

    save_gmm(gmm, os.path.join(out_dir, f"gmm_k{args.K}_{args.source}.pt"))
    gmm_state = GMMState(gmm)

    # -------- Stage II reference --------
    print("\n===== STAGE II: closed-form -> decode -> FID =====", flush=True)
    t0 = time.time()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(99)
    stage2_imgs = np.zeros((args.n_gen, 32, 32, 3), dtype=np.float32)
    gen_bs = 256
    for start in range(0, args.n_gen, gen_bs):
        nb = min(gen_bs, args.n_gen - start)
        x0 = torch.randn(nb, D_LATENT, device=DEVICE, generator=g)
        v = sample_velocity_gpu(x0, gmm_state, generator=g)
        x1_lat_flat = (x0 + v).cpu()
        stage2_imgs[start:start + nb] = decode_latents_to_32(vae, x1_lat_flat, bs=64)
    print(f"  Stage II {args.n_gen} in {time.time()-t0:.0f}s", flush=True)
    stage2_feats = feats_from_hwc(inc, stage2_imgs)
    stage2_fid_5k = compute_fid(feat_real_5k, stage2_feats[:5000])
    stage2_fid_10k = compute_fid(feat_real_10k, stage2_feats[:10000])
    print(f"  Stage II FID 5k={stage2_fid_5k:.3f} 10k={stage2_fid_10k:.3f}", flush=True)
    save_grid(
        stage2_imgs[:100],
        os.path.join(out_dir, f"stage2_latent_{args.tag}.png"),
        f"Stage II ({args.source}) FID10k={stage2_fid_10k:.1f}")

    # -------- Stage III --------
    print(f"\n===== STAGE III: train LatentCorrectionUNet "
          f"({args.train_steps} steps) =====", flush=True)
    model, ema = train_stage3(
        latents, gmm_state, out_dir,
        n_iter=args.train_steps, bs=args.bs,
        base_lr=args.base_lr, min_lr=args.min_lr,
        warmup_steps=args.warmup_steps, unet_base=args.unet_base,
        multi_t=args.multi_t)

    print("\n===== INFERENCE + FID =====", flush=True)
    ema_model = LatentCorrectionUNet(base=args.unet_base).to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()

    results = []
    results_csv = os.path.join(out_dir, f"planG_results_{args.tag}.csv")
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["method", "nfe", "fid_5k", "fid_10k", "fid_50k"])
        csv.writer(f).writerow(
            ["StageII_1step", 0, f"{stage2_fid_5k:.4f}",
             f"{stage2_fid_10k:.4f}", "NA"])

    if args.multi_t:
        # Nested evaluation: (flow_steps, corr_steps) combinations
        # Total NFE = flow_steps × corr_steps
        eval_configs = [(1, 10), (1, 20), (5, 4), (10, 2), (10, 5), (20, 5)] \
            if not args.smoke else [(1, 5), (5, 2)]
    else:
        eval_configs = [(1, s) for s in ([1, 5, 10, 20] if not args.smoke else [1, 5])]
    n_fid = args.n_gen

    for flow_s, corr_s in eval_configs:
        nfe = flow_s * corr_s
        t0 = time.time()
        imgs = generate_corrected(
            gmm_state, ema_model, vae, n=n_fid, corr_steps=corr_s,
            batch_size=256, seed=99,
            multi_t=args.multi_t, flow_steps=flow_s)
        print(f"  [EMA] flow={flow_s} corr={corr_s} (NFE={nfe}) "
              f"gen {n_fid} in {time.time()-t0:.0f}s", flush=True)
        g_feats = feats_from_hwc(inc, imgs)
        f5 = compute_fid(feat_real_5k, g_feats[:5000])
        f10 = compute_fid(feat_real_10k, g_feats[:10000])
        f50 = compute_fid(feat_real_50k, g_feats[:50000]) if n_fid >= 50000 else float("nan")
        print(f"    FID 5k={f5:.3f}  10k={f10:.3f}  50k={f50:.3f}", flush=True)
        results.append((flow_s, corr_s, nfe, f5, f10, f50))
        tag_str = f"f{flow_s}_c{corr_s}" if args.multi_t else f"{corr_s}step"
        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [f"StageIII_EMA_{tag_str}", nfe, f"{f5:.4f}",
                 f"{f10:.4f}", (f"{f50:.4f}" if np.isfinite(f50) else "NA")])
        save_grid(
            imgs[:100],
            os.path.join(out_dir, f"stage3_ema_{tag_str}_{args.tag}.png"),
            f"EMA flow={flow_s} corr={corr_s}  FID10k={f10:.1f}")

    # Summary
    print("\n" + "=" * 70, flush=True)
    print(f"  Plan G ({args.tag}) — HRF2-scale", flush=True)
    print(f"  Stage II ({args.source}):  FID10k={stage2_fid_10k:.3f}", flush=True)
    for row in results:
        if len(row) == 6:
            flow_s, corr_s, nfe, f5, f10, f50 = row
            print(f"  Stage III EMA flow={flow_s} corr={corr_s} (NFE={nfe}):  "
                  f"FID 5k={f5:.3f}  10k={f10:.3f}", flush=True)
        else:
            steps, nfe, f5, f10, f50 = row
            print(f"  Stage III EMA {steps}-step (NFE={nfe}):  "
                  f"FID 5k={f5:.3f}  10k={f10:.3f}", flush=True)
    print(f"  [SD VAE 128 reconstruction floor: 1.714]", flush=True)
    print(f"  [HRF2 pixel best: 3.71]", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
