#!/usr/bin/env python3
"""
MNIST 3-stage CoresetFM — fast GPU-vectorized re-measurement.

Same algorithm and hyperparameters as the paper's Stage I + Stage III,
but all K-component loops are vectorized on GPU so Stage I covariance
learning and Stage III posterior coupling run ~20-50x faster than the
naive version.

Evaluates Inception FID at three sample sizes:
  - 5k generated vs 5k test (paper protocol)
  - 10k generated vs 10k test
  - 50k generated vs 50k train
"""
from __future__ import annotations

import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 784
OUT_DIR = "mnist_pixel_3stage_outputs"
print(f"Device: {DEVICE}", flush=True)


# ===================================================================
# Data
# ===================================================================

def load_mnist(path):
    d = np.load(path)
    return (d["x_train"].astype(np.float32) / 255.0,
            d["y_train"].astype(int),
            d["x_test"].astype(np.float32) / 255.0,
            d["y_test"].astype(int))

def to_flat(x): return x.reshape(-1, D)


# ===================================================================
# Stage I: EMS coreset (already GPU, unchanged)
# ===================================================================

class LowRankGMM:
    def __init__(self, w, mu, L, s2):
        self.weights = w      # (K,) numpy
        self.means = mu       # (K, d) numpy
        self.factors = L      # (K, d, r) numpy
        self.noise_var = s2   # float


def ems_coreset_gpu(x_np, K, lam, nit, rg, chunk_n=None):
    """EMS coreset with soft responsibilities.

    For big (n, K), the full (n, K) distance matrix can exceed GPU memory.
    We auto-chunk over n so each step materializes only (chunk_n, K).
    Behavior is mathematically identical to the unbatched version.
    """
    n = len(x_np)
    d = x_np.shape[1]
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    y = x[rg.choice(n, K, replace=False)].clone()
    w = torch.full((K,), 1.0/K, device=DEVICE)

    if chunk_n is None:
        # Cap each chunk's (chunk_n × K) float32 block at ~2 GiB
        max_elems = 2 * (1024 ** 3) // 4
        chunk_n = max(1024, min(n, max_elems // max(K, 1)))

    def _distances_and_T(y_now, w_now):
        """Iterate once over x in chunks. Returns (T.sum(0), T.T@x, loss_sum)."""
        y_sq = (y_now * y_now).sum(1).unsqueeze(0)      # (1, K)
        log_w = torch.log(w_now + 1e-30).unsqueeze(0)   # (1, K)
        nj_acc = torch.zeros(K, device=DEVICE)
        num_acc = torch.zeros(K, d, device=DEVICE)
        loss_acc = torch.zeros((), device=DEVICE)
        for s in range(0, n, chunk_n):
            xb = x[s:s + chunk_n]                       # (B, d)
            x_sq = (xb * xb).sum(1, keepdim=True)       # (B, 1)
            sq_b = x_sq - 2.0 * (xb @ y_now.T) + y_sq   # (B, K)
            Tb = torch.softmax(log_w - sq_b / lam, dim=1)
            nj_acc += Tb.sum(0)
            num_acc += Tb.T @ xb                        # (K, d)
            loss_acc += (Tb * sq_b).sum()
        return nj_acc, num_acc, loss_acc

    for it in range(nit):
        nj, num, loss_sum = _distances_and_T(y, w)
        nj = nj + 1e-12
        y = num / nj.unsqueeze(1)
        w = nj / nj.sum()
        if (it + 1) % 20 == 0:
            print(f"    EMS {it+1}/{nit}, loss={(loss_sum / n).item():.4f}",
                  flush=True)

    # Final responsibility matrix: materialize on CPU in chunks (avoid a single
    # 51 GB GPU allocation for big n × K).
    T_np = np.zeros((n, K), dtype=np.float32)
    y_sq = (y * y).sum(1).unsqueeze(0)
    log_w = torch.log(w + 1e-30).unsqueeze(0)
    for s in range(0, n, chunk_n):
        xb = x[s:s + chunk_n]
        x_sq = (xb * xb).sum(1, keepdim=True)
        sq_b = x_sq - 2.0 * (xb @ y.T) + y_sq
        Tb = torch.softmax(log_w - sq_b / lam, dim=1)
        T_np[s:s + chunk_n] = Tb.cpu().numpy()
    return y.cpu().numpy(), w.cpu().numpy(), T_np


# ===================================================================
# Stage I: Vectorized low-rank covariance training
# ===================================================================

def learn_lowrank_cov_ppca(x_np, mu, T_np, rank, weights=None,
                            data_batch=None, svd_niter=4, s2_floor=1e-6):
    """Closed-form PPCA covariance fit per GMM component.

    For each component b, the soft-assignment-weighted empirical covariance
        S_b = (1/n_b) sum_i T_{ib} (x_i - mu_b)(x_i - mu_b)^T
    is decomposed via the Tipping-Bishop PPCA MLE:
        sigma_b^2 = (tr S_b - sum_{j=1}^r lambda_j) / (d - r)
        L_b       = U_r diag(sqrt(max(lambda_j - sigma_b^2, 0)))
    where (lambda_j, U_r) are the top-r eigenvalues / eigenvectors of S_b.
    The top-r eigendecomposition is computed via randomized SVD on the
    weighted data matrix W_b (sqrt(T_{ib}/n_b) (x_i - mu_b)), avoiding
    the O(d^3) full eigendecomposition.

    Returns:
      L_all  : (K, d, rank) numpy
      sigma2 : scalar shared noise variance = sum_b w_b sigma_b^2
               (weighted mean of per-component PPCA noise estimates)
    """
    K, d = mu.shape
    n = len(x_np)

    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Push x and centers to GPU once (~15 GB for ImageNet-32). The full
    # responsibility T_np (n, K) is kept on CPU and streamed column-by-column
    # to avoid the K*n*4 byte explosion (51 GB at K=10000, n=1.28M).
    x_gpu = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)      # (n, d)
    c_gpu = torch.tensor(mu, dtype=torch.float32, device=DEVICE)        # (K, d)
    # T_np stays on host; the b-th column is .to(DEVICE) per iteration.
    if weights is None:
        # Compute column sums on host without ever materialising T on GPU.
        col_sums = T_np.sum(axis=0)
        w_gpu = torch.tensor(col_sums / max(col_sums.sum(), 1e-12),
                              dtype=torch.float32, device=DEVICE)
    else:
        w_arr = np.asarray(weights, dtype=np.float32)
        w_gpu = torch.tensor(w_arr / max(w_arr.sum(), 1e-12),
                              dtype=torch.float32, device=DEVICE)

    # Build L_all on CPU to avoid a (K, d, r) GPU allocation when K*d*r is
    # large (K=10000, d=3072, r=80 -> ~9.8 GB).
    L_all_cpu = torch.zeros(K, d, rank, dtype=torch.float32)
    sigma2_per = torch.zeros(K, device=DEVICE)

    chunk = 65536  # for streaming trace; keeps peak memory low
    t0 = time.time()
    for b in range(K):
        # Stream the b-th responsibility column to GPU (n * 4 bytes = 5 MB
        # at n=1.28M). This is the only per-component host->device transfer.
        rb_np = T_np[:, b]
        rb = torch.tensor(rb_np, dtype=torch.float32, device=DEVICE)
        mass = rb.sum() + 1e-12

        # Streaming trace of weighted empirical covariance: never
        # materialise the full (n, d) weighted-diff tensor.
        trace_S = torch.zeros((), device=DEVICE)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d_chunk = x_gpu[s:e] - c_gpu[b]                 # (cs, d)
            nsq = (d_chunk * d_chunk).sum(dim=1)            # (cs,)
            trace_S = trace_S + (rb[s:e] * nsq).sum() / mass
            del d_chunk, nsq

        # Top-r SVD on the high-weight subset (top 2000): bounded memory
        # (2000 * d * 4 = ~25 MB at d=3072).
        top_n = min(2000, n)
        top_idx = torch.argsort(rb, descending=True)[:top_n]
        sqrt_r_top = torch.sqrt(rb[top_idx] / mass).unsqueeze(1)
        W_top = sqrt_r_top * (x_gpu[top_idx] - c_gpu[b])
        _, S_lr, V_lr = torch.svd_lowrank(W_top, q=rank, niter=svd_niter)
        del W_top, sqrt_r_top

        lambdas_top = S_lr * S_lr  # (rank,)
        if d > rank:
            sigma2_b = (trace_S - lambdas_top.sum()) / (d - rank)
        else:
            sigma2_b = torch.tensor(0.0, device=DEVICE)
        sigma2_b = sigma2_b.clamp(min=s2_floor)
        Lambda_r = (lambdas_top - sigma2_b).clamp(min=0.0)
        L_b = V_lr * torch.sqrt(Lambda_r).unsqueeze(0)        # (d, rank)
        L_all_cpu[b] = L_b.detach().cpu()
        sigma2_per[b] = sigma2_b
        del V_lr, S_lr, lambdas_top, Lambda_r, L_b, trace_S, top_idx, rb

        if (b + 1) % max(1, K // 10) == 0:
            print(f"    PPCA {b+1}/{K}, sigma2_avg={sigma2_per[:b+1].mean().item():.5f}",
                  flush=True)

    # Aggregate per-component sigma_b^2 to a single shared sigma^2 via
    # w_b-weighted mean (the MLE under the shared-noise constraint).
    sigma2_shared = (w_gpu * sigma2_per).sum().clamp(min=s2_floor).item()
    print(f"  PPCA done in {time.time()-t0:.0f}s, shared sigma2={sigma2_shared:.6f} "
          f"(per-comp range [{sigma2_per.min().item():.5f}, "
          f"{sigma2_per.max().item():.5f}])", flush=True)

    L_np = L_all_cpu.numpy()
    del L_all_cpu, sigma2_per, x_gpu, c_gpu
    gc.collect(); torch.cuda.empty_cache()
    return L_np, float(sigma2_shared)


def learn_lowrank_cov_fast(x_np, mu, T_np, rank, nit=800, lr=0.005,
                             s2_floor=0.001, wd=1e-4, data_batch=4096,
                             _auto_fallback=True):
    """
    Jointly optimize low-rank factors L ∈ R^{K×d×r} and shared noise sigma²
    under the responsibility-weighted Gaussian NLL.

    All K-component computations are batched on GPU via einsum.

    OOM safety: if the first backward step runs out of GPU memory (typical
    when K*d*r is large, e.g. K=10000 r=160 on 95 GB HBM), the function
    releases all transient allocations and automatically retries with rank
    halved (and data_batch capped). Falls back recursively down to rank=16
    before giving up.
    """
    K, d = mu.shape
    n = len(x_np)

    # Clear any reserved-but-unused pool from prior stages before allocating
    # the (K, d, r) parameter tensor; important when d*r is large.
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        x_gpu = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)      # (n, d)
        c_gpu = torch.tensor(mu, dtype=torch.float32, device=DEVICE)        # (K, d)
        r_gpu = torch.tensor(T_np, dtype=torch.float32, device=DEVICE)      # (n, K)
        eye_r = torch.eye(rank, device=DEVICE)                              # (r, r)

        # Per-component low-rank SVD initialization, stacked into one (K, d, r) tensor.
        # Uses torch.svd_lowrank (stable on GH200 aarch64) instead of torch.linalg.svd
        # because cuSOLVER's gesvdj path falls back to a slow CPU loop on that platform.
        L_init = torch.zeros(K, d, rank, device=DEVICE)
        for b in range(K):
            rb = r_gpu[:, b]
            mass = rb.sum() + 1e-12
            diff = x_gpu - c_gpu[b]
            sqrt_r = torch.sqrt(rb / mass).unsqueeze(1)
            top_idx = torch.argsort(rb, descending=True)[:min(2000, n)]
            wd_sub = sqrt_r[top_idx] * diff[top_idx]
            # torch.svd_lowrank(A, q) returns U (m,q), S (q,), V (n,q) with A ≈ U diag(S) V^T.
            # Our L_b should satisfy L_b L_b^T ≈ cov, so L_b = V * S (shape d × rank).
            _, S_lr, V_lr = torch.svd_lowrank(wd_sub, q=rank, niter=2)
            L_init[b] = V_lr * S_lr.unsqueeze(0)
        del diff, sqrt_r, wd_sub
        torch.cuda.empty_cache()

        L_param = nn.Parameter(L_init)
        log_s2 = nn.Parameter(torch.tensor(math.log(0.01), device=DEVICE))
        opt = torch.optim.Adam([{"params": [L_param], "weight_decay": wd},
                                 {"params": [log_s2]}], lr=lr)
    except torch.cuda.OutOfMemoryError as e:
        # Initialization itself ran out (rare; typically the backward, below, is the
        # first OOM point). Clear and fall through to the retry branch below.
        for name in ("x_gpu", "c_gpu", "r_gpu", "eye_r", "L_init",
                     "L_param", "log_s2", "opt"):
            if name in locals():
                del locals()[name]
        gc.collect(); torch.cuda.empty_cache()
        if _auto_fallback and rank > 16:
            new_rank = max(16, rank // 2)
            print(f"[learn_lowrank_cov_fast] OOM during init at rank={rank}: {e}\n"
                  f"  Retrying with rank={new_rank}, data_batch={min(data_batch, 256)}.",
                  flush=True)
            return learn_lowrank_cov_fast(
                x_np, mu, T_np, new_rank, nit=nit, lr=lr,
                s2_floor=s2_floor, wd=wd,
                data_batch=min(data_batch, 256),
                _auto_fallback=_auto_fallback)
        raise

    try_first_bwd_done = False
    for step in range(nit):
        opt.zero_grad()
        s2 = torch.exp(log_s2).clamp(min=s2_floor)
        perm = torch.randperm(n, device=DEVICE)[:data_batch]
        xb = x_gpu[perm]             # (B, d)
        rb_all = r_gpu[perm]         # (B, K)
        B = xb.shape[0]

        # ||xb - c[k]||² via expansion: ||xb||² + ||c[k]||² - 2 xb·c[k]
        x_sq = (xb * xb).sum(-1, keepdim=True)                  # (B, 1)
        c_sq = (c_gpu * c_gpu).sum(-1).unsqueeze(0)             # (1, K)
        xc = xb @ c_gpu.T                                        # (B, K)
        diff_sq = x_sq + c_sq - 2 * xc                          # (B, K)
        q1 = diff_sq / s2

        # dL[b, k, r] = (xb[b] - c[k]) · L[k]
        xL = torch.einsum("bd,kdr->bkr", xb, L_param)            # (B, K, r)
        cL = torch.einsum("kd,kdr->kr", c_gpu, L_param)          # (K, r)
        dL = xL - cL.unsqueeze(0)

        # M[k] = I_r + L[k]^T L[k] / s²
        LtL = torch.einsum("kdr,kds->krs", L_param, L_param)     # (K, r, r)
        M = eye_r.unsqueeze(0) + LtL / s2
        M_inv = torch.linalg.inv(M)

        # q2[b, k] = dL[b,k] · M_inv[k] · dL[b,k] / s²²
        dL_Mi = torch.einsum("bkr,krs->bks", dL, M_inv)          # (B, K, r)
        q2 = (dL_Mi * dL).sum(-1) / s2**2                        # (B, K)

        # logdet(Σ) = logdet(M) + d·log(s²)
        ld = torch.logdet(M) + d * torch.log(s2)                 # (K,)

        nll_bk = 0.5 * (ld.unsqueeze(0) + q1 - q2)               # (B, K)
        total_nll = (rb_all * nll_bk).sum()
        try:
            (total_nll / B).backward()
        except torch.cuda.OutOfMemoryError as e:
            # First backward is the highest memory-water-mark step (it allocates
            # the K*d*r grad tensor). On OOM here, the checkpoint has not yet
            # been written, so we can safely release and retry from scratch at
            # a smaller rank.
            if not try_first_bwd_done and _auto_fallback and rank > 16:
                # Release the full optimizer-state footprint before recursing.
                del L_param, log_s2, opt, x_gpu, c_gpu, r_gpu, eye_r
                if "L_init" in locals(): del L_init
                gc.collect(); torch.cuda.empty_cache()
                new_rank = max(16, rank // 2)
                new_batch = min(data_batch, 256)
                print(f"[learn_lowrank_cov_fast] OOM on first backward at "
                      f"rank={rank}, data_batch={data_batch}: {e}\n"
                      f"  Retrying with rank={new_rank}, data_batch={new_batch}.",
                      flush=True)
                return learn_lowrank_cov_fast(
                    x_np, mu, T_np, new_rank, nit=nit, lr=lr,
                    s2_floor=s2_floor, wd=wd, data_batch=new_batch,
                    _auto_fallback=_auto_fallback)
            raise
        try_first_bwd_done = True
        opt.step()

        if (step+1) % 100 == 0:
            print(f"    Cov {step+1}/{nit}, NLL={total_nll.item()/B:.2f}, "
                  f"sigma2={torch.exp(log_s2).clamp(min=s2_floor).item():.6f}, "
                  f"rank={rank}", flush=True)

    return L_param.detach().cpu().numpy(), max(torch.exp(log_s2).item(), s2_floor)


# ===================================================================
# GPU-resident GMM state (precomputes Woodbury inverses and log-dets)
# ===================================================================

class GMMState:
    def __init__(self, gmm: LowRankGMM):
        self.K, self.d, self.r = gmm.factors.shape
        self.weights = torch.tensor(gmm.weights, dtype=torch.float32, device=DEVICE)
        self.log_weights = torch.log(self.weights + 1e-30)
        self.means = torch.tensor(gmm.means, dtype=torch.float32, device=DEVICE)
        self.L = torch.tensor(gmm.factors, dtype=torch.float32, device=DEVICE)
        self.s2 = float(gmm.noise_var)
        self.sigma = math.sqrt(self.s2)

        eye_r = torch.eye(self.r, device=DEVICE)
        LtL = torch.einsum("kdr,kds->krs", self.L, self.L)
        self.M = eye_r.unsqueeze(0) + LtL / self.s2
        self.M_inv = torch.linalg.inv(self.M)
        self.log_det = torch.logdet(self.M) + self.d * math.log(self.s2)
        self.mL = torch.einsum("kd,kdr->kr", self.means, self.L)  # precomputed


# ===================================================================
# Stage II: GPU-vectorized sampling
# ===================================================================

@torch.no_grad()
def sample_velocity_gpu(x0, state: GMMState, generator=None):
    """v ~ π̃(·|x₀, 0) = Σ_b w_b N(m_b - x₀, L_b L_b^T + σ²I)."""
    B = x0.shape[0]
    comp = torch.multinomial(state.weights, B, replacement=True, generator=generator)
    sel_mean = state.means[comp]                                      # (B, d)
    sel_L = state.L[comp]                                              # (B, d, r)
    z_r = torch.randn(B, state.r, device=DEVICE, generator=generator)
    z_d = torch.randn(B, state.d, device=DEVICE, generator=generator)
    Lzr = torch.einsum("bdr,br->bd", sel_L, z_r)
    return sel_mean - x0 + Lzr + state.sigma * z_d


def coupled_sample_gpu(v_true, x0, state: GMMState, generator=None):
    """Stochastic posterior coupling for correction training.

    Given (x₀, v_true), compute posterior r_b ∝ w_b·N(v_true; m_b-x₀, Σ_b),
    sample component b* from r, draw v₀ ~ N(m_{b*}-x₀, Σ_{b*}).
    Preserves the marginal π̃(·|x₀, 0).
    """
    B = x0.shape[0]
    w_vec = v_true + x0                                               # diff from m_b

    # q1[b, k] = ||w - m_k||² / s²
    w_sq = (w_vec * w_vec).sum(-1, keepdim=True)
    m_sq = (state.means * state.means).sum(-1).unsqueeze(0)
    wm = w_vec @ state.means.T
    q1 = (w_sq + m_sq - 2 * wm) / state.s2                            # (B, K)

    # dL[b, k, r] = (w[b] - m[k]) · L[k]
    wL = torch.einsum("bd,kdr->bkr", w_vec, state.L)                  # (B, K, r)
    dL = wL - state.mL.unsqueeze(0)                                   # (B, K, r)

    dL_Mi = torch.einsum("bkr,krs->bks", dL, state.M_inv)             # (B, K, r)
    q2 = (dL_Mi * dL).sum(-1) / state.s2**2                           # (B, K)

    log_resp = state.log_weights.unsqueeze(0) - 0.5 * (
        state.log_det.unsqueeze(0) + q1 - q2)                         # (B, K)
    log_resp = log_resp - log_resp.max(dim=-1, keepdim=True).values
    resp = torch.softmax(log_resp, dim=-1)

    comp = torch.multinomial(resp, 1, generator=generator).squeeze(-1)  # (B,)
    sel_mean = state.means[comp]
    sel_L = state.L[comp]
    z_r = torch.randn(B, state.r, device=DEVICE, generator=generator)
    z_d = torch.randn(B, state.d, device=DEVICE, generator=generator)
    Lzr = torch.einsum("bdr,br->bd", sel_L, z_r)
    return sel_mean - x0 + Lzr + state.sigma * z_d


# ===================================================================
# U-Net (identical to paper's arch: 64→128→256)
# ===================================================================

class SinEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
    def forward(self, t):
        h = self.d // 2
        f = torch.exp(-math.log(10000)*torch.arange(h, device=t.device).float()/h)
        e = torch.cat([torch.sin(t[:, None]*f), torch.cos(t[:, None]*f)], 1)
        return self.net(e)

class RB(nn.Module):
    def __init__(self, ci, co, td=128):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.tp = nn.Linear(td, co)
        self.n1 = nn.GroupNorm(min(8, co), co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.tp(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h))) + self.sk(x)


class CorrectionUNet(nn.Module):
    def __init__(self, td=128):
        super().__init__()
        self.tau_emb = SinEmb(td)
        self.t_emb = SinEmb(td)
        self.comb = nn.Linear(2*td, td)
        self.e1 = RB(2, 64, td); self.d1 = nn.Conv2d(64, 64, 3, 2, 1)
        self.e2 = RB(64, 128, td); self.d2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.mid = RB(128, 256, td)
        self.u2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.de2 = RB(256, 128, td)
        self.u1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.de1 = RB(128, 64, td)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, v_tau_img, x0_img, tau, t):
        te = self.comb(torch.cat([self.tau_emb(tau), self.t_emb(t)], 1))
        inp = torch.cat([v_tau_img, x0_img], 1)
        h1 = self.e1(inp, te)
        h2 = self.e2(self.d1(h1), te)
        h = self.mid(self.d2(h2), te)
        h = self.de2(torch.cat([self.u2(h), h2], 1), te)
        h = self.de1(torch.cat([self.u1(h), h1], 1), te)
        return self.out(h)


def train_correction_fast(x_train_flat, state: GMMState, n_iter=40000, bs=64, lr=2e-4):
    """Stage III training — all on GPU, no CPU round-trips per step."""
    model = CorrectionUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_all = torch.tensor(x_train_flat, dtype=torch.float32, device=DEVICE)
    n_data = len(x_all)

    g = torch.Generator(device=DEVICE); g.manual_seed(0)

    t0 = time.time()
    model.train()
    for step in range(n_iter):
        idx = torch.randint(0, n_data, (bs,), device=DEVICE, generator=g)
        x1 = x_all[idx]                                                # (bs, D)
        x0 = torch.randn(bs, D, device=DEVICE, generator=g)
        v_true = x1 - x0

        with torch.no_grad():
            v0 = coupled_sample_gpu(v_true, x0, state, generator=g)

        v_true_img = v_true.reshape(bs, 1, 28, 28)
        v0_img = v0.reshape(bs, 1, 28, 28)
        x0_img = x0.reshape(bs, 1, 28, 28)

        tau = torch.rand(bs, device=DEVICE, generator=g)
        tau_b = tau.view(-1, 1, 1, 1)
        v_tau = (1 - tau_b) * v0_img + tau_b * v_true_img
        target = v_true_img - v0_img

        t_cond = torch.zeros(bs, device=DEVICE)
        pred = model(v_tau, x0_img, tau, t_cond)
        loss = ((pred - target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if (step+1) % 2000 == 0:
            print(f"    Corr step {step+1}/{n_iter}, loss={loss.item():.6f}", flush=True)

    torch.cuda.synchronize()
    model.eval()
    return model, time.time() - t0


@torch.no_grad()
def generate_stage2_fast(state: GMMState, n, batch_size=4096, seed=99):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    out = []
    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D, device=DEVICE, generator=g)
        v = sample_velocity_gpu(x0, state, generator=g)
        x1 = (x0 + v).clamp(0, 1).reshape(nb, 28, 28)
        out.append(x1.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def generate_corrected_fast(state: GMMState, correction, n, corr_steps=10,
                              batch_size=512, seed=99):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    out = []
    for start in range(0, n, batch_size):
        nb = min(batch_size, n - start)
        x0 = torch.randn(nb, D, device=DEVICE, generator=g)
        v = sample_velocity_gpu(x0, state, generator=g)
        v_img = v.reshape(nb, 1, 28, 28)
        x0_img = x0.reshape(nb, 1, 28, 28)
        ds = 1.0 / corr_steps
        for i in range(corr_steps):
            tau = torch.full((nb,), i*ds, device=DEVICE)
            t_cond = torch.zeros(nb, device=DEVICE)
            v_img = v_img + ds * correction(v_img, x0_img, tau, t_cond)
        x1 = (x0_img + v_img).squeeze(1).clamp(0, 1)
        out.append(x1.cpu().numpy())
    return np.concatenate(out)


# ===================================================================
# Inception FID
# ===================================================================

class IncFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.m.eval(); self.m.fc = nn.Identity()
    @torch.no_grad()
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, 299, mode="bilinear", align_corners=False)
        mu = torch.tensor([.485, .456, .406], device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor([.229, .224, .225], device=x.device).view(1, 3, 1, 1)
        return self.m((x - mu) / sd)

@torch.no_grad()
def get_feats(inc, imgs, bs=128):
    fs = []
    for i in range(0, len(imgs), bs):
        batch = torch.tensor(imgs[i:i+bs]).float().unsqueeze(1).to(DEVICE)
        fs.append(inc(batch).cpu().numpy())
    return np.concatenate(fs)

def compute_fid(f1, f2, eps=1e-6):
    m1, m2 = f1.mean(0), f2.mean(0); d = m1 - m2
    c1 = np.cov(f1, rowvar=False) + eps*np.eye(f1.shape[1])
    c2 = np.cov(f2, rowvar=False) + eps*np.eye(f2.shape[1])
    cm = sqrtm(c1 @ c2)
    if np.iscomplexobj(cm): cm = cm.real
    return float(d @ d + np.trace(c1 + c2 - 2*cm))


# ===================================================================
# IO helpers
# ===================================================================

def save_grid(imgs, path, title=""):
    n = min(100, len(imgs)); nr = 10
    fig, ax = plt.subplots(nr, nr, figsize=(10, 10))
    if title: fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    for i in range(nr*nr):
        a = ax[i//nr, i%nr]
        if i < n: a.imshow(np.clip(imgs[i], 0, 1), cmap="gray_r", vmin=0, vmax=1)
        a.axis("off")
    fig.subplots_adjust(wspace=.02, hspace=.02)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}", flush=True)


def save_gmm(gmm, path):
    torch.save({
        "weights": gmm.weights,
        "means": gmm.means,
        "factors": gmm.factors,
        "noise_var": gmm.noise_var,
    }, path, pickle_protocol=5)
    print(f"  Saved GMM: {path}", flush=True)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rg = np.random.default_rng(42); torch.manual_seed(42)

    x_train_2d, _, x_test_2d, _ = load_mnist(
        os.path.join(os.getcwd(), ".data", "mnist.npz"))
    x_train = to_flat(x_train_2d)
    print(f"Train: {x_train.shape}, Test: {x_test_2d.shape}", flush=True)

    N_GEN = 50000
    N_REAL_TRAIN = 50000

    # ---- Inception + real features ----
    print("Loading InceptionV3 + computing real features...", flush=True)
    t = time.time()
    inc = IncFeat().to(DEVICE)
    feat_real_5k = get_feats(inc, x_test_2d[:5000])
    feat_real_10k = get_feats(inc, x_test_2d[:10000])
    feat_real_50k = get_feats(inc, x_train_2d[:N_REAL_TRAIN])
    print(f"  Real features ready ({time.time()-t:.0f}s)", flush=True)

    def fid_three(imgs_50k):
        f_50k = get_feats(inc, imgs_50k)
        return (
            compute_fid(feat_real_5k,  f_50k[:5000]),
            compute_fid(feat_real_10k, f_10k := f_50k[:10000]),  # noqa: F841
            compute_fid(feat_real_50k, f_50k),
        )

    results = []

    # ==== STAGE I ====
    print("\n" + "="*60, flush=True)
    print("STAGE I: EMS coreset (K=500) + vectorized rank-30 low-rank cov", flush=True)
    print("="*60, flush=True)
    K, rank = 500, 30
    t0 = time.time()
    centers, weights, resp = ems_coreset_gpu(x_train, K, lam=1.5, nit=100, rg=rg)
    print(f"  EMS done in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    L_all, sigma2 = learn_lowrank_cov_fast(x_train, centers, resp, rank=rank, nit=800)
    print(f"  Covariance learned in {time.time()-t1:.0f}s, sigma2={sigma2:.6f}", flush=True)

    gmm = LowRankGMM(weights, centers, L_all, sigma2)
    save_gmm(gmm, os.path.join(OUT_DIR, "gmm_k500.pt"))
    state = GMMState(gmm)
    print(f"  Stage I total: {time.time()-t0:.0f}s", flush=True)

    # ==== STAGE II ====
    print("\n" + "="*60, flush=True)
    print(f"STAGE II: Closed-form 1-step generation ({N_GEN} samples)", flush=True)
    print("="*60, flush=True)
    t0 = time.time()
    imgs_s2 = generate_stage2_fast(state, N_GEN, batch_size=4096, seed=99)
    print(f"  Generated {N_GEN} in {time.time()-t0:.0f}s", flush=True)
    f5, f10, f50 = fid_three(imgs_s2)
    print(f"  Stage II: FID 5k={f5:.2f}, 10k={f10:.2f}, 50k={f50:.2f}", flush=True)
    results.append(("StageII_1step", 1, f5, f10, f50))
    save_grid(imgs_s2[:100], os.path.join(OUT_DIR, "stage2_largefid.png"),
              f"Stage II 1-step (FID 50k={f50:.1f})")

    # ==== STAGE III ====
    print("\n" + "="*60, flush=True)
    print("STAGE III: U-Net correction (stochastic posterior coupling, GPU)", flush=True)
    print("="*60, flush=True)
    corr_model, corr_time = train_correction_fast(x_train, state, n_iter=40000, bs=64)
    print(f"  Correction training: {corr_time:.0f}s", flush=True)
    torch.save(corr_model.state_dict(), os.path.join(OUT_DIR, "corr_unet.pt"))
    print(f"  Saved U-Net: {os.path.join(OUT_DIR, 'corr_unet.pt')}", flush=True)

    for steps in [1, 5, 10, 20]:
        t0 = time.time()
        imgs = generate_corrected_fast(state, corr_model, N_GEN,
                                        corr_steps=steps, batch_size=512, seed=99)
        print(f"  Generated {N_GEN} at {steps}-step in {time.time()-t0:.0f}s", flush=True)
        f5, f10, f50 = fid_three(imgs)
        print(f"  Stage III ({steps}-step, {steps+1} NFE): "
              f"FID 5k={f5:.2f}, 10k={f10:.2f}, 50k={f50:.2f}", flush=True)
        results.append((f"StageIII_{steps}step", steps+1, f5, f10, f50))
        save_grid(imgs[:100],
                  os.path.join(OUT_DIR, f"stage3_{steps}step_largefid.png"),
                  f"Stage III {steps}-step (FID 50k={f50:.1f})")

    # ==== WRITE CSV ====
    csv_path = os.path.join(OUT_DIR, "largefid_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "nfe", "fid_5k", "fid_10k", "fid_50k"])
        for row in results:
            w.writerow(row)
    print(f"\nWrote results to {csv_path}", flush=True)

    # ==== SUMMARY ====
    print("\n" + "="*72, flush=True)
    print(f"{'Method':<20} {'NFE':>5} {'FID 5k':>10} {'FID 10k':>10} {'FID 50k':>10}", flush=True)
    print("-"*72, flush=True)
    for m, n, f5, f10, f50 in results:
        print(f"{m:<20} {n:>5} {f5:>10.2f} {f10:>10.2f} {f50:>10.2f}", flush=True)
    print("="*72, flush=True)


if __name__ == "__main__":
    main()
