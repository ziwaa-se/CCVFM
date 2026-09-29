"""Stage I: entropic-Sinkhorn (EMS) coreset and low-rank GMM lift.

The coreset solves

    min_{mu, w, T >= 0}  sum_{ik} T_ik ||x_i - mu_k||^2 + lam * KL(T || (1/n) 1 ⊗ w),
    sum_k T_ik = 1/n,  sum_i T_ik = w_k,

by alternating the closed-form update T_ik ∝ w_k exp(-||x_i - mu_k||^2 / lam)
with barycentric updates of (mu, w). Each atom is then lifted to a Gaussian
N(mu_k, L_k L_k^T + sigma^2 I) by a closed-form probabilistic-PCA fit on its
soft-assigned residuals.
"""
from __future__ import annotations

import math

import torch

from .gmm import LowRankGMM


def _sq_dists(x, mu):
    return ((x * x).sum(-1, keepdim=True) - 2.0 * x @ mu.T
            + (mu * mu).sum(-1).unsqueeze(0)).clamp_min(0.0)


def _init_atoms(x, K, init, generator):
    n = x.shape[0]
    if init == "random":
        return x[torch.randperm(n, generator=generator, device="cpu")[:K].to(x.device)].clone()
    if init != "kmeans++":
        raise ValueError("init must be 'random' or 'kmeans++'")
    first = torch.randint(0, n, (1,), generator=generator).item()
    mu = [x[first]]
    d2 = ((x - x[first]) ** 2).sum(-1)
    for _ in range(1, K):
        probs = (d2 / d2.sum()).cpu()
        j = torch.multinomial(probs, 1, generator=generator).item()
        mu.append(x[j])
        d2 = torch.minimum(d2, ((x - x[j]) ** 2).sum(-1))
    return torch.stack(mu).clone()


def coreset_responsibilities(x, means, weights, lam, chunk=16384):
    """T_ik / (1/n): row-normalised soft assignments -> (n, K)."""
    log_w = torch.log(weights.clamp_min(1e-30)).unsqueeze(0)
    return torch.cat([torch.softmax(log_w - _sq_dists(x[s:s + chunk], means) / lam, -1)
                      for s in range(0, x.shape[0], chunk)], 0)


@torch.no_grad()
def sinkhorn_coreset(x: torch.Tensor, K: int, lam: float | None = None,
                     n_iter: int = 100, generator=None, chunk: int = 16384,
                     init: str = "random", verbose: bool = False):
    """Weighted K-atom coreset of the empirical measure of x (n, d).

    Args:
        lam: entropic bandwidth. ``None`` picks half the mean squared distance
             from a data point to its nearest initial atom, a scale-free default.
        init: "random" (uniform subsample, as in the paper) or "kmeans++".
    Returns:
        means (K, d), weights (K,), lam (float, the bandwidth actually used).
    """
    n, d = x.shape
    mu = _init_atoms(x, K, init, generator)
    w = torch.full((K,), 1.0 / K, device=x.device, dtype=x.dtype)
    if lam is None:
        probe = x[torch.randperm(n, generator=generator, device="cpu")[:min(n, 4096)].to(x.device)]
        lam = 0.5 * _sq_dists(probe, mu).min(1).values.mean().item()
        lam = max(lam, 1e-8)

    for it in range(n_iter):
        mass = torch.zeros(K, device=x.device, dtype=x.dtype)
        num = torch.zeros(K, d, device=x.device, dtype=x.dtype)
        cost = 0.0
        log_w = torch.log(w.clamp_min(1e-30)).unsqueeze(0)
        for s in range(0, n, chunk):
            xb = x[s:s + chunk]
            sq = _sq_dists(xb, mu)
            T = torch.softmax(log_w - sq / lam, -1)
            mass += T.sum(0)
            num += T.T @ xb
            cost += (T * sq).sum().item()
        mass = mass + 1e-12
        mu = num / mass.unsqueeze(1)
        w = mass / mass.sum()
        if verbose and (it + 1) % max(1, n_iter // 5) == 0:
            print(f"  [coreset] iter {it + 1}/{n_iter}  transport cost {cost / n:.4f}", flush=True)
    return mu, w, lam


@torch.no_grad()
def fit_lowrank_covariances(x: torch.Tensor, means: torch.Tensor, resp: torch.Tensor,
                            rank: int, sigma2_floor: float = 1e-6, top_n: int = 2000,
                            exact_max_dim: int = 256):
    """Closed-form PPCA (Tipping & Bishop) per component.

    For the soft-assignment-weighted covariance S_k with eigenvalues l_1 >= ... >= l_d:
        sigma_k^2 = (tr S_k - sum_{j<=r} l_j) / (d - r),
        L_k       = U_r diag(sqrt(max(l_j - sigma_k^2, 0))),
    and the shared noise is sigma^2 = sum_k w_k sigma_k^2.

    For d <= exact_max_dim the eigendecomposition is exact; otherwise the top-r
    pairs come from a randomized SVD on the ``top_n`` highest-responsibility
    points (what the paper scripts do at d = 3072).

    Returns:
        factors (K, d, rank), sigma2 (float)
    """
    n, d = x.shape
    K = means.shape[0]
    rank = min(rank, d)
    mass = resp.sum(0) + 1e-12                      # (K,)
    w = mass / mass.sum()
    L = torch.zeros(K, d, rank, device=x.device, dtype=x.dtype)
    s2 = torch.zeros(K, device=x.device, dtype=x.dtype)
    for k in range(K):
        rk = resp[:, k] / mass[k]
        diff = x - means[k]
        trace = (rk * (diff * diff).sum(-1)).sum()
        if rank == 0:
            s2[k] = trace / d
            continue
        if d <= exact_max_dim:
            S = (diff * rk.unsqueeze(1)).T @ diff
            evals, evecs = torch.linalg.eigh(S)
            lam_top, U = evals.flip(0)[:rank].clamp_min(0), evecs.flip(1)[:, :rank]
        else:
            top = torch.argsort(rk, descending=True)[:min(top_n, n)]
            Wm = torch.sqrt(rk[top]).unsqueeze(1) * diff[top]
            _, S_lr, V = torch.svd_lowrank(Wm, q=rank, niter=4)
            lam_top, U = S_lr * S_lr, V
        s2k = (trace - lam_top.sum()) / (d - rank) if d > rank else trace.new_tensor(0.0)
        s2k = s2k.clamp_min(sigma2_floor)
        L[k] = U * torch.sqrt((lam_top - s2k).clamp_min(0.0)).unsqueeze(0)
        s2[k] = s2k
    sigma2 = float((w * s2).sum().clamp_min(sigma2_floor))
    return L, sigma2


def fit_coreset_gmm(x: torch.Tensor, K: int, rank: int = 0, lam: float | None = None,
                    n_iter: int = 100, seed: int = 0, sigma2_floor: float = 1e-6,
                    init: str = "random", verbose: bool = False) -> LowRankGMM:
    """Stage I end to end: Sinkhorn coreset -> PPCA lift -> LowRankGMM."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means, weights, lam = sinkhorn_coreset(x, K, lam=lam, n_iter=n_iter,
                                           generator=g, init=init, verbose=verbose)
    resp = coreset_responsibilities(x, means, weights, lam)
    factors, sigma2 = fit_lowrank_covariances(x, means, resp, rank, sigma2_floor)
    gmm = LowRankGMM(weights, means, factors, sigma2)
    gmm.lam = lam
    if verbose:
        print(f"  [coreset] K={K} rank={rank} lam={lam:.4g} sigma2={sigma2:.4g}", flush=True)
    return gmm


def w2_to_empirical_upper_bound(x: torch.Tensor, gmm: LowRankGMM) -> float:
    """sqrt(E min_k ||x - mu_k||^2): a quick quantisation-error diagnostic."""
    return math.sqrt(_sq_dists(x, gmm.means).min(1).values.mean().item())
