"""Toy-data helpers for the review-period experiments.

Copied verbatim from the research code that produced the reported numbers:
the Stage-I EMS coreset + closed-form PPCA lift (NumPy), the target samplers,
and the sliced-W2 diagnostic. Only what the scripts in this folder use.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class GaussianMixture:
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_gmm(gmm: GaussianMixture, n: int, rg: np.random.Generator) -> np.ndarray:
    comp = rg.choice(len(gmm.weights), size=n, p=gmm.weights)
    d = gmm.means.shape[1]
    out = np.zeros((n, d))
    for k in range(len(gmm.weights)):
        mask = comp == k
        if np.any(mask):
            out[mask] = rg.multivariate_normal(gmm.means[k], gmm.covs[k], size=mask.sum())
    return out


def make_ring6_target(radius: float = 2.0, sigma: float = 0.20) -> GaussianMixture:
    angles = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    cov = (sigma**2) * np.eye(2)
    covs = np.repeat(cov[None, :, :], len(means), axis=0)
    weights = np.full(len(means), 1.0 / len(means))
    return GaussianMixture(weights=weights, means=means, covs=covs)


def sample_pinwheel(
    n: int,
    rg: np.random.Generator,
    n_arms: int = 5,
    radial_std: float = 0.30,
    tangential_std: float = 0.08,
    rate: float = 0.25,
) -> np.ndarray:
    """Pinwheel distribution: elongated clusters rotated around the origin."""
    per_arm = n // n_arms
    remainder = n - per_arm * n_arms
    pts_list = []
    for k in range(n_arms):
        nk = per_arm + (1 if k < remainder else 0)
        angle = 2.0 * math.pi * k / n_arms
        r = rg.normal(loc=1.5, scale=radial_std, size=nk)
        t = rg.normal(scale=tangential_std, size=nk) + rate * r
        x = r * math.cos(angle) - t * math.sin(angle)
        y = r * math.sin(angle) + t * math.cos(angle)
        pts_list.append(np.stack([x, y], axis=1))
    pts = np.concatenate(pts_list, axis=0)
    return pts[rg.permutation(len(pts))]


def sample_moons(n: int, rg: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    theta1 = rg.uniform(0.0, math.pi, size=n1)
    theta2 = rg.uniform(0.0, math.pi, size=n2)
    moon1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    moon2 = np.stack([1.0 - np.cos(theta2), -np.sin(theta2) - 0.5], axis=1)
    pts = np.concatenate([moon1, moon2], axis=0)
    pts *= 1.7
    pts += rg.normal(scale=noise, size=pts.shape)
    return pts


def sample_helix3d(n: int, rg: np.random.Generator, jitter: float = 0.05) -> np.ndarray:
    t = rg.uniform(0.0, 4.0 * math.pi, size=n)
    x = np.cos(t)
    y = np.sin(t)
    z = (t - 2.0 * math.pi) / math.pi
    pts = np.stack([x, y, z], axis=1)
    pts += rg.normal(scale=jitter, size=pts.shape)
    return pts


def sample_checkerboard(n, rg):
    """4x4 checkerboard on [-2,2]^2, mass on 'black' squares."""
    ij = rg.integers(0, 4, size=(2 * n, 2))
    keep = (ij.sum(1) % 2 == 0)
    ij = ij[keep][:n]
    while len(ij) < n:
        extra = rg.integers(0, 4, size=(2 * n, 2))
        extra = extra[(extra.sum(1) % 2 == 0)]
        ij = np.concatenate([ij, extra])[:n]
    return -2.0 + (ij + rg.uniform(0, 1, size=ij.shape))


def sample_thin_circle(n, rg, radius=1.0, thickness=0.02):
    theta = rg.uniform(0.0, 2.0 * math.pi, size=n)
    r = radius + rg.normal(scale=thickness, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def sample_two_moons(n, rg, noise=0.04):
    n_outer = n // 2
    n_inner = n - n_outer
    th_o = rg.uniform(0.0, math.pi, size=n_outer)
    outer = np.stack([np.cos(th_o), np.sin(th_o)], axis=1)
    th_i = rg.uniform(0.0, math.pi, size=n_inner)
    inner = np.stack([1.0 - np.cos(th_i), 0.5 - np.sin(th_i)], axis=1)
    pts = np.concatenate([outer, inner], axis=0)
    pts += rg.normal(scale=noise, size=pts.shape)
    return pts[rg.permutation(len(pts))]


def ems_coreset(
    x: np.ndarray,
    k: int,
    lam: float,
    n_iter: int,
    rg: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, _ = x.shape
    init_idx = rg.choice(n, size=k, replace=False)
    y = x[init_idx].copy()
    w = np.full(k, 1.0 / k)
    losses = []

    for _ in range(n_iter):
        sqdist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        logits = np.log(w + 1e-300)[None, :] - sqdist / lam
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        tmat = np.exp(logits - log_norm)

        nj = tmat.sum(axis=0) + 1e-12
        y = (tmat.T @ x) / nj[:, None]
        w = nj / nj.sum()

        entropy = np.sum(tmat * (np.log(tmat + 1e-300) - np.log(w + 1e-300)[None, :]))
        loss = np.sum(tmat * sqdist) + lam * entropy
        losses.append(loss / n)

    sqdist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    logits = np.log(w + 1e-300)[None, :] - sqdist / lam
    tmat = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return y, w, np.array(losses), tmat


def coreset_to_gmm(
    x: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    responsibilities: np.ndarray,
    cov_floor: float = 0.0,
    rank: int | None = None,
) -> GaussianMixture:
    """Build the Stage-I GMM by per-component closed-form PPCA.

    Each component covariance is decomposed as
        Sigma_b = L_b L_b^T + sigma_b^2 I_d
    via the Tipping-Bishop PPCA MLE on the soft-assignment-weighted
    empirical covariance:
        sigma_b^2  = mean of the trailing (d - rank) eigenvalues
        L_b        = U_r diag(sqrt(max(lambda_j - sigma_b^2, 0)))
    where U_r and lambda_j are the top-rank eigenvectors and eigenvalues
    of the weighted empirical covariance. For rank = d - 1, this is
    identical to the empirical covariance with sigma_b^2 = smallest
    eigenvalue (no information loss); for smaller rank, the trailing
    eigenvalues are pooled into the isotropic noise floor sigma_b^2 I.

    cov_floor is preserved as a numerical safety floor on sigma_b^2; the
    learned sigma_b^2 is clamped from below by cov_floor**2.
    """
    k, d = centers.shape
    if rank is None:
        rank = max(d - 1, 1)
    rank = int(min(max(rank, 0), d))

    covs = np.zeros((k, d, d))
    for j in range(k):
        r = responsibilities[:, j]
        mass = r.sum() + 1e-12
        diff = x - centers[j]
        # Soft-assignment-weighted empirical covariance
        S = np.einsum("n,ni,nj->ij", r, diff, diff) / mass

        # Eigendecomposition (ascending order from eigh)
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = eigvals[::-1]               # descending
        eigvecs = eigvecs[:, ::-1]
        eigvals = np.clip(eigvals, 0.0, None)  # numerical safety

        # Closed-form PPCA MLE (Tipping-Bishop, 1999)
        if rank < d:
            sigma2 = float(np.mean(eigvals[rank:]))
        else:
            sigma2 = 0.0
        sigma2 = max(sigma2, cov_floor ** 2, 1e-10)
        Lambda_r = np.maximum(eigvals[:rank] - sigma2, 0.0)
        L = eigvecs[:, :rank] * np.sqrt(Lambda_r)[None, :]  # (d, rank)
        covs[j] = L @ L.T + sigma2 * np.eye(d)
    return GaussianMixture(weights=weights, means=centers, covs=covs)


def sliced_w2(x: np.ndarray, y: np.ndarray, n_proj: int = 128, rg: np.random.Generator | None = None) -> float:
    rg = np.random.default_rng(0) if rg is None else rg
    dirs = rg.standard_normal((n_proj, x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = []
    for dvec in dirs:
        px = np.sort(x @ dvec)
        py = np.sort(y @ dvec)
        vals.append(np.mean((px - py) ** 2))
    return float(np.sqrt(np.mean(vals)))


# The submission's five toy targets: name, sampler, dim, K, lam, cov_floor
TARGETS = [
    ("ring6", lambda n, rg_: sample_gmm(make_ring6_target(), n, rg_), 2, 12, 0.10, 0.025),
    ("moons", sample_moons, 2, 18, 0.10, 0.018),
    ("pinwheel", sample_pinwheel, 2, 15, 0.10, 0.015),
    ("checkerboard", sample_checkerboard, 2, 16, 0.10, 0.020),
    ("helix", sample_helix3d, 3, 20, 0.12, 0.020),
]


def build_gmm(x, K, lam, floor, rg_, rank=None):
    centers, weights, _, tmat = ems_coreset(x, K, lam, 45, rg_)
    return coreset_to_gmm(x, centers, weights, tmat, cov_floor=floor, rank=rank)
