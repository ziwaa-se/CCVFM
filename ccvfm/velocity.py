"""Stage II: the closed-form conditional velocity law of the surrogate.

For the straight interpolation X_t = (1 - t) X_0 + t X_1 with X_0 ~ N(0, I) and
X_1 ~ rho1_tilde (a low-rank GMM), the conditional law of V = X_1 - X_0 given
X_t = x is again a K-component Gaussian mixture (HRF2, Thm. 1 / Cor. 1):

    pi_tilde(v | x, t) = sum_k gamma_k(x, t) N(v; m_k(x, t), Lambda_k(t)),
    gamma_k(x, t) ∝ w_k N(x; t mu_k, (1 - t)^2 I + t^2 Sigma_k).

At the generation boundary t = 0 this collapses to pi_tilde(v | x0, 0) =
rho1_tilde(x0 + v): draw a component from Cat(w) and a Gaussian, no network.

Sampling is done exactly through the latent representation
X_1 = mu_k + L_k eps_r + sigma eps_d, which keeps every operation O(d r).
"""
from __future__ import annotations

import math

import torch

from .gmm import LowRankGMM


def interpolant_marginal(gmm: LowRankGMM, t: float) -> LowRankGMM:
    """Law of X_t = (1-t) X_0 + t X_1 under the surrogate: again a low-rank GMM."""
    s2_t = (1.0 - t) ** 2 + t ** 2 * gmm.sigma2
    return LowRankGMM(gmm.weights, t * gmm.means, t * gmm.factors, s2_t)


def component_velocity(gmm: LowRankGMM, x_t: torch.Tensor, t: float,
                       comp: torch.Tensor, generator=None) -> torch.Tensor:
    """v ~ N(m_c(x_t, t), Lambda_c(t)): the velocity law of component c given X_t."""
    if not 0.0 <= t < 1.0:
        raise ValueError("t must lie in [0, 1)")
    B, d = x_t.shape
    s2, r = gmm.sigma2, gmm.rank
    s2_t = (1.0 - t) ** 2 + t ** 2 * s2
    mu = gmm.means[comp]
    diff = x_t - t * mu                                   # = t L eps_r + noise

    x1 = mu.clone()
    if r > 0:
        Lc = gmm.factors[comp]                            # (B, d, r)
        eye = torch.eye(r, device=x_t.device, dtype=x_t.dtype)
        M_t = eye + (t ** 2 / s2_t) * torch.einsum("bdr,bds->brs", Lc, Lc)
        C = torch.linalg.cholesky(M_t)                    # M_t = C C^T
        rhs = (t / s2_t) * torch.einsum("bdr,bd->br", Lc, diff)
        mean_r = torch.cholesky_solve(rhs.unsqueeze(-1), C).squeeze(-1)
        z_r = torch.randn(B, r, 1, device=x_t.device, dtype=x_t.dtype, generator=generator)
        # C^{-T} z has covariance M_t^{-1}
        eps_r = mean_r + torch.linalg.solve_triangular(
            C.transpose(-1, -2), z_r, upper=True).squeeze(-1)
        Leps = torch.einsum("bdr,br->bd", Lc, eps_r)
        diff = diff - t * Leps
        x1 = x1 + Leps
    # eps_d | residual ~ N(t sigma / s2_t * residual, (1-t)^2 / s2_t I)
    sigma = math.sqrt(s2)
    z_d = torch.randn(B, d, device=x_t.device, dtype=x_t.dtype, generator=generator)
    eps_d = (t * sigma / s2_t) * diff + ((1.0 - t) / math.sqrt(s2_t)) * z_d
    x1 = x1 + sigma * eps_d
    return (x1 - x_t) / (1.0 - t)


def velocity_mixture_weights(gmm: LowRankGMM, x_t: torch.Tensor, t: float) -> torch.Tensor:
    """gamma_k(x_t, t) -> (B, K). Equals w_k at t = 0."""
    if t == 0.0:
        return gmm.weights.unsqueeze(0).expand(x_t.shape[0], -1)
    return interpolant_marginal(gmm, t).responsibilities(x_t)


@torch.no_grad()
def sample_velocity(gmm: LowRankGMM, x_t: torch.Tensor, t: float = 0.0,
                    generator=None) -> torch.Tensor:
    """Stage II draw v ~ pi_tilde(. | x_t, t). Zero learned parameters."""
    gamma = velocity_mixture_weights(gmm, x_t, t)
    comp = torch.multinomial(gamma, 1, generator=generator).squeeze(-1)
    return component_velocity(gmm, x_t, t, comp, generator)


@torch.no_grad()
def posterior_coupled_velocity(gmm: LowRankGMM, x0: torch.Tensor, x1: torch.Tensor,
                               t: float = 0.0, generator=None) -> torch.Tensor:
    """Training-time source draw that is anchored to the data point x1.

    The component is drawn from the surrogate posterior P(k | x1) and the
    velocity from that component's conditional law given X_t. When x1 is
    distributed as rho1_tilde, P(k | x_t) = E[P(k | x1) | x_t] = gamma_k(x_t, t),
    so the marginal of the returned v0 is exactly pi_tilde(. | x_t, t): the
    training source and the inference source coincide, while each pair
    (v0, v1) stays inside one mode.
    """
    comp = torch.multinomial(gmm.responsibilities(x1), 1, generator=generator).squeeze(-1)
    x_t = (1.0 - t) * x0 + t * x1
    return component_velocity(gmm, x_t, t, comp, generator)
