"""Stage III: the correction flow in velocity space, and the nested sampler.

The correction network f(v, tau, x_t, t) is trained with the flow-matching loss

    E || f(V_tau, tau, X_t, t) - (V_1 - V_0) ||^2,   V_tau = (1 - tau) V_0 + tau V_1,

where V_1 = X_1 - X_0 is the true velocity and V_0 is drawn from the closed-form
surrogate law pi_tilde(. | X_t, t) instead of N(0, I) (the HRF2 choice). The net
therefore learns a surrogate-to-target correction, not a noise-to-data map.

Model interface used everywhere in this package::

    model(v: (B, d), tau: (B,), x_t: (B, d), t: (B,)) -> (B, d)
"""
from __future__ import annotations

import torch

from .gmm import LowRankGMM
from .velocity import posterior_coupled_velocity, sample_velocity

SOURCES = ("posterior", "independent", "gaussian")


def draw_source(gmm: LowRankGMM, x0, x1, t: float, source: str = "posterior",
                generator=None):
    """Source velocity V_0 for a training batch.

    posterior   : CCVFM default, surrogate law anchored to the component of x1.
    independent : surrogate law drawn independently of x1.
    gaussian    : N(0, I), i.e. the HRF2 baseline (gmm is ignored).
    """
    if source == "posterior":
        return posterior_coupled_velocity(gmm, x0, x1, t, generator)
    if source == "independent":
        x_t = (1.0 - t) * x0 + t * x1
        return sample_velocity(gmm, x_t, t, generator)
    if source == "gaussian":
        return torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    raise ValueError(f"source must be one of {SOURCES}")


def ccvfm_loss(model, gmm: LowRankGMM, x1: torch.Tensor, t: float = 0.0,
               source: str = "posterior", generator=None) -> torch.Tensor:
    """One Monte-Carlo estimate of the Stage III loss at outer time t.

    ``t = 0`` is the recommended J = 1 setting (the one covered by the theory).
    Pass ``t = float(torch.rand(()))`` per step to train for the J > 1 sampler.
    """
    B = x1.shape[0]
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    x_t = (1.0 - t) * x0 + t * x1
    v1 = x1 - x0
    with torch.no_grad():
        v0 = draw_source(gmm, x0, x1, t, source, generator)
    tau = torch.rand(B, device=x1.device, dtype=x1.dtype, generator=generator)
    v_tau = (1.0 - tau).unsqueeze(1) * v0 + tau.unsqueeze(1) * v1
    t_vec = torch.full((B,), float(t), device=x1.device, dtype=x1.dtype)
    pred = model(v_tau, tau, x_t, t_vec)
    return ((pred - (v1 - v0)) ** 2).mean()


@torch.no_grad()
def sample(model, gmm: LowRankGMM, n: int, L: int = 10, J: int = 1,
           source: str = "surrogate", generator=None, batch_size: int = 4096):
    """Nested CCVFM sampler (Algorithm 2 of the paper).

    For J = 1 (recommended): x0 ~ N(0, I); v ~ pi_tilde(. | x0, 0) in closed
    form; L Euler steps of the correction flow; return x0 + v.
    NFE = J * L network evaluations (+ J closed-form draws).

    ``model=None`` or ``L=0`` returns the Stage II (network-free) sample.
    ``source="gaussian"`` starts the inner flow from N(0, I) (HRF2 sampler).
    """
    device, dtype = gmm.device, gmm.means.dtype
    grid = torch.linspace(0.0, 1.0, J + 1).tolist()
    out = []
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        z = torch.randn(b, gmm.dim, device=device, dtype=dtype, generator=generator)
        for j in range(J):
            t = grid[j]
            if source == "gaussian":
                v = torch.randn(b, gmm.dim, device=device, dtype=dtype, generator=generator)
            else:
                v = sample_velocity(gmm, z, t, generator)
            if model is not None and L > 0:
                t_vec = torch.full((b,), t, device=device, dtype=dtype)
                for i in range(L):
                    tau = torch.full((b,), i / L, device=device, dtype=dtype)
                    v = v + model(v, tau, z, t_vec) / L
            z = z + (grid[j + 1] - t) * v
        out.append(z)
    return torch.cat(out, 0)
