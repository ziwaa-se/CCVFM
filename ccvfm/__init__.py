"""CCVFM: Coreset-Induced Conditional Velocity Flow Matching.

Three stages, three calls::

    gmm   = fit_coreset_gmm(x_train, K=..., rank=...)        # Stage I
    x     = sample(None, gmm, n)                              # Stage II (no network)
    loss  = ccvfm_loss(net, gmm, x_batch)                     # Stage III training
    x     = sample(net, gmm, n, L=10)                         # Stage III sampling
"""
from .coreset import (coreset_responsibilities, fit_coreset_gmm,
                      fit_lowrank_covariances, sinkhorn_coreset)
from .flow import ccvfm_loss, draw_source, sample
from .gmm import LowRankGMM
from .nets import EMA, MLPCorrection, UNetCorrection
from .velocity import (component_velocity, interpolant_marginal,
                       posterior_coupled_velocity, sample_velocity,
                       velocity_mixture_weights)

__version__ = "1.0.0"

__all__ = [
    "LowRankGMM", "fit_coreset_gmm", "sinkhorn_coreset", "coreset_responsibilities",
    "fit_lowrank_covariances", "sample_velocity", "posterior_coupled_velocity",
    "component_velocity", "velocity_mixture_weights", "interpolant_marginal",
    "ccvfm_loss", "draw_source", "sample", "MLPCorrection", "UNetCorrection", "EMA",
]
