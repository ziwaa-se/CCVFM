"""Low-rank-plus-isotropic Gaussian mixture, the coreset surrogate of CCVFM.

The surrogate target is

    rho1_tilde(x) = sum_k w_k N(x; mu_k, Sigma_k),   Sigma_k = L_k L_k^T + sigma^2 I_d,

with L_k in R^{d x r}. Every quantity used by CCVFM (densities, posterior
responsibilities, sampling) is evaluated in O(K d r) per point through the
Woodbury identity, so nothing of size d x d is ever materialised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class LowRankGMM:
    """K-component mixture with covariances L_k L_k^T + sigma2 * I.

    Attributes:
        weights: (K,) mixture weights, summing to one.
        means:   (K, d) component means (the coreset atoms).
        factors: (K, d, r) low-rank covariance factors; r may be 0.
        sigma2:  shared isotropic noise variance (a positive float).
    """

    weights: torch.Tensor
    means: torch.Tensor
    factors: torch.Tensor
    sigma2: float

    def __post_init__(self):
        K, d = self.means.shape
        if self.factors.dim() != 3 or self.factors.shape[:2] != (K, d):
            raise ValueError(f"factors must have shape (K={K}, d={d}, r), "
                             f"got {tuple(self.factors.shape)}")
        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        self.weights = self.weights / self.weights.sum()
        self._precompute()

    # ------------------------------------------------------------------
    # shapes / device handling
    # ------------------------------------------------------------------
    @property
    def K(self) -> int:
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def rank(self) -> int:
        return self.factors.shape[2]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def to(self, device) -> "LowRankGMM":
        return LowRankGMM(self.weights.to(device), self.means.to(device),
                          self.factors.to(device), float(self.sigma2))

    def _precompute(self):
        """Cache M_k = I_r + L_k^T L_k / sigma2 and log det Sigma_k."""
        r, s2 = self.rank, self.sigma2
        eye = torch.eye(r, device=self.device, dtype=self.means.dtype)
        LtL = torch.einsum("kdr,kds->krs", self.factors, self.factors)
        self._M = eye.unsqueeze(0) + LtL / s2                      # (K, r, r)
        self._M_chol = torch.linalg.cholesky(self._M) if r > 0 else self._M
        logdet_M = (2.0 * torch.log(torch.diagonal(self._M_chol, dim1=-2, dim2=-1)).sum(-1)
                    if r > 0 else torch.zeros(self.K, device=self.device, dtype=self.means.dtype))
        self._logdet = logdet_M + self.dim * math.log(s2)          # (K,)
        self.log_weights = torch.log(self.weights.clamp_min(1e-30))

    # ------------------------------------------------------------------
    # densities
    # ------------------------------------------------------------------
    def component_log_prob(self, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """log N(x_i; mu_k, Sigma_k) for every point and component -> (n, K)."""
        out = []
        for s in range(0, x.shape[0], chunk):
            out.append(self._component_log_prob(x[s:s + chunk]))
        return torch.cat(out, 0)

    def _component_log_prob(self, x):
        s2, d = self.sigma2, self.dim
        # ||x - mu_k||^2 via the expansion trick, (B, K)
        sq = ((x * x).sum(-1, keepdim=True) - 2.0 * x @ self.means.T
              + (self.means * self.means).sum(-1).unsqueeze(0)).clamp_min(0.0)
        quad = sq / s2
        if self.rank > 0:
            # Woodbury: Sigma^{-1} = (I - L M^{-1} L^T / s2) / s2
            diffL = (torch.einsum("bd,kdr->bkr", x, self.factors)
                     - torch.einsum("kd,kdr->kr", self.means, self.factors).unsqueeze(0))
            # solve M_k y = diffL via the cached Cholesky factor
            y = torch.cholesky_solve(diffL.permute(1, 2, 0), self._M_chol)  # (K, r, B)
            quad = quad - (diffL.permute(1, 2, 0) * y).sum(1).T / s2 ** 2
        return -0.5 * (d * LOG_2PI + self._logdet.unsqueeze(0) + quad)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log rho1_tilde(x) -> (n,)."""
        return torch.logsumexp(self.log_weights.unsqueeze(0) + self.component_log_prob(x), -1)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior P(component = k | x) -> (n, K)."""
        return torch.softmax(self.log_weights.unsqueeze(0) + self.component_log_prob(x), -1)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample_component(self, comp: torch.Tensor, generator=None) -> torch.Tensor:
        """x ~ N(mu_c, Sigma_c) for a vector of component indices c."""
        n = comp.shape[0]
        x = self.means[comp] + math.sqrt(self.sigma2) * torch.randn(
            n, self.dim, device=self.device, dtype=self.means.dtype, generator=generator)
        if self.rank > 0:
            z = torch.randn(n, self.rank, device=self.device, dtype=self.means.dtype,
                            generator=generator)
            x = x + torch.einsum("bdr,br->bd", self.factors[comp], z)
        return x

    def sample(self, n: int, generator=None) -> torch.Tensor:
        comp = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        return self.sample_component(comp, generator)

    def dense_covariances(self) -> torch.Tensor:
        """(K, d, d) covariances. For tests and small d only."""
        eye = torch.eye(self.dim, device=self.device, dtype=self.means.dtype)
        return (torch.einsum("kdr,ker->kde", self.factors, self.factors)
                + self.sigma2 * eye.unsqueeze(0))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"weights": self.weights.cpu(), "means": self.means.cpu(),
                "factors": self.factors.cpu(), "sigma2": float(self.sigma2)}

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu") -> "LowRankGMM":
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        # accept both our keys and the ones written by the paper scripts
        factors = sd.get("factors")
        sigma2 = sd.get("sigma2", sd.get("noise_var"))
        return cls(as_t(sd["weights"]), as_t(sd["means"]), as_t(factors), float(sigma2))

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device="cpu") -> "LowRankGMM":
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False),
                                   device=device)
