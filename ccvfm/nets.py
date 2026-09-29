"""Reference correction networks with the interface model(v, tau, x_t, t)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        ang = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], 1))


class MLPCorrection(nn.Module):
    """Small residual MLP for low-dimensional data (toy experiments)."""

    def __init__(self, dim: int, hidden: int = 256, depth: int = 4, emb: int = 64):
        super().__init__()
        self.tau_emb = SinusoidalEmbedding(emb)
        self.t_emb = SinusoidalEmbedding(emb)
        self.inp = nn.Linear(2 * dim + 2 * emb, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, v, tau, x_t, t):
        h = self.inp(torch.cat([v, x_t, self.tau_emb(tau), self.t_emb(t)], 1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class _ResBlock(nn.Module):
    def __init__(self, ci, co, td):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.tp = nn.Linear(td, co)
        self.n1 = nn.GroupNorm(min(8, co), co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x))) + self.tp(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h))) + self.skip(x)


class UNetCorrection(nn.Module):
    """The MNIST correction U-Net of the paper (base -> 2 base -> 4 base).

    Takes flat vectors and reshapes them to (C, H, W); H and W must be
    divisible by 4. base=128 gives the ~21M-parameter headline network.
    """

    def __init__(self, shape=(1, 28, 28), base: int = 64, td: int = 128):
        super().__init__()
        self.shape = tuple(shape)
        C = self.shape[0]
        c1, c2, c3 = base, 2 * base, 4 * base
        self.tau_emb = SinusoidalEmbedding(td)
        self.t_emb = SinusoidalEmbedding(td)
        self.comb = nn.Linear(2 * td, td)
        self.e1 = _ResBlock(2 * C, c1, td)
        self.d1 = nn.Conv2d(c1, c1, 3, 2, 1)
        self.e2 = _ResBlock(c1, c2, td)
        self.d2 = nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = _ResBlock(c2, c3, td)
        self.u2 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.de2 = _ResBlock(c3, c2, td)
        self.u1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.de1 = _ResBlock(c2, c1, td)
        self.out = nn.Conv2d(c1, C, 1)

    def forward(self, v, tau, x_t, t):
        B = v.shape[0]
        te = self.comb(torch.cat([self.tau_emb(tau), self.t_emb(t)], 1))
        h1 = self.e1(torch.cat([v.view(B, *self.shape), x_t.view(B, *self.shape)], 1), te)
        h2 = self.e2(self.d1(h1), te)
        h = self.mid(self.d2(h2), te)
        h = self.de2(torch.cat([self.u2(h), h2], 1), te)
        h = self.de1(torch.cat([self.u1(h), h1], 1), te)
        return self.out(h).reshape(B, -1)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)
