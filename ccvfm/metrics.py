"""Evaluation helpers: sliced Wasserstein (toy data) and Inception FID (images)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256,
                       seed: int = 0) -> float:
    """Sliced W2 between two equally sized point clouds."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.randn(x.shape[1], n_proj, generator=g).to(x.device)
    theta = theta / theta.norm(dim=0, keepdim=True)
    px, _ = torch.sort(x @ theta, 0)
    py, _ = torch.sort(y @ theta, 0)
    return float(((px - py) ** 2).mean().sqrt())


class InceptionFeatures(nn.Module):
    """2048-d InceptionV3 pool features, the protocol used for all paper FIDs.

    Input: images in [0, 1], shape (B, C, H, W) with C in {1, 3}.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.net.fc = nn.Identity()
        self.net.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)

    @torch.no_grad()
    def features(self, imgs: torch.Tensor, batch_size: int = 128) -> np.ndarray:
        dev = self.mean.device
        return np.concatenate([self(imgs[i:i + batch_size].to(dev)).cpu().numpy()
                               for i in range(0, len(imgs), batch_size)])


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    """FID between two feature sets (n, 2048)."""
    from scipy.linalg import sqrtm
    m1, m2 = f1.mean(0), f2.mean(0)
    c1 = np.cov(f1, rowvar=False) + eps * np.eye(f1.shape[1])
    c2 = np.cov(f2, rowvar=False) + eps * np.eye(f2.shape[1])
    cm = sqrtm(c1 @ c2)
    if np.iscomplexobj(cm):
        cm = cm.real
    d = m1 - m2
    return float(d @ d + np.trace(c1 + c2 - 2.0 * cm))
