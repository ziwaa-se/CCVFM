"""FID helper shared by the CIFAR-10 / ImageNet-32 evaluation scripts."""
from __future__ import annotations

import numpy as np


def compute_fid_robust(f1_np, f2_np, eps=1e-3):
    """Robust FID via numpy eigvals on (c1 @ c2).

    For PSD matrices c1, c2, tr(sqrt(c1 @ c2)) = sum of sqrt of eigenvalues of
    c1 @ c2. The product is NOT symmetric but has real nonneg eigenvalues (same
    as the symmetric conjugate c1^(1/2) c2 c1^(1/2)). Using `np.linalg.eigvals`
    on the product avoids the ill-conditioning failure modes of `eigh` on the
    symmetric conjugate, at the cost of complex arithmetic which we truncate
    to the real part.
    """
    f1 = f1_np.astype(np.float64)
    f2 = f2_np.astype(np.float64)
    m1 = f1.mean(0)
    m2 = f2.mean(0)
    diff = m1 - m2

    c1 = np.cov(f1, rowvar=False)
    c2 = np.cov(f2, rowvar=False)
    dim = c1.shape[0]
    c1 = c1 + eps * np.eye(dim)
    c2 = c2 + eps * np.eye(dim)

    # eigvals of c1 @ c2; truncate imaginary component and clip to nonneg
    prod = c1 @ c2
    eigvals = np.linalg.eigvals(prod)
    # filter out negligible imaginary parts, clip negatives
    eigvals = np.real_if_close(eigvals, tol=1e6).real
    eigvals = np.clip(eigvals, 0.0, None)
    trace_sqrt = float(np.sqrt(eigvals).sum())

    diff_sq = float(diff @ diff)
    tr_c1 = float(np.trace(c1))
    tr_c2 = float(np.trace(c2))
    fid = diff_sq + tr_c1 + tr_c2 - 2.0 * trace_sqrt
    return fid
