"""Correctness tests for the ccvfm package.

Every closed-form quantity is checked against a dense (d x d) reference or a
large Monte-Carlo sample. Runs on CPU in a few minutes:

    python -m pytest -q tests          # or: python tests/test_ccvfm.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ccvfm import (LowRankGMM, MLPCorrection, ccvfm_loss, component_velocity,  # noqa: E402
                   fit_coreset_gmm, posterior_coupled_velocity, sample,
                   sample_velocity, velocity_mixture_weights)

torch.set_default_dtype(torch.float64)


def random_gmm(K=3, d=5, r=2, seed=0, spread=3.0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(K, generator=g) + 0.2
    mu = spread * torch.randn(K, d, generator=g)
    L = 0.7 * torch.randn(K, d, r, generator=g)
    return LowRankGMM(w, mu, L, 0.3)


def mixture_moments(weights, means, covs):
    m = (weights[:, None] * means).sum(0)
    c = sum(w * (S + torch.outer(mu - m, mu - m)) for w, mu, S in zip(weights, means, covs))
    return m, c


def dense_velocity_law(gmm, x, t):
    """Paper Eq. (4): A_k = t^2 I + (1-t)^2 Sigma_k^{-1}, b_k = t x - (1-t) Sigma_k^{-1}(x - mu_k)."""
    d = gmm.dim
    I = torch.eye(d)
    S = gmm.dense_covariances()
    logits, means, covs = [], [], []
    for k in range(gmm.K):
        Si = torch.linalg.inv(S[k])
        A = t ** 2 * I + (1 - t) ** 2 * Si
        Lam = torch.linalg.inv(A)
        b = t * x - (1 - t) * Si @ (x - gmm.means[k])
        means.append(Lam @ b)
        covs.append(Lam)
        cov_xt = (1 - t) ** 2 * I + t ** 2 * S[k]
        logits.append(math.log(gmm.weights[k]) + torch.distributions.MultivariateNormal(
            t * gmm.means[k], cov_xt).log_prob(x))
    gamma = torch.softmax(torch.stack(logits), 0)
    return gamma, torch.stack(means), torch.stack(covs)


def assert_close(a, b, tol, what):
    err = (a - b).abs().max().item()
    assert err < tol, f"{what}: max abs error {err:.4g} >= {tol}"


# ----------------------------------------------------------------------------
def test_log_prob_matches_dense():
    gmm = random_gmm()
    x = torch.randn(50, gmm.dim) * 3
    S = gmm.dense_covariances()
    ref = torch.stack([torch.distributions.MultivariateNormal(gmm.means[k], S[k]).log_prob(x)
                       for k in range(gmm.K)], 1)
    assert_close(gmm.component_log_prob(x), ref, 1e-8, "component log-density")
    r0 = LowRankGMM(gmm.weights, gmm.means, torch.zeros(gmm.K, gmm.dim, 0), 0.5)
    ref0 = torch.stack([torch.distributions.MultivariateNormal(
        gmm.means[k], 0.5 * torch.eye(gmm.dim)).log_prob(x) for k in range(gmm.K)], 1)
    assert_close(r0.component_log_prob(x), ref0, 1e-8, "rank-0 log-density")


def test_mixture_weights_match_dense():
    gmm = random_gmm(seed=1)
    x = torch.randn(gmm.dim) * 2
    for t in (0.0, 0.3, 0.8):
        gamma, _, _ = dense_velocity_law(gmm, x, t)
        assert_close(velocity_mixture_weights(gmm, x[None], t)[0], gamma, 1e-8, f"gamma(t={t})")


def test_component_velocity_matches_closed_form():
    gmm = random_gmm(K=2, d=4, r=2, seed=2)
    x = torch.randn(gmm.dim)
    g = torch.Generator().manual_seed(0)
    n = 100_000
    for t in (0.0, 0.4, 0.9):
        _, m, C = dense_velocity_law(gmm, x, t)
        comp = torch.ones(n, dtype=torch.long)
        v = component_velocity(gmm, x.expand(n, -1), t, comp, generator=g)
        scale = C[1].diagonal().sqrt().max().item()
        assert_close(v.mean(0), m[1], 5 * scale / math.sqrt(n) * 3, f"mean (t={t})")
        assert_close(torch.cov(v.T), C[1], 0.03 * C[1].abs().max().item() + 1e-3, f"cov (t={t})")


def test_stage2_at_t0_is_translated_surrogate():
    gmm = random_gmm(seed=3)
    x0 = torch.randn(gmm.dim)
    n = 100_000
    v = sample_velocity(gmm, x0.expand(n, -1), 0.0, generator=torch.Generator().manual_seed(1))
    m, C = mixture_moments(gmm.weights, gmm.means, gmm.dense_covariances())
    assert_close((x0 + v).mean(0), m, 0.05, "mean of x0 + v")
    assert_close(torch.cov((x0 + v).T), C, 0.03 * C.abs().max().item(), "cov of x0 + v")


def test_source_draws_reproduce_true_joint_law():
    """(X_t, V_0) must have the law of (X_t, X_1 - X_0) when X_1 ~ rho1_tilde,
    for both the independent draw and the posterior (data-anchored) coupling."""
    gmm = random_gmm(K=3, d=3, r=1, seed=4)
    g = torch.Generator().manual_seed(2)
    n = 100_000
    for t in (0.0, 0.5):
        x1 = gmm.sample(n, generator=g)
        x0 = torch.randn(n, gmm.dim, generator=g)
        xt = (1 - t) * x0 + t * x1
        ref = torch.cat([xt, x1 - x0], 1)
        for name, v0 in [("independent", sample_velocity(gmm, xt, t, generator=g)),
                         ("posterior", posterior_coupled_velocity(gmm, x0, x1, t, generator=g))]:
            got = torch.cat([xt, v0], 1)
            tol = 0.03 * ref.var(0).max().item()
            assert_close(got.mean(0), ref.mean(0), 0.05, f"{name} joint mean (t={t})")
            assert_close(torch.cov(got.T), torch.cov(ref.T), tol, f"{name} joint cov (t={t})")


def test_posterior_coupling_shrinks_regression_target():
    gmm = random_gmm(K=4, d=6, r=2, seed=5, spread=6.0)
    g = torch.Generator().manual_seed(3)
    n = 50_000
    x1 = gmm.sample(n, generator=g)
    x0 = torch.randn(n, gmm.dim, generator=g)
    v1 = x1 - x0
    post = ((v1 - posterior_coupled_velocity(gmm, x0, x1, 0.0, generator=g)) ** 2).sum(1).mean()
    ind = ((v1 - sample_velocity(gmm, x0, 0.0, generator=g)) ** 2).sum(1).mean()
    gau = ((v1 - torch.randn(n, gmm.dim, generator=g)) ** 2).sum(1).mean()
    assert post < 0.5 * ind and post < 0.5 * gau, (post, ind, gau)


def test_coreset_recovers_separated_blobs():
    g = torch.Generator().manual_seed(6)
    centers = torch.tensor([[5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, -5.0]])
    x = torch.cat([c + 0.3 * torch.randn(2000, 2, generator=g) for c in centers])
    gmm = fit_coreset_gmm(x, K=4, rank=1, n_iter=50, seed=0, init="kmeans++")
    dist = torch.cdist(centers, gmm.means).min(1).values
    assert dist.max() < 0.2, dist
    assert_close(gmm.weights, torch.full((4,), 0.25), 0.02, "coreset weights")


def test_ppca_recovers_lowrank_covariance():
    g = torch.Generator().manual_seed(7)
    d, r, n = 10, 2, 100_000
    L = torch.randn(d, r, generator=g)
    S = L @ L.T + 0.1 * torch.eye(d)
    x = torch.distributions.MultivariateNormal(torch.zeros(d), S).sample((n,))
    gmm = fit_coreset_gmm(x, K=1, rank=r, n_iter=2, seed=0)
    assert_close(gmm.dense_covariances()[0], S, 0.05 * S.abs().max().item(), "PPCA covariance")
    assert abs(gmm.sigma2 - 0.1) < 0.01


def test_end_to_end_training_smoke():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(8)
    ang = torch.rand(4000, generator=g) * 2 * math.pi
    x = torch.stack([2 * torch.cos(ang), 2 * torch.sin(ang)], 1).float()
    x = x + 0.05 * torch.randn(x.shape, generator=g).float()
    gmm = fit_coreset_gmm(x, K=16, rank=1, seed=0).to("cpu")
    gmm = LowRankGMM(gmm.weights.float(), gmm.means.float(), gmm.factors.float(), gmm.sigma2)
    def train(source, steps):
        net = MLPCorrection(2, hidden=64, depth=2).float()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        before = [q.detach().clone() for q in net.parameters()]
        for _ in range(steps):
            loss = ccvfm_loss(net, gmm, x[torch.randint(0, len(x), (256,))], source=source)
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        moved = any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
        return net, moved

    def eval_loss(model, source):
        g_eval = torch.Generator().manual_seed(123)
        with torch.no_grad():
            return ccvfm_loss(model, gmm, x[:4000], source=source, generator=g_eval).item()

    zero = lambda v, tau, xt, t: torch.zeros_like(v)  # noqa: E731
    # The loop learns: from an N(0, I) source there is a large learnable signal.
    net_g, moved = train("gaussian", 500)
    assert moved
    assert eval_loss(net_g, "gaussian") < 0.8 * eval_loss(zero, "gaussian")
    # CCVFM: the surrogate source already sits on the ring, so the regression target
    # is near its noise floor and there is little left to learn; check the mechanics.
    net, moved = train("posterior", 300)
    assert moved
    assert eval_loss(net, "posterior") < 1.05 * eval_loss(zero, "posterior")
    assert eval_loss(zero, "posterior") < 0.1 * eval_loss(zero, "gaussian")
    out = sample(net, gmm, 500, L=4)
    assert out.shape == (500, 2) and torch.isfinite(out).all()
    out2 = sample(net, gmm, 100, L=2, J=2)
    assert out2.shape == (100, 2) and torch.isfinite(out2).all()


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
