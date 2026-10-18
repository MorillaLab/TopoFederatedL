"""Unit tests for ptopofl_v2.py: implementation properties and the main theoretical results,
checked on the actual code (run:  pytest -q tests)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ptopofl_v2 as P  # noqa: E402


@pytest.fixture(scope="module")
def fed():
    raw = P.make_synthetic("aligned", K=6, seed=0, shift=1.0)
    cl, eta, n = P.build(raw)
    lab, w, pi = P.plain_structure(np.array([c.group for c in cl]), n)
    return cl, eta, n, lab, w, pi


def _pinorm(D, pi):
    return float(np.sqrt(np.sum(pi * np.sum(D ** 2, axis=1))))


# ---------------------------------------------------------------- descriptor
def test_descriptor_dimension_and_invariances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))                      # n <= n_sub: no subsampling, exact checks
    phi = P.ph_descriptor(X, seed=1, n_rep=1)
    assert phi.shape == (46,) and np.all(np.isfinite(phi))
    D = rng.uniform(0.5, 3.0, size=6); b = rng.normal(size=6)
    assert np.allclose(P.ph_descriptor(X * D + b, seed=1, n_rep=1), phi, atol=1e-6)   # coordinate-wise affine
    assert np.allclose(P.ph_descriptor(X[rng.permutation(60)], seed=1, n_rep=1), phi, atol=1e-6)  # records


def test_h1_backends_agree():
    if not P.HAVE_RIPSER:
        pytest.skip("ripser not installed")
    from scipy.spatial.distance import pdist, squareform
    X = np.random.default_rng(3).normal(size=(50, 5))
    Dm = squareform(pdist(X))
    a = np.array(sorted(map(tuple, P._h1_pairs_numpy(Dm))))
    b = np.array(sorted(map(tuple, P._h1_pairs(Dm))))
    assert a.shape == b.shape and np.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------- Theorem 3 (contraction)
def test_round_operator_is_a_q_contraction(fed):
    cl, eta, n, lab, w, pi = fed
    tau = 5; q = (1 - eta * P.CFG["lam"]) ** tau
    rng = np.random.default_rng(1)
    for alpha in (0.0, 0.3, 1.0):
        T1, T2 = rng.normal(size=(2, cl[0].p)) * 2, rng.normal(size=(2, cl[0].p)) * 2
        A = P.run_clustered(cl, lab, w, pi, alpha, 1, tau, eta, 0, Theta0=T1)[2]
        B = P.run_clustered(cl, lab, w, pi, alpha, 1, tau, eta, 0, Theta0=T2)[2]
        assert _pinorm(A - B, pi) <= q * _pinorm(T1 - T2, pi) * (1 + 1e-9)


# ---------------------------------------------------------------- Theorem 4 (drift) and tau = 1
def test_fedavg_with_one_local_step_is_exact():
    cl, eta, n = P.build(P.make_synthetic("aligned", K=4, seed=2, shift=1.0))
    pk = n / n.sum()
    th = P.run_fedavg(cl, 3000, 1, eta, 0)[0][0]
    assert np.linalg.norm(th - P.weighted_optimum(cl, pk)) < 1e-5


def test_drift_bias_below_theorem_bound(fed):
    cl, eta, n, lab, w, pi = fed
    tau, lam = 5, P.CFG["lam"]; q = (1 - eta * lam) ** tau; L = max(c.L for c in cl)
    star = np.array([P.weighted_optimum([cl[k] for k in np.where(lab == j)[0]], w[lab == j]) for j in range(2)])
    That = P.run_clustered(cl, lab, w, pi, 0.0, 4000, tau, eta, 0, Theta0=star)[2]
    for j in range(2):
        C = np.where(lab == j)[0]
        zeta = sum(w[k] * np.linalg.norm(cl[k].grad(star[j])) for k in C)
        c = min(eta * L * (tau - 1) * (1 + eta * lam * tau) / 2, (1 + q) / (1 - q))
        assert np.linalg.norm(That[j] - star[j]) <= c * zeta / lam + 1e-8


# ---------------------------------------------------------------- Proposition 5 (calibration)
@pytest.mark.parametrize("abar", [0.1, 0.3, 0.9])
@pytest.mark.parametrize("q", [0.2, 0.9, 0.99])
def test_alpha_calibration_inverts_effective_blending(abar, q):
    a = P.alpha_from_effective(abar, 1.0, 1, 1.0 - q)   # eta = 1, tau = 1  =>  q = 1 - lam
    assert abs(a / (1 - (1 - a) * q) - abar) < 1e-12


# ---------------------------------------------------------------- Proposition 6 (decomposition)
def test_heterogeneity_decomposition_bounds(fed):
    cl, eta, n, lab, w, pi = fed
    h = P.heterogeneity(cl, lab, w, pi)
    assert 0.0 <= h["Vin"] <= h["Vtot"] + 1e-12
    one = P.heterogeneity(cl, *P.plain_structure(np.zeros(len(cl), int), n))
    alone = P.heterogeneity(cl, *P.plain_structure(np.arange(len(cl)), n))
    assert abs(one["rho"] - 1.0) < 1e-12 and abs(alone["rho"]) < 1e-12


# ---------------------------------------------------------------- Theorem 9 (partition stability)
def test_partition_stable_under_perturbations_below_quarter_margin():
    rng = np.random.default_rng(4)
    V = np.vstack([rng.normal(0, 0.05, (4, 3)), rng.normal(0, 0.05, (4, 3)) + 3.0])
    truth = np.r_[np.zeros(4, int), np.ones(4, int)]
    from scipy.spatial.distance import cdist
    Dm = cdist(V, V); same = truth[:, None] == truth[None, :]
    gamma = Dm[~same].min() - Dm[same].max()
    assert gamma > 0
    for _ in range(20):
        E = rng.normal(size=V.shape); E *= (0.99 * gamma / 4) / np.linalg.norm(E, axis=1, keepdims=True)
        assert P.ari(P.agglomerative(V + E, 2), truth) == 1.0


# ---------------------------------------------------------------- Propositions 11-12 (trust)
def test_mean_std_scores_are_capped_robust_scores_are_not():
    # 6 identical honest descriptors, 2 identical corrupted ones far away (epsilon = 1/4)
    Fh = np.vstack([np.zeros((6, 2)), np.full((2, 2), 100.0)])
    t_std, z_std = P.trust_weights(Fh, rule="std")
    assert np.allclose(z_std[6:], math.sqrt(3.0))                       # sqrt((1-eps)/eps)
    assert np.allclose(t_std[6:], math.exp(1 - math.sqrt(3.0)))        # ~0.48 whatever the distance
    t_mad, _ = P.trust_weights(Fh, rule="mad")
    assert np.all(t_mad[6:] < 1e-6) and np.all(t_mad[:6] == 1.0)


# ---------------------------------------------------------------- Proposition 2 (DP accounting)
def test_dp_budget_split_reproduces_target_epsilon():
    delta, steps = 1e-5, 200
    for eps in (1.0, 8.0, 64.0):
        for f in (0.0, 0.5):
            s_dp, s_phi = P.dp_sigmas(eps, delta, steps, f)
            R = steps / (2 * s_dp ** 2) + (2 / s_phi ** 2 if s_phi > 0 else 0.0)
            assert abs(R + 2 * math.sqrt(R * math.log(1 / delta)) - eps) < 1e-9


def test_analytic_gaussian_mechanism_is_monotone_and_tighter_than_classical():
    e = [P.analytic_gauss_eps(s, 2.0, 1e-5) for s in (5.0, 10.0, 20.0)]
    assert e[0] > e[1] > e[2] > 0
    classical = 2.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 20.0   # valid since < 1
    assert e[2] <= classical


# ---------------------------------------------------------------- baselines and the README example
def test_all_methods_run_and_scaffold_reaches_the_global_optimum():
    cl, eta, n = P.build(P.make_synthetic("aligned", K=4, seed=5, shift=1.0))
    for m in P.METHODS_MAIN:
        models, _ = P.run_method(m, cl, eta, n, 0, 3, 2, 0)
        assert models.shape == (len(cl), cl[0].p) and np.all(np.isfinite(models))
    x = P.run_scaffold(cl, 1500, 2, eta, 0)[0][0]
    assert np.linalg.norm(x - P.weighted_optimum(cl, n / n.sum())) < 1e-4


def test_readme_example():
    raw = P.make_synthetic("aligned", K=8, seed=0, shift=1.0)
    clients, eta, n = P.build(raw)
    models, history, info = P.ptopofl_fit(clients, eta, n, seed=0, T=20, tau=10, ev=10)
    r = P.evaluate(clients, models)
    assert 0 < r["logloss"] < 1 and 0.5 < r["auc"] <= 1
    assert set(np.unique(info["labels"])) == {0, 1} and 0 < info["alpha"] < 0.3


# ---------------------------------------------------------------- real-data loader (offline)
def test_heart_loader_drops_incomplete_records(tmp_path):
    rng = np.random.default_rng(0)
    for _, fn in P.HEART_SITES:
        rows = []
        for i in range(30):
            v = [str(round(x, 1)) for x in rng.uniform(0, 5, 13)] + [str(int(rng.integers(0, 5)))]
            if i % 5 == 0:
                v[4] = "?"                       # missing value in a kept attribute -> dropped
            if i % 7 == 0:
                v[11] = "?"                      # missing value in a dropped attribute (ca) -> kept
            rows.append(",".join(v))
        (tmp_path / fn).write_text("\n".join(rows) + "\n")
    sites = P.load_heart_raw(str(tmp_path))
    assert [len(y) for _, _, y in sites] == [24] * 4 and sites[0][1].shape[1] == 10
