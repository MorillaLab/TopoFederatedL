# -*- coding: utf-8 -*-
"""
pTopoFL v2 -- corrected implementation + all experiments of the revised manuscript.
=====================================================================================

Run in Google Colab (CPU is enough):

    !pip -q install ripser          # fast H1 backend (falls back to pure NumPy otherwise)
    !python ptopofl_v2.py --exp all --seeds 10 --out results_v2

or a smoke test (~3-5 min):

    !python ptopofl_v2.py --exp all --quick --out results_quick

Everything the paper reports is written to  <out>/tables/*.tex  (LaTeX tables and
\\newcommand macros that the manuscript \\input's) and <out>/figures/*.pdf|png.

Differences with respect to the v1 code (see AUDIT / CHANGES documents):
  * local training = tau steps of full-batch gradient descent on an L2-regularised
    logistic loss (all parameters penalised) -> the strong convexity / smoothness
    assumptions of the convergence theorem hold with explicit mu = lam, L = L_k;
  * the server broadcasts the PERSONALISED blended cluster models (as the paper
    describes) and evaluation is personalised (each honest client, its own test split);
  * descriptors are computed ONCE (round 0) from a fixed per-client seed and averaged
    over n_rep subsamples, so partition and weights are fixed, as the theory assumes;
  * cluster masses are trust-weighted; trust uses a robust (median/MAD) z-score;
  * baselines are implemented as published (FedProx proximal objective, SCAFFOLD
    option II control variates, pFedMe bi-level updates, IFCA with distinct inits,
    Ditto); no closed-form "heuristic" substitutes.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

try:  # fast H1 (verified to give identical pairs to the reference implementation)
    from ripser import ripser as _ripser
    HAVE_RIPSER = True
except Exception:  # pragma: no cover
    HAVE_RIPSER = False

# ------------------------------------------------------------------------------------
# Global configuration (single source of truth for the paper's hyper-parameter table)
# ------------------------------------------------------------------------------------
CFG = dict(
    lam=1e-2,          # L2 regularisation  -> mu = lam
    tau=10,            # local GD steps per round
    rounds=150,        # communication rounds T
    M=2,               # number of clusters
    alpha=0.3,         # blending coefficient
    n_sub=80,          # points per PH subsample
    n_rep=5,           # subsamples averaged per descriptor
    n_thr=20,          # Betti-curve resolution
    z0=2.0,            # anomaly threshold on the robust z-score
    mad_floor=0.05,    # robust scale floor (fraction of median distance)
    prox_mu=0.1,       # FedProx
    pfedme_lam=15.0,   # pFedMe
    pfedme_inner=10,   # pFedMe inner prox steps
    ditto_lam=0.1,     # Ditto
    test_frac=0.3,
    eval_every=5,
)


# ====================================================================================
# 1. Persistent-homology descriptor (46 coordinates; v1 layout without its duplicate)
# ====================================================================================
def _h0_pairs(D):
    """Vietoris-Rips H0 via Kruskal / union-find with the elder rule (v1 code)."""
    n = D.shape[0]
    parent = np.arange(n)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    rows, cols = np.triu_indices(n, 1)
    w = D[rows, cols]
    pairs = []
    for idx in np.argsort(w, kind="stable"):
        i, j = int(rows[idx]), int(cols[idx])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        pairs.append((0.0, float(w[idx])))
        if ri < rj:
            parent[rj] = ri
        else:
            parent[ri] = rj
    return np.array(pairs) if pairs else np.zeros((0, 2))


def _h1_pairs_numpy(D, pct=95):
    """Reference Z2 boundary-matrix reduction of the truncated triangle filtration (v1 code)."""
    n = D.shape[0]
    rows, cols = np.triu_indices(n, 1)
    w = D[rows, cols]
    thr = np.percentile(w, pct)
    m = w <= thr
    ei, ej, ew = rows[m], cols[m], w[m]
    o = np.argsort(ew, kind="stable")
    ei, ej, ew = ei[o], ej[o], ew[o]
    eid = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ei, ej))}
    adj = {v: set() for v in range(n)}
    for a, b in zip(ei, ej):
        adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    tris = []
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[a]:
                if c <= b or c not in adj[b]:
                    continue
                e = (eid[(a, b)], eid[(a, c)], eid[(b, c)])
                tris.append((max(ew[list(e)]), e))
    tris.sort(key=lambda t: t[0])
    owner, cols_, out = {}, [], []
    for f, e in tris:
        col = set(e)
        while col:
            piv = max(col)
            if piv not in owner:
                owner[piv] = len(cols_)
                if f > ew[piv]:
                    out.append((float(ew[piv]), float(f)))
                break
            col ^= cols_[owner[piv]]
        cols_.append(col)
    return np.array(out) if out else np.zeros((0, 2))


def _h1_pairs(D, pct=95):
    if HAVE_RIPSER:
        thr = np.percentile(D[np.triu_indices(D.shape[0], 1)], pct)
        d1 = _ripser(D, distance_matrix=True, maxdim=1, thresh=thr)["dgms"][1]
        if len(d1) == 0:
            return np.zeros((0, 2))
        d1 = d1[np.isfinite(d1[:, 1]) & (d1[:, 1] > d1[:, 0])]
        return d1.astype(float)
    return _h1_pairs_numpy(D, pct)


def _scalars(P):
    if len(P) == 0:
        return 0.0, 0.0, 0.0
    pers = P[:, 1] - P[:, 0]
    f = pers[np.isfinite(pers) & (pers > 0)]
    if len(f) == 0:
        return 0.0, 0.0, 0.0
    pr = f / f.sum()
    return float(len(f)), float(-(pr * np.log(pr + 1e-12)).sum()), float(np.sqrt((f ** 2).sum()))


def _betti_curve(P, L):
    if len(P) == 0:
        return np.zeros(L)
    m = np.isfinite(P[:, 1]) & (P[:, 1] > P[:, 0])
    if not m.any():
        return np.zeros(L)
    b, d = P[m, 0], P[m, 1]
    mx = float(np.percentile(d, 95)) or 1.0
    t = np.linspace(0.0, mx, L)
    return np.array([np.sum((b <= s) & (d > s)) for s in t], float)


def ph_descriptor_once(X, n_sub, rng, L=20):
    X = np.asarray(X, float)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    if len(X) < 3:
        return np.zeros(2 * L + 6)
    rg = X.max(0) - X.min(0)
    rg[rg == 0] = 1.0
    D = squareform(pdist((X - X.min(0)) / rg))
    P0, P1 = _h0_pairs(D), _h1_pairs(D)
    b0, e0, a0 = _scalars(P0)
    b1, e1, a1 = _scalars(P1)
    # layout: [BC_H0 (L), BC_H1 (L), N0, N1, E0, E1, A0, A1]  -> 2L + 6 = 46 for L = 20.
    # (v1 stored N_j twice, as "beta_j" and "n_j"; the duplicate is removed.)
    return np.concatenate([_betti_curve(P0, L), _betti_curve(P1, L),
                           [b0, b1], [e0, e1], [a0, a1]])


def ph_descriptor(X, seed, n_sub=None, n_rep=None, L=None):
    """Client-side descriptor: mean of n_rep subsample descriptors (fixed client seed)."""
    n_sub = n_sub or CFG["n_sub"]; n_rep = n_rep or CFG["n_rep"]; L = L or CFG["n_thr"]
    rng = np.random.default_rng(seed)
    return np.mean([ph_descriptor_once(X, n_sub, rng, L) for _ in range(n_rep)], axis=0)


def normalise_rows(F):
    nr = np.linalg.norm(F, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return F / nr


def normalise_desc(F, mode="l2"):
    """'l2': unit-norm rows (default, as in v1).  'zscore': per-coordinate standardisation
    across clients (constant coordinates dropped) followed by unit-norm rows."""
    if mode == "l2":
        return normalise_rows(F)
    sd = F.std(0); keep = sd > 1e-12
    return normalise_rows((F[:, keep] - F[:, keep].mean(0)) / sd[keep])


# ====================================================================================
# 2. Data
# ====================================================================================
def _split(X, y, rng, frac):
    """Stratified train/test split (at least one positive and one negative in test when possible)."""
    idx_te = []
    for c in (0, 1):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        k = int(round(frac * len(ids)))
        if len(ids) >= 2:
            k = min(max(k, 1), len(ids) - 1)
        idx_te.extend(ids[:k])
    te = np.zeros(len(y), bool); te[idx_te] = True
    return X[~te], y[~te], X[te], y[te]


def _standardise(clients):
    """Pooled standardisation (computable with secure aggregation of sums / squares)."""
    Xall = np.vstack([c["Xtr"] for c in clients])
    m, s = Xall.mean(0), Xall.std(0)
    s[s == 0] = 1.0
    for c in clients:
        c["Xtr"] = (c["Xtr"] - m) / s
        c["Xte"] = (c["Xte"] - m) / s
    return clients


def _corrupt(X, y, rng, attack):
    X, y = X.copy(), y.copy()
    if attack in ("poison", "geom_poison"):
        flip = rng.choice(len(y), int(0.4 * len(y)), replace=False)
        y[flip] = 1 - y[flip]
    if attack == "poison":            # v1 'distributional poisoning': shift + 40% flips
        X[:, :10] += rng.normal(2.0, 0.5, size=(len(X), min(10, X.shape[1])))
    elif attack == "geom_poison":     # corrupted acquisition: heavy-tailed noise + 40% flips
        X = X + rng.standard_t(2, size=X.shape)
    elif attack == "labelflip":       # labels inverted, features untouched
        y = 1 - y
    return X, y


def make_synthetic(regime="aligned", K=8, d=20, r_low=3, shift=0.6, s_logit=3.0,
                   n_range=(60, 250), prev_range=(0.10, 0.45), n_adv=0,
                   attack="poison", seed=0, test_frac=None, standardise=True):
    """Clustered heterogeneity with controllable alignment between geometry and concept.

    regime: 'aligned'      -> geometry group == concept group (geometry: intrinsic dimension)
            'aligned_tail' -> geometry group == concept group (geometry: heavy tails)
            'concept'      -> concept groups only (identical feature geometry)
            'geometry'     -> geometry groups only (identical decision rule)
    Geometry group 1 lives near an r_low-dimensional subspace (low intrinsic dimension) or,
    for 'aligned_tail', has Student-t(3) coordinates; group 0 is isotropic Gaussian;
    total variance is matched.  Concept groups use
    beta_g = beta_0 +/- shift * v.  Each client has its own prevalence (intercept)."""
    test_frac = CFG["test_frac"] if test_frac is None else test_frac
    rng = np.random.default_rng(seed)
    g = np.array([0, 1] * (K // 2) + [0] * (K % 2))
    rng.shuffle(g)
    b0 = rng.normal(size=d); b0 /= np.linalg.norm(b0)
    v = rng.normal(size=d); v -= (v @ b0) * b0; v /= np.linalg.norm(v)
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :r_low].T * np.sqrt(d / r_low)
    adv = set(rng.choice(K, n_adv, replace=False).tolist()) if n_adv else set()
    clients = []
    for k in range(K):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        geom = g[k] if regime in ("aligned", "aligned_tail", "geometry") else 0
        conc = g[k] if regime in ("aligned", "aligned_tail", "concept") else 0
        if geom == 1 and regime == "aligned_tail":
            X = rng.standard_t(3, size=(n, d)) / np.sqrt(3.0)
        elif geom == 1:
            X = rng.normal(size=(n, r_low)) @ Q + 0.3 * rng.normal(size=(n, d))
        else:
            X = rng.normal(size=(n, d))
        beta = b0 + (shift if conc == 1 else -shift) * v if regime != "geometry" else b0.copy()
        z = X @ beta
        z = s_logit * (z - z.mean()) / (z.std() + 1e-12)
        prev = rng.uniform(*prev_range)
        z = z - np.quantile(z, 1 - prev)
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
        if k in adv:
            X, y = _corrupt(X, y, rng, attack)
        Xtr, ytr, Xte, yte = _split(X, y, rng, test_frac)
        clients.append(dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, group=int(g[k]),
                            adv=k in adv, name=f"c{k}"))
    return _standardise(clients) if standardise else clients


def make_label_skew(K=10, d=20, n_inf=10, n=150, sep=0.35, seed=0, test_frac=None):
    """Scenario B (revised): class-conditional Gaussians with per-client prevalence ~ U(0.1,0.9).
    Classes overlap (v1 used sep=1.0 with std 0.3-0.5, which is linearly separable: AUC=1)."""
    test_frac = CFG["test_frac"] if test_frac is None else test_frac
    rng = np.random.default_rng(seed)
    clients = []
    for k in range(K):
        prev = rng.uniform(0.1, 0.9)
        npos = max(4, int(round(prev * n))); nneg = max(4, n - npos)
        Xp = np.hstack([rng.normal(sep, 1.2, size=(npos, n_inf)), rng.normal(size=(npos, d - n_inf))])
        Xn = np.hstack([rng.normal(0.0, 1.0, size=(nneg, n_inf)), rng.normal(size=(nneg, d - n_inf))])
        X = np.vstack([Xp, Xn]); y = np.r_[np.ones(npos, int), np.zeros(nneg, int)]
        p = rng.permutation(len(y)); X, y = X[p], y[p]
        Xtr, ytr, Xte, yte = _split(X, y, rng, test_frac)
        clients.append(dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, group=int(prev > 0.5),
                            adv=False, name=f"c{k}"))
    return _standardise(clients)


HEART_SITES = [("Cleveland", "processed.cleveland.data"),
               ("Hungary", "processed.hungarian.data"),
               ("Switzerland", "processed.switzerland.data"),
               ("LongBeachVA", "processed.va.data")]
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"


def load_heart_raw(data_dir="heart_data"):
    """Fed-Heart-Disease (4 hospitals).  Follows FLamby's site partition: keep the 10
    attributes age..oldpeak, drop slope/ca/thal, drop rows with missing values,
    target = 1[num > 0].  Expected site sizes: 303 / 261 / 46 / 130."""
    os.makedirs(data_dir, exist_ok=True)
    sites = []
    for name, fn in HEART_SITES:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            urllib.request.urlretrieve(HEART_URL + fn, path)
        rows = []
        with open(path, "r", encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 14:
                    continue
                feats = parts[:10]
                if any(p.strip() in ("?", "") for p in feats) or parts[13].strip() in ("?", ""):
                    continue
                rows.append([float(p) for p in feats] + [float(parts[13])])
        A = np.array(rows, float)
        sites.append((name, A[:, :10], (A[:, 10] > 0).astype(int)))
    return sites


def make_heart(seed=0, data_dir="heart_data", test_frac=None):
    test_frac = CFG["test_frac"] if test_frac is None else test_frac
    rng = np.random.default_rng(seed)
    clients = []
    for k, (name, X, y) in enumerate(load_heart_raw(data_dir)):
        Xtr, ytr, Xte, yte = _split(X, y, rng, test_frac)
        clients.append(dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, group=-1, adv=False, name=name))
    return _standardise(clients)


# ====================================================================================
# 3. Local objective: L2-regularised logistic regression (intercept penalised too)
# ====================================================================================
def _clip_rows(Xb, C):
    nr = np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb * np.minimum(1.0, C / np.maximum(nr, 1e-12))


class Client:
    def __init__(self, c, lam, clip=None):
        self.Xtr = np.hstack([c["Xtr"], np.ones((len(c["Xtr"]), 1))])
        self.Xte = np.hstack([c["Xte"], np.ones((len(c["Xte"]), 1))])
        self.clip = clip
        if clip is not None:   # per-record preprocessing: ||x~|| <= C  =>  ||grad l_i|| <= C
            self.Xtr, self.Xte = _clip_rows(self.Xtr, clip), _clip_rows(self.Xte, clip)
        self.ytr, self.yte = c["ytr"].astype(int), c["yte"].astype(int)
        self.yt = 2.0 * self.ytr - 1.0
        self.n, self.p = self.Xtr.shape
        self.lam = lam
        self.L = lam + 0.25 * np.linalg.norm(self.Xtr, 2) ** 2 / self.n
        self.adv, self.group, self.name = c["adv"], c["group"], c["name"]
        self.raw = c["Xtr"]

    def loss(self, th):
        z = self.yt * (self.Xtr @ th)
        return float(np.mean(np.logaddexp(0.0, -z)) + 0.5 * self.lam * th @ th)

    def grad(self, th):
        z = self.yt * (self.Xtr @ th)
        s = 0.5 * (1.0 - np.tanh(0.5 * z))
        return -(self.Xtr.T @ (self.yt * s)) / self.n + self.lam * th

    def hess(self, th):
        s = 0.5 * (1.0 + np.tanh(0.5 * (self.Xtr @ th)))
        return (self.Xtr.T * (s * (1 - s))) @ self.Xtr / self.n + self.lam * np.eye(self.p)

    def dp_gd(self, th, eta, tau, sig, rng):
        """DP-GD: full-batch gradient + N(0, (2 C sig / n)^2 I) per step (replace-one adjacency)."""
        s = 2.0 * self.clip * sig / self.n
        for _ in range(tau):
            th = th - eta * (self.grad(th) + s * rng.normal(size=self.p))
        return th

    def sample_losses(self, th, train=True):
        X, y = (self.Xtr, self.ytr) if train else (self.Xte, self.yte)
        return np.logaddexp(0.0, -(2.0 * y - 1.0) * (X @ th))

    def gd(self, th, eta, tau, extra=None):
        """tau steps of GD on F_k (+ optional extra gradient term: function of theta)."""
        for _ in range(tau):
            g = self.grad(th)
            if extra is not None:
                g = g + extra(th)
            th = th - eta * g
        return th


def newton(grad, hess, p, th0=None, tol=1e-10, it=60, value=None):
    th = np.zeros(p) if th0 is None else th0.copy()
    for _ in range(it):
        g = grad(th)
        if np.linalg.norm(g) < tol:
            break
        step = np.linalg.solve(hess(th), g)
        t = 1.0
        if value is not None:
            v0 = value(th)
            while value(th - t * step) > v0 - 0.25 * t * g @ step and t > 1e-8:
                t *= 0.5
        th = th - t * step
    return th


def local_optimum(cl):
    return newton(cl.grad, cl.hess, cl.p, value=cl.loss)


def weighted_optimum(cls, w):
    return newton(lambda t: sum(wi * c.grad(t) for wi, c in zip(w, cls)),
                  lambda t: sum(wi * c.hess(t) for wi, c in zip(w, cls)), cls[0].p,
                  value=lambda t: sum(wi * c.loss(t) for wi, c in zip(w, cls)))


# ====================================================================================
# 4. Metrics (personalised: client k is evaluated with the model it receives)
# ====================================================================================
def _auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(clients, models):
    ys, ss, ll, nn = [], [], [], []
    for k, c in enumerate(clients):
        if c.adv:
            continue
        z = c.Xte @ models[k]
        yt = 2.0 * c.yte - 1.0
        ll.append(float(np.mean(np.logaddexp(0.0, -yt * z)))); nn.append(len(c.yte))
        ys.append(c.yte); ss.append(z)
    y, s = np.concatenate(ys), np.concatenate(ss)
    pred = (s > 0).astype(int)
    tpr = np.mean(pred[y == 1] == 1) if (y == 1).any() else np.nan
    tnr = np.mean(pred[y == 0] == 0) if (y == 0).any() else np.nan
    return dict(logloss=float(np.average(ll, weights=nn)), auc=_auc(y, s),
                bacc=float(0.5 * (tpr + tnr)), per_client=ll)


def mia_auc(clients, local_models):
    """Loss-threshold membership inference (Yeom et al., 2018) against the LAST model each
    honest client transmitted: AUC of -loss for training records (members) vs held-out
    records (non-members), averaged over clients.  0.5 = no leakage detected."""
    out = []
    for k, c in enumerate(clients):
        if c.adv:
            continue
        a = -c.sample_losses(local_models[k], True); b = -c.sample_losses(local_models[k], False)
        out.append(_auc(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    return float(np.mean(out))


def _hist_append(hist, t, T, clients, models, ev):
    if not ev:
        return
    if (t + 1) % ev == 0 or t == T - 1:
        hist.append((t + 1, evaluate(clients, models)))


def _pk(clients):
    n = np.array([c.n for c in clients], float)
    return n / n.sum()


def step_size(clients):
    return 1.0 / max(c.L for c in clients)


# ====================================================================================
# 5. Algorithms (identical local budget: tau GD steps per round, step 1/L_max)
# ====================================================================================
def run_local(clients, T, tau, eta, ev):
    th = [np.zeros(c.p) for c in clients]; hist = []
    for t in range(T):
        th = [c.gd(th[k], eta, tau) for k, c in enumerate(clients)]
        _hist_append(hist, t, T, clients, np.array(th), ev)
    return np.array(th), hist


def run_fedavg(clients, T, tau, eta, ev, agg="mean", trim=0.2):
    pk = _pk(clients); K = len(clients); th = np.zeros(clients[0].p); hist = []
    for t in range(T):
        loc = np.array([c.gd(th, eta, tau) for c in clients])
        if agg == "mean":
            th = pk @ loc
        elif agg == "median":                       # coordinate-wise median (Yin et al. 2018)
            th = np.median(loc, axis=0)
        elif agg == "trimmed":                      # coordinate-wise trimmed mean
            b = int(math.floor(trim * K)); srt = np.sort(loc, axis=0)
            th = srt[b:K - b].mean(axis=0)
        _hist_append(hist, t, T, clients, np.tile(th, (K, 1)), ev)
    return np.tile(th, (K, 1)), hist


def run_fedprox(clients, T, tau, eta, ev, mu_p=None):
    mu_p = CFG["prox_mu"] if mu_p is None else mu_p
    pk = _pk(clients); K = len(clients); th = np.zeros(clients[0].p); hist = []
    eta_p = 1.0 / (1.0 / eta + mu_p)
    for t in range(T):
        g = th.copy()
        th = pk @ np.array([c.gd(g, eta_p, tau, extra=lambda x, g=g: mu_p * (x - g)) for c in clients])
        _hist_append(hist, t, T, clients, np.tile(th, (K, 1)), ev)
    return np.tile(th, (K, 1)), hist


def run_scaffold(clients, T, tau, eta, ev):
    """SCAFFOLD, option II, full participation, global step 1 (Karimireddy et al. 2020)."""
    pk = _pk(clients); K = len(clients); p = clients[0].p
    x = np.zeros(p); c = np.zeros(p); ck = np.zeros((K, p)); hist = []
    for t in range(T):
        dy, dc = np.zeros((K, p)), np.zeros((K, p))
        for k, cl in enumerate(clients):
            y = x.copy()
            for _ in range(tau):
                y = y - eta * (cl.grad(y) - ck[k] + c)
            ck_new = ck[k] - c + (x - y) / (tau * eta)
            dy[k], dc[k] = y - x, ck_new - ck[k]
            ck[k] = ck_new
        x = x + pk @ dy
        c = c + pk @ dc
        _hist_append(hist, t, T, clients, np.tile(x, (K, 1)), ev)
    return np.tile(x, (K, 1)), hist


def _prox(cl, w, lam_p, th0, steps, eta_in):
    th = th0.copy()
    for _ in range(steps):
        th = th - eta_in * (cl.grad(th) + lam_p * (th - w))
    return th


def run_pfedme(clients, T, tau, eta, ev, lam_p=None, inner=None):
    """pFedMe (T Dinh et al. 2020): bi-level Moreau-envelope personalisation, beta = 1."""
    lam_p = CFG["pfedme_lam"] if lam_p is None else lam_p
    inner = CFG["pfedme_inner"] if inner is None else inner
    pk = _pk(clients); K = len(clients); p = clients[0].p
    w = np.zeros(p); th_tilde = np.zeros((K, p)); hist = []
    eta_in = 1.0 / (1.0 / eta + lam_p); eta_out = 0.5 / lam_p
    for t in range(T):
        W = np.zeros((K, p))
        for k, cl in enumerate(clients):
            wk = w.copy()
            for _ in range(tau):
                th_tilde[k] = _prox(cl, wk, lam_p, th_tilde[k], inner, eta_in)
                wk = wk - eta_out * lam_p * (wk - th_tilde[k])
            W[k] = wk
        w = pk @ W
        if ev and ((t + 1) % ev == 0 or t == T - 1):
            pers = np.array([_prox(cl, w, lam_p, th_tilde[k], 50, eta_in) for k, cl in enumerate(clients)])
            hist.append((t + 1, evaluate(clients, pers)))
    pers = np.array([_prox(cl, w, lam_p, th_tilde[k], 200, eta_in) for k, cl in enumerate(clients)])
    return pers, hist


def run_ditto(clients, T, tau, eta, ev, lam_d=None):
    """Ditto (Li et al. 2021): FedAvg global model + proximally regularised personal models."""
    lam_d = CFG["ditto_lam"] if lam_d is None else lam_d
    pk = _pk(clients); K = len(clients); p = clients[0].p
    w = np.zeros(p); v = np.zeros((K, p)); hist = []
    eta_v = 1.0 / (1.0 / eta + lam_d)
    for t in range(T):
        wt = w.copy()
        w = pk @ np.array([c.gd(wt, eta, tau) for c in clients])
        v = np.array([c.gd(v[k], eta_v, tau, extra=lambda x, g=wt: lam_d * (x - g))
                      for k, c in enumerate(clients)])
        _hist_append(hist, t, T, clients, v, ev)
    return v, hist


def run_ifca(clients, T, tau, eta, ev, M=None, seed=0):
    """IFCA (Ghosh et al. 2020): client-side cluster identity by minimum training loss;
    cluster models initialised from M distinct randomly chosen clients (20 local steps)."""
    M = CFG["M"] if M is None else M
    rng = np.random.default_rng(seed + 7); K = len(clients); n = np.array([c.n for c in clients], float)
    init = rng.choice(K, M, replace=False)
    Th = np.array([clients[i].gd(np.zeros(clients[0].p), eta, 20) for i in init]); hist = []
    assign = np.zeros(K, int)
    for t in range(T):
        assign = np.array([int(np.argmin([c.loss(Th[j]) for j in range(M)])) for c in clients])
        for j in range(M):
            S = np.where(assign == j)[0]
            if len(S):
                Th[j] = (n[S] / n[S].sum()) @ np.array([clients[k].gd(Th[j], eta, tau) for k in S])
        _hist_append(hist, t, T, clients, Th[assign], ev)
    assign = np.array([int(np.argmin([c.loss(Th[j]) for j in range(M)])) for c in clients])
    return Th[assign], hist


def run_clustered(clients, labels, w, pi, alpha, T, tau, eta, ev, Theta0=None, keep_traj=False,
                  dp_sigma=0.0, rng=None):
    """Blended clustered local GD -- the operator Psi analysed in Theorem 1.
    M = 1 recovers FedAvg with weights w.  dp_sigma > 0 runs DP-GD locally (Prop. 9)."""
    M = int(labels.max()) + 1; p = clients[0].p
    Theta = np.zeros((M, p)) if Theta0 is None else Theta0.copy()
    B = (1.0 - alpha) * np.eye(M) + alpha * np.outer(np.ones(M), pi)
    hist, traj = [], []
    last = np.zeros((len(clients), p))
    for t in range(T):
        U = np.zeros((M, p))
        for k, c in enumerate(clients):
            if dp_sigma > 0:
                last[k] = c.dp_gd(Theta[labels[k]], eta, tau, dp_sigma, rng)
            else:
                last[k] = c.gd(Theta[labels[k]], eta, tau)
            U[labels[k]] += w[k] * last[k]
        Theta = B @ U
        if keep_traj:
            traj.append(Theta.copy())
        if ev:
            _hist_append(hist, t, T, clients, Theta[labels], ev)
    return Theta[labels], hist, Theta, traj, last


# ------------------------------------------------------------------------------------
# pTopoFL server-side structure (computed once, from descriptors only)
# ------------------------------------------------------------------------------------
def descriptors(clients_raw_X, seed):
    return np.array([ph_descriptor(X, seed=10_000 * seed + 97 * k + 1)
                     for k, X in enumerate(clients_raw_X)])


def agglomerative(Fh, M):
    from sklearn.cluster import AgglomerativeClustering
    if len(Fh) <= M:
        return np.arange(len(Fh))
    return AgglomerativeClustering(n_clusters=M, linkage="average").fit_predict(Fh)


def trust_weights(Fh, rule="mad", z0=None):
    z0 = CFG["z0"] if z0 is None else z0
    K = len(Fh)
    Dm = squareform(pdist(Fh))
    delta = Dm.sum(1) / (K - 1)
    if rule == "none":
        return np.ones(K), np.zeros(K)
    if rule == "std":                          # v1 rule: mean/std z-score, threshold 1
        z = (delta - delta.mean()) / (delta.std() + 1e-12)
        return np.exp(-np.maximum(z - 1.0, 0.0)), z
    med = np.median(delta)
    scale = max(1.4826 * np.median(np.abs(delta - med)), CFG["mad_floor"] * med, 1e-12)
    z = (delta - med) / scale
    return np.exp(-np.maximum(z - z0, 0.0)), z


def ptopofl_structure(Fh, n, M, trust="mad", labels=None):
    if labels is None:
        labels = agglomerative(Fh, M)
    labels = np.unique(labels, return_inverse=True)[1]
    t, z = trust_weights(Fh, trust)
    w = np.zeros(len(n)); Mx = labels.max() + 1; pi = np.zeros(Mx)
    for j in range(Mx):
        C = np.where(labels == j)[0]
        cen = Fh[C].mean(0)
        raw = n[C] * np.exp(-np.linalg.norm(Fh[C] - cen, axis=1)) * t[C]
        w[C] = raw / raw.sum()
        pi[j] = (n[C] * t[C]).sum()
    return labels, w, pi / pi.sum(), t, z


def plain_structure(labels, n):
    labels = np.unique(labels, return_inverse=True)[1]
    Mx = labels.max() + 1; w = np.zeros(len(n)); pi = np.zeros(Mx)
    for j in range(Mx):
        C = np.where(labels == j)[0]
        w[C] = n[C] / n[C].sum(); pi[j] = n[C].sum()
    return labels, w, pi / pi.sum()


def subspace_labels(clients, M, r=3):
    """PACFL-style one-shot clustering (Vahidian et al., 2023): each client sends the top-r
    left singular vectors of its (features x samples) data matrix; the server clusters on
    the sum of principal angles with average linkage."""
    from sklearn.cluster import AgglomerativeClustering
    U = [np.linalg.svd(np.asarray(c.raw, float).T, full_matrices=False)[0][:, :r] for c in clients]
    K = len(U); D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            s = np.clip(np.linalg.svd(U[i].T @ U[j], compute_uv=False), -1.0, 1.0)
            D[i, j] = D[j, i] = float(np.arccos(s).sum())
    if K <= M:
        return np.arange(K)
    return AgglomerativeClustering(n_clusters=M, metric="precomputed",
                                   linkage="average").fit_predict(D)


def oracle_labels(clients, M, seed=0):
    """Privacy-INCOMPATIBLE reference: k-means on the clients' exact local optima."""
    from sklearn.cluster import KMeans
    W = np.array([local_optimum(c) for c in clients])
    if len(W) <= M:
        return np.arange(len(W))
    return KMeans(n_clusters=M, n_init=20, random_state=seed).fit_predict(W)


# ------------------------------------------------------------------------------------
# Theory diagnostics (Proposition 4): heterogeneity decomposition of local optima
# ------------------------------------------------------------------------------------
def heterogeneity(clients, labels, w, pi, wloc=None):
    wloc = np.array([local_optimum(c) for c in clients]) if wloc is None else wloc
    om = np.array([pi[labels[k]] * w[k] for k in range(len(clients))])
    wbar = om @ wloc
    Vtot = float(om @ np.sum((wloc - wbar) ** 2, 1))
    Vin = 0.0
    for j in range(labels.max() + 1):
        C = np.where(labels == j)[0]
        wj = w[C] @ wloc[C]
        Vin += pi[j] * float(w[C] @ np.sum((wloc[C] - wj) ** 2, 1))
    return dict(Vtot=Vtot, Vin=Vin, rho=Vin / Vtot if Vtot > 0 else float("nan"))


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


# ====================================================================================
# 6. Helpers shared by the experiments
# ====================================================================================
def alpha_from_effective(abar, eta, tau, lam):
    """Invert alpha_eff = alpha / (1 - (1 - alpha) q)  with  q = (1 - eta*lam)^tau  (Prop. 3)."""
    q = (1.0 - eta * lam) ** tau
    return abar * (1.0 - q) / (1.0 - abar * q)


def build(raw):
    cl = [Client(c, CFG["lam"]) for c in raw]
    return cl, step_size(cl), np.array([c.n for c in cl], float)


def ptopofl_fit(cl, eta, n, seed, T, tau, ev, abar=0.3, M=None, trust="mad",
                noise=0.0, Fh=None, labels=None):
    M = CFG["M"] if M is None else M
    if Fh is None:
        Fh = normalise_rows(descriptors([c.raw for c in cl], seed))
    if noise > 0:
        Fh = Fh + np.random.default_rng(seed + 123).normal(0.0, noise, size=Fh.shape)
    lab, w, pi, t, z = ptopofl_structure(Fh, n, M, trust=trust, labels=labels)
    a = alpha_from_effective(abar, eta, tau, CFG["lam"])
    models, hist, Theta, _, _ = run_clustered(cl, lab, w, pi, a, T, tau, eta, ev)
    return models, hist, dict(labels=lab, w=w, pi=pi, trust=t, z=z, alpha=a, Fh=Fh)


METHODS_MAIN = ["Local", "FedAvg", "FedProx", "SCAFFOLD", "pFedMe", "Ditto", "IFCA", "PACFL", "pTopoFL"]


def run_method(name, cl, eta, n, seed, T, tau, ev, **kw):
    if name == "Local":
        return run_local(cl, T, tau, eta, ev)
    if name == "FedAvg":
        return run_fedavg(cl, T, tau, eta, ev)
    if name == "Median":
        return run_fedavg(cl, T, tau, eta, ev, agg="median")
    if name == "Trimmed":
        return run_fedavg(cl, T, tau, eta, ev, agg="trimmed")
    if name == "FedProx":
        return run_fedprox(cl, T, tau, eta, ev)
    if name == "SCAFFOLD":
        return run_scaffold(cl, T, tau, eta, ev)
    if name == "pFedMe":
        return run_pfedme(cl, T, tau, eta, ev)
    if name == "Ditto":
        return run_ditto(cl, T, tau, eta, ev)
    if name == "IFCA":
        return run_ifca(cl, T, tau, eta, ev, seed=seed)
    if name == "PACFL":
        lab, w, pi = plain_structure(subspace_labels(cl, kw.get("M", CFG["M"])), n)
        a = alpha_from_effective(0.3, eta, tau, CFG["lam"])
        m, h, _, _, _ = run_clustered(cl, lab, w, pi, a, T, tau, eta, ev)
        return m, h
    if name.startswith("pTopoFL"):
        m, h, _ = ptopofl_fit(cl, eta, n, seed, T, tau, ev, **kw)
        return m, h
    raise ValueError(name)


def scenario_raw(name, seed, **kw):
    if name == "A":
        return make_synthetic("aligned", K=8, seed=seed, shift=1.0, **kw)
    if name == "B":
        return make_label_skew(K=10, seed=seed)
    if name == "HD":
        return make_heart(seed=seed)
    raise ValueError(name)


def wilcoxon_holm(ref, others, lower_is_better=True):
    """Two-sided paired Wilcoxon tests of ref vs each baseline, Holm-corrected.
    Returns {name: (p_adj, sign)} with sign=+1 if ref is better, -1 if worse."""
    names = list(others)
    raw = []
    for nm in names:
        d = np.asarray(ref) - np.asarray(others[nm])
        if np.allclose(d, 0):
            raw.append(1.0)
        else:
            try:
                raw.append(float(wilcoxon(d, zero_method="wilcox").pvalue))
            except ValueError:
                raw.append(1.0)
    order = np.argsort(raw); m = len(raw); adj = np.empty(m); prev = 0.0
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (m - r) * raw[i])); adj[i] = prev
    out = {}
    for i, nm in enumerate(names):
        md = np.median(np.asarray(ref) - np.asarray(others[nm]))
        better = (md < 0) if lower_is_better else (md > 0)
        out[nm] = (float(adj[i]), 1 if better else -1)
    return out


def fmt(m, s, nd=3, bold=False):
    t = f"{m:.{nd}f}\\pm{s:.{nd}f}"
    return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"


class Out:
    def __init__(self, root):
        self.root = root
        for sub in ("tables", "figures", "json"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.macros = {}

    def json(self, name, obj):
        with open(os.path.join(self.root, "json", name + ".json"), "w") as f:
            json.dump(obj, f, indent=1, default=float)

    def table(self, name, text):
        with open(os.path.join(self.root, "tables", name + ".tex"), "w") as f:
            f.write(text)

    def macro(self, key, val):
        self.macros[key] = val
        path = os.path.join(self.root, "tables", "results_macros.tex")
        with open(path, "w") as f:
            f.write("% auto-generated by ptopofl_v2.py -- do not edit by hand\n")
            for k in sorted(self.macros):
                f.write(f"\\newcommand{{\\{k}}}{{{self.macros[k]}}}\n")

    def fig(self, fig, name):
        import matplotlib.pyplot as plt
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.root, "figures", f"{name}.{ext}"), bbox_inches="tight", dpi=200)
        plt.close(fig)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


COL = {"PACFL": "#bc80bd", "pTopoFL": "#2ca02c", "FedAvg": "#1f77b4", "FedProx": "#d62728", "SCAFFOLD": "#8c564b",
       "pFedMe": "#9467bd", "Ditto": "#e377c2", "IFCA": "#ff7f0e", "Local": "#7f7f7f",
       "Median": "#17becf", "Trimmed": "#bcbd22", "Oracle": "#000000", "Random": "#aaaaaa",
       "pTopoFL (no trust)": "#98df8a", "pTopoFL (v1 trust)": "#006400"}


# ====================================================================================
# 7. Experiments
# ====================================================================================
def exp_theory(out, T=300, quick=False):
    """E1: numerical validation of Theorem 1, Theorem 2 and Proposition 3."""
    plt = _mpl()
    cl, eta, n = build(scenario_raw("A", 0))
    Fh = normalise_rows(descriptors([c.raw for c in cl], 0))
    lab, w, pi, t, z = ptopofl_structure(Fh, n, CFG["M"])
    Mx = lab.max() + 1
    lam = CFG["lam"]
    theta_star = np.array([weighted_optimum([cl[k] for k in np.where(lab == j)[0]],
                                            w[lab == j]) for j in range(Mx)])

    def fixed_point(alpha, eta_, tau, Th0=None, tol=1e-12, itmax=60000):
        Th = theta_star.copy() if Th0 is None else Th0.copy()
        B = (1 - alpha) * np.eye(Mx) + alpha * np.outer(np.ones(Mx), pi)
        for _ in range(itmax):
            U = np.zeros_like(Th)
            for k, c in enumerate(cl):
                U[lab[k]] += w[k] * c.gd(Th[lab[k]], eta_, tau)
            Tn = B @ U
            if np.sqrt(np.sum(pi * np.sum((Tn - Th) ** 2, 1))) < tol:
                return Tn
            Th = Tn
        return Th

    pnorm = lambda D: float(np.sqrt(np.sum(pi * np.sum(D ** 2, 1))))
    res = {"contraction": {}, "bias": [], "blend": {}}
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    # (a) contraction
    for tau, col in zip([1, 10, 50], ["#1f77b4", "#2ca02c", "#d62728"]):
        a = alpha_from_effective(0.3, eta, tau, lam)
        _, _, _, traj, _ = run_clustered(cl, lab, w, pi, a, T if not quick else 100, tau, eta, 0,
                                         keep_traj=True)
        That = fixed_point(a, eta, tau, Th0=traj[-1])
        err = np.array([pnorm(Th - That) for Th in traj])
        e0 = pnorm(np.zeros_like(That) - That)
        q = (1 - eta * lam) ** tau
        tt = np.arange(1, len(err) + 1)
        ax[0].semilogy(tt, err, color=col, lw=1.6, label=f"$\\tau={tau}$")
        ax[0].semilogy(tt, e0 * q ** tt, color=col, lw=1, ls="--")
        # compare only where the bound is above the precision of the computed fixed point
        valid = e0 * q ** tt > 1e-8
        ratio = float(np.max(err[valid] / (e0 * q ** tt[valid]))) if valid.any() else float("nan")
        res["contraction"][tau] = dict(q=q, max_ratio=ratio, alpha=a)
    ax[0].set_ylim(1e-10, None); ax[0].set_xlabel("round $t$")
    ax[0].set_ylabel(r"$\|\Theta^t-\hat\Theta\|_\pi$")
    ax[0].set_title("(a) linear convergence (dashed: $q^t$ bound)")
    ax[0].legend(frameon=False)
    # (b) drift bias at alpha = 0 versus eta (tau - 1)
    for ef, mk in zip([1.0, 0.5, 0.25], ["o", "s", "^"]):
        xs, ys, bs = [], [], []
        for tau in ([1, 2, 5, 10, 20] if not quick else [1, 2, 5]):
            e_ = ef * eta
            That = fixed_point(0.0, e_, tau)
            b = pnorm(That - theta_star)
            q = (1 - e_ * lam) ** tau
            bj = []
            for j in range(Mx):
                C = np.where(lab == j)[0]
                zeta = sum(w[k] * np.linalg.norm(cl[k].grad(theta_star[j])) for k in C)
                Lm = max(cl[k].L for k in C)
                B1 = e_ * Lm * (tau - 1) * (1 + e_ * lam * tau) / (2 * lam) * zeta
                B2 = (1 + q) / (lam * (1 - q)) * zeta
                bj.append(min(B1, B2))
            bound = float(np.sqrt(np.sum(pi * np.array(bj) ** 2)))
            res["bias"].append(dict(eta_factor=ef, tau=tau, bias=b, bound=bound))
            if tau > 1:
                xs.append(e_ * (tau - 1)); ys.append(b); bs.append(bound)
        ax[1].loglog(xs, ys, marker=mk, color="#2ca02c", lw=1.2, label=f"measured, $\\eta={ef}/L$")
        ax[1].loglog(xs, bs, marker=mk, color="#7f7f7f", lw=0.8, ls=":", mfc="none")
    ax[1].set_xlabel(r"$\eta(\tau-1)$"); ax[1].set_ylabel(r"$\|\hat\Theta_0-\Theta^\star\|_\pi$")
    ax[1].set_title("(b) drift bias at $\\alpha=0$ (dotted: Thm. 2 bound)")
    ax[1].legend(frameon=False, fontsize=6.5)
    # (c) effective blending: retained personalisation D(Theta_alpha)/D(Theta_0)
    def disp(Th):
        m = pi @ Th
        return float(np.sqrt(np.sum(pi * np.sum((Th - m) ** 2, 1))))
    H = [sum(w[k] * cl[k].hess(theta_star[j]) for k in np.where(lab == j)[0]) for j in range(Mx)]
    ev_all = np.concatenate([np.linalg.eigvalsh(h) for h in H])
    hmin, hmed = ev_all.min(), np.median(ev_all)
    agrid = np.array([0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]) if not quick else np.array([0, 0.03, 0.3, 1.0])
    for tau, col in zip([1, 10, 50], ["#1f77b4", "#2ca02c", "#d62728"]):
        T0 = fixed_point(0.0, eta, tau); D0 = disp(T0)
        r = [disp(fixed_point(a, eta, tau, Th0=T0)) / D0 for a in agrid]
        res["blend"][tau] = dict(alpha=agrid.tolist(), retained=r)
        ax[2].semilogx(np.maximum(agrid, 1e-3), r, "o-", color=col, lw=1.4, ms=3, label=f"$\\tau={tau}$")
        aa = np.logspace(-3, 0, 100)
        for h, ls in [(hmin, "--"), (hmed, ":")]:
            q = (1 - eta * h) ** tau
            ax[2].semilogx(aa, (1 - aa) * (1 - q) / (1 - (1 - aa) * q), color=col, lw=0.8, ls=ls)
    ax[2].set_xlabel(r"blending coefficient $\alpha$")
    ax[2].set_ylabel("retained personalisation")
    ax[2].set_title("(c) effective blending (lines: Prop. 3)")
    ax[2].legend(frameon=False)
    fig.tight_layout(); out.fig(fig, "fig_theory")
    out.json("theory", res)
    worst = max(v["max_ratio"] for v in res["contraction"].values())
    out.macro("ThContractionWorst", f"{worst:.3f}")
    # tau = 1: Theorem 2 gives bias exactly 0; the measured value is the fixed-point solver
    # precision (tol / (1 - q) ~ 1e-10), hence the absolute slack below.
    okb = all(r_["bias"] <= r_["bound"] * (1 + 1e-9) + 1e-8 for r_ in res["bias"])
    out.macro("ThBiasTauOne", f"{max(r_['bias'] for r_ in res['bias'] if r_['tau'] == 1):.0e}")
    out.macro("ThBiasBoundHolds", "yes" if okb else "NO")
    tightness = np.median([r_["bound"] / max(r_["bias"], 1e-16) for r_ in res["bias"] if r_["tau"] > 1])
    out.macro("ThBiasTightness", f"{tightness:.0f}")
    q10 = (1 - eta * lam) ** 10
    out.macro("ThQten", f"{q10:.3f}")
    out.macro("ThAlphaCalTen", f"{alpha_from_effective(0.3, eta, 10, lam):.3f}")
    out.macro("ThAlphaEffVone", f"{0.3 / (1 - 0.7 * q10):.2f}")
    out.macro("ThKappa", f"{max(c.L for c in cl) / lam:.0f}")
    print(f"[E1] contraction worst ratio {worst:.3f}; bias bound holds: {okb}; "
          f"median bound/bias {tightness:.0f}")
    return res


def exp_main(out, seeds, T, tau, ev, scenarios=("A", "B", "HD")):
    """E2: main comparison (personalised log-loss / AUC on honest clients) + per-client view."""
    plt = _mpl()
    allres = {}
    for sc in scenarios:
        per = {m: [] for m in METHODS_MAIN + ["Random", "Oracle"]}
        curves = {m: [] for m in METHODS_MAIN}
        extra = dict(ari=[], ari_pacfl=[], rho_topo=[], rho_rand=[], rho_orac=[], K=None, sizes=None)
        try:
            for s in seeds:
                cl, eta, n = build(scenario_raw(sc, s))
                extra["K"] = len(cl); extra["sizes"] = [int(c.n + len(c.yte)) for c in cl]
                for m in METHODS_MAIN:
                    models, hist = run_method(m, cl, eta, n, s, T, tau, ev)
                    per[m].append(evaluate(cl, models)); curves[m].append(hist)
                _, _, info = ptopofl_fit(cl, eta, n, s, 1, tau, 0)
                lab = info["labels"]; a = info["alpha"]
                rl = np.random.default_rng(s + 99).permutation(lab)
                l2, w2, p2 = plain_structure(rl, n)
                per["Random"].append(evaluate(cl, run_clustered(cl, l2, w2, p2, a, T, tau, eta, 0)[0]))
                lo = oracle_labels(cl, CFG["M"], s); l3, w3, p3 = plain_structure(lo, n)
                per["Oracle"].append(evaluate(cl, run_clustered(cl, l3, w3, p3, a, T, tau, eta, 0)[0]))
                wl = np.array([local_optimum(c) for c in cl])
                # rho compares PARTITIONS: use sample-proportional weights (omega_k = p_k) for every
                # partition, so that V_tot is common; the algorithm's own weights are kept for reference.
                l1, w1, p1 = plain_structure(lab, n)
                extra["rho_topo"].append(heterogeneity(cl, l1, w1, p1, wl)["rho"])
                extra.setdefault("rho_topo_algweights", []).append(
                    heterogeneity(cl, lab, info["w"], info["pi"], wl)["rho"])
                extra["rho_rand"].append(heterogeneity(cl, l2, w2, p2, wl)["rho"])
                extra["rho_orac"].append(heterogeneity(cl, l3, w3, p3, wl)["rho"])
                g = np.array([c.group for c in cl])
                if (g >= 0).all():
                    extra["ari"].append(ari(lab, g))
                    extra["ari_pacfl"].append(ari(subspace_labels(cl, CFG["M"]), g))
                print(f"  [E2:{sc}] seed {s} done", flush=True)
        except Exception as exc:  # e.g. no internet for the real dataset
            print(f"  [E2:{sc}] skipped: {exc}")
            continue
        allres[sc] = dict(per=per, extra=extra, curves=curves)
        if sc == "HD" and extra["sizes"]:
            names_ = [n_ for n_, _ in HEART_SITES]
            print("  [E2:HD] site sizes after preprocessing (train+test): " +
                  ", ".join(f"{a_}={b_}" for a_, b_ in zip(names_, extra["sizes"])) +
                  "   (expected 303/261/46/130)", flush=True)
    out.json("main", {sc: dict(per=v["per"], extra=v["extra"]) for sc, v in allres.items()})
    names = METHODS_MAIN + ["Random", "Oracle"]
    label = {"A": "Synthetic (clustered)", "B": "Label skew", "HD": "Fed-Heart-Disease"}
    cols = [sc for sc in scenarios if sc in allres]
    lines = ["\\begin{tabular}{l" + "cc" * len(cols) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{label[sc]}}}" for sc in cols) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             "Method & " + " & ".join(["Log-loss $\\downarrow$ & AUC $\\uparrow$"] * len(cols)) + " \\\\",
             "\\midrule"]
    stats = {}
    for sc in cols:
        per = allres[sc]["per"]
        ll = {m: [r["logloss"] for r in per[m]] for m in names}
        au = {m: [r["auc"] for r in per[m]] for m in names}
        tests = wilcoxon_holm(ll["pTopoFL"], {m: ll[m] for m in METHODS_MAIN if m != "pTopoFL"})
        stats[sc] = dict(ll=ll, au=au, tests=tests,
                         best_ll=min(np.mean(ll[m]) for m in METHODS_MAIN),
                         best_au=max(np.mean(au[m]) for m in METHODS_MAIN))
    for m in names:
        if m == "Random":
            lines.append("\\midrule")
        disp = {"Random": "Clustered, random partition$^\\diamond$",
                "Oracle": "Clustered, oracle partition$^\\diamond$",
                "PACFL": "Subspace clustering (PACFL-style)",
                "pTopoFL": "\\textbf{\\ptopofl}"}.get(m, m)
        row = [disp]
        for sc in cols:
            S = stats[sc]; mll, sll = np.mean(S["ll"][m]), np.std(S["ll"][m])
            mau, sau = np.mean(S["au"][m]), np.std(S["au"][m])
            c1 = fmt(mll, sll, bold=(m in METHODS_MAIN and abs(mll - S["best_ll"]) < 5e-4))
            if m in S["tests"]:
                p, sgn = S["tests"][m]
                if p < 0.05:
                    c1 = c1[:-1] + ("^{+}" if sgn > 0 else "^{-}") + "$"
            row += [c1, fmt(mau, sau, bold=(m in METHODS_MAIN and abs(mau - S["best_au"]) < 5e-4))]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_main", "\n".join(lines) + "\n")
    # per-client view: share of honest clients made worse off, worst-client loss
    pc = ["\\begin{tabular}{l" + "ccc" * len(cols) + "}", "\\toprule",
          " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{label[sc]}}}" for sc in cols) + " \\\\",
          " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(cols))),
          "Method & " + " & ".join(["worse than FedAvg & worse than Local & worst client"] * len(cols)) + " \\\\",
          "\\midrule"]
    for m in ["Local", "FedAvg", "IFCA", "Ditto", "PACFL", "pTopoFL"]:
        row = [{"pTopoFL": "\\ptopofl", "PACFL": "PACFL-style"}.get(m, m)]
        for sc in cols:
            P = allres[sc]["per"]
            wf = np.mean([np.mean(np.array(P[m][i]["per_client"]) > np.array(P["FedAvg"][i]["per_client"]) + 1e-12)
                          for i in range(len(P[m]))])
            wl_ = np.mean([np.mean(np.array(P[m][i]["per_client"]) > np.array(P["Local"][i]["per_client"]) + 1e-12)
                           for i in range(len(P[m]))])
            worst = np.mean([np.max(P[m][i]["per_client"]) for i in range(len(P[m]))])
            row += [f"{wf:.2f}", f"{wl_:.2f}", f"{worst:.3f}"]
            if m == "pTopoFL":
                key = {"A": "A", "B": "B", "HD": "HD"}[sc]
                out.macro(f"PC{key}WorseFedavg", f"{wf:.2f}"); out.macro(f"PC{key}WorseLocal", f"{wl_:.2f}")
        pc.append(" & ".join(row) + " \\\\")
    pc += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_perclient", "\n".join(pc) + "\n")
    for sc in cols:
        S = stats[sc]; key = {"A": "A", "B": "B", "HD": "HD"}[sc]
        for m in ["pTopoFL", "FedAvg", "IFCA", "Ditto", "Local", "Oracle", "Random", "PACFL", "pFedMe",
                  "FedProx", "SCAFFOLD"]:
            mk = {"pTopoFL": "Ptopo", "FedAvg": "Fedavg", "IFCA": "Ifca", "PACFL": "Pacfl",
                  "pFedMe": "Pfedme", "FedProx": "Fedprox", "SCAFFOLD": "Scaffold"}.get(m, m)
            out.macro(f"Res{key}{mk}LL", f"{np.mean(S['ll'][m]):.3f}")
            out.macro(f"Res{key}{mk}AUC", f"{np.mean(S['au'][m]):.3f}")
        ex = allres[sc]["extra"]
        out.macro(f"Res{key}RhoTopo", f"{np.mean(ex['rho_topo']):.2f}")
        out.macro(f"Res{key}RhoRand", f"{np.mean(ex['rho_rand']):.2f}")
        out.macro(f"Res{key}RhoOrac", f"{np.mean(ex['rho_orac']):.2f}")
        out.macro(f"Res{key}RhoCmp", "below" if np.mean(ex["rho_topo"]) < np.mean(ex["rho_rand"]) else "not below")
        if ex["ari"]:
            out.macro(f"Res{key}ARI", f"{np.mean(ex['ari']):.2f}")
            out.macro(f"Res{key}ARIpacfl", f"{np.mean(ex['ari_pacfl']):.2f}")
        out.macro(f"Res{key}NBetter", str(sum(1 for (p, sg) in S["tests"].values() if p < 0.05 and sg > 0)))
        out.macro(f"Res{key}NWorse", str(sum(1 for (p, sg) in S["tests"].values() if p < 0.05 and sg < 0)))
        best = min(METHODS_MAIN, key=lambda mm: np.mean(S["ll"][mm]))
        out.macro(f"Res{key}Best", {"pTopoFL": "\\ptopofl", "PACFL": "PACFL-style"}.get(best, best))
        if sc == "HD" and ex["sizes"]:
            out.macro("HDsizes", "/".join(str(s_) for s_ in ex["sizes"]))
    fig, axs = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 2.7), squeeze=False)
    for i, sc in enumerate(cols):
        ax = axs[0, i]
        for m in METHODS_MAIN:
            H = allres[sc]["curves"][m]
            if not H:
                continue
            rounds = [h[0] for h in H[0]]
            Y = np.array([[h[1]["logloss"] for h in hh] for hh in H])
            mu_, sd_ = Y.mean(0), Y.std(0)
            ax.plot(rounds, mu_, color=COL[m], lw=2.0 if m == "pTopoFL" else 1.1, label=m)
            ax.fill_between(rounds, mu_ - sd_, mu_ + sd_, color=COL[m], alpha=0.04, lw=0)
        ax.set_title(label[sc]); ax.set_xlabel("round"); ax.set_ylabel("personalised test log-loss")
        lo_, hi_ = np.inf, -np.inf
        for m in METHODS_MAIN:
            if allres[sc]["curves"][m]:
                Y = np.array([[h[1]["logloss"] for h in hh] for hh in allres[sc]["curves"][m]]).mean(0)
                lo_, hi_ = min(lo_, Y[len(Y) // 3:].min()), max(hi_, Y[len(Y) // 3:].max())
        ax.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.6 * (hi_ - lo_))
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); out.fig(fig, "fig_curves")
    return allres


def exp_alignment(out, seeds, T, tau):
    """E3: when does topology help?  Which geometric differences does it see?"""
    plt = _mpl()
    regs = [("aligned", "Aligned (dim.)"), ("aligned_tail", "Aligned (tails)"),
            ("concept", "Concept only"), ("geometry", "Geometry only")]
    meths = ["FedAvg", "IFCA", "Ditto", "PACFL", "pTopoFL"]
    R = {}
    for reg, _ in regs:
        r = {m: [] for m in meths + ["Random", "Oracle"]}
        r.update(ari=[], ari_pacfl=[], rho_topo=[], rho_rand=[], rho_orac=[], gain=[])
        for s in seeds:
            cl, eta, n = build(make_synthetic(reg, K=8, seed=s, shift=1.0))
            for m in meths:
                r[m].append(evaluate(cl, run_method(m, cl, eta, n, s, T, tau, 0)[0])["logloss"])
            _, _, info = ptopofl_fit(cl, eta, n, s, 1, tau, 0)
            lab, a = info["labels"], info["alpha"]
            rl = np.random.default_rng(s + 99).permutation(lab); l2, w2, p2 = plain_structure(rl, n)
            r["Random"].append(evaluate(cl, run_clustered(cl, l2, w2, p2, a, T, tau, eta, 0)[0])["logloss"])
            lo = oracle_labels(cl, CFG["M"], s); l3, w3, p3 = plain_structure(lo, n)
            r["Oracle"].append(evaluate(cl, run_clustered(cl, l3, w3, p3, a, T, tau, eta, 0)[0])["logloss"])
            wl = np.array([local_optimum(c) for c in cl])
            l1, w1, p1 = plain_structure(lab, n)          # common weights: V_tot identical across partitions
            r["rho_topo"].append(heterogeneity(cl, l1, w1, p1, wl)["rho"])
            r["rho_rand"].append(heterogeneity(cl, l2, w2, p2, wl)["rho"])
            r["rho_orac"].append(heterogeneity(cl, l3, w3, p3, wl)["rho"])
            g = np.array([c.group for c in cl])
            r["ari"].append(ari(lab, g)); r["ari_pacfl"].append(ari(subspace_labels(cl, CFG["M"]), g))
            r["gain"].append(r["FedAvg"][-1] - r["pTopoFL"][-1])
        R[reg] = r
        print(f"  [E3] {reg}: ARI topo {np.mean(r['ari']):.2f} / subspace {np.mean(r['ari_pacfl']):.2f}, "
              f"rho_topo {np.mean(r['rho_topo']):.2f}", flush=True)
    out.json("alignment", R)
    lines = ["\\begin{tabular}{l" + "c" * len(regs) + "}", "\\toprule",
             " & " + " & ".join(lbl for _, lbl in regs) + " \\\\", "\\midrule"]
    def row(name, vals):
        lines.append(name + " & " + " & ".join(vals) + " \\\\")
    row("ARI, PH-descriptor partition", [f"${np.mean(R[r_]['ari']):.2f}$" for r_, _ in regs])
    row("ARI, subspace partition", [f"${np.mean(R[r_]['ari_pacfl']):.2f}$" for r_, _ in regs])
    row("$\\rho(\\mathcal{C})$, PH descriptor", [f"${np.mean(R[r_]['rho_topo']):.2f}$" for r_, _ in regs])
    row("$\\rho(\\mathcal{C})$, random", [f"${np.mean(R[r_]['rho_rand']):.2f}$" for r_, _ in regs])
    row("$\\rho(\\mathcal{C})$, oracle$^\\diamond$", [f"${np.mean(R[r_]['rho_orac']):.2f}$" for r_, _ in regs])
    lines.append("\\midrule")
    for m in meths + ["Random", "Oracle"]:
        nm = {"Random": "Clustered, random", "Oracle": "Clustered, oracle$^\\diamond$",
              "pTopoFL": "\\ptopofl", "PACFL": "PACFL-style"}.get(m, m)
        row(nm, [fmt(np.mean(R[r_][m]), np.std(R[r_][m])) for r_, _ in regs])
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_alignment", "\n".join(lines) + "\n")
    for r_, key in [("aligned", "Al"), ("aligned_tail", "Ta"), ("concept", "Co"), ("geometry", "Ge")]:
        out.macro(f"Align{key}ARI", f"{np.mean(R[r_]['ari']):.2f}")
        out.macro(f"Align{key}ARIsub", f"{np.mean(R[r_]['ari_pacfl']):.2f}")
        out.macro(f"Align{key}RhoTopo", f"{np.mean(R[r_]['rho_topo']):.2f}")
        out.macro(f"Align{key}RhoOrac", f"{np.mean(R[r_]['rho_orac']):.2f}")
        out.macro(f"Align{key}Gain", f"{np.mean(R[r_]['gain']):.3f}")
        for m, mk in [("pTopoFL", "Ptopo"), ("FedAvg", "Fedavg"), ("IFCA", "Ifca"), ("PACFL", "Pacfl"),
                      ("Ditto", "Ditto")]:
            out.macro(f"Align{key}{mk}", f"{np.mean(R[r_][m]):.3f}")
    allx = np.concatenate([1 - np.array(R[r_]["rho_topo"]) for r_, _ in regs])
    ally = np.concatenate([R[r_]["gain"] for r_, _ in regs])
    out.macro("AlignCorr", f"{np.corrcoef(allx, ally)[0, 1]:.2f}")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for (r_, lbl), mk in zip(regs, ["o", "D", "s", "^"]):
        ax.scatter(1 - np.array(R[r_]["rho_topo"]), R[r_]["gain"], marker=mk, s=18, label=lbl)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$1-\rho(\mathcal{C})$ for the PH partition")
    ax.set_ylabel("log-loss gain over FedAvg")
    ax.legend(frameon=False, fontsize=6.5); fig.tight_layout(); out.fig(fig, "fig_alignment")
    return R


def exp_robust(out, seeds, T, tau, fracs=(0.0, 0.1, 0.2, 0.3, 0.4),
               attacks=("poison", "geom_poison", "labelflip")):
    """E4: corrupted clients (honest clients' personalised log-loss) and detection rates.
    Attack types can be run in separate invocations (--attacks); results are merged."""
    plt = _mpl()
    meths = ["FedAvg", "Median", "Trimmed", "IFCA", "Ditto", "pTopoFL", "pTopoFL (no trust)",
             "pTopoFL (v1 trust)"]
    path = os.path.join(out.root, "json", "robust.json")
    R = {}
    if os.path.exists(path):
        with open(path) as fh:
            R = {att: {float(f): v for f, v in d.items()} for att, d in json.load(fh).items()}
    for att in attacks:
        R[att] = {}
        for f in fracs:
            K = 10; na = int(round(f * K))
            r = {m: [] for m in meths}; r.update(tpr=[], fpr=[], tpr_v1=[], fpr_v1=[])
            for s in seeds:
                cl, eta, n = build(make_synthetic("aligned", K=K, seed=s, shift=1.0, n_adv=na, attack=att))
                adv = np.array([c.adv for c in cl])
                Fh = normalise_rows(descriptors([c.raw for c in cl], s))
                for m in meths:
                    if m.startswith("pTopoFL"):
                        tr = {"pTopoFL": "mad", "pTopoFL (no trust)": "none", "pTopoFL (v1 trust)": "std"}[m]
                        mod, _, info = ptopofl_fit(cl, eta, n, s, T, tau, 0, trust=tr, Fh=Fh)
                        if na > 0 and tr in ("mad", "std"):
                            flagged = info["trust"] < 1 - 1e-9
                            r["tpr" if tr == "mad" else "tpr_v1"].append(float(flagged[adv].mean()))
                            r["fpr" if tr == "mad" else "fpr_v1"].append(float(flagged[~adv].mean()))
                    else:
                        mod = run_method(m, cl, eta, n, s, T, tau, 0)[0]
                    r[m].append(evaluate(cl, mod)["logloss"])
            R[att][f] = r
            print(f"  [E4] {att} frac={f} done", flush=True)
    out.json("robust", R)
    order = [a_ for a_ in ("poison", "geom_poison", "labelflip") if a_ in R]
    fr = sorted(R[order[0]].keys())
    titles = {"poison": "shift + 40% label flips (earlier attack)",
              "geom_poison": "heavy-tailed corruption + 40% flips", "labelflip": "label inversion only"}
    fig, axs = plt.subplots(1, len(order), figsize=(3.5 * len(order), 2.7), squeeze=False)
    for i, att in enumerate(order):
        ax = axs[0, i]
        for m in meths:
            mu_ = [np.mean(R[att][f][m]) for f in fr]
            ax.plot(np.array(fr) * 100, mu_, marker="o", ms=3, color=COL.get(m, "k"),
                    lw=2.0 if m == "pTopoFL" else 1.0, ls="--" if "(" in m else "-", label=m)
        ax.set_title(titles[att]); ax.set_xlabel("corrupted clients (%)")
        ax.set_ylabel("honest clients' log-loss")
        if i == 0:
            ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); out.fig(fig, "fig_robust")
    fpos = [f for f in fr if f > 0]
    lines = ["\\begin{tabular}{l" + "c" * len(fpos) + "}", "\\toprule",
             "Attack / score & " + " & ".join(f"{int(round(100 * f))}\\%" for f in fpos) + " \\\\",
             "\\midrule"]
    short = {"poison": "shift + flips", "geom_poison": "heavy-tailed + flips", "labelflip": "label inversion"}
    for att in order:
        for rule, kt, kf in [("robust", "tpr", "fpr"), ("mean/std", "tpr_v1", "fpr_v1")]:
            vals = [f"{np.mean(R[att][f][kt]):.2f} / {np.mean(R[att][f][kf]):.2f}" for f in fpos]
            lines.append(f"{short[att]}, {rule} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_detection", "\n".join(lines) + "\n")
    allt = [v for att in order for f in fpos for v in R[att][f]["tpr"]]
    allf = [v for att in order for f in fpos for v in R[att][f]["fpr"]]
    out.macro("RobTPR", f"{np.mean(allt):.2f}"); out.macro("RobFPR", f"{np.mean(allf):.2f}")
    f2 = 0.2 if 0.2 in fr else fr[-1]; f4 = fr[-1]
    out.macro("RobFracMid", f"{int(round(100 * f2))}"); out.macro("RobFracMax", f"{int(round(100 * f4))}")
    for att, key in [("poison", "Po"), ("geom_poison", "Ge"), ("labelflip", "Lf")]:
        if att in R:
            for m, mk in [("pTopoFL", "Ptopo"), ("FedAvg", "Fedavg"), ("Ditto", "Ditto"), ("IFCA", "Ifca"),
                          ("pTopoFL (no trust)", "PtopoNT"), ("Median", "Median"), ("Trimmed", "Trimmed")]:
                out.macro(f"Rob{key}{mk}", f"{np.mean(R[att][f2][m]):.3f}")
                out.macro(f"Rob{key}{mk}Max", f"{np.mean(R[att][f4][m]):.3f}")
            out.macro(f"Rob{key}TPR", f"{np.mean([v for f in fpos for v in R[att][f]['tpr']]):.2f}")
    return R


def analytic_gauss_eps(sigma, sens, delta):
    """Smallest eps such that N(0, sigma^2) is (eps, delta)-DP for L2-sensitivity sens
    (exact characterisation of Balle & Wang, 2018), found by bisection."""
    from scipy.stats import norm
    def dlt(eps):
        a = sens / (2 * sigma); b = eps * sigma / sens
        return norm.cdf(a - b) - math.exp(min(eps + norm.logcdf(-a - b), 700.0))
    lo, hi = 0.0, 1.0
    while dlt(hi) > delta and hi < 1e4:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dlt(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def dp_sigmas(eps, delta, steps, f_phi):
    """Split an (eps, delta) target between the one-shot descriptor release (fraction f_phi of
    the RDP budget R) and DP-GD; R from eps = R + 2 sqrt(R log(1/delta)) (Prop. 9)."""
    if not np.isfinite(eps):
        return 0.0, 0.0
    c = math.log(1.0 / delta)
    R = (math.sqrt(c + eps) - math.sqrt(c)) ** 2
    sig_dp = math.sqrt(steps / (2.0 * (1.0 - f_phi) * R))
    sig_phi = math.sqrt(2.0 / (f_phi * R)) if f_phi > 0 else 0.0
    return sig_dp, sig_phi


def exp_privacy(out, seeds, eps_grid=(np.inf, 64.0, 16.0, 8.0, 4.0), delta=1e-5, T=40, tau=5,
                clip=5.0, noise_grid=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)):
    """E5: (a) partition stability under descriptor noise (Thm 6);
           (b) record-level DP sweep for the full transmitted pair (Prop. 9);
           (c) membership inference on transmitted local models; (d) property inference."""
    plt = _mpl()
    from sklearn.linear_model import LogisticRegression
    # ---------------- (a) descriptor noise vs partition ----------------
    Ra = {s_: dict(ari=[], ll=[]) for s_ in noise_grid}
    gaps, loo, inter = [], [], []
    for s in seeds:
        cl, eta, n = build(scenario_raw("A", s))
        Fh = normalise_rows(descriptors([c.raw for c in cl], s))
        lab0 = agglomerative(Fh, CFG["M"])
        D = squareform(pdist(Fh)); inter.append(float(np.median(pdist(Fh))))
        same = (lab0[:, None] == lab0[None, :]); off = ~np.eye(len(D), dtype=bool)
        din = D[same & off].max() if (same & off).any() else 0.0
        gaps.append(float(D[~same].min() - din))
        for sg in noise_grid:
            mod, _, info = ptopofl_fit(cl, eta, n, s, CFG["rounds"], CFG["tau"], 0, Fh=Fh, noise=sg)
            Ra[sg]["ari"].append(ari(info["labels"], lab0)); Ra[sg]["ll"].append(evaluate(cl, mod)["logloss"])
        if s == seeds[0]:
            X0 = cl[0].raw; base = Fh[0]
            for i in range(min(len(X0), 40)):
                fi = ph_descriptor(np.delete(X0, i, axis=0), seed=10_000 * s + 1)
                loo.append(float(np.linalg.norm(fi / np.linalg.norm(fi) - base)))
    m_dim = 2 * CFG["n_thr"] + 6
    gp = np.array(gaps); pos = gp[gp > 0]
    safe = float(np.median(pos) / (4 * (np.sqrt(m_dim) + np.sqrt(2 * np.log(8 / 0.05))))) if len(pos) else float("nan")
    # ---------------- (b)+(c) DP sweep ----------------
    S_steps = T * tau
    meths = ["DP-FedAvg", "DP-pTopoFL", "DP-pTopoFL (descriptor not private)"]
    Rb = {e: {m: dict(ll=[], mia=[], ari=[]) for m in meths} for e in eps_grid}
    loc_ll = []
    feats, groups = [], []
    for s in seeds:
        raw = make_synthetic("aligned", K=8, seed=s, shift=1.0, n_range=(400, 1200), standardise=False)
        cl = [Client(c, CFG["lam"], clip=clip) for c in raw]
        eta = step_size(cl); n = np.array([c.n for c in cl], float)
        g = np.array([c.group for c in cl])
        Fh = normalise_rows(descriptors([c.raw for c in cl], s))
        feats.append(Fh); groups.append(g)
        lab_clean = agglomerative(Fh, CFG["M"])
        loc_ll.append(evaluate(cl, run_local(cl, T, tau, eta, 0)[0])["logloss"])
        for e in eps_grid:
            for m in meths:
                f_phi = 0.5 if m == "DP-pTopoFL" else 0.0
                sig_dp, sig_phi = dp_sigmas(e, delta, S_steps, f_phi)
                rng = np.random.default_rng(1000 * s + 17)
                if m == "DP-FedAvg":
                    lab = np.zeros(len(cl), int); w = n / n.sum(); pi = np.array([1.0]); a = 0.0
                else:
                    F_rel = Fh + (np.random.default_rng(s + 5).normal(0, sig_phi, Fh.shape) if sig_phi > 0 else 0.0)
                    lab, w, pi, _, _ = ptopofl_structure(F_rel, n, CFG["M"])
                    a = alpha_from_effective(0.3, eta, tau, CFG["lam"])
                mod, _, _, _, last = run_clustered(cl, lab, w, pi, a, T, tau, eta, 0, dp_sigma=sig_dp, rng=rng)
                Rb[e][m]["ll"].append(evaluate(cl, mod)["logloss"])
                Rb[e][m]["mia"].append(mia_auc(cl, last))
                Rb[e][m]["ari"].append(ari(lab, g) if m != "DP-FedAvg" else 0.0)
        print(f"  [E5] seed {s} done", flush=True)
    # ---------------- (d) property inference from descriptors ----------------
    Fall, gall = np.vstack(feats), np.concatenate(groups)
    def loo_acc(F):
        acc = []
        for i in range(len(F)):
            msk = np.ones(len(F), bool); msk[i] = False
            mu, sd = F[msk].mean(0), F[msk].std(0) + 1e-12
            clf = LogisticRegression(C=1.0, max_iter=2000).fit((F[msk] - mu) / sd, gall[msk])
            acc.append(float(clf.predict(((F[i] - mu) / sd)[None])[0] == gall[i]))
        return float(np.mean(acc))
    prop = {}
    for e in eps_grid:
        _, sphi = dp_sigmas(e, delta, S_steps, 0.5)
        Fn = Fall + (np.random.default_rng(7).normal(0, sphi, Fall.shape) if sphi > 0 else 0.0)
        prop[e] = loo_acc(Fn)
    out.json("privacy", dict(noise=Ra, gaps=gaps, loo=loo,
                             dp={str(e): v for e, v in Rb.items()}, local_ll=loc_ll,
                             prop={str(e): v for e, v in prop.items()}))
    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.7))
    xs = np.array([max(sg, 1e-4) for sg in noise_grid])
    ax[0].semilogx(xs, [np.mean(Ra[sg]["ari"]) for sg in noise_grid], "o-", color=COL["pTopoFL"])
    if np.isfinite(safe):
        ax[0].axvline(safe, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel(r"descriptor noise $\sigma_\phi$"); ax[0].set_ylabel("ARI (noisy vs. clean partition)")
    ax[0].set_title("(a) partition stability (dashed: Thm. 6)")
    elab = ["$\\infty$" if not np.isfinite(e) else f"{e:g}" for e in eps_grid]
    xx = np.arange(len(eps_grid))
    for m, c_ in zip(meths, ["#1f77b4", "#2ca02c", "#98df8a"]):
        ax[1].plot(xx, [np.mean(Rb[e][m]["ll"]) for e in eps_grid], "o-", color=c_, label=m, lw=1.4)
        ax[2].plot(xx, [np.mean(Rb[e][m]["mia"]) for e in eps_grid], "o-", color=c_, label=m, lw=1.4)
    ax[1].axhline(np.mean(loc_ll), color="#7f7f7f", ls=":", lw=1, label="Local (nothing released)")
    ax[2].axhline(0.5, color="k", lw=0.6)
    for a_ in ax[1:]:
        a_.set_xticks(xx); a_.set_xticklabels(elab); a_.set_xlabel(r"record-level $\varepsilon$ ($\delta=10^{-5}$)")
    ax[1].set_ylabel("personalised log-loss"); ax[1].set_title("(b) utility under DP")
    ax[2].set_ylabel("membership-inference AUC"); ax[2].set_title("(c) attack on transmitted models")
    ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); out.fig(fig, "fig_privacy")
    # ---------------- table + macros ----------------
    lines = ["\\begin{tabular}{l" + "c" * len(eps_grid) + "}", "\\toprule",
             "$\\varepsilon$ & " + " & ".join(elab) + " \\\\", "\\midrule"]
    tex_name = {m: m.replace("pTopoFL", "\\ptopofl{}") for m in meths}   # no backslash inside f-string braces (Python <= 3.11)
    for m in meths:
        lines.append(tex_name[m] + " log-loss & " +
                     " & ".join(f"${np.mean(Rb[e][m]['ll']):.3f}$" for e in eps_grid) + " \\\\")
    for m in meths:
        lines.append(tex_name[m] + " MIA AUC & " +
                     " & ".join(f"${np.mean(Rb[e][m]['mia']):.3f}$" for e in eps_grid) + " \\\\")
    lines.append("DP-\\ptopofl{} ARI (partition vs.\\ latent groups) & " +
                 " & ".join(f"${np.mean(Rb[e]['DP-pTopoFL']['ari']):.2f}$" for e in eps_grid) + " \\\\")
    lines.append("Property inference from descriptor (accuracy) & " +
                 " & ".join(f"${prop[e]:.2f}$" for e in eps_grid) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_privacy", "\n".join(lines) + "\n")
    out.macro("DPsafeSigma", f"{safe:.4f}"); out.macro("DPgapPos", f"{np.mean(gp > 0):.2f}")
    out.macro("DPlooMed", f"{np.median(loo):.4f}"); out.macro("DPlooMax", f"{np.max(loo):.4f}")
    out.macro("DPinterMed", f"{np.median(inter):.4f}")
    ok = [sg for sg in noise_grid if sg > 0 and np.mean(Ra[sg]["ari"]) >= 0.9]
    out.macro("DPsigmaOK", f"{max(ok):.4f}" if ok else "0")
    e_ok = analytic_gauss_eps(max(ok), 2.0, delta) if ok else float("inf")
    out.macro("DPepsOK", f"{e_ok:.0f}")
    out.macro("DPsteps", str(S_steps)); out.macro("DPclip", f"{clip:g}")
    fin = [e for e in eps_grid if np.isfinite(e)]
    e8 = 8.0 if 8.0 in eps_grid else fin[len(fin) // 2]
    out.macro("DPepsMid", f"{e8:g}")
    for m, mk in [("DP-FedAvg", "Fedavg"), ("DP-pTopoFL", "Ptopo"), ("DP-pTopoFL (descriptor not private)", "PtopoNP")]:
        out.macro(f"DP{mk}LLinf", f"{np.mean(Rb[np.inf][m]['ll']):.3f}")
        out.macro(f"DP{mk}LLmid", f"{np.mean(Rb[e8][m]['ll']):.3f}")
        out.macro(f"DP{mk}MIAinf", f"{np.mean(Rb[np.inf][m]['mia']):.3f}")
        out.macro(f"DP{mk}MIAmid", f"{np.mean(Rb[e8][m]['mia']):.3f}")
    out.macro("DPPtopoARImid", f"{np.mean(Rb[e8]['DP-pTopoFL']['ari']):.2f}")
    out.macro("DPPtopoARIinf", f"{np.mean(Rb[np.inf]['DP-pTopoFL']['ari']):.2f}")
    out.macro("DPLocalLL", f"{np.mean(loc_ll):.3f}")
    out.macro("PropInfClean", f"{prop[np.inf]:.2f}"); out.macro("PropInfMid", f"{prop[e8]:.2f}")
    _, sphi_mid = dp_sigmas(e8, delta, S_steps, 0.5); out.macro("DPsigPhiMid", f"{sphi_mid:.2f}")
    sdp_mid, _ = dp_sigmas(e8, delta, S_steps, 0.0); out.macro("DPsigDPMid", f"{sdp_mid:.1f}")
    return Rb


def exp_sensitivity(out, seeds, T):
    """E6b: sensitivity to M, n_sub, n_rep and descriptor normalisation (reviewer request)."""
    tau = CFG["tau"]; rows = []; res = {}
    configs = [("$M=1$", dict(M=1)), ("$M=2$ (default)", dict(M=2)), ("$M=3$", dict(M=3)), ("$M=4$", dict(M=4)),
               ("$n_{\\mathrm{sub}}=40$", dict(n_sub=40)), ("$n_{\\mathrm{sub}}=160$", dict(n_sub=160)),
               ("one subsample ($n_{\\mathrm{rep}}=1$)", dict(n_rep=1)),
               ("per-coordinate $z$-score normalisation", dict(norm="zscore"))]
    for name, cfg in configs:
        ll, aa = [], []
        for s in seeds:
            cl, eta, n = build(scenario_raw("A", s))
            F = np.array([ph_descriptor(c.raw, seed=10_000 * s + 97 * k + 1, n_sub=cfg.get("n_sub"),
                                        n_rep=cfg.get("n_rep")) for k, c in enumerate(cl)])
            Fh = normalise_desc(F, cfg.get("norm", "l2"))
            mod, _, info = ptopofl_fit(cl, eta, n, s, T, tau, 0, M=cfg.get("M", CFG["M"]), Fh=Fh)
            ll.append(evaluate(cl, mod)["logloss"]); aa.append(ari(info["labels"], np.array([c.group for c in cl])))
        res[name] = dict(ll=ll, ari=aa)
        rows.append(f"{name} & {fmt(np.mean(ll), np.std(ll))} & ${np.mean(aa):.2f}$ \\\\")
        print(f"  [E6b] {name}: {np.mean(ll):.4f}  ARI {np.mean(aa):.2f}", flush=True)
    out.json("sensitivity", res)
    out.table("tab_sensitivity", "\n".join(["\\begin{tabular}{lcc}", "\\toprule",
                                            "Variant & Log-loss $\\downarrow$ & ARI \\\\", "\\midrule"] + rows +
                                           ["\\bottomrule", "\\end{tabular}"]) + "\n")
    return res


def exp_ablation(out, seeds, T):
    """E6: component ablation and the (alpha, tau) interaction."""
    rows = []
    cfgs = [("Full \\ptopofl{} ($\\bar\\alpha=0.3$, $\\tau=10$)", dict()),
            ("No clustering ($M=1$)", dict(M=1)),
            ("Uniform within-cluster weights ($w_k\\propto n_k$)", dict(uniform=True)),
            ("No trust weights", dict(trust="none")),
            ("v1 trust rule (mean/std $z$)", dict(trust="std")),
            ("Uncalibrated $\\alpha=0.3$ (v1 default)", dict(alpha_raw=0.3)),
            ("$\\alpha=0$ (no blending)", dict(alpha_raw=0.0)),
            ("$\\alpha=1$ (consensus only)", dict(alpha_raw=1.0))]
    R = {}
    for name, cfg in cfgs:
        vals = []
        for s in seeds:
            cl, eta, n = build(scenario_raw("A", s))
            Fh = normalise_rows(descriptors([c.raw for c in cl], s))
            M = cfg.get("M", CFG["M"])
            lab, w, pi, t, z = ptopofl_structure(Fh, n, M, trust=cfg.get("trust", "mad"))
            if cfg.get("uniform"):
                lab, w, pi = plain_structure(lab, n)
            a = cfg.get("alpha_raw", alpha_from_effective(0.3, eta, CFG["tau"], CFG["lam"]))
            vals.append(evaluate(cl, run_clustered(cl, lab, w, pi, a, T, CFG["tau"], eta, 0)[0])["logloss"])
        R[name] = vals
        rows.append(f"{name} & {fmt(np.mean(vals), np.std(vals))} \\\\")
        print(f"  [E6] {name}: {np.mean(vals):.4f}", flush=True)
    grid = {}
    for tau in (1, 10, 50):
        for abar in (0.0, 0.1, 0.3, 0.6, 1.0):
            vals = []
            for s in seeds:
                cl, eta, n = build(scenario_raw("A", s))
                Fh = normalise_rows(descriptors([c.raw for c in cl], s))
                lab, w, pi, t, z = ptopofl_structure(Fh, n, CFG["M"])
                a = alpha_from_effective(abar, eta, tau, CFG["lam"]) if abar < 1 else 1.0
                Tt = max(int(T * CFG["tau"] / tau), 30)   # equal total local computation
                vals.append(evaluate(cl, run_clustered(cl, lab, w, pi, a, Tt, tau, eta, 0)[0])["logloss"])
            grid[(tau, abar)] = vals
    out.json("ablation", dict(R=R, grid={f"{k[0]}_{k[1]}": v for k, v in grid.items()}))
    tab = ["\\begin{tabular}{lc}", "\\toprule", "Variant (Synthetic, clustered) & Log-loss $\\downarrow$ \\\\",
           "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_ablation", "\n".join(tab) + "\n")
    g = ["\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule",
         "$\\tau$ \\textbackslash{} $\\bar\\alpha$ & 0 & 0.1 & 0.3 & 0.6 & 1 \\\\", "\\midrule"]
    for tau in (1, 10, 50):
        g.append(f"{tau} & " + " & ".join(f"${np.mean(grid[(tau, ab)]):.3f}$" for ab in (0.0, 0.1, 0.3, 0.6, 1.0)) + " \\\\")
    g += ["\\bottomrule", "\\end{tabular}"]
    out.table("tab_alpha_tau", "\n".join(g) + "\n")
    keys = list(R)
    out.macro("AblFull", f"{np.mean(R[keys[0]]):.3f}"); out.macro("AblNoClust", f"{np.mean(R[keys[1]]):.3f}")
    out.macro("AblUnif", f"{np.mean(R[keys[2]]):.3f}"); out.macro("AblNoTrust", f"{np.mean(R[keys[3]]):.3f}")
    out.macro("AblVoneAlpha", f"{np.mean(R[keys[5]]):.3f}"); out.macro("AblAlphaZero", f"{np.mean(R[keys[6]]):.3f}")
    out.macro("AblAlphaOne", f"{np.mean(R[keys[7]]):.3f}")
    return R


def exp_continual(out, seeds, K=6, rounds=20):
    """E7 (appendix): drift monitoring.  Two clients' feature geometry drifts progressively
    (intrinsic dimension 20 -> 3); descriptors recomputed every round on fresh samples."""
    plt = _mpl()
    aucs, tprs, fprs, traces = [], [], [], None
    d = 20
    for s in seeds:
        rng = np.random.default_rng(s)
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :3].T * np.sqrt(d / 3)
        drift = rng.choice(K, 2, replace=False)
        phis = np.zeros((rounds, K, 2 * CFG["n_thr"] + 6))
        for r in range(rounds):
            for k in range(K):
                X = rng.normal(size=(120, d))
                if k in drift:
                    lam_r = min(1.0, r / (rounds - 1))
                    X = (1 - lam_r) * X + lam_r * (rng.normal(size=(120, 3)) @ Q + 0.3 * rng.normal(size=(120, d)))
                phis[r, k] = ph_descriptor(X, seed=10_000 * s + 100 * r + k, n_rep=1)
        ph = phis / np.linalg.norm(phis, axis=2, keepdims=True)
        D = np.mean(np.linalg.norm(ph[1:] - ph[:1], axis=2), axis=0)
        y = np.isin(np.arange(K), drift).astype(int)
        aucs.append(_auc(y, D))
        flag = D > np.percentile(D, 75)
        tprs.append(float(flag[y == 1].mean())); fprs.append(float(flag[y == 0].mean()))
        if traces is None:
            traces = (ph[:, :, 42].tolist(), ph[:, :, 43].tolist(), drift.tolist())
    out.json("continual", dict(auc=aucs, tpr=tprs, fpr=fprs))
    out.macro("ContAUC", f"{np.mean(aucs):.2f}"); out.macro("ContTPR", f"{np.mean(tprs):.2f}")
    out.macro("ContFPR", f"{np.mean(fprs):.2f}")
    h0, h1, dr = traces
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k in range(K):
        ls = "--" if k in dr else "-"
        ax[0].plot(np.arange(1, rounds + 1), np.array(h0)[:, k], ls=ls, lw=1.2)
        ax[1].plot(np.arange(1, rounds + 1), np.array(h1)[:, k], ls=ls, lw=1.2)
    ax[0].set_title("(a) normalised $H_0$ entropy coordinate"); ax[1].set_title("(b) normalised $H_1$ entropy coordinate")
    for a_ in ax:
        a_.set_xlabel("round")
    fig.tight_layout(); out.fig(fig, "fig_continual")
    print(f"  [E7] drift detection AUC {np.mean(aucs):.2f}", flush=True)


def exp_cost(out, seed=0):
    """E8 (appendix): descriptor computation time and per-round communication."""
    cl, eta, n = build(scenario_raw("A", seed))
    X = cl[0].raw
    t0 = time.time(); ph_descriptor(X, seed=1, n_rep=1); t_desc = time.time() - t0
    p = cl[0].p; M = CFG["M"]
    t0 = time.time(); ph_descriptor_once(X, CFG["n_sub"], np.random.default_rng(0), CFG["n_thr"]); t1 = time.time() - t0
    t0 = time.time(); D = squareform(pdist(X[:CFG["n_sub"]])); _h1_pairs_numpy(D); t_np = time.time() - t0
    out.macro("CostDescOne", f"{t1:.3f}"); out.macro("CostHoneNumpy", f"{t_np:.2f}")
    out.macro("CostDescSec", f"{t_desc:.3f}")
    out.macro("CostBackend", "ripser" if HAVE_RIPSER else "NumPy reference")
    out.macro("CostP", str(p))
    out.table("tab_comm", "\n".join([
        "\\begin{tabular}{lcc}", "\\toprule",
        "Method & uplink / client / round & downlink / client / round \\\\", "\\midrule",
        f"FedAvg, FedProx, Ditto & $p$ & $p$ \\\\",
        f"SCAFFOLD & $2p$ & $2p$ \\\\",
        f"pFedMe & $p$ & $p$ \\\\",
        f"IFCA ($M$ clusters) & $p$ (+ cluster id) & $Mp$ \\\\",
        f"\\ptopofl & $p$ (+ $46$ floats once) & $p$ \\\\",
        "\\bottomrule", "\\end{tabular}"]) + "\n")


def write_hparams(out):
    rows = [("$\\lambda$ (L2, all parameters)", CFG["lam"]), ("$\\tau$ (local GD steps)", CFG["tau"]),
            ("$T$ (rounds)", CFG["rounds"]), ("$\\eta$", "$1/\\max_k L_k$"),
            ("$M$ (clusters)", CFG["M"]), ("$\\bar\\alpha$ (target effective blending)", 0.3),
            ("$n_{\\mathrm{sub}}$, subsamples averaged", f"{CFG['n_sub']}, {CFG['n_rep']}"),
            ("$L$ (Betti-curve resolution)", CFG["n_thr"]), ("$z_0$ (robust $z$ threshold)", CFG["z0"]),
            ("FedProx $\\mu$", CFG["prox_mu"]), ("pFedMe $\\lambda$, inner steps", f"{CFG['pfedme_lam']}, {CFG['pfedme_inner']}"),
            ("Ditto $\\lambda$", CFG["ditto_lam"]), ("test fraction", CFG["test_frac"])]
    out.table("tab_hparams", "\n".join(["\\begin{tabular}{ll}", "\\toprule", "Hyper-parameter & Value \\\\",
                                        "\\midrule"] + [f"{a} & {b} \\\\" for a, b in rows] +
                                       ["\\bottomrule", "\\end{tabular}"]) + "\n")


# ====================================================================================
# 8. Entry point
# ====================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    help="comma list: theory,main,alignment,robust,privacy,ablation,sensitivity,continual,cost | all")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--quick", action="store_true", help="smoke test (2 seeds, short runs)")
    ap.add_argument("--no-heart", action="store_true", help="skip the real dataset (no internet)")
    ap.add_argument("--attacks", default="poison,geom_poison,labelflip",
                    help="robustness experiment: attack types to run (results are merged)")
    a = ap.parse_args(argv)
    out = Out(a.out)
    if os.path.exists(os.path.join(a.out, "tables", "results_macros.tex")):
        with open(os.path.join(a.out, "tables", "results_macros.tex")) as fh:
            for line in fh:
                if line.startswith("\\newcommand"):
                    k = line.split("{\\")[1].split("}")[0]; v = line.split("}{", 1)[1].rsplit("}", 1)[0]
                    out.macros[k] = v
    seeds = list(range(2 if a.quick else a.seeds))
    T = 40 if a.quick else CFG["rounds"]
    tau, ev = CFG["tau"], CFG["eval_every"]
    exps = ["theory", "main", "alignment", "robust", "privacy", "ablation", "sensitivity", "continual",
            "cost"] if a.exp == "all" else a.exp.split(",")
    print(f"pTopoFL v2 | ripser={HAVE_RIPSER} | seeds={len(seeds)} | T={T} | exps={exps}")
    write_hparams(out)
    out.macro("NumSeeds", str(len(seeds)))
    t0 = time.time()
    if "theory" in exps:
        exp_theory(out, quick=a.quick)
    if "main" in exps:
        exp_main(out, seeds, T, tau, ev, scenarios=("A", "B") if a.no_heart else ("A", "B", "HD"))
    if "alignment" in exps:
        exp_alignment(out, seeds, T, tau)
    if "robust" in exps:
        exp_robust(out, seeds, T, tau, fracs=(0.0, 0.2, 0.4) if a.quick else (0.0, 0.1, 0.2, 0.3, 0.4),
                   attacks=tuple(a.attacks.split(",")))
    if "privacy" in exps:
        exp_privacy(out, seeds, T=20 if a.quick else 40)
    if "ablation" in exps:
        exp_ablation(out, seeds, T)
    if "sensitivity" in exps:
        exp_sensitivity(out, seeds, T)
    if "continual" in exps:
        exp_continual(out, seeds[:5] if not a.quick else seeds)
    if "cost" in exps:
        exp_cost(out)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}/")


if __name__ == "__main__":
    main()
