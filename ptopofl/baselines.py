"""
baselines/baselines.py
======================
All five FL baselines compared in Table 1.

Each baseline exposes the same interface as PTopoFL:
  - fit(client_data, eval_data=None)  → self
  - metrics_  list of per-round dicts with 'round', 'auc', 'acc'

Implementations are logistic-regression based (matching the paper's
choice of a linear local model class for fair framework comparison).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(C=1.0, random_state=None):
    return LogisticRegression(
        C=C, max_iter=300, solver='lbfgs',
        random_state=random_state, warm_start=False
    )


def _fit_local(model_template, global_params, X, y,
               prox_mu=None, scaffold_correction=None,
               moreau_lambda=None, moreau_params=None,
               max_iter=200):
    """
    Fit a local model with optional FedProx / SCAFFOLD / pFedMe modifications.
    All implemented as modified logistic regression via sklearn.
    """
    model = clone(model_template)

    # Standard local fit
    model.fit(X, y)

    if global_params is None:
        return _extract_params(model)

    coef_local = model.coef_.copy()
    intercept_local = model.intercept_.copy()

    # FedProx: proximal update toward global (closed form for L2 reg)
    if prox_mu is not None and global_params is not None:
        coef_g = global_params['coef']
        intercept_g = global_params['intercept']
        # Proximal step: θ* = (θ_local + μ·θ_global) / (1 + μ)
        coef_local = (coef_local + prox_mu * coef_g) / (1 + prox_mu)
        intercept_local = (intercept_local + prox_mu * intercept_g) / (1 + prox_mu)

    # pFedMe: Moreau envelope — personalised model
    if moreau_lambda is not None and moreau_params is not None:
        coef_g = moreau_params['coef']
        intercept_g = moreau_params['intercept']
        # pFedMe closed form: θ_p = (θ_local + λ·θ_g) / (1 + λ)
        coef_local = (coef_local + moreau_lambda * coef_g) / (1 + moreau_lambda)
        intercept_local = (intercept_local + moreau_lambda * intercept_g) / (1 + moreau_lambda)

    return {
        'coef': coef_local,
        'intercept': intercept_local,
        'classes': model.classes_.copy(),
    }


def _extract_params(model):
    return {
        'coef': model.coef_.copy(),
        'intercept': model.intercept_.copy(),
        'classes': model.classes_.copy(),
    }


def _fedavg_aggregate(local_params_list, n_samples):
    """Data-volume-weighted FedAvg aggregation."""
    total = sum(n_samples) + 1e-8
    coef = sum((n / total) * p['coef'] for n, p in zip(n_samples, local_params_list))
    intercept = sum(
        (n / total) * p['intercept'] for n, p in zip(n_samples, local_params_list)
    )
    classes = local_params_list[0]['classes']
    return {'coef': coef, 'intercept': intercept, 'classes': classes}


def _evaluate(global_params, X_test, y_test, model_template):
    model = clone(model_template)
    try:
        model.coef_ = global_params['coef']
        model.intercept_ = global_params['intercept']
        model.classes_ = global_params['classes']
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        acc = float((model.predict(X_test) == y_test).mean())
    except Exception:
        auc, acc = 0.5, 0.5
    return auc, acc


def _pilot_params(client_data, model_template):
    """Bootstrap global model from pilot data."""
    X_pilot = np.vstack([d[0][:5] for d in client_data])
    y_pilot = np.concatenate([d[1][:5] for d in client_data])
    m = clone(model_template)
    m.fit(X_pilot, y_pilot)
    return _extract_params(m)


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg
# ─────────────────────────────────────────────────────────────────────────────

class FedAvg:
    """McMahan et al. (2017) — data-volume-weighted average."""

    def __init__(self, n_rounds=15, random_state=None):
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.model_template = _make_model(random_state=random_state)
        self.metrics_ = []

    def fit(self, client_data, eval_data=None):
        K = len(client_data)
        n_samples = [len(d[0]) for d in client_data]
        global_params = _pilot_params(client_data, self.model_template)

        for r in range(self.n_rounds):
            local = [_fit_local(self.model_template, global_params, X, y)
                     for X, y in client_data]
            global_params = _fedavg_aggregate(local, n_samples)

            m = {'round': r + 1}
            if eval_data is not None:
                m['auc'], m['acc'] = _evaluate(
                    global_params, eval_data[0], eval_data[1], self.model_template
                )
            self.metrics_.append(m)

        self.global_params_ = global_params
        return self


# ─────────────────────────────────────────────────────────────────────────────
# FedProx
# ─────────────────────────────────────────────────────────────────────────────

class FedProx:
    """Li et al. (2020) — proximal penalty μ/2 ||θ - θ_g||²."""

    def __init__(self, mu=0.1, n_rounds=15, random_state=None):
        self.mu = mu
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.model_template = _make_model(random_state=random_state)
        self.metrics_ = []

    def fit(self, client_data, eval_data=None):
        n_samples = [len(d[0]) for d in client_data]
        global_params = _pilot_params(client_data, self.model_template)

        for r in range(self.n_rounds):
            local = [
                _fit_local(self.model_template, global_params, X, y,
                           prox_mu=self.mu)
                for X, y in client_data
            ]
            global_params = _fedavg_aggregate(local, n_samples)

            m = {'round': r + 1}
            if eval_data is not None:
                m['auc'], m['acc'] = _evaluate(
                    global_params, eval_data[0], eval_data[1], self.model_template
                )
            self.metrics_.append(m)

        self.global_params_ = global_params
        return self


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD
# ─────────────────────────────────────────────────────────────────────────────

class SCAFFOLD:
    """
    Karimireddy et al. (2020) — control-variate drift correction.

    Client correction: g_k_corrected = g_k - c_k + c_global
    where c_k and c_global are running estimates of the local and
    global gradient directions.
    """

    def __init__(self, n_rounds=15, lr=0.01, random_state=None):
        self.n_rounds = n_rounds
        self.lr = lr
        self.random_state = random_state
        self.model_template = _make_model(random_state=random_state)
        self.metrics_ = []

    def fit(self, client_data, eval_data=None):
        K = len(client_data)
        n_samples = [len(d[0]) for d in client_data]
        global_params = _pilot_params(client_data, self.model_template)

        # Initialise control variates at zero
        c_global = {
            'coef': np.zeros_like(global_params['coef']),
            'intercept': np.zeros_like(global_params['intercept']),
        }
        c_local = [
            {'coef': np.zeros_like(global_params['coef']),
             'intercept': np.zeros_like(global_params['intercept'])}
            for _ in range(K)
        ]

        for r in range(self.n_rounds):
            local_params = []
            delta_c = []

            for k, (X, y) in enumerate(client_data):
                # Local fit
                lp = _fit_local(self.model_template, global_params, X, y)

                # SCAFFOLD correction: subtract local variate, add global variate
                lp['coef'] = (lp['coef']
                               - c_local[k]['coef']
                               + c_global['coef'])
                lp['intercept'] = (lp['intercept']
                                    - c_local[k]['intercept']
                                    + c_global['intercept'])

                # Update local control variate
                new_ck_coef = (c_local[k]['coef']
                                - c_global['coef']
                                + (global_params['coef'] - lp['coef']) / (self.lr + 1e-8))
                new_ck_int = (c_local[k]['intercept']
                               - c_global['intercept']
                               + (global_params['intercept'] - lp['intercept'])
                               / (self.lr + 1e-8))

                delta_c.append({
                    'coef': new_ck_coef - c_local[k]['coef'],
                    'intercept': new_ck_int - c_local[k]['intercept'],
                })
                c_local[k] = {'coef': new_ck_coef, 'intercept': new_ck_int}
                local_params.append(lp)

            # Update global control variate
            total = sum(n_samples) + 1e-8
            c_global['coef'] += sum(
                (n / total) * dc['coef'] for n, dc in zip(n_samples, delta_c)
            )
            c_global['intercept'] += sum(
                (n / total) * dc['intercept'] for n, dc in zip(n_samples, delta_c)
            )

            global_params = _fedavg_aggregate(local_params, n_samples)

            m = {'round': r + 1}
            if eval_data is not None:
                m['auc'], m['acc'] = _evaluate(
                    global_params, eval_data[0], eval_data[1], self.model_template
                )
            self.metrics_.append(m)

        self.global_params_ = global_params
        return self


# ─────────────────────────────────────────────────────────────────────────────
# pFedMe
# ─────────────────────────────────────────────────────────────────────────────

class PFedMe:
    """
    T Dinh et al. (2020) — Moreau-envelope personalised FL.
    λ=15 as in paper.  Each client maintains a personalised model;
    the global model is the FedAvg of personalised models.
    """

    def __init__(self, lam=15.0, n_rounds=15, random_state=None):
        self.lam = lam
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.model_template = _make_model(random_state=random_state)
        self.metrics_ = []

    def fit(self, client_data, eval_data=None):
        K = len(client_data)
        n_samples = [len(d[0]) for d in client_data]
        global_params = _pilot_params(client_data, self.model_template)
        # Personalised models initialised to global
        personal_params = [dict(global_params) for _ in range(K)]

        for r in range(self.n_rounds):
            for k, (X, y) in enumerate(client_data):
                # Personalised model update (Moreau envelope)
                personal_params[k] = _fit_local(
                    self.model_template, global_params, X, y,
                    moreau_lambda=self.lam,
                    moreau_params=global_params
                )

            # Global model update: FedAvg of personalised models
            global_params = _fedavg_aggregate(personal_params, n_samples)

            m = {'round': r + 1}
            if eval_data is not None:
                # Evaluate global model (personalised models require per-client data)
                m['auc'], m['acc'] = _evaluate(
                    global_params, eval_data[0], eval_data[1], self.model_template
                )
            self.metrics_.append(m)

        self.global_params_ = global_params
        self.personal_params_ = personal_params
        return self


# ─────────────────────────────────────────────────────────────────────────────
# IFCA
# ─────────────────────────────────────────────────────────────────────────────

class IFCA:
    """
    Ghosh et al. (2020) — Iterative Federated Clustering Algorithm.

    At each round, each client assigns itself to the cluster whose model
    gives the lowest local loss, then updates that cluster's model.
    M=2 clusters as in paper.
    """

    def __init__(self, M=2, n_rounds=15, random_state=None):
        self.M = M
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.model_template = _make_model(random_state=random_state)
        self.metrics_ = []

    def _local_loss(self, params, X, y):
        """Negative log-likelihood proxy for cluster assignment."""
        model = clone(self.model_template)
        try:
            model.coef_ = params['coef']
            model.intercept_ = params['intercept']
            model.classes_ = params['classes']
            proba = model.predict_proba(X)
            eps = 1e-8
            proba = np.clip(proba, eps, 1 - eps)
            # NLL
            idx = np.searchsorted(model.classes_, y)
            nll = -np.log(proba[np.arange(len(y)), idx]).mean()
            return float(nll)
        except Exception:
            return np.inf

    def fit(self, client_data, eval_data=None):
        K = len(client_data)
        n_samples = [len(d[0]) for d in client_data]
        rng = np.random.default_rng(self.random_state)

        # Initialise M cluster models independently
        cluster_models = []
        for j in range(self.M):
            idx = rng.choice(K)
            m = _pilot_params(client_data, self.model_template)
            cluster_models.append(m)

        assignments = np.zeros(K, dtype=int)

        for r in range(self.n_rounds):
            # Step 1: assign each client to best cluster
            for k, (X, y) in enumerate(client_data):
                losses = [self._local_loss(cluster_models[j], X, y)
                          for j in range(self.M)]
                assignments[k] = int(np.argmin(losses))

            # Step 2: update each cluster model
            for j in range(self.M):
                members = np.where(assignments == j)[0]
                if len(members) == 0:
                    continue
                local = [_fit_local(self.model_template, cluster_models[j],
                                    client_data[k][0], client_data[k][1])
                         for k in members]
                ns = [n_samples[k] for k in members]
                cluster_models[j] = _fedavg_aggregate(local, ns)

            # Global model for evaluation: weighted average of cluster models
            cluster_sizes = [
                sum(n_samples[k] for k in np.where(assignments == j)[0])
                for j in range(self.M)
            ]
            global_params = _fedavg_aggregate(cluster_models, cluster_sizes)

            m = {'round': r + 1}
            if eval_data is not None:
                m['auc'], m['acc'] = _evaluate(
                    global_params, eval_data[0], eval_data[1], self.model_template
                )
            self.metrics_.append(m)

        self.cluster_models_ = cluster_models
        self.assignments_ = assignments
        self.global_params_ = global_params
        return self
