"""
ptopofl/ptopopfl.py
====================
Full pTopoFL algorithm (Algorithm 1, paper §3.3–3.6).

Protocol (§3.6): each client transmits both the local model
parameters θ_k and the 48-dim descriptor φ_k.  The server uses φ_k
exclusively for topology-guided operations; θ_k is used only for
model aggregation (identical in structure to FedAvg).
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .descriptor import PHDescriptor          # ← fixed: relative import


# ─────────────────────────────────────────────────────────────────────────────
# Local training step (§3.2)
# ─────────────────────────────────────────────────────────────────────────────

def _local_train(model_params, X_train, y_train, phi, model_template,
                 augment_features=True):
    """
    One round of local training on client data.

    Parameters
    ----------
    model_params : dict or None
        Current global model parameters (warm-start).
    X_train, y_train : arrays
        Local training data.
    phi : ndarray (48,)
        Topological descriptor for feature augmentation (§3.2).
    model_template : sklearn estimator
    augment_features : bool
        If True, append 4 topology-derived features to X_train.

    Returns
    -------
    local_params : dict  {coef, intercept, classes, augmented}
    """
    model = clone(model_template)

    if augment_features and len(X_train) > 0:
        # Four topology-derived features per sample (§3.2):
        # topo-centroid distance, H0 entropy, H1 entropy, median Betti number
        # Descriptor layout: [bc_H0(20), bc_H1(20), β0,β1, H0,H1, A0,A1, n0,n1]
        topo_centroid_dist = float(np.linalg.norm(phi))
        h0_entropy  = float(phi[42])          # index 42 = H0 entropy
        h1_entropy  = float(phi[43])          # index 43 = H1 entropy
        median_betti = float(np.median(phi[:20]))  # median of H0 Betti curve
        topo_feats = np.full(
            (len(X_train), 4),
            [topo_centroid_dist, h0_entropy, h1_entropy, median_betti]
        )
        X_aug = np.hstack([X_train, topo_feats])
    else:
        X_aug = X_train

    # Warm-start from global model if shapes match
    if model_params is not None:
        try:
            model.coef_       = model_params['coef'].copy()
            model.intercept_  = model_params['intercept'].copy()
            model.classes_    = model_params['classes'].copy()
            model.warm_start  = True
        except (KeyError, ValueError, AttributeError):
            pass

    try:
        model.fit(X_aug, y_train)
    except Exception:
        model = clone(model_template)
        model.fit(X_aug, y_train)

    return {
        'coef':      model.coef_.copy(),
        'intercept': model.intercept_.copy(),
        'classes':   model.classes_.copy(),
        'augmented': augment_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection (§3.4)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trust_weights(descriptors, tau=2.0):
    """
    Trust weights  t_k = exp(-max(z_k - 1, 0))   (Eq. 8 + §3.4).

    Returns
    -------
    trust    : ndarray (K,) — values in (0, 1]
    z_scores : ndarray (K,)
    """
    K = len(descriptors)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    phi_hat = descriptors / norms

    delta = np.array([
        np.linalg.norm(phi_hat - phi_hat[k], axis=1).sum() / (K - 1)
        if K > 1 else 0.0
        for k in range(K)
    ])

    mu_d    = delta.mean()
    sigma_d = delta.std() + 1e-8
    z_scores = (delta - mu_d) / sigma_d
    trust    = np.exp(-np.maximum(z_scores - 1.0, 0.0))
    return trust, z_scores


# ─────────────────────────────────────────────────────────────────────────────
# Clustering (§3.3, Step 1)
# ─────────────────────────────────────────────────────────────────────────────

def _topology_clustering(descriptors, M):
    """
    Hierarchical average-linkage clustering on L2-normalised descriptors.
    Returns cluster labels (K,).
    """
    K = len(descriptors)
    if K <= M:
        return np.arange(K)

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    phi_hat = descriptors / norms

    labels = AgglomerativeClustering(
        n_clusters=M, linkage='average', metric='euclidean'
    ).fit_predict(phi_hat)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (§3.3, Steps 2–3)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(local_params, descriptors, cluster_labels, n_samples,
               trust_weights, alpha_blend=0.3):
    """
    Two-level topology-weighted aggregation (Eqs. 6–7).

    Returns
    -------
    cluster_models : dict  cluster_id → params
    global_model   : dict  data-volume-weighted average of cluster models
    personalised   : dict  cluster_id → blended params
    """
    M = len(np.unique(cluster_labels))
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    phi_hat = descriptors / norms

    cluster_models = {}
    cluster_sizes  = {}

    for j in range(M):
        members = np.where(cluster_labels == j)[0]
        if len(members) == 0:
            continue

        centroid = phi_hat[members].mean(axis=0)

        # w_k ∝ n_k · exp(-||φ̂_k - φ̂_{Cj}||) · t_k  (Eq. 6)
        raw_w = np.array([
            n_samples[k]
            * np.exp(-np.linalg.norm(phi_hat[k] - centroid))
            * trust_weights[k]
            for k in members
        ])
        raw_w = np.maximum(raw_w, 1e-8)
        w = raw_w / raw_w.sum()

        coef      = sum(w[i] * local_params[k]['coef']      for i, k in enumerate(members))
        intercept = sum(w[i] * local_params[k]['intercept'] for i, k in enumerate(members))
        classes   = local_params[members[0]]['classes']

        cluster_models[j] = {'coef': coef, 'intercept': intercept, 'classes': classes}
        cluster_sizes[j]  = sum(n_samples[k] for k in members)

    # Global consensus (Eq. 7) — data-volume-weighted average of cluster models
    total = sum(cluster_sizes.values()) + 1e-8
    global_coef = sum(
        (cluster_sizes[j] / total) * cluster_models[j]['coef']
        for j in cluster_models
    )
    global_intercept = sum(
        (cluster_sizes[j] / total) * cluster_models[j]['intercept']
        for j in cluster_models
    )
    ref_classes  = next(iter(cluster_models.values()))['classes']
    global_model = {'coef': global_coef, 'intercept': global_intercept,
                    'classes': ref_classes}

    # Personalised models: blend cluster model with global consensus  (Eq. 7)
    personalised = {}
    for j, cm in cluster_models.items():
        personalised[j] = {
            'coef':      (1 - alpha_blend) * cm['coef']      + alpha_blend * global_coef,
            'intercept': (1 - alpha_blend) * cm['intercept'] + alpha_blend * global_intercept,
            'classes':   cm['classes'],
        }

    return cluster_models, global_model, personalised


# ─────────────────────────────────────────────────────────────────────────────
# Topological drift tracking (§3.5)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_drift(descriptors_history):
    """
    Topological drift Δ_k (Eq. 9).

    Parameters
    ----------
    descriptors_history : list of ndarray (K, 48), one per round

    Returns
    -------
    drift : ndarray (K,)
    """
    if len(descriptors_history) < 2:
        return np.zeros(len(descriptors_history[0]))
    phi_0  = descriptors_history[0]
    deltas = np.array([
        np.linalg.norm(phi - phi_0, axis=1)
        for phi in descriptors_history[1:]
    ])
    return deltas.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# pTopoFL trainer
# ─────────────────────────────────────────────────────────────────────────────

class PTopoFL:
    """
    pTopoFL federated learning algorithm (Algorithm 1).

    Parameters
    ----------
    n_clusters : int
        Number of topology-guided clusters M (paper: 2).
    alpha_blend : float
        Inter-cluster blending coefficient α (paper: 0.3).
    tau : float
        Anomaly detection z-score threshold τ (paper: 2.0).
    n_rounds : int
        Communication rounds (paper: 15).
    n_sub : int
        Points subsampled per client for TDA (paper: 80).
    augment_features : bool
        Augment local features with TDA statistics (§3.2).
    random_state : int or None
        Master seed — controls descriptor subsampling and pilot fitting.
    """

    def __init__(self, n_clusters=2, alpha_blend=0.3, tau=2.0,
                 n_rounds=15, n_sub=80, augment_features=True,
                 random_state=None):
        self.n_clusters       = n_clusters
        self.alpha_blend      = alpha_blend
        self.tau              = tau
        self.n_rounds         = n_rounds
        self.n_sub            = n_sub
        self.augment_features = augment_features
        self.random_state     = random_state

        self.descriptor_fn = PHDescriptor(
            n_sub=n_sub, n_thresholds=20, random_state=random_state
        )
        self.model_template = LogisticRegression(
            C=1.0, max_iter=300, solver='lbfgs',
            random_state=random_state, warm_start=True
        )

        self._descriptor_history = []
        self._cluster_labels     = None
        self.metrics_            = []   # list of per-round dicts

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, client_data, eval_data=None):
        """
        Run pTopoFL for self.n_rounds communication rounds.

        Parameters
        ----------
        client_data : list of (X_train, y_train) — one tuple per client
        eval_data   : (X_test, y_test) or None

        Returns
        -------
        self
        """
        K         = len(client_data)
        n_samples = np.array([len(d[0]) for d in client_data])

        # Pilot global model on a small union of client data
        X_pilot = np.vstack([d[0][:5] for d in client_data])
        y_pilot = np.concatenate([d[1][:5] for d in client_data])
        pilot   = clone(self.model_template)
        pilot.fit(X_pilot, y_pilot)
        global_params = {
            'coef':      pilot.coef_.copy(),
            'intercept': pilot.intercept_.copy(),
            'classes':   pilot.classes_.copy(),
        }

        for r in range(self.n_rounds):

            # ── Step 1: client-side — compute descriptor + local training ────
            local_params = {}
            descriptors  = np.zeros((K, PHDescriptor.DIM))

            for k, (X_k, y_k) in enumerate(client_data):
                phi_k          = self.descriptor_fn.compute(X_k)
                descriptors[k] = phi_k
                local_params[k] = _local_train(
                    global_params, X_k, y_k, phi_k,
                    self.model_template,
                    augment_features=self.augment_features
                )

            self._descriptor_history.append(descriptors.copy())

            # ── Step 2: round-0 clustering; drift-triggered re-clustering ────
            if r == 0:
                self._cluster_labels = _topology_clustering(
                    descriptors, self.n_clusters
                )
            else:
                drift = _compute_drift(self._descriptor_history)
                drifted = np.where(drift > np.percentile(drift, 75))[0]
                if len(drifted) > 0:
                    new_labels = _topology_clustering(descriptors, self.n_clusters)
                    for k in drifted:
                        self._cluster_labels[k] = new_labels[k]

            # ── Step 3: anomaly detection ─────────────────────────────────────
            trust_weights, z_scores = _compute_trust_weights(descriptors, self.tau)

            # ── Step 4: topology-weighted aggregation ─────────────────────────
            _, global_model, _ = _aggregate(
                local_params, descriptors, self._cluster_labels,
                n_samples, trust_weights, self.alpha_blend
            )
            global_params = global_model

            # ── Step 5: evaluation ────────────────────────────────────────────
            metrics = {'round': r + 1}
            if eval_data is not None:
                X_test, y_test = eval_data
                if self.augment_features:
                    mean_phi  = descriptors.mean(axis=0)
                    topo_feats = np.full(
                        (len(X_test), 4),
                        [float(np.linalg.norm(mean_phi)),
                         float(mean_phi[42]),
                         float(mean_phi[43]),
                         float(np.median(mean_phi[:20]))]
                    )
                    X_eval = np.hstack([X_test, topo_feats])
                else:
                    X_eval = X_test

                eval_model = self._params_to_model(global_params)
                if eval_model is not None:
                    try:
                        proba = eval_model.predict_proba(X_eval)[:, 1]
                        metrics['auc'] = float(roc_auc_score(y_test, proba))
                        metrics['acc'] = float(
                            (eval_model.predict(X_eval) == y_test).mean()
                        )
                    except Exception:
                        metrics['auc'] = 0.5
                        metrics['acc'] = 0.5

            self.metrics_.append(metrics)

        self.global_params_   = global_params
        self.descriptors_     = descriptors
        self.trust_weights_   = trust_weights
        self.cluster_labels_  = self._cluster_labels
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _params_to_model(self, params):
        model = clone(self.model_template)
        try:
            model.coef_      = params['coef']
            model.intercept_ = params['intercept']
            model.classes_   = params['classes']
        except Exception:
            return None
        return model

    def predict_proba(self, X):
        if self.augment_features:
            mean_phi  = self.descriptors_.mean(axis=0)
            topo_feats = np.full(
                (len(X), 4),
                [float(np.linalg.norm(mean_phi)),
                 float(mean_phi[42]),
                 float(mean_phi[43]),
                 float(np.median(mean_phi[:20]))]
            )
            X_eval = np.hstack([X, topo_feats])
        else:
            X_eval = X
        return self._params_to_model(self.global_params_).predict_proba(X_eval)
