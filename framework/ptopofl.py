"""
ptopofl.py  Personalised Topology-Aware Federated Learning
============================================================
The improved TopoFederatedL algorithm combining three innovations:

  1. Topology-Guided Clustering:
       Clients are grouped by PH descriptor similarity before aggregation.
       Each cluster maintains its own sub-global model.

  2. Topology-Weighted Sample Training:
       Per-client sample weights derived from distance to the topological
       centroid (H0 persistence amplitude scale). Samples near topological
       noise get downweighted â†’ cleaner gradient signal.

  3. Two-Level Aggregation with Inter-Cluster Blending:
       Intra-cluster: topology-similarity Ã— sample-size Ã— trust weights
       Inter-cluster: controlled blending coefficient Î± blends cluster
       models toward the global consensus, preventing cluster divergence.

Together these fix the three root-cause failures of the original TopoFL:
  Bug 1: shape mismatch on warm-start (fixed by per-cluster model tracking)
  Bug 2: uniform topology weights (fixed by clustering + similarity weights)
  Bug 3: noisy per-sample features (fixed by persistence-amplitude weighting)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

from framework.tda import compute_topological_descriptor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _topo_sample_weights(X, desc, clip_lo=0.2, clip_hi=5.0):
    """
    Per-sample weights based on topological distance to cluster centroid.
    Scale set by H0 persistence amplitude (characteristic scale of H0).
    """
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    mean_pt = Xs.mean(0)
    dists = np.linalg.norm(Xs - mean_pt, axis=1)
    scale = max(desc['h0_amplitude'], 0.1)
    w = np.exp(-dists / scale)
    w = w / (w.mean() + 1e-8)
    return np.clip(w, clip_lo, clip_hi)


class pTopoFLClient:
    """Client for Personalised Topology-Aware FL."""

    def __init__(self, client_id, X_train, y_train, X_test=None, y_test=None,
                 n_tda_sample=80, random_state=42):
        self.client_id   = client_id
        self.X_train     = X_train
        self.y_train     = y_train.astype(float)
        self.X_test      = X_test
        self.y_test      = y_test
        self.n_tda_sample = n_tda_sample
        self.rng         = np.random.RandomState(random_state + client_id)
        self.scaler      = StandardScaler()
        self.scaler.fit(X_train)
        self.desc        = None
        self.coef_       = None
        self.intercept_  = None
        self.classes_    = np.array([0., 1.])
        self._round      = 0

    def compute_tda(self):
        """Compute (or return cached) topological descriptor."""
        if self.desc is None:
            self.desc = compute_topological_descriptor(
                self.X_train,
                n_sample=self.n_tda_sample,
                random_state=int(self.rng.randint(0, 10000))
            )
        return self.desc

    def get_descriptor(self):
        """Privacy-safe descriptor for server."""
        d = self.compute_tda()
        return {
            'client_id':    self.client_id,
            'feature_vector': d['feature_vector'].copy(),
            'h0_entropy':   d['h0_entropy'],
            'h1_entropy':   d['h1_entropy'],
            'h0_amplitude': d['h0_amplitude'],
            'n_samples':    len(self.X_train),
        }

    def train_local(self, cluster_params=None, seed_offset=0):
        """
        Local training with topology-guided sample weights.
        Warm-starts from cluster model (same feature space).
        """
        X  = self.scaler.transform(self.X_train)
        y  = self.y_train
        sw = _topo_sample_weights(self.X_train, self.compute_tda())

        lr = LogisticRegression(
            max_iter=300, C=1.0,
            random_state=int(self.rng.randint(0, 100000)),
            solver='lbfgs'
        )

        # Warm start from cluster model (no shape mismatch  same feature space)
        if cluster_params is not None:
            try:
                lr.coef_      = cluster_params['coef'].copy()
                lr.intercept_ = cluster_params['intercept'].copy()
                lr.classes_   = self.classes_.copy()
                lr.warm_start = True
            except Exception:
                pass

        lr.fit(X, y, sample_weight=sw)
        self.coef_      = lr.coef_
        self.intercept_ = lr.intercept_
        self._round    += 1
        return self

    def get_params(self):
        if self.coef_ is None:
            return None
        return {
            'coef':      self.coef_.copy(),
            'intercept': self.intercept_.copy(),
            'n_samples': len(self.X_train),
        }

    def evaluate(self):
        if self.coef_ is None or self.X_test is None or len(self.X_test) == 0:
            return {'accuracy': 0.0, 'auc': 0.5}
        Xt     = self.scaler.transform(self.X_test)
        logits = Xt @ self.coef_.T + self.intercept_
        probs  = _sigmoid(logits[:, 0])
        preds  = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(self.y_test, probs)
        except Exception:
            auc = 0.5
        return {'accuracy': accuracy_score(self.y_test, preds), 'auc': auc}


class pTopoFLServer:
    """
    Server for Personalised Topology-Aware FL.

    Maintains per-cluster sub-global models and blends them
    with a global consensus at each round.
    """

    def __init__(self, n_clusters=2, alpha_blend=0.3,
                 anomaly_threshold=2.0, random_state=42):
        self.n_clusters        = n_clusters
        self.alpha_blend       = alpha_blend
        self.anomaly_threshold = anomaly_threshold
        self.cluster_models    = {}   # cluster_id -> {'coef','intercept'}
        self.cluster_labels_   = None
        self.feat_norm_        = None
        self.round_logs        = []
        self._clustered        = False

    #  Clustering (called once after first round of descriptors) ################”€
    def fit_clusters(self, descriptors):
        feat = np.array([d['feature_vector'] for d in descriptors])
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        self.feat_norm_ = feat / norms
        D = euclidean_distances(self.feat_norm_)

        n = len(descriptors)
        k = min(self.n_clusters, n - 1)
        if k < 2:
            self.cluster_labels_ = np.zeros(n, dtype=int)
        else:
            clust = AgglomerativeClustering(
                n_clusters=k, metric='precomputed', linkage='average'
            )
            self.cluster_labels_ = clust.fit_predict(D)

        self._clustered = True
        return self.cluster_labels_

    # Trust scores (adversarial detection) #############################”€
    def _trust_weights(self, descriptors):
        n = len(descriptors)
        if n < 3:
            return np.ones(n)
        feat = np.array([d['feature_vector'] for d in descriptors])
        norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        fn = feat / norms
        D  = euclidean_distances(fn)
        mean_d = D.sum(1) / (n - 1)
        z = (mean_d - mean_d.mean()) / (mean_d.std() + 1e-8)
        trust = np.exp(-np.maximum(z - 1.0, 0))
        return np.clip(trust, 0.05, 1.0)

    #  Main aggregation round #############################
        def aggregate(self, clients, round_num=0):
        descriptors = [c.get_descriptor() for c in clients]

        # Cluster on first round
        if not self._clustered:
            self.fit_clusters(descriptors)

        labels  = self.cluster_labels_
        trust_w = self._trust_weights(descriptors)

        n_clusters_actual = len(set(labels))
        new_cluster_models = {}

        for c_id in range(n_clusters_actual):
            members = [i for i, l in enumerate(labels) if l == c_id]
            if not members:
                continue

            # Topology-similarity weight within cluster
            feat_members = self.feat_norm_[members]
            centroid = feat_members.mean(0)
            topo_sim = np.exp(
                -np.linalg.norm(feat_members - centroid, axis=1)
            )

            params = [clients[i].get_params() for i in members]
            valid  = [(j, p) for j, p in enumerate(params) if p is not None]
            if not valid:
                continue

            # Check shape consistency
            shapes = set(p['coef'].shape for _, p in valid)
            if len(shapes) > 1:
                # fallback: plain average
                coefs = [p['coef']      for _, p in valid]
                ints  = [p['intercept'] for _, p in valid]
                new_cluster_models[c_id] = {
                    'coef':      np.mean(coefs, axis=0),
                    'intercept': np.mean(ints,  axis=0),
                }
                continue

            ns = np.array([p['n_samples'] for _, p in valid], dtype=float)
            ts = np.array([trust_w[members[j]] for j, _ in valid])
            si = np.array([topo_sim[j] for j, _ in valid])

            w  = ns * si * ts
            w /= w.sum() + 1e-8

            agg_coef = sum(w[j] * p['coef']      for j, (_, p) in enumerate(valid))
            agg_int  = sum(w[j] * p['intercept'] for j, (_, p) in enumerate(valid))
            new_cluster_models[c_id] = {'coef': agg_coef, 'intercept': agg_int}

        # ”€ Inter-cluster blending ##########################################
        if new_cluster_models:
            all_coefs = [v['coef']      for v in new_cluster_models.values()]
            all_ints  = [v['intercept'] for v in new_cluster_models.values()]
            # Weighted global (by cluster size)
            cluster_sizes = np.array([
                sum(1 for l in labels if l == c_id)
                for c_id in new_cluster_models
            ], dtype=float)
            cw = cluster_sizes / (cluster_sizes.sum() + 1e-8)
            global_coef = sum(w * c for w, c in zip(cw, all_coefs))
            global_int  = sum(w * i for w, i in zip(cw, all_ints))

            Î± = self.alpha_blend
            for c_id in new_cluster_models:
                new_cluster_models[c_id]['coef'] = (
                    (1 - Î±) * new_cluster_models[c_id]['coef'] + Î± * global_coef
                )
                new_cluster_models[c_id]['intercept'] = (
                    (1 - Î±) * new_cluster_models[c_id]['intercept'] + Î± * global_int
                )

        self.cluster_models = new_cluster_models

        # Push cluster params to clients
        for i, c in enumerate(clients):
            c_id = labels[i]
            if c_id in self.cluster_models:
                c.coef_      = self.cluster_models[c_id]['coef'].copy()
                c.intercept_ = self.cluster_models[c_id]['intercept'].copy()

        log = {
            'round':           round_num,
            'cluster_labels':  labels.tolist(),
            'trust_weights':   trust_w.tolist(),
            'n_clusters':      n_clusters_actual,
            'flagged':         [i for i, t in enumerate(trust_w) if t < 0.5],
        }
        self.round_logs.append(log)
        return self.cluster_models, log


def run_ptopofl_rounds(clients, server, n_rounds=15, verbose=False):
    """Standard runner compatible with experiment harness."""
    round_accs, round_aucs = [], []
    for r in range(n_rounds):
        # Local training with cluster params
        for i, c in enumerate(clients):
            lbl  = server.cluster_labels_[i] if server._clustered else None
            cpar = server.cluster_models.get(lbl) if lbl is not None else None
            c.train_local(cluster_params=cpar)

        server.aggregate(clients, round_num=r)

        accs = [c.evaluate()['accuracy'] for c in clients
                if c.X_test is not None and len(c.X_test) > 0]
        aucs = [c.evaluate()['auc'] for c in clients
                if c.X_test is not None and len(c.X_test) > 0]
        round_accs.append(np.mean(accs) if accs else 0.0)
        round_aucs.append(np.mean(aucs) if aucs else 0.5)

        if verbose:
            log = server.round_logs[-1]
            print(f"  Round {r+1:2d} | Acc={round_accs[-1]:.3f} | "
                  f"AUC={round_aucs[-1]:.3f} | Flagged={log['flagged']}")

    return round_accs, round_aucs
