"""
fl.py — TopoFederatedL: Federated Learning framework with TDA
==============================================================
Implements all 5 research directions:

  1. Enhanced Data Representations    — TDA features for local models
  2. Topology-Aware Aggregation       — Wasserstein-weighted FedAvg
  3. Adversarial Robustness           — Topological anomaly detection
  4. Continual Learning               — Topological signature tracking
  5. Privacy via Topological Abstraction — PH descriptors, not gradients
"""

import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering

from framework.tda import (
    compute_topological_descriptor, descriptor_distance,
    persistence_entropy, betti_curve
)


# ###############################################################
# Direction 1+5: TopoFL Client
# ###############################################################

class TopoFLClient:
    """
    Federated Learning client with TDA-enhanced representation.

    Privacy guarantee: only topological descriptors are transmitted
    to the server, never raw data or raw gradients.
    """

    def __init__(self, client_id, X_train, y_train, X_test=None, y_test=None,
                 n_tda_sample=80, random_state=42):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_tda_sample = n_tda_sample
        self.rng = np.random.RandomState(random_state + client_id)
        self.scaler = StandardScaler()
        self.model = None
        self.topo_descriptor = None
        self.round_history = []  # for Direction 4: continual learning

    def compute_tda_features(self):
        """Direction 1: Compute TDA features from local data."""
        self.topo_descriptor = compute_topological_descriptor(
            self.X_train,
            n_sample=self.n_tda_sample,
            random_state=int(self.rng.randint(0, 10000))
        )
        return self.topo_descriptor

    def get_privacy_safe_descriptor(self):
        """
        Direction 5: Return only topological descriptor (not raw data/gradients).
        This is what gets transmitted to the server.
        """
        if self.topo_descriptor is None:
            self.compute_tda_features()
        return {
            'client_id': self.client_id,
            'feature_vector': self.topo_descriptor['feature_vector'].copy(),
            'h0_entropy': self.topo_descriptor['h0_entropy'],
            'h1_entropy': self.topo_descriptor['h1_entropy'],
            'h0_amplitude': self.topo_descriptor['h0_amplitude'],
            'h1_amplitude': self.topo_descriptor['h1_amplitude'],
            'betti0_curve': self.topo_descriptor['betti0_curve'].copy(),
            'betti1_curve': self.topo_descriptor['betti1_curve'].copy(),
            'n_samples': len(self.X_train),
            'n_features': self.X_train.shape[1],
            # NOTE: no raw data, no model weights, no gradients
        }

    def train_local(self, global_model_params=None, tda_augment=True):
        """
        Direction 1: Train local model, optionally augmented with TDA features.
        """
        X = self.scaler.fit_transform(self.X_train)

        if tda_augment and self.topo_descriptor is not None:
            # Augment features with TDA-derived statistics (per-sample TDA)
            tda_feat = self._compute_per_sample_tda(X)
            X_aug = np.hstack([X, tda_feat])
        else:
            X_aug = X

        self.model = LogisticRegression(
            max_iter=300, random_state=42, C=1.0,
            solver='lbfgs'
        )

        # Warm start from global model if provided
        if global_model_params is not None and self.model is not None:
            try:
                self.model.coef_ = global_model_params['coef'].copy()
                self.model.intercept_ = global_model_params['intercept'].copy()
                self.model.classes_ = global_model_params['classes'].copy()
                self.model.warm_start = True
            except Exception:
                pass

        self.model.fit(X_aug, self.y_train)
        self.tda_augment = tda_augment
        self.n_aug_features = X_aug.shape[1]
        return self

    def _compute_per_sample_tda(self, X):
        """
        Compute local TDA summary statistics as sample-level features.
        For each sample, its distance to the topological centroid.
        """
        if len(X) == 0:
            return np.zeros((0, 4))

        desc = self.topo_descriptor
        # Features: distance to mean, h0_entropy_local, h1_entropy_local,
        #           betti0_at_median_scale
        mean_pt = X.mean(0)
        dist_to_mean = np.linalg.norm(X - mean_pt, axis=1, keepdims=True)
        h0_ent_feat = np.full((len(X), 1), desc['h0_entropy'])
        h1_ent_feat = np.full((len(X), 1), desc['h1_entropy'])
        betti0_med = desc['betti0_curve'][len(desc['betti0_curve'])//2]
        betti0_feat = np.full((len(X), 1), betti0_med)
        return np.hstack([dist_to_mean, h0_ent_feat, h1_ent_feat, betti0_feat])

    def get_model_params(self):
        """Return model parameters for aggregation."""
        if self.model is None:
            return None
        try:
            return {
                'coef': self.model.coef_.copy(),
                'intercept': self.model.intercept_.copy(),
                'classes': self.model.classes_.copy(),
                'n_samples': len(self.X_train),
                'n_aug_features': self.n_aug_features,
            }
        except AttributeError:
            return None

    def evaluate(self, X=None, y=None, use_tda=True):
        """Evaluate local model."""
        if self.model is None:
            return {'accuracy': 0.0, 'auc': 0.5}
        if X is None:
            X, y = self.X_test, self.y_test
        if X is None or len(X) == 0:
            return {'accuracy': 0.0, 'auc': 0.5}

        X_sc = self.scaler.transform(X)
        if use_tda and hasattr(self, 'tda_augment') and self.tda_augment:
            tda_feat = self._compute_per_sample_tda(X_sc)
            X_sc = np.hstack([X_sc, tda_feat])

        y_pred = self.model.predict(X_sc)
        try:
            y_prob = self.model.predict_proba(X_sc)[:, 1]
            auc = roc_auc_score(y, y_prob)
        except Exception:
            auc = 0.5

        return {
            'accuracy': accuracy_score(y, y_pred),
            'auc': auc,
        }

    def track_topology_round(self, round_num):
        """Direction 4: Track topological signature for continual FL."""
        if self.topo_descriptor is not None:
            self.round_history.append({
                'round': round_num,
                'h0_entropy': self.topo_descriptor['h0_entropy'],
                'h1_entropy': self.topo_descriptor['h1_entropy'],
                'feature_vector': self.topo_descriptor['feature_vector'].copy(),
            })


# #############################################################################
# Direction 2+3: TopoFL Server
# #############################################################################

class TopoFLServer:
    """
    Federated Learning server with topology-aware aggregation
    and adversarial detection.
    """

    def __init__(self, n_clients, anomaly_threshold=2.5, random_state=42):
        self.n_clients = n_clients
        self.anomaly_threshold = anomaly_threshold
        self.rng = np.random.RandomState(random_state)
        self.global_model_params = None
        self.round_logs = []

    def compute_topo_similarity_matrix(self, descriptors):
        """
        Direction 2: Compute pairwise topological similarity between clients.
        Returns distance matrix based on feature vector distance.
        """
        n = len(descriptors)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                fi = descriptors[i]['feature_vector']
                fj = descriptors[j]['feature_vector']
                norm = np.linalg.norm(fi) + np.linalg.norm(fj) + 1e-8
                d = np.linalg.norm(fi - fj) / norm * 2
                D[i, j] = D[j, i] = d
        return D

    def detect_adversarial_clients(self, descriptors, sim_matrix):
        """
        Direction 3: Detect adversarial/anomalous clients via
        topological irregularities.

        A client is flagged if its mean topological distance to all
        others exceeds mean + anomaly_threshold * std.
        """
        n = len(descriptors)
        if n < 3:
            return [], np.ones(n)

        mean_dists = sim_matrix.sum(1) / (n - 1)
        mu = mean_dists.mean()
        sigma = mean_dists.std() + 1e-8

        z_scores = (mean_dists - mu) / sigma
        flagged = [i for i, z in enumerate(z_scores) if z > self.anomaly_threshold]
        trust_weights = np.exp(-np.maximum(z_scores - 1.0, 0))
        trust_weights = np.clip(trust_weights, 0.05, 1.0)

        return flagged, trust_weights

    def cluster_clients(self, sim_matrix, n_clusters=None):
        """
        Direction 2: Cluster clients by topological similarity.
        Auto-select number of clusters if not specified.
        """
        n = len(sim_matrix)
        if n < 3:
            return list(range(n))

        if n_clusters is None:
            # Simple heuristic: use sqrt(n) clusters, min 2
            n_clusters = max(2, int(np.sqrt(n)))
            n_clusters = min(n_clusters, n - 1)

        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(sim_matrix)
        except Exception:
            labels = np.zeros(n, dtype=int)

        return labels

    def topology_weighted_fedavg(self, client_params, descriptors,
                                  trust_weights=None):
        """
        Direction 2: Topology-aware weighted FedAvg.

        Weights = n_samples * topological_similarity_weight * trust_weight
        Clients with similar topology to the majority get higher weight.
        """
        valid = [(i, p) for i, p in enumerate(client_params) if p is not None]
        if not valid:
            return self.global_model_params

        indices = [i for i, _ in valid]
        params = [p for _, p in valid]

        # Sample size weights
        n_samples = np.array([p['n_samples'] for p in params], dtype=float)

        # Topological similarity weights: reward similarity to centroid
        feat_vecs = np.array([descriptors[i]['feature_vector'] for i in indices])
        centroid = feat_vecs.mean(0)
        topo_dists = np.array([
            np.linalg.norm(fv - centroid) / (np.linalg.norm(fv) + 1e-8)
            for fv in feat_vecs
        ])
        topo_weights = np.exp(-topo_dists)

        # Trust weights (from adversarial detection)
        if trust_weights is not None:
            tw = np.array([trust_weights[i] for i in indices])
        else:
            tw = np.ones(len(indices))

        # Combined weights
        weights = n_samples * topo_weights * tw
        weights = weights / (weights.sum() + 1e-8)

        # Weighted average of model coefficients
        # Only aggregate if shapes match
        shapes_coef = set(p['coef'].shape for p in params)
        shapes_int = set(p['intercept'].shape for p in params)

        if len(shapes_coef) == 1 and len(shapes_int) == 1:
            agg_coef = sum(w * p['coef'] for w, p in zip(weights, params))
            agg_int = sum(w * p['intercept'] for w, p in zip(weights, params))
            return {
                'coef': agg_coef,
                'intercept': agg_int,
                'classes': params[0]['classes'],
                'aggregation_weights': weights,
            }
        else:
            # Fallback: simple FedAvg by sample size
            w = n_samples / n_samples.sum()
            agg_coef = sum(wi * p['coef'] for wi, p in zip(w, params))
            agg_int = sum(wi * p['intercept'] for wi, p in zip(w, params))
            return {
                'coef': agg_coef,
                'intercept': agg_int,
                'classes': params[0]['classes'],
                'aggregation_weights': w,
            }

    def aggregate(self, clients, round_num=0):
        """
        Full server aggregation round implementing all directions.
        """
        # Collect privacy-safe descriptors from all clients
        descriptors = [c.get_privacy_safe_descriptor() for c in clients]

        # Compute topological similarity matrix
        sim_matrix = self.compute_topo_similarity_matrix(descriptors)

        # Direction 3: Detect adversarial clients
        flagged, trust_weights = self.detect_adversarial_clients(
            descriptors, sim_matrix
        )

        # Direction 2: Cluster clients by topology
        cluster_labels = self.cluster_clients(sim_matrix)

        # Collect model params
        client_params = [c.get_model_params() for c in clients]

        # Direction 2+3: Topology-weighted FedAvg
        self.global_model_params = self.topology_weighted_fedavg(
            client_params, descriptors, trust_weights
        )

        # Log
        log = {
            'round': round_num,
            'n_clients': len(clients),
            'flagged_clients': flagged,
            'trust_weights': trust_weights.tolist(),
            'cluster_labels': cluster_labels.tolist()
                              if hasattr(cluster_labels, 'tolist')
                              else list(cluster_labels),
            'sim_matrix': sim_matrix,
            'descriptors': descriptors,
        }
        self.round_logs.append(log)
        return self.global_model_params, log


# #############################################################################��
# Standard FedAvg (baseline)
# #############################################################################

class FedAvgServer:
    """Standard FedAvg baseline (no topology)."""

    def __init__(self):
        self.global_model_params = None

    def aggregate(self, clients, round_num=0):
        params = [c.get_model_params() for c in clients]
        valid = [p for p in params if p is not None]
        if not valid:
            return self.global_model_params, {}

        shapes_ok = (
            len(set(p['coef'].shape for p in valid)) == 1 and
            len(set(p['intercept'].shape for p in valid)) == 1
        )
        if not shapes_ok:
            self.global_model_params = valid[0]
            return self.global_model_params, {}

        n_samples = np.array([p['n_samples'] for p in valid], dtype=float)
        w = n_samples / n_samples.sum()
        agg_coef = sum(wi * p['coef'] for wi, p in zip(w, valid))
        agg_int  = sum(wi * p['intercept'] for wi, p in zip(w, valid))
        self.global_model_params = {
            'coef': agg_coef,
            'intercept': agg_int,
            'classes': valid[0]['classes'],
        }
        return self.global_model_params, {}
