"""
experiments/data.py
===================
Synthetic data generation for Scenario A (Healthcare) and
Scenario B (Benchmark pathological non-IID).

Matches §4.1 exactly:
  Scenario A: K=8 clients, 10 informative features out of 20,
              mortality rate 10–45 %, 60–250 samples per client,
              2 adversarial clients (distributional poisoning).
  Scenario B: K=10 clients, 12 informative features out of 20,
              class imbalance Uniform(0.1, 0.9) per client.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def _make_client_data(n_samples, n_features, n_informative,
                       class_weight, rng, noise_std=0.3):
    """
    Generate a single client's labelled dataset with a given class imbalance.

    Parameters
    ----------
    n_samples : int
    n_features : int
    n_informative : int
    class_weight : float
        Fraction of positive samples (mortality rate or imbalance ratio).
    rng : np.random.Generator
    noise_std : float
        Standard deviation of feature noise.
    """
    n_pos = max(2, int(n_samples * class_weight))
    n_neg = n_samples - n_pos

    # Positive class (high-risk / minority)
    X_pos = rng.normal(loc=1.0, scale=noise_std + 0.2, size=(n_pos, n_informative))
    # Negative class (low-risk / majority)
    X_neg = rng.normal(loc=0.0, scale=noise_std, size=(n_neg, n_informative))

    # Noise features
    n_noise = n_features - n_informative
    if n_noise > 0:
        noise_pos = rng.normal(size=(n_pos, n_noise))
        noise_neg = rng.normal(size=(n_neg, n_noise))
        X_pos = np.hstack([X_pos, noise_pos])
        X_neg = np.hstack([X_neg, noise_neg])

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _apply_distributional_poisoning(X, y, rng, feature_shift=2.0,
                                     label_flip_rate=0.4):
    """
    Distributional poisoning: shift features AND flip labels (§3.4).
    This produces detectable topological anomalies.
    """
    X_p = X.copy()
    y_p = y.copy()

    # Feature shift on informative dimensions
    n_inf = min(10, X.shape[1])
    X_p[:, :n_inf] += rng.normal(loc=feature_shift, scale=0.5,
                                   size=(len(X), n_inf))

    # Label flip
    flip_idx = rng.choice(len(y), int(label_flip_rate * len(y)), replace=False)
    y_p[flip_idx] = 1 - y_p[flip_idx]

    return X_p, y_p


def make_healthcare(K=8, n_adv=2, n_features=20, n_informative=10,
                    random_state=None):
    """
    Scenario A — Healthcare (non-IID, adversarial).

    Returns
    -------
    client_data : list of K (X_train, y_train) tuples
    eval_data : (X_test, y_test)
    adv_indices : list of adversarial client indices
    """
    rng = np.random.default_rng(random_state)

    mortality_rates = np.linspace(0.10, 0.45, K)
    rng.shuffle(mortality_rates)

    # Variable client sizes: 60–250 patients
    sizes = rng.integers(60, 251, size=K)

    client_data = []
    for k in range(K):
        X, y = _make_client_data(
            int(sizes[k]), n_features, n_informative,
            mortality_rates[k], rng
        )
        client_data.append((X, y))

    # Select adversarial clients (last n_adv by default, shuffled)
    adv_indices = list(rng.choice(K, n_adv, replace=False))
    for k in adv_indices:
        X, y = client_data[k]
        X_p, y_p = _apply_distributional_poisoning(X, y, rng)
        client_data[k] = (X_p, y_p)

    # Held-out evaluation set drawn from a balanced distribution
    X_test, y_test = _make_client_data(
        400, n_features, n_informative, 0.30, rng
    )
    scaler = StandardScaler()
    X_all = np.vstack([d[0] for d in client_data])
    scaler.fit(X_all)

    client_data = [(scaler.transform(X), y) for X, y in client_data]
    X_test = scaler.transform(X_test)

    return client_data, (X_test, y_test), adv_indices


def make_benchmark(K=10, n_features=20, n_informative=12,
                   n_samples_per_client=150, random_state=None):
    """
    Scenario B — Benchmark (pathological non-IID).

    Class imbalance per client drawn Uniform(0.1, 0.9).

    Returns
    -------
    client_data : list of K (X_train, y_train) tuples
    eval_data : (X_test, y_test)
    """
    rng = np.random.default_rng(random_state)

    imbalances = rng.uniform(0.1, 0.9, size=K)

    client_data = []
    for k in range(K):
        X, y = _make_client_data(
            n_samples_per_client, n_features, n_informative,
            imbalances[k], rng
        )
        client_data.append((X, y))

    X_test, y_test = _make_client_data(
        400, n_features, n_informative, 0.5, rng
    )
    scaler = StandardScaler()
    X_all = np.vstack([d[0] for d in client_data])
    scaler.fit(X_all)

    client_data = [(scaler.transform(X), y) for X, y in client_data]
    X_test = scaler.transform(X_test)

    return client_data, (X_test, y_test)


def make_continual(K=6, n_rounds=20, n_features=15, n_informative=10,
                   drift_clients=None, random_state=None):
    """
    Continual FL scenario for §4.4: data distributions evolve across rounds.

    Returns
    -------
    round_data : list of K (X, y) per round (n_rounds × K)
    drift_clients : list of client indices that experience concept drift
    """
    rng = np.random.default_rng(random_state)
    if drift_clients is None:
        drift_clients = [K - 1, K - 2]

    round_data = []
    for r in range(n_rounds):
        client_data = []
        for k in range(K):
            imbalance = 0.3 + 0.1 * (k % 3)
            # Drift clients experience progressive distribution shift
            if k in drift_clients:
                shift = 0.05 * r  # gradual drift
                imbalance = min(0.9, imbalance + shift)
            X, y = _make_client_data(
                100, n_features, n_informative, imbalance, rng
            )
            client_data.append((X, y))
        round_data.append(client_data)

    return round_data, drift_clients
