"""
data.py â€” Synthetic data generation for TopoFederatedL experiments
===================================================================
Two scenarios:

  Scenario A: Healthcare / Clinical (non-IID)
    - 8 hospital clients with heterogeneous patient populations
    - Binary classification: Y1 mortality risk post-lung-transplant
    - Non-IID: each hospital has different age/comorbidity distribution
    - Includes 2 adversarial clients with poisoned data

  Scenario B: Benchmark (MNIST-inspired, pathological non-IID)
    - 10 clients, each with only 2 of 5 classes
    - Classic FL non-IID stress test
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# #############################################################################”€
# Scenario A: Healthcare (non-IID clinical data)
# #############################################################################”€

def make_healthcare_federated(
    n_clients=8,
    n_samples_per_client=None,
    n_features=20,
    n_informative=10,
    adversarial_clients=None,
    adversarial_fraction=0.3,
    global_test_size=500,
    random_state=42
):
    """
    Generate non-IID federated healthcare dataset.

    Each client (hospital) has a distinct patient distribution:
      - Different class imbalance (mortality rates 0.1â€“0.45)
      - Different feature covariance structure (comorbidity profiles)
      - Different feature means (age/severity shifted)

    adversarial_clients: list of client indices that send poisoned updates
    """
    rng = np.random.RandomState(random_state)

    if n_samples_per_client is None:
        # Realistic hospital size variation: 60â€“250 patients
        n_samples_per_client = rng.randint(60, 250, size=n_clients).tolist()

    if adversarial_clients is None:
        adversarial_clients = []

    client_data = []

    # Shared global feature structure
    X_global, y_global = make_classification(
        n_samples=10000,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=random_state
    )

    # Global test set (IID)
    idx_test = rng.choice(len(X_global), global_test_size, replace=False)
    X_test_global = X_global[idx_test]
    y_test_global = y_global[idx_test]

    # Per-client non-IID data
    mortality_rates = np.linspace(0.10, 0.45, n_clients)
    rng.shuffle(mortality_rates)

    for c in range(n_clients):
        n = n_samples_per_client[c]
        rate = mortality_rates[c]

        # Non-IID: different distribution shift per client
        shift = rng.randn(n_features) * 0.5  # feature mean shift
        scale = rng.uniform(0.7, 1.4, n_features)  # feature scale

        # Sample from class-conditioned distributions
        n_pos = int(n * rate)
        n_neg = n - n_pos

        # Positive class (mortality = 1)
        pos_idx = np.where(y_global == 1)[0]
        neg_idx = np.where(y_global == 0)[0]

        pos_sample = rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=True)
        neg_sample = rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=True)

        X_pos = X_global[pos_sample] * scale + shift
        X_neg = X_global[neg_sample] * scale + shift * 0.3

        X_c = np.vstack([X_pos, X_neg])
        y_c = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        # Shuffle
        perm = rng.permutation(len(X_c))
        X_c, y_c = X_c[perm], y_c[perm]

        # Adversarial poisoning: flip labels for fraction of samples
        if c in adversarial_clients:
            n_flip = int(adversarial_fraction * len(y_c))
            flip_idx = rng.choice(len(y_c), n_flip, replace=False)
            y_c[flip_idx] = 1 - y_c[flip_idx]

        # Local test set
        n_local_test = max(30, int(n * 0.2))
        test_idx = rng.choice(len(X_c), n_local_test, replace=False)
        train_idx = np.setdiff1d(np.arange(len(X_c)), test_idx)

        client_data.append({
            'client_id': c,
            'X_train': X_c[train_idx],
            'y_train': y_c[train_idx],
            'X_test': X_c[test_idx],
            'y_test': y_c[test_idx],
            'is_adversarial': c in adversarial_clients,
            'mortality_rate': rate,
            'n_train': len(train_idx),
        })

    return client_data, X_test_global, y_test_global


# #############################################################################”€
# Scenario B: Pathological non-IID benchmark
# #############################################################################”€

def make_benchmark_federated(
    n_clients=10,
    n_samples_total=5000,
    n_features=20,
    n_classes=5,
    classes_per_client=2,
    random_state=42
):
    """
    Classic pathological non-IID: each client holds data from only
    classes_per_client out of n_classes.
    """
    rng = np.random.RandomState(random_state)

    X, y = make_classification(
        n_samples=n_samples_total,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Binary collapse for simplicity
    y_bin = (y >= n_classes // 2).astype(int)

    # Assign class subsets to clients
    client_data = []
    class_indices = {c: np.where(y_bin == c)[0] for c in range(2)}

    for c in range(n_clients):
        # Each client gets a biased subset
        frac_pos = rng.uniform(0.1, 0.9)
        n_c = n_samples_total // n_clients

        n_pos = int(n_c * frac_pos)
        n_neg = n_c - n_pos

        pos_idx = rng.choice(class_indices[1], min(n_pos, len(class_indices[1])),
                             replace=True)
        neg_idx = rng.choice(class_indices[0], min(n_neg, len(class_indices[0])),
                             replace=True)

        X_c = np.vstack([X[pos_idx], X[neg_idx]])
        y_c = np.hstack([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

        # Add client-specific feature noise
        X_c += rng.randn(*X_c.shape) * 0.3

        perm = rng.permutation(len(X_c))
        X_c, y_c = X_c[perm], y_c[perm]

        split = int(len(X_c) * 0.8)
        client_data.append({
            'client_id': c,
            'X_train': X_c[:split],
            'y_train': y_c[:split],
            'X_test': X_c[split:],
            'y_test': y_c[split:],
            'is_adversarial': False,
            'class_imbalance': frac_pos,
            'n_train': split,
        })

    # Global test (balanced)
    n_test = 500
    test_pos = rng.choice(class_indices[1], n_test // 2, replace=True)
    test_neg = rng.choice(class_indices[0], n_test // 2, replace=True)
    X_test = np.vstack([X[test_pos], X[test_neg]])
    y_test = np.hstack([np.ones(n_test // 2), np.zeros(n_test // 2)])

    return client_data, X_test, y_test
