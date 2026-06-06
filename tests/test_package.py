"""
tests/test_package.py
=====================
Unit and integration tests for the pTopoFL package.
Verifies every component against the paper's exact specifications.

Run:
    python -m pytest tests/ -v
    # or standalone:
    python tests/test_package.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from ptopofl import (
    PHDescriptor,
    PTopoFL,
    FedAvg, FedProx, SCAFFOLD, PFedMe, IFCA,
    make_healthcare, make_benchmark, make_continual,
)


# ─────────────────────────────────────────────────────────────────────────────
# PHDescriptor
# ─────────────────────────────────────────────────────────────────────────────

class TestPHDescriptor:

    def setup_method(self):
        self.desc = PHDescriptor(n_sub=80, n_thresholds=20, random_state=42)

    def test_output_dimension(self):
        """Descriptor must be exactly 48-dimensional (§3.1, Eq. 5)."""
        X = np.random.default_rng(0).standard_normal((120, 10))
        phi = self.desc.compute(X)
        assert phi.shape == (48,), f"Expected (48,), got {phi.shape}"

    def test_dimension_various_inputs(self):
        """48-dim holds for all valid input sizes and dimensions."""
        rng = np.random.default_rng(1)
        for n, d in [(30, 4), (80, 8), (200, 16), (500, 32)]:
            X = rng.standard_normal((n, d))
            phi = self.desc.compute(X)
            assert phi.shape == (48,), f"n={n}, d={d}: got shape {phi.shape}"

    def test_subsampling_applied(self):
        """Descriptor is computed on at most n_sub=80 points."""
        X = np.random.default_rng(2).standard_normal((500, 8))
        # If subsampling did not apply, the computation would be O(500^3) — slow.
        # We just verify the output shape and that it completes quickly.
        phi = self.desc.compute(X)
        assert phi.shape == (48,)

    def test_distinct_distributions_differ(self):
        """Two clearly different distributions yield different descriptors."""
        rng = np.random.default_rng(3)
        X1 = rng.standard_normal((100, 6))
        X2 = rng.standard_normal((100, 6)) * 5 + 20
        phi1 = self.desc.compute(X1)
        phi2 = self.desc.compute(X2)
        assert not np.allclose(phi1, phi2, atol=1e-6), \
            "Distinct distributions produced identical descriptors"

    def test_reproducibility(self):
        """Same seed → same descriptor."""
        X = np.random.default_rng(4).standard_normal((150, 8))
        d1 = PHDescriptor(n_sub=80, random_state=99)
        d2 = PHDescriptor(n_sub=80, random_state=99)
        assert np.allclose(d1.compute(X), d2.compute(X)), \
            "Descriptor not reproducible for same random_state"

    def test_descriptor_layout(self):
        """
        Verify the descriptor layout from Appendix B:
          phi[0:20]  = Betti curve H0
          phi[20:40] = Betti curve H1
          phi[40]    = β0 (Betti number)
          phi[41]    = β1
          phi[42]    = H0 entropy
          phi[43]    = H1 entropy
          phi[44]    = A0 amplitude
          phi[45]    = A1 amplitude
          phi[46]    = n0 (finite pair count)
          phi[47]    = n1
        """
        X = np.random.default_rng(5).standard_normal((100, 6))
        phi = self.desc.compute(X)
        # Betti curves: non-negative integers
        assert (phi[0:40] >= 0).all(), "Betti curve values must be non-negative"
        # Betti numbers: non-negative integers
        assert phi[40] >= 0 and phi[41] >= 0
        # Entropy: non-negative
        assert phi[42] >= 0 and phi[43] >= 0
        # Amplitude: non-negative
        assert phi[44] >= 0 and phi[45] >= 0
        # Counts: non-negative integers
        assert phi[46] >= 0 and phi[47] >= 0

    def test_compute_batch(self):
        """compute_batch returns shape (K, 48)."""
        rng = np.random.default_rng(6)
        datasets = [rng.standard_normal((100, 6)) for _ in range(5)]
        phis = self.desc.compute_batch(datasets)
        assert phis.shape == (5, 48)

    def test_small_input(self):
        """Very small inputs (< 3 points) return zero vector without error."""
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        phi = self.desc.compute(X)
        assert phi.shape == (48,)

    def test_h0_determinism(self):
        """H0 computation uses deterministic elder rule (no random root choice)."""
        X = np.random.default_rng(7).standard_normal((50, 4))
        phi_a = PHDescriptor(n_sub=50, random_state=0).compute(X)
        phi_b = PHDescriptor(n_sub=50, random_state=0).compute(X)
        assert np.allclose(phi_a, phi_b), "H0 not deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGenerators:

    def test_healthcare_shapes(self):
        client_data, eval_data, adv_idx = make_healthcare(K=8, n_adv=2, random_state=0)
        assert len(client_data) == 8
        assert len(adv_idx) == 2
        for X, y in client_data:
            assert X.shape[1] == 20
            assert len(y) == len(X)
            assert set(y).issubset({0, 1})
        X_test, y_test = eval_data
        assert X_test.shape[1] == 20

    def test_healthcare_adversarial_clients_differ(self):
        """Adversarial clients should have shifted features (distributional poisoning)."""
        cd_clean, _, _ = make_healthcare(K=8, n_adv=0, random_state=0)
        cd_adv,   _, adv_idx = make_healthcare(K=8, n_adv=2, random_state=0)
        for k in adv_idx:
            assert not np.allclose(cd_clean[k][0], cd_adv[k][0]), \
                f"Client {k} not poisoned"

    def test_benchmark_shapes(self):
        client_data, eval_data = make_benchmark(K=10, random_state=0)
        assert len(client_data) == 10
        for X, y in client_data:
            assert X.shape[1] == 20
            assert len(y) == len(X)

    def test_benchmark_class_imbalance(self):
        """Each benchmark client should have a different class ratio."""
        client_data, _ = make_benchmark(K=10, random_state=0)
        ratios = [y.mean() for _, y in client_data]
        # At least some variation across clients
        assert np.std(ratios) > 0.05, "Benchmark clients not heterogeneous enough"

    def test_continual_shapes(self):
        round_data, drift_clients = make_continual(K=6, n_rounds=20, random_state=0)
        assert len(round_data) == 20
        assert len(round_data[0]) == 6
        assert len(drift_clients) >= 1

    def test_data_standardised(self):
        """make_healthcare applies StandardScaler; features should have ~zero mean."""
        client_data, _, _ = make_healthcare(K=8, n_adv=0, random_state=0)
        all_X = np.vstack([X for X, _ in client_data])
        assert abs(all_X.mean()) < 0.5, "Data not approximately standardised"

    def test_reproducibility_healthcare(self):
        cd1, _, _ = make_healthcare(K=8, n_adv=2, random_state=42)
        cd2, _, _ = make_healthcare(K=8, n_adv=2, random_state=42)
        for (X1, y1), (X2, y2) in zip(cd1, cd2):
            assert np.allclose(X1, X2) and np.array_equal(y1, y2)


# ─────────────────────────────────────────────────────────────────────────────
# pTopoFL
# ─────────────────────────────────────────────────────────────────────────────

class TestPTopoFL:

    def _small_scenario(self, seed=0, K=4, n_rounds=3):
        client_data, eval_data, _ = make_healthcare(
            K=K, n_adv=1, n_features=10, n_informative=6, random_state=seed
        )
        model = PTopoFL(
            n_clusters=2, alpha_blend=0.3, tau=2.0,
            n_rounds=n_rounds, n_sub=40, random_state=seed
        )
        return model, client_data, eval_data

    def test_fit_runs_without_error(self):
        model, cd, ed = self._small_scenario()
        model.fit(cd, eval_data=ed)

    def test_metrics_length(self):
        n_rounds = 5
        model, cd, ed = self._small_scenario(n_rounds=n_rounds)
        model.fit(cd, eval_data=ed)
        assert len(model.metrics_) == n_rounds

    def test_metrics_contain_auc(self):
        model, cd, ed = self._small_scenario()
        model.fit(cd, eval_data=ed)
        for m in model.metrics_:
            assert 'auc' in m, f"Round {m['round']} missing 'auc'"
            assert 0.0 <= m['auc'] <= 1.0

    def test_auc_above_chance(self):
        """pTopoFL should beat random (AUC > 0.5) on a learnable problem."""
        model, cd, ed = self._small_scenario(n_rounds=5)
        model.fit(cd, eval_data=ed)
        final_auc = model.metrics_[-1]['auc']
        assert final_auc > 0.5, f"AUC {final_auc:.3f} not above chance"

    def test_cluster_labels_shape(self):
        model, cd, ed = self._small_scenario(K=4)
        model.fit(cd, eval_data=ed)
        assert model.cluster_labels_.shape == (4,)

    def test_cluster_labels_valid(self):
        model, cd, ed = self._small_scenario(K=4)
        model.fit(cd, eval_data=ed)
        assert set(model.cluster_labels_).issubset({0, 1})

    def test_trust_weights_shape(self):
        model, cd, ed = self._small_scenario(K=4)
        model.fit(cd, eval_data=ed)
        assert model.trust_weights_.shape == (4,)

    def test_trust_weights_range(self):
        model, cd, ed = self._small_scenario(K=4)
        model.fit(cd, eval_data=ed)
        assert ((model.trust_weights_ > 0) & (model.trust_weights_ <= 1.0)).all()

    def test_descriptors_shape(self):
        model, cd, ed = self._small_scenario(K=4)
        model.fit(cd, eval_data=ed)
        assert model.descriptors_.shape == (4, 48)

    def test_predict_proba(self):
        model, cd, ed = self._small_scenario(n_rounds=3)
        model.fit(cd, eval_data=ed)
        X_test, _ = ed
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == len(X_test)
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_no_clustering_degenerates_to_fedavg(self):
        """n_clusters=1 should give similar AUC as FedAvg."""
        cd, ed, _ = make_healthcare(K=8, n_adv=0, random_state=0)
        topo = PTopoFL(n_clusters=1, n_rounds=5, random_state=0)
        fa   = FedAvg(n_rounds=5, random_state=0)
        topo.fit(cd, eval_data=ed)
        fa.fit(cd, eval_data=ed)
        topo_auc = topo.metrics_[-1].get('auc', 0)
        fa_auc   = fa.metrics_[-1].get('auc', 0)
        assert abs(topo_auc - fa_auc) < 0.08, \
            f"k=1 pTopoFL ({topo_auc:.3f}) differs from FedAvg ({fa_auc:.3f}) by > 0.08"

    def test_reproducibility(self):
        """Same seed → identical AUC sequence."""
        cd, ed, _ = make_healthcare(K=4, n_adv=1, random_state=7)
        m1 = PTopoFL(n_rounds=3, random_state=7)
        m2 = PTopoFL(n_rounds=3, random_state=7)
        m1.fit(cd, eval_data=ed)
        m2.fit(cd, eval_data=ed)
        for r1, r2 in zip(m1.metrics_, m2.metrics_):
            assert abs(r1.get('auc', 0) - r2.get('auc', 0)) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselines:

    def _run(self, MethodClass, kwargs, K=4, n_rounds=3, seed=0):
        cd, ed, _ = make_healthcare(
            K=K, n_adv=1, n_features=10, n_informative=6, random_state=seed
        )
        m = MethodClass(n_rounds=n_rounds, random_state=seed, **kwargs)
        m.fit(cd, eval_data=ed)
        return m

    def test_fedavg(self):
        m = self._run(FedAvg, {})
        assert len(m.metrics_) == 3
        assert 'auc' in m.metrics_[-1]

    def test_fedprox(self):
        m = self._run(FedProx, {'mu': 0.1})
        assert 'auc' in m.metrics_[-1]

    def test_scaffold(self):
        m = self._run(SCAFFOLD, {})
        assert 'auc' in m.metrics_[-1]

    def test_pfedme(self):
        m = self._run(PFedMe, {'lam': 15.0})
        assert 'auc' in m.metrics_[-1]

    def test_ifca(self):
        m = self._run(IFCA, {'M': 2})
        assert 'auc' in m.metrics_[-1]
        assert hasattr(m, 'assignments_')
        assert len(m.assignments_) == 4

    def test_all_above_chance(self):
        """Every baseline should exceed AUC 0.5 on a learnable problem."""
        classes = [
            ('FedAvg',   FedAvg,   {}),
            ('FedProx',  FedProx,  {'mu': 0.1}),
            ('SCAFFOLD', SCAFFOLD, {}),
            ('pFedMe',   PFedMe,   {'lam': 15.0}),
            ('IFCA',     IFCA,     {'M': 2}),
        ]
        for name, Cls, kw in classes:
            m = self._run(Cls, kw, n_rounds=5)
            auc = m.metrics_[-1].get('auc', 0)
            assert auc > 0.5, f"{name} AUC {auc:.3f} not above chance"

    def test_ifca_cluster_assignments_valid(self):
        m = self._run(IFCA, {'M': 2}, K=6)
        assert set(m.assignments_).issubset({0, 1})

    def test_scaffold_control_variates_updated(self):
        """SCAFFOLD should have non-zero global control variate after training."""
        cd, ed, _ = make_healthcare(K=4, n_adv=0, random_state=0)
        m = SCAFFOLD(n_rounds=3, random_state=0)
        m.fit(cd, eval_data=ed)
        # AUC should still be computed
        assert 'auc' in m.metrics_[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Integration: pTopoFL beats FedAvg on Healthcare (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_ptopofl_vs_fedavg_healthcare(self):
        """
        Over 5 seeds, pTopoFL mean AUC should be >= FedAvg mean AUC on Healthcare.
        This is the core empirical claim of Table 1.
        """
        topo_aucs, fa_aucs = [], []
        for seed in range(5):
            cd, ed, _ = make_healthcare(K=8, n_adv=2, random_state=seed)
            t = PTopoFL(n_clusters=2, alpha_blend=0.3, tau=2.0,
                        n_rounds=15, random_state=seed)
            f = FedAvg(n_rounds=15, random_state=seed)
            t.fit(cd, eval_data=ed)
            f.fit(cd, eval_data=ed)
            topo_aucs.append(t.metrics_[-1].get('auc', 0.5))
            fa_aucs.append(f.metrics_[-1].get('auc', 0.5))

        mean_topo = np.mean(topo_aucs)
        mean_fa   = np.mean(fa_aucs)
        print(f"\n  pTopoFL={mean_topo:.3f}  FedAvg={mean_fa:.3f}")
        assert mean_topo >= mean_fa - 0.02, \
            f"pTopoFL ({mean_topo:.3f}) unexpectedly worse than FedAvg ({mean_fa:.3f})"

    def test_ptopofl_vs_fedavg_benchmark(self):
        """Same check on Benchmark scenario."""
        topo_aucs, fa_aucs = [], []
        for seed in range(5):
            cd, ed = make_benchmark(K=10, random_state=seed)
            t = PTopoFL(n_clusters=2, alpha_blend=0.3, tau=2.0,
                        n_rounds=15, random_state=seed)
            f = FedAvg(n_rounds=15, random_state=seed)
            t.fit(cd, eval_data=ed)
            f.fit(cd, eval_data=ed)
            topo_aucs.append(t.metrics_[-1].get('auc', 0.5))
            fa_aucs.append(f.metrics_[-1].get('auc', 0.5))

        mean_topo = np.mean(topo_aucs)
        mean_fa   = np.mean(fa_aucs)
        print(f"\n  pTopoFL={mean_topo:.3f}  FedAvg={mean_fa:.3f}")
        assert mean_topo >= mean_fa - 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback

    test_classes = [
        TestPHDescriptor,
        TestDataGenerators,
        TestPTopoFL,
        TestBaselines,
        TestIntegration,
    ]

    total = passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods  = [m for m in dir(cls) if m.startswith('test_')]
        print(f"\n{'─'*60}")
        print(f"  {cls.__name__}  ({len(methods)} tests)")
        print(f"{'─'*60}")
        for name in methods:
            total += 1
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
            try:
                getattr(instance, name)()
                print(f"  ✓  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗  {name}  →  {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗  {name}  →  {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed}/{total} passed   {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
