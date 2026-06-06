"""
experiments/run_experiments.py
==============================
Reproduces all main results in the manuscript:
  Table 1  — Healthcare and Benchmark final AUC-ROC (10 seeds)
  Figure 2 — AUC-ROC training curves (saved as JSON → plots.py)
  Figure 3 — Final AUC bar chart + convergence speed
  Figure 4 — Adversarial robustness under distributional poisoning
  Figure 5 — Topological signature stability (continual FL)
  Figure 6 — SNR reconstruction-surface analysis (Theorem 6)
  Figure 7 — Ablation study

Usage
-----
  # Full reproduction (10 seeds, all experiments, ~15 min on a laptop CPU)
  python run_experiments.py --scenario all --seeds 10 --out results/

  # Single experiment
  python run_experiments.py --scenario table1 --seeds 10 --out results/

  # Quick smoke test (2 seeds)
  python run_experiments.py --scenario all --seeds 2 --out results_quick/
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# ── Path setup: allow running as a script from any directory ────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Package imports (all names fully qualified) ──────────────────────────────
from ptopofl import (
    PHDescriptor,
    PTopoFL,
    FedAvg, FedProx, SCAFFOLD, PFedMe, IFCA,
    make_healthcare, make_benchmark, make_continual,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (match §4.1 exactly)
# ─────────────────────────────────────────────────────────────────────────────

HPARAMS = {
    'n_sub'       : 80,       # points subsampled per client for TDA
    'n_thresholds': 20,       # Betti-curve resolution
    'M'           : 2,        # number of clusters
    'alpha_blend' : 0.3,      # inter-cluster blending coefficient
    'tau_compare' : 2.0,      # anomaly detection threshold (comparison)
    'tau_robust'  : 1.8,      # threshold for robustness experiment
    'mu_fedprox'  : 0.1,      # FedProx proximal penalty
    'lam_pfedme'  : 15.0,     # pFedMe Moreau-envelope lambda
    'n_rounds_main'    : 15,  # rounds for Table 1 / Figures 2-3
    'n_rounds_robust'  : 10,  # rounds for Figure 4
    'n_rounds_continual': 20, # rounds for Figure 5
    'K_healthcare' : 8,
    'K_benchmark'  : 10,
    'K_continual'  : 6,
    'n_adv'        : 2,       # adversarial clients (Healthcare)
}

# Method registry: (display_name, class, extra_kwargs)
METHODS = [
    ('pTopoFL', PTopoFL, {'n_clusters': HPARAMS['M'],
                          'alpha_blend': HPARAMS['alpha_blend'],
                          'tau': HPARAMS['tau_compare']}),
    ('IFCA',    IFCA,    {'M': HPARAMS['M']}),
    ('FedProx', FedProx, {'mu': HPARAMS['mu_fedprox']}),
    ('pFedMe',  PFedMe,  {'lam': HPARAMS['lam_pfedme']}),
    ('FedAvg',  FedAvg,  {}),
    ('SCAFFOLD',SCAFFOLD,{}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(seed_metrics_list, n_rounds):
    """
    Aggregate per-seed metric lists into mean ± std per round.

    Parameters
    ----------
    seed_metrics_list : list[list[dict]]
        Outer = seeds; inner = per-round metric dicts from method.metrics_.
    n_rounds : int

    Returns
    -------
    dict with keys:
      auc_mean, auc_std, acc_mean, acc_std  – lists of length n_rounds
      final_auc, final_auc_std              – scalars
      final_acc, final_acc_std              – scalars
      conv_round_mean, conv_round_std       – scalars
    """
    n_seeds = len(seed_metrics_list)
    auc_mat = np.full((n_seeds, n_rounds), np.nan)
    acc_mat = np.full((n_seeds, n_rounds), np.nan)

    for s, metrics in enumerate(seed_metrics_list):
        for m in metrics:
            r = m['round'] - 1
            if r < n_rounds:
                auc_mat[s, r] = m.get('auc', np.nan)
                acc_mat[s, r] = m.get('acc', np.nan)

    auc_mean = np.nanmean(auc_mat, axis=0).tolist()
    auc_std  = np.nanstd( auc_mat, axis=0).tolist()
    acc_mean = np.nanmean(acc_mat, axis=0).tolist()
    acc_std  = np.nanstd( acc_mat, axis=0).tolist()

    # Convergence: first round reaching 95 % of final AUC (per seed)
    conv_rounds = []
    for s in range(n_seeds):
        final = auc_mat[s, -1]
        if np.isnan(final):
            conv_rounds.append(n_rounds)
            continue
        threshold = 0.95 * final
        reached   = np.where(auc_mat[s] >= threshold)[0]
        conv_rounds.append(int(reached[0]) + 1 if len(reached) > 0 else n_rounds)

    return {
        'auc_mean'       : auc_mean,
        'auc_std'        : auc_std,
        'acc_mean'       : acc_mean,
        'acc_std'        : acc_std,
        'final_auc'      : float(auc_mean[-1]) if auc_mean else np.nan,
        'final_auc_std'  : float(auc_std[-1])  if auc_std  else np.nan,
        'final_acc'      : float(acc_mean[-1]) if acc_mean else np.nan,
        'final_acc_std'  : float(acc_std[-1])  if acc_std  else np.nan,
        'conv_round_mean': float(np.mean(conv_rounds)),
        'conv_round_std' : float(np.std(conv_rounds)),
    }


def _print_table1(all_results):
    """Pretty-print Table 1 to stdout."""
    print("\n" + "=" * 76)
    print(f"{'Method':<12} {'HC AUC':>14} {'BM AUC':>14} {'HC Acc':>12} {'BM Acc':>12}")
    print("-" * 76)
    for name, _, _ in METHODS:
        hc = all_results.get('Healthcare', {}).get(name, {})
        bm = all_results.get('Benchmark',  {}).get(name, {})
        star = '*' if name == 'pTopoFL' else ' '
        print(
            f"{star}{name:<11} "
            f"{hc.get('final_auc', 0):.3f}±{hc.get('final_auc_std', 0):.3f}  "
            f"{bm.get('final_auc', 0):.3f}±{bm.get('final_auc_std', 0):.3f}  "
            f"{hc.get('final_acc', 0):.3f}±{hc.get('final_acc_std', 0):.3f}  "
            f"{bm.get('final_acc', 0):.3f}±{bm.get('final_acc_std', 0):.3f}"
        )
    print("=" * 76)
    print("* = pTopoFL (proposed)   HC = Healthcare   BM = Benchmark")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: Table 1 + Figures 2–3
# ─────────────────────────────────────────────────────────────────────────────

def experiment_table1(n_seeds=10, out_dir='results'):
    """
    Reproduce Table 1 and the data for Figures 2–3.
    Saves results/table1.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    scenarios = [
        ('Healthcare', make_healthcare,
         {'K': HPARAMS['K_healthcare'], 'n_adv': HPARAMS['n_adv']},
         HPARAMS['n_rounds_main']),
        ('Benchmark', make_benchmark,
         {'K': HPARAMS['K_benchmark']},
         HPARAMS['n_rounds_main']),
    ]

    for scenario_label, scenario_fn, scenario_kwargs, n_rounds in scenarios:
        print(f"\n── {scenario_label}  ({n_seeds} seeds, {n_rounds} rounds) ──")
        scenario_results = {}

        for name, MethodClass, method_kwargs in METHODS:
            print(f"  {name:<10}", end='', flush=True)
            seed_metrics = []

            for seed in range(n_seeds):
                print('.', end='', flush=True)
                out        = scenario_fn(random_state=seed, **scenario_kwargs)
                client_data, eval_data = out[0], out[1]

                method = MethodClass(
                    n_rounds=n_rounds, random_state=seed, **method_kwargs
                )
                method.fit(client_data, eval_data=eval_data)
                seed_metrics.append(method.metrics_)

            summary = _summarise(seed_metrics, n_rounds)
            scenario_results[name] = summary
            print(f"  AUC {summary['final_auc']:.3f}±{summary['final_auc_std']:.3f}")

        all_results[scenario_label] = scenario_results

    _print_table1(all_results)

    path = os.path.join(out_dir, 'table1.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {path}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: Figure 4 — Adversarial robustness
# ─────────────────────────────────────────────────────────────────────────────

def experiment_adversarial(attack_rates=None, n_seeds=10, out_dir='results'):
    """
    Figure 4: final AUC vs. distributional-poisoning attack rate (0–50 %).
    Also records per-round curves at 0 %, 30 %, 50 % for panel (B).
    Saves results/adversarial.json.
    """
    if attack_rates is None:
        attack_rates = [0, 10, 20, 30, 40, 50]
    os.makedirs(out_dir, exist_ok=True)

    adv_methods = [
        ('pTopoFL', PTopoFL, {'n_clusters': 2, 'alpha_blend': 0.3,
                              'tau': HPARAMS['tau_robust']}),
        ('FedAvg',  FedAvg,  {}),
        ('IFCA',    IFCA,    {'M': 2}),
        ('FedProx', FedProx, {'mu': 0.1}),
        ('SCAFFOLD',SCAFFOLD,{}),
    ]

    results     = {}
    curve_rates = {0, 30, 50}     # save full curves for these rates

    for rate in attack_rates:
        n_adv = max(0, round(HPARAMS['K_healthcare'] * rate / 100))
        print(f"  Attack {rate:>3}% ({n_adv} adversarial clients)")

        rate_results = {}
        for name, MethodClass, kwargs in adv_methods:
            seed_final  = []
            seed_curves = []

            for seed in range(n_seeds):
                client_data, eval_data, _ = make_healthcare(
                    K=HPARAMS['K_healthcare'], n_adv=n_adv, random_state=seed
                )
                m = MethodClass(
                    n_rounds=HPARAMS['n_rounds_robust'],
                    random_state=seed, **kwargs
                )
                m.fit(client_data, eval_data=eval_data)
                seed_final.append(m.metrics_[-1].get('auc', 0.5))
                if rate in curve_rates:
                    seed_curves.append([me.get('auc', 0.5) for me in m.metrics_])

            rate_results[name] = {
                'mean': float(np.mean(seed_final)),
                'std' : float(np.std(seed_final)),
            }
            if rate in curve_rates:
                curve_mat = np.array(seed_curves)
                rate_results[name]['curve_mean'] = np.nanmean(curve_mat, 0).tolist()
                rate_results[name]['curve_std']  = np.nanstd( curve_mat, 0).tolist()

        results[rate] = rate_results
        print("   " + "  ".join(
            f"{n}: {rate_results[n]['mean']:.3f}"
            for n, _, _ in adv_methods
        ))

    path = os.path.join(out_dir, 'adversarial.json')
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"Saved → {path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: Figure 5 — Topological signature stability
# ─────────────────────────────────────────────────────────────────────────────

def experiment_continual(n_seeds=5, out_dir='results'):
    """
    Figure 5: H0 and H1 persistence entropy per client over 20 FL rounds.
    Saves results/continual.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    desc_fn = PHDescriptor(
        n_sub=HPARAMS['n_sub'], n_thresholds=20, random_state=0
    )

    # Descriptor layout: [bc_H0(20), bc_H1(20), β0,β1, H0,H1, A0,A1, n0,n1]
    IDX_H0_ENTROPY = 42
    IDX_H1_ENTROPY = 43

    all_seed_data = []

    for seed in range(n_seeds):
        round_data, drift_clients = make_continual(
            K=HPARAMS['K_continual'],
            n_rounds=HPARAMS['n_rounds_continual'],
            random_state=seed
        )
        K = HPARAMS['K_continual']
        h0_per_client = {k: [] for k in range(K)}
        h1_per_client = {k: [] for k in range(K)}

        for r_data in round_data:
            for k, (X, y) in enumerate(r_data):
                phi = desc_fn.compute(X)
                h0_per_client[k].append(float(phi[IDX_H0_ENTROPY]))
                h1_per_client[k].append(float(phi[IDX_H1_ENTROPY]))

        # Compute per-client drift (Eq. 9) for annotation
        drift = {}
        for k in range(K):
            h0_arr = np.array(h0_per_client[k])
            drift[k] = float(np.mean(np.abs(h0_arr - h0_arr[0])))

        all_seed_data.append({
            'seed'         : seed,
            'h0_entropy'   : {str(k): v for k, v in h0_per_client.items()},
            'h1_entropy'   : {str(k): v for k, v in h1_per_client.items()},
            'drift'        : {str(k): v for k, v in drift.items()},
            'drift_clients': [int(c) for c in drift_clients],
        })

    # Mean drift across seeds
    mean_drift = float(np.mean([
        np.mean(list(sd['drift'].values())) for sd in all_seed_data
    ]))
    print(f"  Mean normalised topological drift Δ = {mean_drift:.4f}  "
          f"(paper reports 0.55)")

    path = os.path.join(out_dir, 'continual.json')
    with open(path, 'w') as f:
        json.dump(all_seed_data, f, indent=2)
    print(f"Saved → {path}")
    return all_seed_data


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: Figure 6 — SNR reconstruction-surface analysis
# ─────────────────────────────────────────────────────────────────────────────

def experiment_privacy(n_seeds=5, out_dir='results'):
    """
    Figure 6: SNR comparison between descriptor and gradient channels
    (Theorem 6).

    SNR_φ = Δ_φ² / (σ²·m)
    SNR_G = 4B² / (n_k²·σ²·p)

    Δ_φ is measured empirically via leave-one-out subsampling.
    Saves results/privacy.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    desc_fn = PHDescriptor(n_sub=HPARAMS['n_sub'], n_thresholds=20, random_state=0)
    rng     = np.random.default_rng(0)

    m      = 48          # descriptor dimension
    p      = 10_000      # representative model parameter count (logistic reg)
    B      = 1.0         # bounded gradient norm (Assumption 1 of Theorem 6)
    sigma2 = 1.0         # noise variance (relative comparison; cancels in ratio)
    N_LOO  = 10          # leave-one-out samples per client

    records = []
    for seed in range(n_seeds):
        client_data, _, _ = make_healthcare(
            K=HPARAMS['K_healthcare'], n_adv=0, random_state=seed
        )
        for k, (X, y) in enumerate(client_data):
            n_k  = len(X)
            phi  = desc_fn.compute(X)

            # Empirical Δ_φ: mean leave-one-out descriptor change
            delta_phi_vals = []
            for _ in range(N_LOO):
                i       = int(rng.integers(n_k))
                X_loo   = np.delete(X, i, axis=0)
                phi_loo = desc_fn.compute(X_loo)
                delta_phi_vals.append(float(np.linalg.norm(phi - phi_loo)))
            delta_phi = float(np.mean(delta_phi_vals))

            delta_g  = 2.0 * B / n_k
            snr_phi  = delta_phi ** 2 / (sigma2 * m)
            snr_g    = (4.0 * B ** 2) / (n_k ** 2 * sigma2 * p)
            snr_ratio = snr_phi / (snr_g + 1e-20)

            records.append({
                'seed'     : seed,
                'client'   : k,
                'n_k'      : n_k,
                'delta_phi': delta_phi,
                'delta_g'  : delta_g,
                'snr_phi'  : snr_phi,
                'snr_g'    : snr_g,
                'snr_ratio': snr_ratio,          # < 1  means descriptor is narrower
            })

    mean_ratio = float(np.mean([r['snr_ratio'] for r in records]))
    print(f"  Mean SNR_φ / SNR_G = {mean_ratio:.3f}  "
          f"(< 1 means descriptor channel is narrower; "
          f"paper reports ~0.22, i.e. 4.5× reduction)")

    path = os.path.join(out_dir, 'privacy.json')
    with open(path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved → {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Experiment: Figure 7 — Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def experiment_ablation(n_seeds=10, out_dir='results'):
    """
    Figure 7: ablation on Healthcare.
    Conditions: no clustering (k=1), no blending (α=0), full pTopoFL.
    Saves results/ablation.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    conditions = [
        ('No clustering (k=1)',
         {'n_clusters': 1, 'alpha_blend': 0.3, 'tau': 2.0}),
        ('No blending (α=0)',
         {'n_clusters': 2, 'alpha_blend': 0.0, 'tau': 2.0}),
        ('Full pTopoFL',
         {'n_clusters': 2, 'alpha_blend': 0.3, 'tau': 2.0}),
    ]

    results = {}
    for label, kwargs in conditions:
        print(f"  {label:<25}", end='', flush=True)
        seed_aucs = []

        for seed in range(n_seeds):
            client_data, eval_data, _ = make_healthcare(
                K=HPARAMS['K_healthcare'],
                n_adv=HPARAMS['n_adv'],
                random_state=seed
            )
            method = PTopoFL(
                n_rounds=HPARAMS['n_rounds_main'],
                random_state=seed, **kwargs
            )
            method.fit(client_data, eval_data=eval_data)
            seed_aucs.append(method.metrics_[-1].get('auc', 0.5))

        results[label] = {
            'mean': float(np.mean(seed_aucs)),
            'std' : float(np.std(seed_aucs)),
            'all' : [float(v) for v in seed_aucs],
        }
        print(f"  AUC {results[label]['mean']:.3f}±{results[label]['std']:.3f}")

    path = os.path.join(out_dir, 'ablation.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='pTopoFL — reproduce manuscript results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--scenario', default='all',
        choices=['table1', 'adversarial', 'continual', 'privacy', 'ablation', 'all'],
        help='Which experiment to run (default: all)',
    )
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of random seeds (default: 10)')
    parser.add_argument('--out',   default='results',
                        help='Output directory (default: results/)')
    args = parser.parse_args()

    print(f"\npTopoFL experiment runner")
    print(f"  scenario : {args.scenario}")
    print(f"  seeds    : {args.seeds}")
    print(f"  out      : {args.out}\n")

    if args.scenario in ('table1', 'all'):
        experiment_table1(n_seeds=args.seeds, out_dir=args.out)

    if args.scenario in ('adversarial', 'all'):
        experiment_adversarial(n_seeds=args.seeds, out_dir=args.out)

    if args.scenario in ('continual', 'all'):
        experiment_continual(n_seeds=min(args.seeds, 5), out_dir=args.out)

    if args.scenario in ('privacy', 'all'):
        experiment_privacy(n_seeds=min(args.seeds, 5), out_dir=args.out)

    if args.scenario in ('ablation', 'all'):
        experiment_ablation(n_seeds=args.seeds, out_dir=args.out)

    print(f"\nAll done.  Results saved to {args.out}/")
    print("Run  python experiments/plots.py --results results/  to generate figures.")
