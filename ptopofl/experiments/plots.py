"""
experiments/plots.py
====================
Generate all manuscript figures (2–7) from JSON results produced by
run_experiments.py.

Usage
-----
  # After running run_experiments.py:
  python experiments/plots.py --results results/ --out figures/

  # Generate a single figure:
  python experiments/plots.py --results results/ --fig 2
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Aesthetic constants (consistent with manuscript) ────────────────────────
COLOURS = {
    'pTopoFL' : '#2ca02c',   # green
    'IFCA'    : '#ff7f0e',   # orange
    'FedProx' : '#d62728',   # red
    'pFedMe'  : '#9467bd',   # purple
    'FedAvg'  : '#1f77b4',   # blue
    'SCAFFOLD': '#8c564b',   # brown
}
LS = {
    'pTopoFL' : '-',
    'IFCA'    : '--',
    'FedProx' : '-.',
    'pFedMe'  : ':',
    'FedAvg'  : '--',
    'SCAFFOLD': (0, (5, 2)),
}
LW = 2.0

METHOD_ORDER = ['pTopoFL', 'IFCA', 'FedProx', 'pFedMe', 'FedAvg', 'SCAFFOLD']

plt.rcParams.update({
    'font.size'       : 11,
    'axes.titlesize'  : 11,
    'axes.labelsize'  : 11,
    'legend.fontsize' : 9,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'figure.dpi'      : 150,
})


def _save(fig, path, fmt='pdf'):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path + '.' + fmt, bbox_inches='tight')
    fig.savefig(path + '.png',    bbox_inches='tight', dpi=150)
    print(f"  Saved → {path}.{fmt}  +  {path}.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — AUC-ROC training curves (mean ± std, 10 seeds)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_curves(data, out_dir):
    """Two-panel AUC-ROC curves for Healthcare (A) and Benchmark (B)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    fig.suptitle(
        'AUC-ROC comparison across 15 FL rounds  (mean ± std, 10 seeds)',
        fontsize=12, fontweight='bold'
    )

    panels = [
        ('Healthcare', 'A', '(A) Healthcare  — 8 non-IID hospitals, 2 adversarial'),
        ('Benchmark',  'B', '(B) Benchmark   — 10 clients, pathological non-IID'),
    ]

    for ax, (scenario, _, title) in zip(axes, panels):
        scenario_data = data.get(scenario, {})
        for method in METHOD_ORDER:
            md = scenario_data.get(method, {})
            if not md:
                continue
            mean = np.array(md['auc_mean'])
            std  = np.array(md['auc_std'])
            rounds = np.arange(1, len(mean) + 1)
            ax.plot(rounds, mean,
                    label=method, color=COLOURS[method],
                    linestyle=LS[method], linewidth=LW,
                    marker='o' if method == 'pTopoFL' else None,
                    markersize=4)
            ax.fill_between(rounds, mean - std, mean + std,
                            alpha=0.12, color=COLOURS[method])

        ax.set_title(title, pad=6)
        ax.set_xlabel('Communication round')
        ax.set_ylabel('AUC-ROC')
        ax.set_xlim(1, len(mean))
        ax.grid(True, alpha=0.25)
        ax.legend(loc='lower right')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fig2_curves'))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Final AUC bars + convergence speed
# ─────────────────────────────────────────────────────────────────────────────

def fig3_bars(data, out_dir):
    """Two-panel bar chart: final AUC (C) and convergence round (D)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Final AUC and convergence speed across six methods  (10 seeds)',
        fontsize=12, fontweight='bold'
    )

    methods     = METHOD_ORDER
    x           = np.arange(len(methods))
    bar_w       = 0.35
    hc_auc      = [data.get('Healthcare', {}).get(m, {}).get('final_auc', 0)      for m in methods]
    hc_auc_std  = [data.get('Healthcare', {}).get(m, {}).get('final_auc_std', 0)  for m in methods]
    bm_auc      = [data.get('Benchmark',  {}).get(m, {}).get('final_auc', 0)      for m in methods]
    bm_auc_std  = [data.get('Benchmark',  {}).get(m, {}).get('final_auc_std', 0)  for m in methods]
    hc_conv     = [data.get('Healthcare', {}).get(m, {}).get('conv_round_mean', 0) for m in methods]
    bm_conv     = [data.get('Benchmark',  {}).get(m, {}).get('conv_round_mean', 0) for m in methods]
    colours     = [COLOURS[m] for m in methods]

    # ── Panel C: final AUC ───────────────────────────────────────────────────
    ax = axes[0]
    bars_hc = ax.bar(x - bar_w/2, hc_auc, bar_w, yerr=hc_auc_std,
                     capsize=4, color=colours, alpha=0.95,
                     label='Healthcare (HC)', linewidth=0)
    bars_bm = ax.bar(x + bar_w/2, bm_auc, bar_w, yerr=bm_auc_std,
                     capsize=4, color=colours, alpha=0.45,
                     label='Benchmark (BM)', linewidth=0)

    # Significance asterisk on pTopoFL HC bar
    hc_idx = methods.index('pTopoFL')
    ax.text(hc_idx - bar_w/2, hc_auc[hc_idx] + hc_auc_std[hc_idx] + 0.002,
            '*', ha='center', va='bottom', fontsize=13, color='black')
    ax.text(hc_idx - bar_w/2 + 0.02, hc_auc[hc_idx] - 0.006,
            'p<0.05', ha='center', va='top', fontsize=6.5, color='#d62728')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('(C) Final-round AUC-ROC')
    ax.set_ylim(0.70, 0.96)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)

    # ── Panel D: convergence round ───────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - bar_w/2, hc_conv, bar_w, color=colours, alpha=0.95,
           label='Healthcare (HC)', linewidth=0)
    ax.bar(x + bar_w/2, bm_conv, bar_w, color=colours, alpha=0.45,
           label='Benchmark (BM)', linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('Round reaching 95 % of final AUC')
    ax.set_title('(D) Convergence speed')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fig3_bars'))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Adversarial robustness
# ─────────────────────────────────────────────────────────────────────────────

def fig4_adversarial(data, out_dir):
    """
    Panel (A): final AUC vs. attack rate for all methods.
    Panel (B): AUC curves at 0 %, 30 %, 50 % for pTopoFL and FedAvg.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        'Adversarial robustness — distributional-poisoning attacks',
        fontsize=12, fontweight='bold'
    )

    rates = sorted(int(k) for k in data.keys())
    adv_methods = ['pTopoFL', 'IFCA', 'FedProx', 'FedAvg', 'SCAFFOLD']

    # ── Panel A: final AUC vs attack rate ────────────────────────────────────
    ax = axes[0]
    for m in adv_methods:
        means = [data[str(r)].get(m, {}).get('mean', 0.5) for r in rates]
        stds  = [data[str(r)].get(m, {}).get('std',  0.0) for r in rates]
        ax.plot(rates, means, color=COLOURS.get(m, 'grey'),
                linestyle=LS.get(m, '-'), linewidth=LW,
                marker='o', markersize=5, label=m)
        ax.fill_between(rates,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.10, color=COLOURS.get(m, 'grey'))

    ax.set_xlabel('Fraction of adversarial clients [%]')
    ax.set_ylabel('Final AUC-ROC')
    ax.set_title('(A) Final AUC vs. attack rate')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(15, ax.get_ylim()[0] + 0.01,
            'Attack: distributional poisoning\n(feature + label)',
            fontsize=7.5, style='italic', color='grey')

    # ── Panel B: training curves at 0/30/50 % ───────────────────────────────
    ax = axes[1]
    curve_methods = ['pTopoFL', 'FedAvg']
    curve_rates   = [0, 30, 50]
    ls_by_rate    = {0: '-', 30: '--', 50: ':'}
    alpha_by_rate = {0: 0.95, 30: 0.70, 50: 0.55}

    for m in curve_methods:
        for r in curve_rates:
            r_data = data.get(str(r), {}).get(m, {})
            curve  = r_data.get('curve_mean', [])
            if not curve:
                continue
            rounds = np.arange(1, len(curve) + 1)
            label  = f"{m} ({r}% adv)" if r in (0, 50) else f"({r}% adv)"
            ax.plot(rounds, curve,
                    color=COLOURS.get(m, 'grey'),
                    linestyle=ls_by_rate[r],
                    linewidth=LW,
                    alpha=alpha_by_rate[r],
                    label=label)

    ax.set_xlabel('Communication round')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('(B) Training curves at 0%, 30%, 50% attack')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fig4_adversarial'))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Topological signature stability
# ─────────────────────────────────────────────────────────────────────────────

def fig5_continual(data, out_dir):
    """H0 and H1 persistence entropy per client over 20 FL rounds."""
    seed_0 = data[0]   # plot the first seed for clarity
    K = len(seed_0['h0_entropy'])

    client_colours = plt.cm.tab10(np.linspace(0, 0.9, K))
    drift_clients  = seed_0.get('drift_clients', [])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(
        'Topological signature stability over 20 FL rounds',
        fontsize=12, fontweight='bold'
    )

    for ax, key, ylabel in [
        (axes[0], 'h0_entropy', 'H₀ persistence entropy'),
        (axes[1], 'h1_entropy', 'H₁ persistence entropy'),
    ]:
        for k_str, vals in seed_0[key].items():
            k      = int(k_str)
            rounds = np.arange(1, len(vals) + 1)
            is_drift = k in drift_clients
            ax.plot(rounds, vals,
                    color=client_colours[k],
                    linewidth=1.8,
                    linestyle='--' if is_drift else '-',
                    alpha=0.9,
                    label=f'Client {k + 1}' + (' *' if is_drift else ''))

        ax.axvline(x=1, color='grey', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.text(1.2, ax.get_ylim()[1] * 0.96, 'Round-0\ncluster\nassignment',
                fontsize=7, color='grey', va='top')

        ax.set_xlabel('FL round')
        ax.set_ylabel(ylabel)
        ax.set_title(
            f'({"A" if "H₀" in ylabel else "B"}) {ylabel} per client'
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=7.5, ncol=2)

    if drift_clients:
        axes[0].text(0.02, 0.03,
                     'Dashed: clients with elevated drift\n(candidates for re-clustering)',
                     transform=axes[0].transAxes,
                     fontsize=7.5, color='grey', style='italic')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fig5_continual'))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Reconstruction surface (SNR analysis, Theorem 6)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_privacy(data, out_dir):
    """Three-panel: SNR comparison, sensitivity, dimensionality."""
    clients  = sorted(set(r['client'] for r in data))
    snr_phi  = np.array([r['snr_phi']   for r in data])
    snr_g    = np.array([r['snr_g']     for r in data])
    d_phi    = np.array([r['delta_phi'] for r in data])
    d_g      = np.array([r['delta_g']   for r in data])
    n_ks     = np.array([r['n_k']       for r in data])

    # Aggregate by client (mean across seeds)
    client_ids = np.array([r['client'] for r in data])
    K = len(clients)
    mean_snr_phi = np.array([snr_phi[client_ids == k].mean() for k in range(K)])
    mean_snr_g   = np.array([snr_g[client_ids == k].mean()   for k in range(K)])
    mean_d_phi   = np.array([d_phi[client_ids == k].mean()   for k in range(K)])
    mean_d_g     = np.array([d_g[client_ids == k].mean()     for k in range(K)])

    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    fig.suptitle(
        'Descriptor reconstruction surface vs. gradient channel  (Theorem 6)',
        fontsize=12, fontweight='bold'
    )

    x = np.arange(1, K + 1)

    # ── Panel A: SNR (log scale) ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.semilogy(x, mean_snr_g,   'b-o', linewidth=LW, markersize=5,
                label='SNR$_G$ (gradient)')
    ax.semilogy(x, mean_snr_phi, 'g--s', linewidth=LW, markersize=5,
                label='SNR$_φ$ (descriptor)')
    ratio = float(np.mean(mean_snr_phi / (mean_snr_g + 1e-20)))
    ax.text(0.05, 0.07,
            f'Descriptor channel\nconsistently narrower\n(×{ratio:.2f} in mean SNR)',
            transform=ax.transAxes, fontsize=8, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.set_xlabel('Client index')
    ax.set_ylabel('Adversarial reconstruction SNR  (log scale)')
    ax.set_title('(A) Reconstruction SNR comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.25, which='both')

    # ── Panel B: sensitivity ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.plot(x, mean_d_g,   'b-o', linewidth=LW, markersize=5,
            label='$\\Delta_G = 2B/n_k$ (gradient)')
    ax.plot(x, mean_d_phi, 'g--s', linewidth=LW, markersize=5,
            label='$\\Delta_\\phi \\leq c_\\delta$ (descriptor)')
    ax.axhline(y=mean_d_phi.mean(), color='green', linestyle=':',
               alpha=0.6, linewidth=1.2, label=f'bounded: $c_\\delta\\approx{mean_d_phi.mean():.3f}$')
    ax.set_xlabel('Client index')
    ax.set_ylabel('Leave-one-out sensitivity')
    ax.set_title('(B) Sensitivity vs. client size')
    ax.legend(loc='upper right', fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # ── Panel C: dimensionality ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    bar_colours = ['#2ca02c', '#1f77b4']
    ax.bar(['Descriptor\n($\\phi_k$)', 'LR model\n($\\theta_k$)'],
           [48, 10_000], color=bar_colours, width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Transmitted dimensions  (log scale)')
    ax.set_title('(C) Transmitted dimensionality')
    ax.text(0, 48 * 1.4, '48', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#2ca02c')
    ax.text(1, 10_000 * 1.4, '10 000', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#1f77b4')
    ax.grid(True, axis='y', alpha=0.25)

    _save(fig, os.path.join(out_dir, 'fig6_privacy'))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def fig7_ablation(data, out_dir):
    """Bar chart with error bars for the three ablation conditions."""
    conditions = list(data.keys())
    means = [data[c]['mean'] for c in conditions]
    stds  = [data[c]['std']  for c in conditions]
    colours = ['#7f7f7f', '#aec7e8', '#2ca02c']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(conditions, means, yerr=stds, capsize=6,
           color=colours, width=0.5, linewidth=0)

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f'{m:.3f}±{s:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # FedAvg baseline
    fedavg_val = means[0]   # No-clustering condition = FedAvg
    ax.axhline(fedavg_val, color='#1f77b4', linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'FedAvg baseline ({fedavg_val:.3f})')

    # Delta annotation
    delta = means[-1] - means[0]
    ax.annotate('', xy=(2, means[-1] + 0.005), xytext=(0, means[0] + 0.005),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.2))
    ax.text(1, (means[0] + means[-1]) / 2 + 0.008,
            f'Δ = {delta:.3f} AUC', ha='center', fontsize=9, color='grey')

    ax.set_ylim(0.760, max(means) + max(stds) + 0.022)
    ax.set_ylabel('AUC-ROC (Healthcare scenario)')
    ax.set_title(
        'Ablation study — pTopoFL components\n(mean ± std, 10 seeds)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fig7_ablation'))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load(path, label):
    if not os.path.exists(path):
        print(f"  [skip] {label}: file not found ({path})")
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate manuscript figures from JSON results'
    )
    parser.add_argument('--results', default='results',
                        help='Directory containing JSON files (default: results/)')
    parser.add_argument('--out', default='figures',
                        help='Output directory for figures (default: figures/)')
    parser.add_argument('--fig', type=int, default=0,
                        help='Generate only this figure number (0 = all)')
    args = parser.parse_args()

    R   = args.results
    out = args.out
    os.makedirs(out, exist_ok=True)

    print(f"\nGenerating figures from {R}/ → {out}/\n")

    if args.fig in (0, 2, 3):
        d = _load(os.path.join(R, 'table1.json'), 'Table1 / Figs 2-3')
        if d:
            print("Figure 2 — AUC training curves")
            fig2_curves(d, out)
            print("Figure 3 — Final AUC bars")
            fig3_bars(d, out)

    if args.fig in (0, 4):
        d = _load(os.path.join(R, 'adversarial.json'), 'Fig 4')
        if d:
            print("Figure 4 — Adversarial robustness")
            fig4_adversarial(d, out)

    if args.fig in (0, 5):
        d = _load(os.path.join(R, 'continual.json'), 'Fig 5')
        if d:
            print("Figure 5 — Topological signature stability")
            fig5_continual(d, out)

    if args.fig in (0, 6):
        d = _load(os.path.join(R, 'privacy.json'), 'Fig 6')
        if d:
            print("Figure 6 — Reconstruction surface")
            fig6_privacy(d, out)

    if args.fig in (0, 7):
        d = _load(os.path.join(R, 'ablation.json'), 'Fig 7')
        if d:
            print("Figure 7 — Ablation study")
            fig7_ablation(d, out)

    print(f"\nAll figures saved to {out}/")
