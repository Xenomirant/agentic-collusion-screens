"""Static report figures from the saved E0-E2 results (no recomputation)."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

# reference palette, light mode (validated slots 1-3); neutrals for reference lines
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
    'lines.linewidth': 2, 'legend.frameon': False,
})


def fig_e1():
    A = pd.read_csv(os.path.join(RES, 'e1', 'auc_by_time.csv'), index_col=0)
    t = A.index.values / 1e6
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    series = [('phi2_lag1', 'factorization defect (lag-1 $\\varphi^2$)', BLUE),
              ('variance', 'price variance', ORANGE),
              ('pearson', 'Pearson correlation', AQUA)]
    for col, name, c in series:
        ax.plot(t, A[col].values, color=c, label=name)
    ax.axhline(0.5, color=INK2, lw=1, ls='--')
    ax.text(t[-1], 0.52, 'chance', ha='right', fontsize=8, color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('learning period (millions)')
    ax.set_ylabel('AUC, collusive vs myopic')
    ax.set_title('E1: when can price structure separate collusive from myopic Q-learners?', loc='left', fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e1_auc_over_time.png'), dpi=200)

    r = json.load(open(os.path.join(RES, 'e1', 'summary.json')))
    auc = pd.Series(r['auc_vs_control']['myopic']).drop(['Delta', 'price_mean'])
    names = {'phi2_k2': '$\\varphi^2$, k=2 windows', 'phi2_lag1': '$\\varphi^2$, lag 1',
             'rho_dcca_large': '$\\rho_{DCCA}$, s$\\geq$100', 'hgr_lag1': 'HGR max-corr, lag 1',
             'ar1': 'AR(1)', 'phi2_k1': '$\\varphi^2$, k=1', 'rho_dcca': '$\\rho_{DCCA}$, all s',
             'variance': 'variance', 'pearson': 'Pearson'}
    auc = auc.sort_values()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    y = np.arange(len(auc))
    ax.barh(y, auc.values, height=0.6, color=BLUE)
    ax.set_yticks(y, [names[k] for k in auc.index])
    for yi, v in zip(y, auc.values):
        ax.text(v + 0.01, yi, '{:.2f}'.format(v), va='center', fontsize=8, color=INK)
    ax.axvline(0.5, color=INK2, lw=1, ls='--')
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('AUC (converged play, 20 vs 20 runs)')
    ax.set_title('E1: collusive vs myopic, prices only', loc='left', fontsize=9)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e1_auc_converged.png'), dpi=200)


def fig_e2():
    r = json.load(open(os.path.join(RES, 'e2', 'summary.json')))
    P = pd.DataFrame(r['rejection_rate_copy'])
    eps = P.index.astype(float).values
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    for col, name, c in [('p_phi2_k2_within', 'factorization defect', BLUE),
                         ('p_dcca_within', '$\\rho_{DCCA}$', ORANGE),
                         ('p_pearson_within', 'Pearson', AQUA)]:
        ax.plot(eps, P[col].values, '-o', ms=4, color=c, label=name)
    ax.axhline(0.05, color=INK2, lw=1, ls='--')
    ax.text(eps[-1], 0.08, '$\\alpha$ = 0.05', ha='right', fontsize=8, color=INK2)
    ax.set_xlabel('coupling probability $\\epsilon$ (lagged copy)')
    ax.set_ylabel('detection rate (30 seeds)')
    ax.set_title('E2a: power vs coupling', loc='left', fontsize=9)
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(-0.03, 1.03)

    X = pd.DataFrame(r['xor_rejection_rate']).loc['0.3']
    labels = ['Pearson\n0 vs 2', '$\\rho_{DCCA}$\n0 vs 2', 'defect\n0 vs 2', 'defect\n1 vs 2', 'defect\n{0,1} vs 2']
    vals = X[['p_pearson_02', 'p_dcca_02', 'p_pair_02', 'p_pair_12', 'p_group_01_2']].values
    ax = axes[1]
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.6, color=BLUE)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, '{:.2f}'.format(v), ha='center', fontsize=8, color=INK)
    ax.axhline(0.05, color=INK2, lw=1, ls='--')
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title('E2b: XOR triad, $\\epsilon$ = 0.3', loc='left', fontsize=9)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e2_power_xor.png'), dpi=200)


def fig_e0():
    r = json.load(open(os.path.join(RES, 'e0', 'summary.json')))['cascade']
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(r['q'], r['h_theory'], color=INK2, lw=1.5, ls='--', label='theory')
    ax.plot(r['q'], r['h_mfdfa'], 'o', ms=5, color=BLUE, label='batched MFDFA')
    ax.set_xlabel('q')
    ax.set_ylabel('h(q)')
    ax.set_title('E0: binomial cascade, p = 0.3', loc='left', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e0_cascade_hq.png'), dpi=200)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    fig_e0()
    fig_e1()
    fig_e2()
    print(sorted(os.listdir(FIG)))
