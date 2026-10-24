"""Threshold = max of a same-size exchangeable null: how many false positives should one expect?"""
import json, numpy as np
d = json.load(open('/workspace/projects/TenDeFlAn/experiments/results/e14/summary_N1M.json'))['detectors']
n_test_clean = 9_500_000 - 479_081 - 8_000  # approx: pairs minus benign minus planted
n_null = 9_500_000 - 479_081                 # null world, non-benign
print('analytic P(>=1 clean FP) =', n_test_clean / (n_test_clean + n_null), ' E[#FP] =', n_test_clean / (n_null + 1))
rng = np.random.default_rng(0); N = 200_000; K = 2000
fp = np.array([(rng.standard_normal(N) > rng.standard_normal(N).max()).sum() for _ in range(K)])
print(f'simulated (N={N}, {K} reps): P(0 FP) = {(fp == 0).mean():.3f}, mean FP = {fp.mean():.3f}, P(FP>=2) = {(fp >= 2).mean():.3f}')
for k in ('v1', 'v2_oneway', 'v2_mutual', 'raw_lag1', 'rho_dcca', 'rho_q_best', 'rho_q_absmax'):
    f = d[k]['fwer']; print(k, 'clean FP', f['false_flags_clean'], 'benign FP', f['false_flags_benign'], 'true', f['true_flags'],
                            'xor recall', round(f['recall_xor_triad'], 3), 'n_pairs', d[k]['n_pairs'])
f = d['v1']['fwer']; nx = round(1145 / 1)
print('implied XOR-triad true flags for v1 ~', round(f['recall_xor_triad'] * 3 * 0.02 * 50000 * 0.98), '(~2,940 XOR pairs x recall); copy-pair ~', round(f['recall_copy_pair'] * 963))
