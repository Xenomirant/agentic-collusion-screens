"""Head-to-head in Bonjour, Aggarwal & Bhargava's own setting: iterated RPS with assistant collusion.

Question: in the task Bonjour et al. (UAI 2022) designed their net-influence method for, do the
classical cross-correlation baselines (rho_DCCA, MFDCCA rho_q) or our v2 coupling scores beat it?

Settings (experiments/sims/rps.py)
---------------------------------
(i)  3 players: 0 random, 1 assistant of 0 (plays the move 0 beats w.p. CP), 2 random.
     CP in {0.1, 0.2, 0.3}, T in {200, 500, 1000}.
(ii) 4 players: seeded pair (0, 1) as above, players 2, 3 best-respond to the public state
     (previous round's modal action) w.p. rho in {0, 0.3, 0.6} (common-state confound).
     CP in {0.1, 0.2, 0.3}, T in {500, 1000}.
Ground truth pair (0, 1). Matched null runs: CP = 0, same rho and T (players 0, 1 random).
REPS replications per cell, in SEEDS blocks (block = seed); the same for the null runs.

Methods (every method gives one score per unordered pair; top-1 = true pair strictly first)
-------
Gamma_joint_bits   Bonjour net influence, state = previous joint action (3^N values), bits
                   (the configuration that matches the paper's Table 2, E5).
Gamma_pub_bits / Gamma_pub_nats   the same with state = public state (3 values).
z_perm, z_markov   influence_z, I(a_i; a_j | public state) vs a within-state permutation /
                   per-agent Markov bootstrap null (199 surrogates).
Gamma_none_bits    Bonjour net influence with the trivial state (S = None, i.e. unconditional
                   gamma(i;j) = I(a_i; a_j)), bits: Bonjour's strongest configuration when there is no
                   common-state confound (same contingency-table information as Cramer's V).
z_none             influence_z with S = None (plain permutation null, 199 surrogates), Bonferroni.
cramer             Cramer's V (no state).
dcca[enc,clean,scales,agg,lags]   rho_DCCA (= rho_q at q = 2) between action series:
                   enc  code   = centred action code a - 1 in {-1, 0, 1} (the E6 Pearson code);
                        onehot = the 3 action indicators, pair score = max over the 3 x 3 indicator pairs;
                   clean raw | st = each series' public-state-conditional mean removed (regression on
                        the one-hot public state; the classical analogue of state conditioning);
                   scales s4_16 = (4, 8, 16) | s8_64 = (8, 16, 32, 64);
                   agg  signed = |mean_s rho(s)| | abs = mean_s |rho(s)|;
                   lags l0 | l01 (lag 1 two-sided, max over directions); channels combined by the
                        max of robust z, each lag channel scaled by the median / MAD of the matched
                        null runs (not of the test batch, which holds many colluding pairs).
rhoq[q, enc, clean]  MFDCCA rho_q (Kristoufek signed, baselines_mf.rho_q), q in {1, 4} (q = 2 is
                   dcca), scales 4-16, signed scale mean, lags 0-1.
v2[mode, enc, clean]  our detector: scale.market_lag_channels (lags 0-3, market_mean=False: markets
                   of 3-4 agents) -> coupling_scores logic (oneway: max(|C_ij|, |C_ji|); mutual:
                   |C_ij + C_ji| / sqrt 2; lag 0 |C|), robust z per lag against the null runs, max
                   over lags and the parity channel (run_e14_threats.detectors' v2). No global
                   common-mode removal (replications are independent games).

Thresholds (5%)
---------------
own:   Gamma - Bonjour's rule (pairs with min(Gamma_ij, Gamma_ji) >= 0.05);
       z     - Bonferroni, surrogate p <= 0.05 / n_pairs;
       classical and v2 - empirical null: 95% quantile of the per-run max pair score over the
             matched CP = 0 runs (family-wise over pairs; no pair is singled out as lawful).
null:  the empirical-null threshold applied to every method (uniform calibration).
det  = the true pair is the only pair flagged (Bonjour's "exactly one pair" semantics);
fp   = some other pair is flagged (CP > 0 runs); alarm0 = any pair flagged on CP = 0 runs
       (for empirical thresholds cross-fitted: threshold from the other seed blocks).
null_top1_pair01 = share of CP = 0 runs in which pair (0, 1) is ranked first (chance 1/n_pairs): a
       method whose top-1 is high here ranks the pair first for reasons other than collusion
       (e.g. Gamma with the 81-state joint action at rho = 0.6, where the two non-responders win).

Run:  cd fathon && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python ../experiments/run_bonjour_h2h_rps.py
Out:  experiments/results/bonjour_h2h/rps.json
"""
import itertools
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
sys.path.insert(0, HERE)
import rps  # noqa: E402
from fathon.tensor import influence as inf  # noqa: E402

OUT = os.path.join(HERE, 'results', 'bonjour_h2h')
REPS = int(os.environ.get('H2H_REPS', 200))
SEEDS = 4
NPERM = 199
ALPHA = 0.05
SET1 = dict(N=3, cps=[0.1, 0.2, 0.3], rhos=[0.0], Ts=[200, 500, 1000])
SET2 = dict(N=4, cps=[0.1, 0.2, 0.3], rhos=[0.0, 0.3, 0.6], Ts=[500, 1000])
TRUE = (0, 1)


def strategies(N, cp, rho):
    st = [{'kind': 'random'}, {'kind': 'assistant', 'partner': 0, 'cp': cp} if cp > 0 else {'kind': 'random'}]
    if N == 3:
        return st + [{'kind': 'random'}]
    return st + [{'kind': 'responder', 'rho': rho}, {'kind': 'responder', 'rho': rho}]


def pairvec(M, pairs, sym_min=True):
    return np.array([min(M[i, j], M[j, i]) if sym_min else M[i, j] for i, j in pairs])


def job(args):
    N, cp, rho, T, block, rep = args
    seed = 7_000_001 * (N - 2) + 100_003 * block + 1009 * rep + 10_007 * T + int(100 * cp) * 31 + int(100 * rho) * 7
    r = rps.simulate(strategies(N, cp, rho), T, seed=seed)
    A, s, J = r['actions'], r['state'], r['joint_prev']
    pairs = list(itertools.combinations(range(N), 2))
    out = {}
    out['Gamma_joint_bits'] = pairvec(inf.net_influence(inf.influence_matrix(A, S=J, base=2.0)), pairs)
    out['Gamma_pub_bits'] = pairvec(inf.net_influence(inf.influence_matrix(A, S=s, base=2.0)), pairs)
    out['Gamma_pub_nats'] = pairvec(inf.net_influence(inf.influence_matrix(A, S=s)), pairs)
    out['Gamma_none_bits'] = pairvec(inf.net_influence(inf.influence_matrix(A, S=None, base=2.0)), pairs)
    out['cramer'] = pairvec(inf.action_correlation(A), pairs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Z, P = inf.influence_z(A, S=s, nPerm=NPERM, seed=seed % 2**31, warn=False)
        Zm, Pm = inf.influence_z(A, S=s, nPerm=NPERM, seed=seed % 2**31 + 1, null='markov')
        Zn, Pn = inf.influence_z(A, S=None, nPerm=NPERM, seed=seed % 2**31 + 2, warn=False)
    out['z_perm'], out['p_perm'] = pairvec(Z, pairs), pairvec(P, pairs, False)
    out['z_markov'], out['p_markov'] = pairvec(Zm, pairs), pairvec(Pm, pairs, False)
    out['z_none'], out['p_none'] = pairvec(Zn, pairs), pairvec(Pn, pairs, False)
    return dict(N=N, cp=cp, rho=rho, T=T, block=block, rep=rep, A=A.astype(np.int8), S=s.astype(np.int8), **out)


# ---------------------------------------------------------------------------------------------
# GPU: classical and v2 scores for a batch of runs (same N, T)
# ---------------------------------------------------------------------------------------------

def encode(A, S, enc, clean):
    """A (M, N, T) int, S (M, T) -> X (M, n, T) float64 torch; n = N (code) or 3N (onehot)."""
    import torch
    A = torch.as_tensor(A, dtype=torch.long, device='cuda')
    if enc == 'code':
        X = (A - 1).double()
    else:
        X = torch.nn.functional.one_hot(A, 3).permute(0, 1, 3, 2).reshape(A.shape[0], -1, A.shape[2]).double()
    if clean == 'st':
        S = torch.as_tensor(S, dtype=torch.long, device='cuda')
        D = torch.nn.functional.one_hot(S, 3).double()                  # (M, T, 3)
        cnt = D.sum(1).clamp_min(1.0)                                     # (M, 3)
        mu = (X @ D) / cnt[:, None, :]                                   # (M, n, 3) state-conditional means
        X = X - torch.gather(mu, 2, S[:, None, :].expand(-1, X.shape[1], -1))
    return X


def agent_pairs(Sser, N, enc, directed):
    """Series-level scores (M, n, n) -> agent pair scores (M, n_pairs); onehot: max over 3 x 3."""
    import torch
    pairs = list(itertools.combinations(range(N), 2))
    k = 1 if enc == 'code' else 3
    out = []
    for a, b in pairs:
        blk = Sser[:, a * k:(a + 1) * k, b * k:(b + 1) * k]
        if directed:
            blk = torch.maximum(blk, Sser[:, b * k:(b + 1) * k, a * k:(a + 1) * k].transpose(1, 2))
        out.append(blk.reshape(blk.shape[0], -1).max(1).values)
    return torch.stack(out, 1)


def lag_rho(X, lag, scales, qs):
    """rho_q(s) between x_i(t) (later) and x_j(t - lag): (nq, ns, M, n, n)."""
    import torch
    import baselines_mf as bmf
    M, n, T = X.shape
    if lag == 0:
        return bmf.rho_q(X.reshape(M * n, T), n, scales=scales, qs=qs, agg='none')
    Z = torch.cat([X[:, :, lag:], X[:, :, :-lag]], 1)
    R = bmf.rho_q(Z.reshape(M * 2 * n, T - lag), 2 * n, scales=scales, qs=qs, agg='none')
    return R[..., :n, n:]


def nz(v, null_rows):
    """Robust z with median / MAD of the null runs' pair scores."""
    ref = v[null_rows].reshape(-1)
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826
    return (v - med) / (mad + 1e-12)


def gpu_scores(A, S, N, null_rows):
    import torch
    from fathon.tensor import scale
    null_rows = torch.as_tensor(null_rows, device='cuda')
    out = {}
    SC = {'s4_16': (4, 8, 16), 's8_64': (8, 16, 32, 64)}
    for enc in ('code', 'onehot'):
        for clean in ('raw', 'st'):
            X = encode(A, S, enc, clean)
            M, n, T = X.shape
            # rho_DCCA (q = 2) and rho_q (q = 1, 4)
            for sc, scales in SC.items():
                qs = (1, 2, 4) if sc == 's4_16' else (2,)
                per_lag = {lag: lag_rho(X, lag, scales, qs) for lag in (0, 1)}
                for iq, q in enumerate(qs):
                    aggs = ('signed', 'abs') if q == 2 else ('signed',)
                    for agg in aggs:
                        ch = {}
                        for lag, R in per_lag.items():
                            Rq = R[iq]
                            Sser = Rq.mean(0).abs() if agg == 'signed' else Rq.abs().mean(0)
                            ch[lag] = nz(agent_pairs(Sser, N, enc, directed=lag > 0), null_rows)
                        tag = f'{enc},{clean},{sc},{agg}'
                        if q == 2:
                            out[f'dcca[{tag},l0]'] = ch[0]
                            out[f'dcca[{tag},l01]'] = torch.maximum(ch[0], ch[1])
                        elif sc == 's4_16':
                            out[f'rhoq{q}[{enc},{clean},l01]'] = torch.maximum(ch[0], ch[1])
            # v2: multi-lag two-sided coupling + parity, our detector
            Xf = X.reshape(M * n, T).float()
            C = scale.market_lag_channels(Xf, n, lags=(0, 1, 2, 3), market_mean=False)
            par = scale.market_pair_scores(Xf, n, lag=1, parity=True, market_mean=False)['parity'].double()
            zp = nz(agent_pairs(par, N, enc, directed=False), null_rows)
            for mode in ('oneway', 'mutual'):
                best = None
                for lag, Cl in C.items():
                    Cl = Cl.double()
                    if lag == 0:
                        Sser, d = Cl.abs(), False
                    elif mode == 'oneway':
                        Sser, d = Cl.abs(), True                   # max(|C_ij|, |C_ji|) via directed
                    else:
                        k = 1 if enc == 'code' else 3
                        # mutual per indicator pair: |C[a c, b d] + C[b d, a c]| / sqrt 2
                        Sser = None
                        pairs = list(itertools.combinations(range(N), 2))
                        vals = []
                        for a, b in pairs:
                            blk = Cl[:, a * k:(a + 1) * k, b * k:(b + 1) * k] + \
                                Cl[:, b * k:(b + 1) * k, a * k:(a + 1) * k].transpose(1, 2)
                            vals.append((blk.abs() * 0.7071067811865476).reshape(M, -1).max(1).values)
                        z = nz(torch.stack(vals, 1), null_rows)
                        best = z if best is None else torch.maximum(best, z)
                        continue
                    z = nz(agent_pairs(Sser, N, enc, directed=d), null_rows)
                    best = z if best is None else torch.maximum(best, z)
                out[f'v2[{mode},{enc},{clean}]'] = torch.maximum(best, zp)
    return {k: v.float().cpu().numpy().astype(np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------

def evaluate(scores, flags_own, meta, N):
    """scores: method -> (R, P) array; flags_own: method -> (R, P) bool or None; meta: list of dicts."""
    pairs = list(itertools.combinations(range(N), 2))
    ti = pairs.index(TRUE)
    cp = np.array([m['cp'] for m in meta])
    rho = np.array([m['rho'] for m in meta])
    T = np.array([m['T'] for m in meta])
    blk = np.array([m['block'] for m in meta])
    res = {}
    for meth, Sc in scores.items():
        others = np.delete(Sc, ti, 1)
        top1 = Sc[:, ti] > others.max(1)
        conf_top = None
        if N == 4:
            ci = pairs.index((2, 3))
            conf_top = Sc[:, ci] >= np.delete(Sc, ci, 1).max(1)
        rows = []
        for r_ in sorted(set(rho.tolist())):
            for t_ in sorted(set(T.tolist())):
                g0 = (cp == 0) & (rho == r_) & (T == t_)
                mx0 = Sc[g0].max(1)
                b0 = blk[g0]
                thr_all = np.quantile(mx0, 0.95)
                # cross-fitted null alarm of the empirical threshold
                alarm0_emp = np.mean([np.mean(mx0[b0 == b] > np.quantile(mx0[b0 != b], 0.95)) for b in range(SEEDS)])
                fo = flags_own.get(meth)
                alarm0_own = None if fo is None else float(fo[g0].any(1).mean())
                null_top1 = float(top1[g0].mean())      # pair (0, 1) ranked first although CP = 0 (chance 1/n_pairs)
                for c_ in sorted(set(cp.tolist()) - {0.0}):
                    g = (cp == c_) & (rho == r_) & (T == t_)
                    rec = dict(cp=c_, rho=r_, T=t_, n=int(g.sum()))
                    bb = blk[g]

                    def by_block(v):
                        per = [float(np.mean(v[bb == b])) for b in range(SEEDS) if np.any(bb == b)]
                        return float(np.mean(v)), float(np.std(per, ddof=1)) if len(per) > 1 else 0.0, per
                    rec['top1'], rec['top1_sd_seeds'], rec['top1_seeds'] = by_block(top1[g])
                    if conf_top is not None:
                        rec['conf_top'] = float(conf_top[g].mean())
                    fl = Sc[g] > thr_all
                    det = fl[:, ti] & ~np.delete(fl, ti, 1).any(1)
                    rec['det_null'], rec['det_null_sd_seeds'], _ = by_block(det)
                    rec['hit_null'] = float(fl[:, ti].mean())
                    rec['fp_null'] = float(np.delete(fl, ti, 1).any(1).mean())
                    rec['alarm0_null'] = float(alarm0_emp)
                    rec['thr_null'] = float(thr_all)
                    rec['null_top1_pair01'] = null_top1
                    if fo is None:                       # own rule = empirical null threshold
                        rec['det_own'], rec['det_own_sd_seeds'] = rec['det_null'], rec['det_null_sd_seeds']
                        rec['fp_own'], rec['alarm0_own'] = rec['fp_null'], rec['alarm0_null']
                    else:
                        f2 = fo[g]
                        d2 = f2[:, ti] & ~np.delete(f2, ti, 1).any(1)
                        rec['det_own'], rec['det_own_sd_seeds'], _ = by_block(d2)
                        rec['fp_own'] = float(np.delete(f2, ti, 1).any(1).mean())
                        rec['alarm0_own'] = alarm0_own
                    rows.append(rec)
        res[meth] = rows
    return res


def run_setting(cfg, pool, name):
    N = cfg['N']
    jobs = []
    for rho in cfg['rhos']:
        for T in cfg['Ts']:
            for cp in [0.0] + cfg['cps']:
                for b in range(SEEDS):
                    for rep in range(REPS // SEEDS):
                        jobs.append((N, cp, rho, T, b, rep))
    t0 = time.time()
    runs = pool.map(job, jobs, chunksize=8)
    print(f'[{name}] {len(runs)} runs simulated + info stats in {time.time() - t0:.0f} s', flush=True)
    info = ['Gamma_joint_bits', 'Gamma_pub_bits', 'Gamma_pub_nats', 'Gamma_none_bits', 'cramer',
            'z_perm', 'z_markov', 'z_none']
    scores = {k: np.zeros((len(runs), N * (N - 1) // 2)) for k in info}
    for k in info:
        scores[k] = np.stack([r[k] for r in runs])
    flags = {
        'Gamma_joint_bits': scores['Gamma_joint_bits'] >= ALPHA,
        'Gamma_pub_bits': scores['Gamma_pub_bits'] >= ALPHA,
        'Gamma_pub_nats': scores['Gamma_pub_nats'] >= ALPHA,
        'Gamma_none_bits': scores['Gamma_none_bits'] >= ALPHA,
        'z_perm': np.stack([r['p_perm'] for r in runs]) <= ALPHA / (N * (N - 1) // 2),
        'z_markov': np.stack([r['p_markov'] for r in runs]) <= ALPHA / (N * (N - 1) // 2),
        'z_none': np.stack([r['p_none'] for r in runs]) <= ALPHA / (N * (N - 1) // 2),
    }
    meta = [{k: r[k] for k in ('cp', 'rho', 'T', 'block', 'rep')} for r in runs]
    # GPU scores, grouped by (rho, T) so that each group carries its own null rows
    t0 = time.time()
    gs = {}
    idx_all = np.arange(len(runs))
    for rho in cfg['rhos']:
        for T in cfg['Ts']:
            idx = idx_all[[(m['rho'] == rho and m['T'] == T) for m in meta]]
            A = np.stack([runs[i]['A'] for i in idx])
            S = np.stack([runs[i]['S'] for i in idx])
            null_rows = np.flatnonzero([meta[i]['cp'] == 0 for i in idx])
            d = gpu_scores(A, S, N, null_rows)
            for k, v in d.items():
                gs.setdefault(k, np.zeros((len(runs), v.shape[1])))[idx] = v
    scores.update(gs)
    print(f'[{name}] GPU classical / v2 scores in {time.time() - t0:.0f} s', flush=True)
    return evaluate(scores, flags, meta, N)


def main():
    import torch
    torch.set_num_threads(8)
    os.makedirs(OUT, exist_ok=True)
    res = dict(settings=dict(reps=REPS, seed_blocks=SEEDS, n_perm=NPERM, alpha=ALPHA, set1=SET1, set2=SET2,
                             true_pair=TRUE, doc=__doc__))
    with Pool(12) as pool:
        res['three_player'] = run_setting(SET1, pool, '3p')
        res['four_player'] = run_setting(SET2, pool, '4p')
    with open(os.path.join(OUT, 'rps.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print('saved', os.path.join(OUT, 'rps.json'))


if __name__ == '__main__':
    main()
