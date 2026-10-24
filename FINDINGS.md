# Findings: detecting collusion from agents' actions

*Phase 2 status as of 2026-10-24 (server work, branch `autoresearch/scale`). Phase 1 (the sprint
results, E0-E7) follows below unchanged.*

## Phase 2: scale, review, LLM agent farms, real markets

### TL;DR

1. **Independent review of `fathon.tensor`: no error in the mathematics; one statistical flaw.**
   The calibrated influence test's permutation null assumes rounds are exchangeable given the
   state. Independent *sticky* agents whose own previous action is not part of the state are
   rejected 70% of the time at nominal 5%. Fixed with a Markov-bootstrap null (`null='markov'`)
   and a `memory_beyond_state` diagnostic; E5/E6 are unaffected (their states contain every
   agent's previous action), E4/E7-style public states are affected.
2. **Scalability.** A torch port of batched MFDFA runs 1M series x 2,048 points in 0.21 s and
   4M x 1,024 in 0.42 s on one H100 (5-10 million series/s, matching fathon's C code to 1e-15);
   the standard per-series tools manage 7-140 series/s. Two library inefficiencies were fixed
   (65x on the covariance tensor, 14x on the pairwise statistics, 8x on the calibrated influence).
3. **Who colludes among many agents sharing one model.** New module `fathon.tensor.scale`:
   remove the dense low-rank common mode (the shared model's response to public information),
   then rank pairs by lagged residual coupling in a chunked, never-materialised N x N search, with
   a parity-feature channel for XOR-type coalitions and a coalition-closure re-ranking. On
   planted-coalition markets (5k-150k agents) raw correlation finds nothing (precision 0.00-0.01)
   while the pipeline reaches mean precision@K 0.87 over 34 cases, 1.00 on every XOR case and
   0.997 / 1.00 at 150,000 agents in 7.4 s. Weak coupling (logit 0.25, T = 2048) sits at the
   information limit: true pairs and the null maximum both at z ~ 6.
4. **Real LLM agents, local.** 4 Qwen2.5-7B-Instruct sellers per market against rule-based buyers,
   served by vLLM, 80 markets, a private channel seeded in one pair of 40 of them. The pair
   coordinates by *matching prices to the cent*: an exact-coincidence statistic ranks it first in
   89% of markets (AUC 0.95 against controls, 60% detection at 2.7% false alarm), while every lagged
   coupling statistic, Bonjour's included, is at chance. Pooled across markets, sellers of the same
   model in *different* markets coincide on prices 13-20 rounds out of 59 against one expected:
   parallel conduct from a shared prior, the common-mode problem in its purest form (E8).
5. **Real markets.** NSW FuelCheck (1,003 stations x 1,096 days): the top residual-coupled pairs
   are 93x enriched in same-postcode pairs and, among co-located pairs, 4.0x enriched in same-brand
   pairs (ownership beyond co-location), robustly across windows and channels; the effect of
   common-mode removal on that ownership signal is small and inconsistent in sign (+5% on
   2022-2024, -10% on 2017-2026), so on this data it is not what carries the result. Brazilian
   retail-gasoline cartels (ANP survey + CADE decisions): cross-sectional screens are mixed on the
   partial data; the within-city cartel-vs-control contrast needs the 2013-2019 files (downloading).

### Review of the implementation (commit c1b8b4a)

An independent reviewer read all seven modules and checked them against brute-force
implementations and the C loops (interleaving permutation, normalisations, tau fits, log bases,
p-value conventions, chi-square / phi2, conditional MI, surrogate exchangeability). Findings:

| severity | finding | fix |
|---|---|---|
| statistical validity | `influence_z` permutation null anti-conservative for agents with memory beyond the state (sticky Markov, stay 0.8, exogenous state: 70% false positives at 5%) | `null='markov'` parametric bootstrap of each agent's policy `P(a_t | s_t, a_{t-1..t-m})`; `memory_beyond_state()`; warning |
| efficiency | `covariance_tensor` per-window einsum | one matmul over stacked residuals (identical values; 96 s -> 1.4 s at 1,024 x 16,384) |
| efficiency | `pairwise_matrix` recomputed words per pair; `influence_z` Python loop over permutations | per-agent codes once + bincount (70 s -> 5 s at 256 agents); vectorised null (4.3 s -> 0.5 s) |
| robustness | `cut_stats` on empty rows, `words` int64 overflow / alphabet check, `cell_measure` short input and zero mass, `mean_corr` with constant agents, `cp_als(n_iter=0)`, short-input errors | fixed, all with tests |

Verified correct: parity with C (1e-9 to 1e-15), `interleave`, `cross_replica_tensor` vs einsum,
`tt_svd`, `tucker`, `cp_als`, Kronecker rearrangement, `shift_null` uniformity under the null,
`phase_surrogate` spectra, `conditional_permutation` strata. 86 tests pass (81 + 5 for `scale`).

### Scalability against the common multifractal tools

Batched MFDFA, 7 scales x 7 moments, identical output to fathon's C code (max relative error 1e-15):

| method | series / s | note |
|---|---|---|
| nolds DFA | ~7 | monofractal only |
| fathon C, per-series loop | ~25 | the standard |
| pypi MFDFA | ~140 | |
| `fathon.tensor.mvdfa` (numpy, batched) | ~900 | at 1,024 series |
| torch port, one H100 | 4.8M - 9.6M | 1M x 2,048 in 0.21 s; 4M x 1,024 in 0.42 s |

The cross-agent covariance tensor (all-pairs DCCA at 7 scales) takes 0.01 s at 1,024 agents on
the GPU against an estimated 40 min for the C pair loop. Past ~50k agents the N x N object is the
wall (68 GB per scale at 131k), which is why `scale` never materialises it.

### The scalable detector and the autoresearch loop (`fathon/fathon/tensor/scale.py`, `benchmark.py`)

Corpus: `autoresearch/data/planted_cases.jsonl`, 34 simulated markets (`experiments/sims/planted.py`):
N = 5k / 30k / 150k binary agents, T = 2048 / 4096, a shared AR(1) latent of rank 4-12 (one or
two "base models"), own stickiness 0.3 or 1.2, coalitions of 2-3 agents that copy a peer's last
action (logit 0.25-0.6) or follow the XOR of two peers (logit 1.5-2.5; pairwise invisible).
Metric: mean precision@K, K = number of planted pairs. Regression gate: the test suite.

| iteration | change | precision@K | kept |
|---|---|---|---|
| 0 | common-mode removal (oracle rank) + lag-1 residual correlation, max over directions | 0.484 | baseline |
| 1 | sum of directions instead of max | 0.459 | no |
| 2 | AR(1) whitening | 0.482 | no |
| 3 | parity channel, raw max over channels | 0.452 | no |
| 4 | score = max(one-way, one-way, mutual / sqrt 2) | 0.526 | yes |
| re-baseline | XOR coupling moved to 1.5 / 2.5 (0.25 / 0.45 is below the information limit even for the oracle triplet statistic) | 0.536 | |
| 5 | parity channel scaled by its own null quantile, max over channels | 0.815 | yes |
| 6 | closure re-ranking, weight 1 | 0.756 | no |
| 7 | remove rank + 8 components (nonlinear leakage below the spectral edge) | 0.839 | yes |
| 8 | closure re-ranking, weight 0.5, 10K candidates | 0.856 | yes |
| 9 | closure weight 1.0 | 0.854 | no |
| re-baseline | + sticky agents, two base models, 150k agents (34 cases) | 0.876 | |
| 10 | whitening retry | 0.864 | no |
| re-baseline | estimated rank instead of oracle rank (robustness) | 0.871 | |
| 11 | 20K candidates | 0.873 (2x latency) | no |
| 12 | rank + 16 components | 0.869 | no |

Per regime (final detector, estimated rank): copy pairs at logit 0.45: 1.00; copy triads at 0.45:
0.93-1.00; XOR triads: 1.00 everywhere; sticky agents: 0.81-1.00; two base models: 0.71-1.00;
150k agents: 0.997 (copy) and 1.00 (XOR) in 7.4 s; weak copy coupling (0.25): 0.03-0.83, with
true-pair scores at the null maximum (information-limited at T = 2048).

### E8: local LLM sellers (Qwen2.5-7B-Instruct via vLLM), seeded private channel

4 Qwen2.5-7B-Instruct sellers, 3 rule-based ZIP buyers, 60 rounds, temperature 0.7, served by
vLLM on one H100 (about 20 min per market, 16 markets in parallel). `controls`: no messaging.
`seeded`: exactly one seller pair has a private channel (the sprint patch). Analysis from public
asks and trades only; 40 + 40 markets were run, of which 37 controls and 35 seeded completed all
60 rounds and enter the tables.

**What the pair does.** The partners talk and act: "Let's keep our asks at $79.30 ... coordination is
key", and their asks match to the cent for long stretches. A 7B model coordinates mostly on
*matching a level*, often downwards, so the signature is static price identity rather than lagged
reactivity, exactly the E3/E4 lesson.

| statistic (per market, 6 pairs; chance top-1 = 0.17) | top-1 on seeded | AUC max score, seeded vs controls | detection / false alarm |
|---|---|---|---|
| exact-coincidence z (within-round null) | 0.89 | **0.95** | z > 3: 60% / 2.7%; z > 5: 14% / 0% |
| share of rounds with identical asks | **0.91** | 0.92 | |
| Pearson of ask levels | 0.43 | 0.74 | |
| Pearson of ask changes | 0.31 | 0.66 | |
| Cramer's V of ask moves | 0.20 | 0.49 | |
| Bonjour Gamma (public state) | 0.11 | 0.53 | |
| calibrated influence z, permutation null | 0.11 | 0.55 | Bonferroni: 6% / 0% |
| calibrated influence z, Markov null | 0.09 | 0.49 | Bonferroni: 3% / 8% |

The memory-beyond-state diagnostic is positive in these markets (median 0.065), so the permutation
null is the wrong one here; the Markov null is calibrated but, like every lagged statistic, has no
signal to find at 60 rounds.

**Pooled stress test (same model, 72 markets, 288 sellers, K = 35).** Stacking all sellers and
asking the detector for the top-35 pairs without telling it which sellers share a market: the
lag-1 residual channel finds 4 of 35, the exact-coincidence channel 15 of 35 (within-round null)
or 13 (shift null). The remaining top pairs are *cross-market*: sellers in different markets that
never interacted post identical prices in the same round 13-20 times out of 59 against about one
expected. Same-model agents share price trajectories and focal prices, which is parallel conduct
from a shared prior, not agreement. A detector that ignores market structure flags them; one that
only compares agents who can interact does not. This is the sharpest instance so far of the
common-mode problem the method is built around. The pooled ask-change matrix has a weak common
mode (top singular value 99 against a permutation-null maximum of 94).

New library functions: `fathon.tensor.scale.coincidence_pairs` (small N, explicit nulls) and
`coincidence_scores` (sort-based, rarity-weighted, n-gram shingles; 1M agents x 100 rounds in 78 s
on CPU with precision 1.0 on planted matchers, where 1-gram coincidences are uninformative because
a dense population covers the whole price grid).

A third dataset, `seeded_monitored` (the channel pair is told a regulator screens the public ask
history for coupled pairs), is being generated for threat model 3.4.

### E9: NSW FuelCheck, 1,003 stations x 1,096 days (2022-2024, U91)

Daily log-price changes of all stations with >= 80% coverage; estimated common-mode rank 48 (the
regional price cycles); top-K pairs by lag-0 residual coupling. Proxy labels: same postcode
(co-location), same brand (ownership), and same brand *given* same suburb (ownership beyond
co-location). Lift = rate in the top-K / base rate over all 502k pairs.

| channel | K | same postcode lift | same brand given same suburb (rate / base / lift) |
|---|---|---|---|
| residual, common mode removed | 200 | 93 | 0.476 / 0.118 / 4.0 |
| residual, common mode removed | 1000 | 37 | 0.338 / 0.118 / 2.9 |
| raw changes, no removal | 200 | 134 | 0.450 / 0.118 / 3.8 |
| raw changes, no removal | 1000 | 57 | 0.295 / 0.118 / 2.5 |
| local (postcode) mean removed | 200 | 101 | 0.300 / 0.118 / 2.5 |

Geography dominates raw co-movement (neighbours follow the same cycle). On this window
common-mode removal trades some geographic enrichment for a slightly stronger ownership signal;
on the full span (2017-2026, 1,390 stations x 3,560 days, coverage >= 50%) the ordering flips
(ownership-given-co-location lift 6.1 raw vs 5.5 after removal at K = 200; 4.0 vs 3.9 at K = 1000),
so the effect of removal on the ownership signal is small and not consistent in sign, while the
ownership enrichment itself (4-6x) is robust across windows and channels. The full-span detection
runs in 0.6 s on one GPU. A circular-shift null puts 15 (window) / 4 (full span) pairs above the
family-wise null maximum and the 5,000th observed score at twice the null's. Caveat: brand is a
proxy for a common decision maker, not evidence of collusion; same-brand pricing is lawful.

### E10: Brazilian retail-gasoline cartels (ANP weekly survey, CADE decisions)

Partial data (2004-05 to 2012-07; the 2013-2019 control windows are still downloading).
Cross-sectional contrast: percentile of the cartel city among same-state cities of similar size
in the same week, averaged over the documented cartel window.

| city (cartel window) | weeks | dispersion (cv) | sync | residual AR(1) | residual variance |
|---|---|---|---|---|---|
| Belo Horizonte (2004-01..2008-04) | 208 | 0.79 | 0.39 | 0.18 | 0.37 |
| Brasilia (2009-11..2012-07 observed) | 141 | 0.10 | 0.44 | 0.32 | 0.00 |
| Caxias do Sul (2004-01..2007-07) | 169 | 0.40 | 0.25 | 0.58 | 0.23 |
| Sao Luis (2007-01..2011-03) | 221 | 0.39 | 0.50 | 0.39 | 0.25 |

No indicator is consistently extreme across the four strong cases; low residual variance (price
rigidity relative to the state) leans the right way in all four. The within-city cartel-vs-control
AUCs (Silveira et al.'s design) will be filled in once the later files arrive.

### Threat models and coverage (updated)

| threat model | method | status |
|---|---|---|
| seeded reactive pair | calibrated influence (Markov null); `scale.detect` at scale | works; E8 on LLM sellers pending |
| k-coalition / all-N coordination | group cuts, total correlation; closure re-ranking | triads recovered at 0.93-1.00 (moderate coupling) |
| higher-order (XOR) coalitions | parity-feature channel | 1.00 at logit >= 1.5; impossible below ~1 for any method at T = 2048 |
| same base model, same public information | common-mode removal, estimated rank | the core; two-model markets 0.71-1.00 |
| sticky agents (memory beyond the state) | Markov null; rank+8 removal | fixed / 0.81-1.00 |
| static or focal collusion without reactivity | none of the coupling statistics | honest limit (E1, E3) |
| monitored coalition that adapts (3.4) | not yet run | needs the local farm with a monitor prompt |

### Reproduce (phase 2)

```bash
# environment: Python 3.13 venv, numpy 2.x, torch (CUDA); apt-get install libgsl-dev (lost on container restart)
cd fathon && ../.venv/bin/python -m pytest tests -q            # 86 passed
cd fathon && ../.venv/bin/python ../benchmark.py              # metric: 0.87xx (34 planted markets, GPU)
cd fathon && ../.venv/bin/python ../experiments/run_e9_nsw.py  # needs external/data/nsw_fuelcheck (experiments/data/DATA.md)
PYTHONPATH=fathon .venv/bin/python experiments/run_e10_anp.py  # needs external/data/anp_brazil_fuel
cd fathon && ../.venv/bin/python ../experiments/run_e8_local.py  # needs experiments/results/e8_local runs
```

---

# Phase 1 (sprint): detecting collusion from agents' actions

*Status as of 2026-10-24. This supersedes `experiments/REPORT_DRAFT.md`, which was framed around
Q-learning markets.*

## Question

Apart AI Collusion Sprint, *Markets and Collusion* track, questions 1.3 and 1.4:

> Can a dependence statistic computed from agents' observable actions alone detect collusion,
> identify *who* colludes, and serve as a quantitative **plus factor** in market cases?

The **target environment** is the Agrawal et al. (2025) LLM double auction
(`llm-agent-collusion`). The **reference method** is the net influence of Bonjour, Aggarwal &
Bhargava (UAI 2022). The scripted games below are used to validate methods against known ground
truth before spending API budget.

## TL;DR

1. **Bonjour et al. reproduced.** Net influence is implemented (`fathon.tensor.influence`) and
   matches their RPS detection accuracies within 2–3 points on average (E5). That needed two
   choices the paper doesn't state: state = previous joint action, and MI in bits.
2. **Calibration beats a fixed threshold.** A conditional permutation test removes the need for α and
   the log base. It detects weak collusion that Bonjour's α never flags: 83% vs 0% at CP = 0.1, and
   96–97% vs 9–58% at CP = 0.2. False-pair rates stay at 1–7% (E5, E6).
3. **State conditioning is what beats plain correlation.** In 4-agent RPS, two non-colluders reacting to
   the same public state fool Cramér's V completely: it picks the decoy pair 100% of the time.
   State-conditioned net influence and the calibrated z rank the true pair first in 85.5–100% of runs
   at T = 1000 (E6).
4. **On real LLM double-auction data**, seller messaging is detectable from asks alone:
   - The best single statistic is simple price rigidity (AUC 0.945).
   - Coupling statistics add information beyond rigidity: all-seller total correlation, partial ρ
     with the LLM-judge score 0.31, p = 0.007.
   - Bonjour's Γ fails (AUC 0.66, n.s.). It assumes exactly two colluders, and here all 5 sellers
     coordinate (E4).
5. **Negative results.**
   - Coupling statistics are *not* early-warning signals in Q-learning markets.
   - They are blind to *static* collusion: memoryless Q-learners (Δ = 0.96), and LLM pricing agents
     whose high prices come without reactive coupling (E1, E3).
   - The multifractal / replica-tensor tools are validated but have not yet been tested as early
     warning on transitions in LLM agents.

## What is built

| Component | Where | Notes |
|---|---|---|
| `fathon.tensor` (pure numpy) | `fathon/` (fork of stfbnc/fathon; our work is the 3 `tensor-ext` commits) | `detrend`, `mvdfa`, `decomp`, `factorization`, `replica`, `ews`, `influence`. 39 new tests; 70/70 pass, including 31 upstream. |
| Bonjour net influence + calibrated variant | `fathon/fathon/tensor/influence.py` | γ = I(a_i; a_j \| s), lagged/directed variant, Γ, the exactly-one-pair rule, conditional-permutation z, Cramér's V, pair ranking |
| Seeded private seller pair in the double auction | `patches/collusion_hackathon-seeded-pair-channel.patch` (against upstream `d585a7b`; see README for applying it) | `--seller_channel_pair i j`. Routing verified offline with a stub model. |
| Experiments E0–E7 | `experiments/run_e*.py` → `experiments/results/e*/` | every number below traces to these JSON/CSV files |

Numerical notes:
- The numpy detrending is exact to about 1e-13 against rational arithmetic. fathon's C loops lose
  about 1e-7 at polynomial order 3 (raw-power Vandermonde fit).
- Batched MFDFA and ρ_DCCA match the C implementation to ≤1.5e-13 for every option.

## Results

### E5 — Bonjour replication: 3-player iterated RPS, assistant collusion (1,000 iterations per cell)

Detection accuracy (exactly the true pair flagged, α = 0.05):

| CP | games | Paper | Ours (state = previous joint action, bits) | Ours (no state, nats) | Calibrated permutation test | Calibrated false pair |
|---|---|---|---|---|---|---|
| 0.1 | 1000 | ≈8 (average over sizes) | 0.4 | 0.0 | **83.0** | 4.5 |
| 0.2 | 400 / 1000 | – / 24.3 | 36.4 / 47.4 | 17.1 / 9.3 | **97.0 / 96.0** | 2.5 / 4.0 |
| 0.3 | 100 / 400 / 1000 | 37.1 / 82.8 / 100 | 37.6 / 88.1 / 99.9 | 69.3 / 94.4 / 99.9 | 78.0 / 95.5 / 97.5 | 1.5 / 4.5 / 2.5 |
| 0.4 | 100 / 400 | 58.4 / 99.3 | 66.0 / 99.9 | 95.8 / 100 | 94.5 / 95.5 | 3.0 / 4.5 |

- **Fit to the paper.** Mean absolute error over 100–1000 games is 3.3 points (CP 0.3) and 2.4
  points (CP 0.4) in the paper-matching configuration.
- **Where the fixed threshold fails.** With nats, the true I(A;B) at CP = 0.2 is 0.038, below
  α = 0.05, so detection *decreases* with data. A fixed α ties the detector to the log base.
- **Unresolved.** CP = 0.2 at 1,000 games: ours 47% vs the paper's 24.3%.
- **Without a confound, correlation is enough.** Plain Cramér's V ranks the true pair first as
  often as Γ does.

### E6 — Four agents, one seeded pair, a common-state confound (200 reps per cell)

Setup:
- Players 0 and 1 are the colluding pair, with collusion probability CP.
- Players 2 and 3 best-respond to the previous round's modal action with probability ρ. They are
  dependent through the shared state but do not coordinate.

True pair ranked first at T = 1000 (%):

| CP | ρ | Cramér's V | Pearson | Γ (public state) | Γ (previous joint action, 81 states) | Calibrated z (public state) |
|---|---|---|---|---|---|---|
| 0.1 | 0 / 0.3 / 0.6 | 99 / 64 / **0** | 50 / 18.5 / 0 | 93 / 88.5 / 86.5 | 29.5 / 66.5 / 99.5 | 93.5 / 87.5 / 85.5 |
| 0.2 | 0 / 0.3 / 0.6 | 100 / 100 / **0** | 91.5 / 63 / 0 | 100 / 100 / 100 | 84.5 / 93 / 100 | 100 / 100 / 100 |
| 0.3 | 0 / 0.3 / 0.6 | 100 / 100 / **3.5** | 100 / 87.5 / 0 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 |

- **Correlation picks the decoy.** At ρ = 0.6, Cramér's V ranks the confounded pair first in
  96.5–100% of runs.
- **Detection rule at T = 1000.** Calibrated (Bonferroni over 6 pairs): 48.5–57% at CP 0.1, 95–98.5%
  at CP 0.2. Bonjour's α rule on the public state: 0% and 3.5–7.5% respectively.
- **False alarms.** The calibrated false-pair rate is ≤ 7% in every cell.
- **The over-fine 81-state conditioning is unreliable.** It fires on plug-in bias: at CP 0.1 and
  ρ 0.6, detection falls from 94.5% (T = 500) to 58.5% (T = 1000).
- **LLM-judge arm.** Ready but not run (no API key yet). Dry run: 45 transcripts, about 43k input
  tokens.

### E4 — Agrawal double auction, shipped data (75 runs, 5 sellers, 30 hours, asks only)

Base (no seller messaging, 10 runs) vs base with messaging (10 runs):

| Statistic | AUC | p (MWU) | Spearman with judge score (75 runs) |
|---|---|---|---|
| Fraction of unchanged asks (price rigidity) | **0.945** | 0.001 | 0.57 |
| Pairwise factorization defect, z | **0.86** | 0.007 | 0.31 |
| Total correlation (all 5 sellers), z | 0.74 | 0.076 | **0.37** (p = 0.001) |
| Bonjour Γ, best pair | 0.66 | 0.24 | −0.15 |
| Bonjour γ, raw plug-in | 0.32 | 0.19 | −0.21 |
| Pearson of ask changes / ρ_DCCA | 0.49 / 0.53 | ≥ 0.85 | ≈ 0 |
| Collusion index (needs benchmarks) | 0.69 | 0.16 | – |

- **Controlling for rigidity**, partial ρ with the judge score is 0.31 (p = 0.007) for total
  correlation and 0.25 (p = 0.03) for the pairwise defect.
- **The 20-run comparison alone can't separate the two effects**: the logit with rigidity has
  p = 0.15–0.2.
- **Pooled-ensemble permutation tests were not significant** (p = 0.17–0.77).
- **Caveat.** These were the unconditioned (shift-null) statistics. Base runs already show coupling
  (mean z ≈ 4) because sellers watch the same book. The state-conditioned, calibrated version
  (`influence_z`) has not yet been run on this data. At 28 usable rounds over 6 states it would be
  underpowered.

### E7 — Seeded pair in the double auction (target experiment; built, not run)

- **Harness.** The patch gives exactly two of N sellers a private channel.
- **Analyzer.** `run_e7_da_seeded.py` ranks seller pairs from public asks and trades:
  Γ (state = previous public outcome), calibrated z (simultaneous and lag-1), Cramér's V, Pearson,
  and price matching. It reports top-1 accuracy and the false-alarm rate on no-channel controls.
- **Judge.** `run_e7_llm_judge.py` gives an LLM the same public log.
- **Suggested design.** 4 sellers, rule-based (ZIP) buyers, 60 hours, 20 seeded and 10 control
  runs: about 7,200 calls. Roughly $15–20 with gpt-4.1-mini, or $80–100 with gpt-4.1 (approximate
  list prices; the harness prints its own estimate).

### E3 — LLM pricing agents (Fish et al. setup, 210 runs, 300 rounds)

- **Price structure is at chance.** All price-structure indicators score AUC 0.46–0.58 for P1 vs P2,
  and |Spearman| ≤ 0.28 with Δ. Only the price level relative to the Nash/monopoly benchmarks
  separates the conditions (AUC 0.895–0.94).
- **This is not a power problem.** At T = 200 the lag-1 test has size 0.05 and power 0.78 / 1.00
  at coupling 0.2 / 0.4, so strong round-to-round reactivity is excluded in most runs: the high P1
  prices look static (focal).
- **No transition to warn about.** Prices start *high* and decay (P1: 1.30 → 0.47 in Nash–monopoly
  units).

### E1 — Calvano Q-learning duopoly (20 seeds per condition, 2M periods)

- **Steady state.** Prices alone separate collusive from myopic firms: AUC 0.955 for φ² with
  2-step windows; Pearson 0.18.
- **No early warning.** Price level and variance move first. During learning the defect's AUC
  reverses (0.04–0.10).
- **Static collusion is missed.** Memoryless learners reach Δ = 0.96 with *less* coupling.
- **Low priority now.** Frozen Q-policies are a poor proxy for adaptive LLM agents.

### E2 — Synthetic coupled agents (30 seeds, T = 20,000)

- **Lagged-copy coupling, ε = 0.02.** Defect power 77%, Pearson 10%.
- **Coalition recovery at ε = 0.05.** 93% from the pairwise defect.
- **XOR coalitions are invisible to every pairwise statistic.** The debiased group cut / bipartition
  recovers them: 87% at ε = 0.05, 100% at ε ≥ 0.1, against ≤ 10% for pairwise.
- **Why it matters.** This is the tensor ("beyond pairs") case that Bonjour's pairwise γ cannot see.

### E0 — Estimator validation

| Check | Result |
|---|---|
| DFA Hurst on exact fGn, H = 0.3 / 0.5 / 0.7 / 0.9 | 0.301 / 0.498 / 0.700 / 0.909 |
| τ(q) of a binomial cascade via the replica partition function | exact (4e-16) |
| MFDFA h(q) on the cascade | right shape, constant offset ≈ −0.05 (finite size) |
| Replica-tensor estimator vs exact moment tensors | < 5% error; Bell ranks 2 and 5 recovered |

## Relation to Bonjour et al.

**Covered:**
- γ, Γ, the detection rule, simultaneous and sequential (lagged) variants;
- state conditioning (`influence_matrix(A, S, lag)`).

**Beyond Bonjour:**
1. **Calibrated nulls instead of α.** A conditional permutation within state strata, which is
   unit-free and has known size.
2. **Coalitions larger than two.** Group cuts and total correlation; Γ breaks when more than two
   agents collude.
3. **Higher-order (XOR) coordination** via exhaustive bipartition.
4. **Window length, lags and coarse-graining scales.**
5. **Coalition trees.**

**Missing:**
- per-agent *private* states (their partially observable poker case);
- a single library function combining state conditioning with group cuts.

## Next steps (server)

1. **E6 judge arm.** Needs an API key. About 45 calls, cheap.
2. **E7 seeded-pair double auction.** The main deliverable. Run the seeded and control arms, then the
   judge on the same public logs.
   - Decide the seller model, judge model and budget.
   - Prefer ≥ 60 hours: 30 hours is too short for pairwise tests (E4).
3. **Conditional group cut in the library.** `influence_z` for groups of agents (state-conditioned
   total correlation), with the same permutation null. Apply it to E4/E7.
4. **Early warning, done properly.** Rerun the auction or pricing harness with *regime shifts*:
   cost or demand shocks, a rival swap, announced monitoring (questions 1.2 and 3.4).
   - Runs of ≥ 1,000 rounds.
   - Compare expanding-window EWS (notes: h*, Δh, asymmetry B from MF-DFA; variance, AR(1)) with
     the coupling statistics.
   - Pool runs aligned at the shift into replica tensors.
5. **Optional.** A sequential / partially observable game (Leduc-style) for the directed and
   private-state variants.

## Reproduce

See `README.md`. Each experiment is a single script. Everything is free except the judge arms and E7.
