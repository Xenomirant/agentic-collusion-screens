# Findings: detecting collusion from agents' actions

*Phase 2 as of 2026-10-24 (branch `autoresearch/scale`). Phase 1, the sprint results E0-E7, follows
below unchanged. Every number here traces to a JSON file under `experiments/results/`.*

## Phase 2: scale, threat model, multifractal baselines, real agents

### Bottom line

The question was whether a tensorized multifractal method can screen millions of agents that share a
base model, cover the realistic threat models, and do something the common multifractal methods
cannot. After an adversarial review that overturned several of our own claims, the answer is:

* **Scale: yes.** One streamed decomposition of the population (Gram + eigh, rank from a
  parallel-analysis null) and within-market scoring process **4 million agents (38 million candidate
  pairs) in 6 s of GPU time** (E11). Nothing is ever `N x N`.
* **Things the common multifractal methods cannot do, shown against their strongest configurations**
  (signed q = 2 rho_DCCA and Kristoufek's rho_q from MFDCCA on the same cleaned data, per-agent MFDFA
  spectra, multifractal early-warning indicators):
  1. **Higher-order (XOR) coalitions**: every pair inside is uncorrelated. AUC **1.00** with the parity
     channel; every pairwise linear or multifractal statistic is at chance (0.49-0.54).
  2. **Calibrated screening at scale**: with the threshold set at the maximum over 9.5 million pairs of a
     coalition-free world, the detector flags **1,145 colluding pairs with 0 false positives** among
     1 million agents; every classical baseline flags **0** colluding pairs at that level (E14).
  3. **Early warning**: 86% of newly formed coalitions are flagged within 256 rounds at 1% false
     alarms (rho_DCCA 55%); the multifractal early-warning indicators (rolling Hurst exponent and
     spectrum width) stay at chance throughout (E13).
  4. **Strong shared-model common mode**: AUC 0.982 vs 0.911 for the same channels without
     common-mode removal and 0.71 for cleaned rho_DCCA (E11, 4M agents).
  5. **Anti-coordination and delayed copying** (turn-taking, lagged following): 0.99 / 0.99 with the
     multi-lag two-sided channels; raw correlation 0.06 / 0.88 (E14).
  6. **Slow timing coordination** (coalitions that agree *when* to be active on scales of 64-512
     rounds, directions independent): the one-decomposition QTT pyramid scores 0.75 vs 0.71 for the
     best cleaned rho_DCCA variant at 1 million agents, and doubles its recall at FPR 1e-3 (E12).
* **Where it does not win, stated plainly:**
  * **Fast timing coordination** (blocks of 4-64 rounds): signed rho_DCCA on cleaned log-activity is
    better, 0.98 vs 0.92; at medium scales they tie. Much of the timing-world advantage claimed earlier
    came from preprocessing (log coordinates + correct common-mode removal), which helps rho_DCCA just
    as much.
  * **Benign interaction**: lawful one-way reaction to a rival is coupling too. At a pair false-positive
    rate of 1e-3 the false discovery rate is 0.74-0.83; the reciprocal score cuts the benign share of
    false flags from 55% to 12% at some cost on ring-shaped coalitions.
  * **Precision at realistic prevalence**: pairwise screening at moderate thresholds has a PPV of a few
    percent at 1% coalition prevalence; only family-wise thresholds (above) or a market-first screen
    (PPV about 13% at 1% prevalence, 62% at 10%) are usable.
  * **Real LLM agents** (local Qwen2.5-7B sellers) coordinate by matching prices to the cent; the
    exact-coincidence channel finds them (AUC 0.95), coupling channels add little except under
    monitoring (E8).
  * **By design**, coalitions across markets (one operator in several markets) are not scored, and the
    method needs known market membership.

### What changed after the adversarial review (corrections to our own earlier claims)

| earlier claim | correction | evidence |
|---|---|---|
| "classical multifractal tools cannot see timing coordination" | false as stated: signed rho_DCCA on cleaned log-activity reaches 0.92-0.98 | `results/baselines_mf_pilot.json`, `results/e12/` |
| E11 lead comes from common-mode removal | in the weak-drive world it comes from the parity channel (raw corr + parity 0.940 vs method 0.962); removal matters with a strong shared drive (0.911 -> 0.982) | `results/e11/` |
| E12 best classical 0.73 | the cleaned baseline had been wrecked by an overcounting rank estimator; with a correct estimator it is the strongest classical method | review, `results/e12/bands_*.json` |
| planted pairs were fair | coalitions sat at fixed positions with model = agent parity, so positives were cross-model while 47% of negatives were same-model; randomized everywhere | `experiments/sims/planted_markets.py` (`layout`) |
| early-warning 93% | used a common mode fitted on the full horizon (look-ahead); causal version 86% | `results/e13/` |
| fathon C: 3 series/s, 1 pair/s | OpenMP oversubscription; idle core: 11 series/s, 12.5 pairs/s | review |
| 4M run "in progress" | the quadratic AUC routine was OOM-killed at 240 GB; replaced by a rank-based AUC | `fathon/fathon/tensor/ews.py` |
| market_level_coupling is "the agent cut of the cross-replica tensor, tensorized" | it was multiscale log-energy correlation with per-level SVDs; now rebuilt as one decomposition + dyadic contraction (below) and described as such | `scale.market_level_coupling_qtt` |

### The method as it now stands (`fathon/fathon/tensor/scale.py`, `linalg.py`, `tt.py`)

1. **Population common mode, one decomposition per mode.** Stream all agents once: Gram `X^T X`
   (T x T, float64 accumulation) and the parallel-analysis null of row-wise circular-shift surrogates
   in closed form (centred-Toeplitz expectation with Percival's bias correction, Wishart finite-size
   spread, deflation, two calibration surrogates). The rank is the leading run above the null; it
   recovers the true common-mode rank in every planted world (4, 12 and 8) where the white-noise
   Marchenko-Pastur edge says up to 37 for sticky agents. `fit_common_mode`.
2. **Within-market channels**, never materializing `N x N`: multi-lag (0-3) two-sided coupling with a
   one-way and a reciprocal variant; a parity channel `s_i(t)s_i(t-2)` vs `s_k(t-1)s_k(t-2)` that turns
   XOR coordination into a pairwise identity; exact-value coincidence by sorting and n-gram shingles
   (1 million agents x 100 rounds in 78 s); and the multiscale activity pyramid.
3. **Multiscale pyramid in QTT form.** Log energy once on cells of 4 rounds; market mean and the single
   population decomposition removed there; every coarser level is the dyadic average, i.e. the
   contraction of the trailing binary modes with [1/2, 1/2]. Linear averaging commutes with the
   agent-side projection, so all levels inherit the cleaning without their own SVD (20-200x faster
   than per-level SVDs). Extra leakage components are removed only at levels with enough cells.
4. **Library reuse.** Tensor-train work goes through torchtt (TT-SVD, rounding, DMRG cross) and teneva
   (cross, arithmetic); tall-skinny algebra through torch (streamed Gram is the fastest float64-accurate
   option: 4e6 x 256 in 0.1 s; Householder TSQR is 5-7x slower) with CANS (polar-decomposition SVD)
   as an optional backend that beats torch's SVD on the small R factor at T = 1024. Found on the way:
   teneva 0.14.11's `svd`/`truncate` put singular values on the wrong side (93% error at eps 0.3 on an
   exact rank-6 tensor), and this container enables TF32 by default, making float32 GEMMs about 1000x
   less accurate than IEEE (guarded inside the library).
5. **QTT replica tensors.** TT-cross on the exact cascade law recovers the Bell-type ranks of the
   moment tensors at q = 4, depth 14 (ranks 8 and 15) in 14 s, a tensor with about 10^17 dense entries.
   On *sampled* data every method (old and new) fails beyond depth 12: entrywise noise grows like
   `(E W^{2q} / (E W^q)^2)^{n/2} / sqrt(S)`, so deep empirical replica tensors need astronomically many
   samples. The QTT machinery is not the bottleneck; sampling is.

### E11: scale, copy and XOR coalitions (4 million agents, 200,000 markets, strong shared drive)

| detector | AUC | copy pair | copy triad | XOR triad | onset pair | precision@K |
|---|---|---|---|---|---|---|
| method (lag + parity, common mode removed) | **0.982** | 0.997 | 0.982 | 0.999 | 0.914 | **0.555** |
| raw correlation + parity (no removal) | 0.911 | 0.943 | 0.850 | 1.000 | 0.800 | 0.407 |
| raw lag-1 correlation | 0.731 | 0.952 | 0.867 | 0.491 | 0.818 | 0.048 |
| rho_DCCA, cleaned | 0.714 | 0.952 | 0.798 | 0.541 | 0.744 | 0.066 |
| rho_DCCA, raw | 0.666 | 0.879 | 0.728 | 0.520 | 0.703 | 0.022 |
| per-agent MFDFA width (agent level) | 0.525 | | | | | |

Detection: 6.1 s of GPU time for 38 million within-market pairs (simulation 231 s, baselines on 4 of 80
chunks 3.9 s). With a weak shared drive (1M agents, w = 0.8) the method reaches 0.96 and the parity
channel alone explains most of the gap to raw correlation.

### E14: threat-model coverage, specificity and calibration (1 million agents, 9.5 million pairs)

Seven coalition types, each in 2% of markets; benign one-way followers in half the markets; a second,
coalition-free world sets the thresholds.

| detector | AUC vs clean | AUC vs benign | anti | lag-2 | match | XOR | FDR at FPR 1e-3 | family-wise: true / false flags |
|---|---|---|---|---|---|---|---|---|
| v1 (lag-1 + parity) | 0.909 | 0.701 | 0.47 | 0.77 | 0.89 | 1.00 | 0.83 | **1,145 / 0** |
| v2 one-way (lags 0-3, two-sided) | **0.943** | 0.823 | 0.99 | 0.99 | 0.78 | 1.00 | 0.78 | 828 / 1 |
| v2 reciprocal | 0.903 | **0.851** | 0.99 | 1.00 | 0.81 | 1.00 | **0.74** | 384 / 2 |
| raw lag-1 correlation | 0.687 | 0.492 | 0.06 | 0.88 | 0.71 | 0.49 | 0.98 | 0 / 1 |
| rho_DCCA (lag 0, 1) | 0.703 | 0.569 | 0.87 | 0.84 | 0.56 | 0.50 | 0.98 | 0 / 1 |
| MFDCCA rho_q, q = 4 | 0.668 | 0.503 | 0.08 | 0.88 | 0.71 | 0.49 | 0.99 | 0 / 5 |
| MFDCCA max over q of abs(rho_q) | 0.680 | 0.581 | 0.85 | 0.78 | 0.57 | 0.49 | 0.99 | 0 / 1 |

The robust z-scores are not N(0,1) (the z > 4 tail is 100-500x heavier), so thresholds must come
from an empirical null. Screening markets first (market score = max pair score): AUC about 0.70 for
our detectors vs 0.49-0.50 for every classical baseline. Detection time 5 s.

### E13: early warning of coalition onset (1 million agents, causal common mode)

Detection rate of newly formed pairs at 1% false alarms (AUC in brackets), by rounds after onset:

| indicator | +0 | +128 | +256 | +512 |
|---|---|---|---|---|
| sliding-window residual coupling (ours) | 0.01 (0.50) | 0.17 (0.90) | **0.86 (0.99)** | 0.86 (0.99) |
| expanding window | 0.01 (0.50) | 0.02 (0.66) | 0.04 (0.76) | 0.09 (0.87) |
| rho_DCCA, raw window | 0.01 (0.51) | 0.05 (0.68) | 0.55 (0.94) | 0.54 (0.94) |
| rho_DCCA, cleaned window | 0.01 (0.51) | 0.02 (0.62) | 0.35 (0.90) | 0.34 (0.89) |
| rolling Hurst exponent change | 0.01 (0.51) | 0.01 (0.52) | 0.03 (0.55) | 0.02 (0.53) |
| rolling MFDFA width change | 0.01 (0.50) | 0.01 (0.51) | 0.01 (0.51) | 0.01 (0.51) |

5,000 onset pairs against 5.7 million null pairs per window; the pre-onset false-alarm rate is at its
nominal 1%.

### E12: timing coordination at an unknown band of scales

1 million agents (50,000 markets), coalition cascades misaligned with the detector's dyadic cells
(offset 37). Every method sees the same population-wide cleaning (market mean, then the
parallel-analysis rank). For rho_DCCA we report its best variant per band, chosen with the labels
(in the baseline's favour); the a-priori union fixes the rank-only, scales 8-512 variant.

| coalition shares | QTT pyramid (ours) | best rho_DCCA variant | a-priori union | recall at FPR 1e-3: ours / best rho_DCCA |
|---|---|---|---|---|
| slow scales (blocks 64-512) | **0.752** | 0.710 | 0.743 | 0.023 / 0.012 |
| medium (16-256) | 0.929 | 0.936 | **0.938** | 0.389 / 0.388 |
| fast (4-64) | 0.915 | **0.983** | 0.937 | 0.290 / 0.655 |

Scoring 1 million agents takes 0.3-0.4 s for the pyramid against 4.2-4.5 s for rho_DCCA at seven
scales; the two population fits take 4-7 s each. No single statistic wins every band: the pyramid is
best when coordination is slow, rho_DCCA when it is fast, and they tie in between. A pyramid
without the level-dependent rule (or per-level SVDs) is worse on slow bands; see
`results/e12/bands_pilot*.json` for the 80,000-agent ablations.

### E8: real LLM agents (local Qwen2.5-7B-Instruct sellers, 120 markets)

| detector | seeded pairs: top-1 / AUC | monitored pairs: top-1 / AUC |
|---|---|---|
| exact-price coincidence | **0.90 / 0.95** | 0.80 / 0.78 |
| reciprocal coupling of ask levels (global common mode removed) | 0.42 / 0.68 | 0.28 / 0.70 |
| combined | 0.90 / 0.95 | 0.78 / 0.79 |
| calibrated influence z (Markov null) | 0.08 / 0.49 | 0.33 / 0.49 |

Told that a regulator screens for coupled asks, the pair stops matching to the cent (detection at
z > 3 falls from 62% to 25% at 2.5% false alarms) while its market's asks end higher (p = 0.02).
Pooled across markets, same-model sellers in *different* markets coincide on prices 13-20 rounds out
of 59 against about one expected: parallel conduct from a shared prior. Market-mean removal must be
skipped in markets of 4 sellers.

### E9 / E10: real markets

* **NSW FuelCheck** (1,390 stations x 3,560 days): top residual-coupled pairs are 4-6x enriched in
  same-brand pairs within a suburb (ownership beyond co-location), robust across windows; the effect
  of common-mode removal on that signal is small and not consistent in sign.
* **Brazil (ANP survey + CADE cartel decisions)**: the 2013-2019 control-window files are still
  downloading (VPN limit of 3 transfers); cross-sectional screens on the 2004-2012 cartel windows are
  mixed.

### Reproduce (phase 2)

```bash
# Python 3.13 venv, numpy 2.x, torch (CUDA); pip install -e "./fathon[tensor]"; apt-get install libgsl-dev
cd fathon && ../.venv/bin/python -m pytest tests -q                     # 182 passed
cd fathon && ../.venv/bin/python ../experiments/run_e11_scale.py --markets 200000 --eps_copy 0.75 --w_global 2.0
cd fathon && ../.venv/bin/python ../experiments/run_e14_threats.py --markets 50000
cd fathon && ../.venv/bin/python ../experiments/run_e13_ews.py --markets 50000 --T 2048
cd fathon && ../.venv/bin/python ../experiments/run_e12c_scale.py --markets 50000
cd fathon && ../.venv/bin/python ../experiments/run_e8_scale.py         # needs results/e8_local runs
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
