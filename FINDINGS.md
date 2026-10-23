# Findings: detecting collusion from agents' actions

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
| `fathon.tensor` (pure numpy) | `fathon/` (fork of stfbnc/fathon; our work is the 3 `tensor-ext` commits) | `detrend`, `mvdfa`, `decomp`, `factorization`, `replica`, `ews`, `influence`. 44 new tests; 70/70 pass, including upstream. |
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
