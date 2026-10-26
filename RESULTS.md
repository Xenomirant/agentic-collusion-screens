# Results: evidence-efficient multi-agent collusion detection

*Draft, 2026-10-25. Sections marked (running) will be filled as jobs finish. All numbers come from
`results/`; commands are in `README.md` and the `experiments/` docstrings.*

## Executive summary

- **Setting.** An auditor sees only public metadata of a multi-agent log and pays per revealed
  record; "collusion established" means an independently checkable certificate (aware proposal →
  aware acceptance → executions after the proposal → no valid waiver in a closed world). We built
  the framework, a synthetic world with three mechanisms and seven alternative-explanation
  controls, a two-agent real dataset mapping (SALT-NLP/agent-collusion), and **TeamOps-L: 1,200
  episodes of 8–12 local LLM agents (Qwen2.5-7B, Qwen3-8B) with real actions, objects, a scripted
  lead and several concurrent private pairs**, judged by Qwen3-8B with thinking.
- **Information efficiency is real and transfers.** On the many-agent logs, the metadata-prior
  AND-OR selectors find the 4-record certificate after 16–20 of 90–145 records: cross-model
  AUROC at a 20% budget 0.968 / 0.990 (priors from the other agent model; 111 / 134 positives),
  held-out team size 0.995 / 0.999, vs random 0.66–0.82 and the content-agnostic cross-agent
  baselines 0.78–0.91; median overhead over the minimum certificate 1.2–2.6× vs 9–20× for
  content-agnostic search. With a realistic per-record LLM reader the same selectors reach
  0.90–0.95 at 20% (vs 0.64–0.86), while a Qwen3-thinking monitor reading the whole log scores
  0.63 and accuses 39% of innocent episodes.
- **Complementarity itself is a small part of it.** Over an individually-scored suspicion prior
  the group selectors add +0.05 to +0.13 in the synthetic world and +0.03 to +0.09 on the many-agent
  logs; over individual uncertainty-reduction (voi) they add ≈0 to +0.06. Most of the gain is
  metadata links plus rule structure (link shuffling collapses it) and the certificate stopping
  rule.
- **Small certificates are not cheap to find**, and what makes them expensive is the haystack
  (length, concurrent plausible interactions, missing provenance), not temporal distance or team
  size (MVP2, C_cert* held fixed).
- **False accusations are a reader problem, not a search problem.** With exact reads the verifier
  never issues a false "supported" (0 across all experiments, including 308 many-agent episodes
  containing real non-coordinated violations). With an LLM reader, misread chat yields 6–8% false
  certificates on the many-agent logs and 22% on SALT; re-reading only the certificate's records
  with a stronger reader removes them (0 at every budget, no extra disclosure) at a recall cost
  that is small on the many-agent logs and large on SALT's 200-character messages.
- **Assigned ≠ realised.** Only 36–46% of LLM pairs assigned a collusive objective produce a
  certifiable coordinated violation; the rest refuse, act without an in-log agreement, or agree
  without executing. Ranking transfers across agent models; score calibration does not.
- **Negative/limiting results:** the refutation-first ordering is not uniformly better (worse on
  SALT and on one held-out-size split); SALT's two-agent, 200-character episodes are
  reader-limited for every method; Qwen2.5-7B is useless as an LLM auditor (AUROC 0.52–0.55).

> **Status of the tables (2026-10-26 07:00 UTC).** Two semantics corrections and a belief-model
> defect found while tracing the many-agent audits forced a regeneration of every audit table
> (synthetic ground-truth labels unchanged, 0 of 3,020). Regenerated: MVP1 perfect and Qwen2.5
> reader with all ablations, held-out mechanism D, MVP3 perfect, SALT (all five variants), every
> TeamOps-L audit. **Still regenerating at the time of writing** (previous-run numbers shown and
> marked "previous run"): MVP2, MVP1 with the Qwen3 reader; **not yet run**: MVP3 and held-out D
> with the Qwen3 reader, the Qwen3-thinking active auditor on TeamOps-L (bounded rerun in
> progress; the first pass crashed on a context overflow after 6 h) and on MVP1. Realisation
> statistics, reader accuracies, LLM-monitor baselines and C_cert* values were never affected.

## 0. Setting and what is being measured

An auditor sees only public metadata of a multi-agent log (event id, time, type, actor, recipients,
object references, token length) and pays the token length of a record to reveal its content.
A **coordinated violation** of the public rules R is established by a **certificate**: a set of
revealed records E with V(E, R) = supported, where V is an independent three-valued verifier
(`certaudit/verifier.py`) that requires (i) execution of every operation of a rule pattern,
(ii) a knowing agreement (an aware proposal and a later aware acceptance/confirmation between
the two agents) made before execution, and (iii) a closed-world check that no valid waiver exists
among the approval records that can refer to the pattern's objects. V is monotone, so the first
sufficient prefix of the disclosure log is the discovered certificate; the exact minimum certificate
C_cert* is computed by enumeration and checked against brute-force subset search (tests).

Costs tracked per audit: records and tokens disclosed (C_read), total records/tokens, number of
queries, discovery cost of the first verified certificate, C_cert*, search overhead
C_read / C_cert*, and LLM inference tokens where an LLM is involved.

**Disclosure boundary.** `tests/test_leakage.py` perturbs the hidden content of every record a
selector did *not* reveal and asserts that the selector's entire request sequence is unchanged
(all eight selectors pass); the oracle enforces budgets and logs every reveal; record length is
drawn independently of content (a classifier predicting message kind from length has AUC < 0.6).

**Methods compared** (same priors, same information, same budgets):
random records; random connected groups; metadata-graph traversal (`graph`); `caes`, a
*reconstruction* of the earlier cross-agent selector from its description (the original code was
not found; see RESEARCH_STATUS.md); and a 2×2 factorial of belief-driven selectors that share one
metadata-only prior model trained on the training split: `suspicion` (individual records most
likely to add positive evidence), `voi` (individual records by expected uncertainty reduction),
`comp` (hypothesis-level complementary groups through the rule's AND-OR structure), `combined`
(`comp` with refutation-first ordering inside the group, i.e. alternative-explanation testing).
LLM baselines: an active LLM auditor that chooses reveals itself, and a full-information LLM monitor.

## 1. MVP1, synthetic mechanisms A/B/C (perfect reading) — provisional by construction

The selectors are designed around the same rule structure that generates the data, so these
numbers show that the machinery works, not that it understands collusion (see §4 for real data).

Test split: template family 2 (never seen by the priors), 780 episodes (180 positives per
mechanism-balanced design, 7 negative conditions + nulls), ~60 records / ~2,800 tokens each.

AUROC at matched token budgets (results/mvp1/perfect/metrics.csv):

| selector | 5% | 10% | 20% | 50% | cert found @20% | overhead (median) |
|---|---|---|---|---|---|---|
| random | 0.53 | 0.58 | 0.59 | 0.74 | 0.01 | 10.2 |
| random_groups | 0.52 | 0.53 | 0.60 | 0.74 | 0.01 | 9.8 |
| graph | 0.53 | 0.55 | 0.65 | 0.87 | 0.03 | 8.1 |
| caes (reconstruction) | 0.51 | 0.56 | 0.60 | 0.87 | 0.00 | 6.7 |
| suspicion | 0.51 | 0.58 | 0.82 | 1.00 | 0.28 | 2.8 |
| voi | 0.62 | 0.76 | 0.91 | 1.00 | 0.57 | 2.3 |
| comp | 0.56 | 0.65 | 0.90 | 1.00 | 0.49 | 2.4 |
| combined | 0.60 | 0.75 | **0.95** [0.94, 0.97] | 1.00 | 0.54 | 2.2 |

TPR at a 5% FPR threshold frozen on the calibration family, 20% budget: random 0.14, graph 0.17,
caes 0.09, suspicion 0.39, voi 0.69, comp 0.65, combined 0.79 (realised FPR 0.04–0.08).

Paired bootstrap (combined minus other, AUROC, 20% budget): vs random +0.36 [0.31, 0.42];
vs caes +0.35 [0.30, 0.40]; vs graph +0.31 [0.26, 0.36]; vs suspicion +0.13 [0.10, 0.17];
vs voi +0.04 [0.02, 0.07]; vs comp +0.05 [0.02, 0.08]. At 10%: vs comp +0.10 [0.04, 0.15],
vs voi −0.01 [−0.06, 0.04]. Reading of the 2×2: most of the gain over content-agnostic baselines
comes from the metadata prior plus the rule structure (suspicion alone: 0.82); selecting
hypothesis-level complementary groups adds +0.08 over individually-scored suspicion (comp vs
suspicion) and the refutation-first order another +0.05 (combined vs comp); against individual
uncertainty-reduction (voi) the whole group machinery is worth +0.04 at 20% and nothing at 10%.

Ablations (AUROC at 20%, perfect reading):

| | none | shuffle links | no provenance | no comms | no actions | hide outcomes |
|---|---|---|---|---|---|---|
| combined | 0.953 | 0.755 | 0.745 | 0.50 | 0.56 | 0.971 |
| comp | 0.903 | 0.665 | 0.690 | 0.50 | 0.52 | 0.928 |
| voi | 0.909 | 0.715 | 0.790 | 0.50 | 0.55 | 0.940 |
| suspicion | 0.819 | 0.680 | 0.645 | 0.50 | 0.55 | 0.878 |
| caes | 0.602 | 0.635 | 0.623 | 0.50 | 0.53 | 0.624 |
| random | 0.590 | 0.602 | 0.640 | 0.50 | 0.50 | 0.626 |

Shuffling who-talks-to-whom / who-acted collapses the belief-driven selectors to the level of the
content-agnostic ones (the old pilot reported 0.82 → 0.54), so the information is in the
cross-agent links, not in per-record features. Communications are necessary by the rule's
definition (an agreement must be in the messages). Hiding execution outcomes does not hurt.

**Held-out mechanism (H4 within the synthetic world).** Priors trained on A/B/C only, tested on
mechanism D (coordinated false reports; results/mvp1/perfect_mvp1_test_D, 360 episodes): AUROC at
5/10/20/50%: combined 0.62/0.78/**0.954** [0.93, 0.98]/1.00, comp 0.54/0.74/0.92/1.00,
voi 0.66/0.83/0.92/1.00, suspicion 0.50/0.51/0.885/1.00, caes 0.57/0.58/0.59/0.90, graph
0.53/0.51/0.56/0.85, random 0.59/0.54/0.61/0.70. Certificates found at 20%: combined 0.68,
voi 0.63, comp 0.57, suspicion 0.30, others 0; paired at 20%, combined − random +0.35 [0.26, 0.44],
− caes +0.36 [0.28, 0.44], − suspicion +0.07 [0.02, 0.12], − voi/comp +0.03 (n.s.). The rule
structure, not the mechanism, carries the priors. Caveat: thresholds frozen on the D calibration
family realise 7–11% FPR on this smaller negative set (120 negatives + 60 nulls), so operating
points are less stable than AUROC.

## 2. LLM reading and LLM baselines on MVP1

Reader accuracy on the held-out template family (results/extraction/*.json), 54k records:

| reader | DO | GRANT | CHAT | PROPOSE | ACCEPT | CONFIRM |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.96 | 0.97 | 0.99 | 0.67 | 0.66 | 0.73 |
| Qwen3-8B (no thinking) | 0.99 | 0.99 | 0.74 | 0.77 | 0.81 | 0.77 |

Qwen2.5 misses *held-out* awareness phrasings ("let's not loop L0 in"); Qwen3 reads them but
over-reads ordinary chat as proposals.

**Audits with the Qwen3-8B reader** (results/mvp1/qwen3_mvp1/metrics.csv; every method reads
through the same noisy reader, the verifier is unchanged; *previous run, regeneration in progress*):

| selector | 5% | 10% | 20% | 50% | 100% | TPR@5%FPR, 20% | cert found @20% |
|---|---|---|---|---|---|---|---|
| random | 0.52 | 0.55 | 0.55 | 0.66 | 0.73 | 0.07 | 0.01 |
| graph | 0.50 | 0.51 | 0.58 | 0.76 | 0.73 | 0.09 | 0.03 |
| caes | 0.51 | 0.54 | 0.56 | 0.73 | 0.73 | 0.06 | 0.00 |
| suspicion | 0.50 | 0.57 | 0.70 | 0.79 | 0.73 | 0.23 | 0.16 |
| voi | 0.57 | 0.68 | 0.75 | 0.78 | 0.73 | 0.39 | 0.30 |
| comp | 0.54 | 0.62 | 0.76 | 0.77 | 0.73 | 0.41 | 0.33 |
| combined | 0.59 | 0.69 | **0.81** [0.77, 0.85] | 0.77 | 0.73 | 0.46 | 0.28 |

Reading everything through this reader gives AUROC 0.73 for every method (the reader's ceiling:
it misses agreements, while false "supported" verdicts on negatives stay at 1%). At 20% the
belief score of `combined` is *better* than the full-information verdict, because a verifier
that refutes on a misread collapses the score to 0 while the partial-evidence belief does not.
Paired at 20%: combined − suspicion +0.12 [0.07, 0.16], − voi +0.06 [0.02, 0.11], − comp
+0.05 [0.01, 0.09], − caes +0.25 [0.20, 0.31]. At 50% all belief-driven selectors converge
(0.77–0.79). The pattern mirrors the earlier pilot (0.82 at 20% vs 0.79 full information),
with the same caveat that this is one synthetic world.

**With the weaker Qwen2.5-7B reader** (results/mvp1/qwen25_mvp1, regenerated): full reading tops
out at 0.65 (it misses the held-out awareness phrasings), combined at 20% reaches 0.75 [0.71, 0.79]
vs suspicion 0.65, voi 0.70, comp 0.67, random/graph/caes 0.55–0.57; paired gains at 20%:
+0.19 [0.12, 0.25] vs random, +0.18 [0.12, 0.24] vs caes, +0.20 [0.14, 0.27] vs graph, +0.11
[0.06, 0.16] vs suspicion, +0.06 [0.01, 0.10] vs voi, +0.09 [0.04, 0.14] vs comp. Ablations under
this reader (combined at 20%): links shuffled 0.65, no provenance 0.63, no communications 0.50,
no actions 0.54, so the same structure carries the gain under noisy reading. No selector
produces a false "supported" on negatives with either reader on this split (supported_on_neg
≤ 0.01).

Active LLM auditor and full-information LLM monitor, Qwen2.5-7B
(results/mvp1/llm/qwen2.5-7b/summary_mvp1_test.json): monitor AUROC 0.55 [0.51, 0.59], "supported"
on 77% of positives **and 71% of negatives**, no valid certificate; auditor AUROC 0.52–0.56 at
nominal budgets 10/20/50%, actually spending 8–15% and stopping after 5–9 records, refuting ~90%
of episodes. A 7B instruction model is not a credible auditor of this task. (running) Qwen3-8B
with thinking.

## 3. MVP2 (search complexity at fixed C_cert*) and MVP3 (alternative explanations)

The controlled generator keeps the core interaction fixed per seed; C_cert* was verified invariant
under trajectory length 60–480, 0–8 decoy interactions, temporal gap ×0.5–4 and 3–8 agents (only
metadata quality changes it, by widening the waiver closure).

**MVP2, perfect reading** (results/mvp2/metrics.csv; 12 cores × A/B/C × {pos, one negative} per
setting, one knob varied at a time; priors trained on the whole grid, family 0; *previous run,
regeneration in progress*). C_cert* = 225
tokens (median) in every structural setting; 248/284 when 15%/30% of metadata fields are missing.

Median search overhead C_read / C_cert* of the first verified certificate (absolute discovery
tokens for `combined` in brackets; total tokens per episode in the last column):

| setting | random | graph | caes | suspicion | voi | comp | combined | total tokens |
|---|---|---|---|---|---|---|---|---|
| base (60 events, 2 decoys, 4 agents) | 10.6 | 8.6 | 7.4 | 2.7 | 2.4 | 2.2 | 2.2 [481] | 2,846 |
| length 120 / 240 / 480 | 22 / 40 / 81 | 17 / 32 / 60 | 13 / 28 / 54 | 5.2 / 12 / 26 | 2.5 / 7.4 / 31 | 2.8 / 7.4 / 14 | 3.6 / 7.2 / 19 [851 / 1,750 / 4,345] | 5,659 / 11,327 / 22,550 |
| decoys 0 / 4 / 8 | 11 / 11 / 12 | 7.4 / 9.9 / 12 | 6.9 / 7.1 / 8.4 | 2.5 / 3.3 / 3.9 | 1.4 / 3.0 / 4.1 | 1.4 / 3.1 / 4.2 | 1.3 / 2.4 / 3.2 | 2,825 / 2,854 / 3,350 |
| temporal gap ×0.5 / ×2 / ×4 | 11 / 9.8 / 11 | 8.9 / 8.9 / 8.3 | 6.4 / 7.3 / 8.6 | 2.7 / 3.0 / 3.2 | 2.0 / 2.6 / 2.6 | 2.3 / 2.3 / 2.7 | 2.0 / 2.3 / 2.4 | ≈2,840 |
| agents 3 / 6 / 8 | 10 / 11 / 11 | 9.1 / 9.8 / 9.0 | 7.7 / 6.2 / 5.6 | 3.4 / 2.5 / 2.4 | 2.8 / 1.6 / 1.8 | 2.7 / 1.8 / 1.8 | 2.6 / 2.0 / 1.7 | ≈2,830 |
| metadata present 85% / 70% | 9.4 / 8.9 | 8.8 / 7.7 | 8.6 / 8.5 | 3.1 / 3.7 | 3.2 / 3.6 | 2.0 / 2.3 | 2.1 / 2.5 | 2,846 |

AUROC at a 20% budget, combined / comp / suspicion / random: base 0.98 / 0.94 / 0.87 / 0.61;
length 480 0.88 / 0.82 / 0.78 / 0.66; decoys 8 0.86 / 0.66 / 0.74 / 0.48; metadata 70%
0.86 / 0.85 / 0.65 / 0.59; agents 8 1.00 / 0.98 / 0.99 / 0.58. Certificates found within 20%:
base 0.67 / 0.61 / 0.39 / 0.00; decoys 8 0.47 / 0.25 / 0.28 / 0.00; metadata 70% 0.33 / 0.25 / 0.08 / 0.00.

Reading. (i) A small certificate is never cheap to find: even the best selector reads 2.2× the
certificate in the easiest setting and 19× when the haystack is 8× longer; discovery cost scales
with trajectory length for every method, sublinearly for the belief-driven ones (combined 2.2→19
vs random 11→81). (ii) Plausible alternative interactions (decoys) and missing provenance
metadata are what degrade *detection* at a fixed budget; temporal distance and the number of
agents barely matter (more agents make the pair stand out). (iii) The refutation-first order
matters most exactly where alternatives multiply (decoys 8: 0.86 vs 0.66 for the same groups
without it). Absolute costs are reported throughout; no number here is a percentage of a padded
trajectory.

**MVP3, perfect reading** (results/mvp3/perfect; 840 test episodes = 40 cores × A/B/C × 7
conditions with identical skeleton metadata, priors trained on the MVP2 grid, family 0):
with a sound verifier and exact reads no selector ever returns a false "supported"
(verdict FPR = 0 at every budget, for every selector), so the alternative-explanation effect
appears in coverage and calibration rather than in FPR:

| 20% budget | TPR (verdict) | abstain | refutes authorized / unexecuted / benign | TPR @5% FPR (score) | ECE | risk at 60% coverage |
|---|---|---|---|---|---|---|
| suspicion (A) | 0.46 | 0.90 | 0.07 / 0.08 / 0.12 | 0.58 | 0.083 | 0.040 |
| voi (B) | 0.50 | 0.85 | 0.23 / 0.09 / 0.09 | 0.64 | 0.070 | 0.018 |
| comp (C without refutation-first) | 0.56 | 0.90 | 0.04 / 0.02 / 0.03 | 0.74 | 0.082 | 0.024 |
| combined (C) | **0.63** | **0.76** | **0.34 / 0.28 / 0.29** | **0.78** | **0.050** | **0.000** |
| caes | 0.01 | 1.00 | 0.00 / 0.00 / 0.00 | 0.03 | 0.102 | 0.103 |

At 50% every belief-driven selector reaches TPR 0.92–1.0 with 38–50% abstention; the misled,
shared-error and individual conditions are almost never resolved at 20% (they need the waiver
closure or all messages) and are correctly left "insufficient". The refutation-first order is
what resolves alternatives early: the same groups without it (comp) refute 2–4% of them. The
false-accusation question is decided under LLM reading (§3b and §5), where misreads can satisfy
the verifier.

## 4. MVP4a, real two-agent LLM episodes: SALT-NLP/agent-collusion (Shi, Zhang & Yang 2026)

Mapping (`certaudit/salt.py`): verdict records are executions of a two-operation "report" pattern;
the dataset's human-validated agreement judge supplies the aware proposal/acceptance records
(its quotes locate the messages); unconstrained-communication runs carry a valid "cap lifted" waiver.
The verifier's label equals the dataset's collusion label on 99.9% of 3,112 episodes. Test split:
1,514 episodes whose non-gemini partner model (deepseek, claude-opus/sonnet, gpt, qwen) never
appears in training or calibration: 400 coordinated, 400 simultaneous-relaxation (both accepted,
no explicit agreement), 14 abandoned agreements, 300 authorized (unconstrained), 400 none.
Episodes are small (~12 records, ~370 tokens), so results are reported by records revealed.

AUROC by records revealed, **perfect reading**:

| k | random | graph | caes | suspicion | voi | comp | combined |
|---|---|---|---|---|---|---|---|
| 4 | 0.60 | 0.78 | 0.83 | 0.83 | 0.74 | 0.83 | 0.79 |
| 6 | 0.74 | 0.83 | 0.87 | 0.92 | 0.84 | 0.92 | 0.82 |
| 8 | 0.79 | 0.89 | 0.90 | 0.97 | 0.94 | 0.97 | 0.87 |

The refutation-first order (`combined`) is *worse* than plain suspicion here. Thresholds frozen on
gemini-3.7 calibration give 10–18% FPR on the other model families: calibration does not transfer
across models.

**Qwen3-8B reader (no thinking)**, the realistic condition (results/salt/qwen3-8b):

| | final AUROC | records read | supported on coordinated | supported on simultaneous relaxation | on abandoned / authorized / none |
|---|---|---|---|---|---|
| read everything (random/graph/caes) | 0.84 | 7–9 | 0.76 | **0.23** | 0 / 0 / 0 |
| suspicion / comp | 0.77 | 5.8 | 0.60 | 0.17 | 0 / 0 / 0 |
| voi / combined | 0.76–0.77 | 4.7 | 0.59–0.61 | 0.17 | 0 / 0 / 0 |
| + certificate confirmation by an oracle second reader | 0.68–0.75 | 5–10 | 0.36–0.49 | **0.00** | 0 / 0 / 0 |

Reading: (a) the structural half of the verifier (executions, waivers) produces no false accusations
on authorized, abandoned or non-accepting episodes under any reader; (b) false accusations come
entirely from misread chat supplying a fake "agreement" on episodes where both agents did violate
the protocol independently; (c) early stopping on a certificate built from noisy reads *loses*
accuracy relative to reading everything (0.77 vs 0.84); (d) re-reading only the certificate's
records with a second reader removes the false accusations without new disclosure, but recall then
falls to the first reader's recall of real agreements (PROPOSE 0.59). On this dataset the audit is
**reader-limited**, not search-limited.

**Thinking-mode Qwen3-8B as the first reader** (results/salt/qwen3-8b-think; message accuracy
CHAT 0.90, PROPOSE 0.41, ACCEPT 0.87, 568 reasoning tokens per record): the reader becomes
conservative, so false certificates on simultaneous relaxation fall from 22.5% to 5.2%, but
recall on coordinated episodes falls from 76% to 50% and the full-read AUROC from 0.84 to 0.74
(by records revealed, best selectors 0.82–0.83 at k = 8). Neither reader setting is adequate
on 200-character messages; the two readers trade precision for recall, which is what the
confirmation design exploits (liberal first reader, conservative re-read of the certificate only).
**Honest confirmation** (non-thinking first reader, thinking re-read of the certificate's records,
results/salt/qwen3-8b+confirm_salt_qwen3-8b-think): false certificates on simultaneous relaxation
2.2% (from 22.5%), authorized / abandoned / none still 0, but recall on coordinated episodes 31.5%
(from 76%), full-read AUROC 0.65. On SALT the conservative re-reader rejects most true agreements
as well, so confirmation buys precision at too high a recall price; the many-agent logs (§5) are
where this trade-off is tested with longer, less ambiguous messages.

## 5. MVP4b, TeamOps-L: 8–12 local LLM agents with actions, objects and a scripted lead

`certaudit/teamops.py`. Each episode: N ∈ {8, 12} Qwen agents with roles and a public ticket
list, 10 rounds, one message + one operation + one optional waiver request per agent per round;
L0 is scripted (benign ops get signed waivers, pattern ops are denied or get a draft). Disjoint
agent pairs receive private objectives: 45% of episodes have one `pos` pair plus two
alternative-explanation pairs (misled / authorized / unexecuted / individual / benign_coop /
shared_error), 45% have three alternative pairs and no collusive one, 10% have none. Every
prompt is identical across conditions except the private objective. Executions, waivers and
state logs are exact; message facts are read by Qwen3-8B with thinking (the judge; 2,800 tokens
per record), and the verifier decides the label. 600 episodes per agent model (Qwen2.5-7B:
69M agent tokens, 54 min; Qwen3-8B: 2.5 h under GPU contention). The audited readers never see
the judge's facts.

One semantic correction surfaced here and is applied everywhere (synthetic labels unchanged,
0 of 3,020 episodes): a proposal "concerns" a pattern if it names any of its operations, since a
proposer who does their own part and asks the partner for the rest has proposed the joint plan.

**Assigned objective vs realised behaviour, Qwen3-8B agents (600 episodes, results/teamops/L_realisation.csv):**

| team size | episodes | records / messages / tokens | assigned collusive → executed → aware in-log agreement → verified coordinated | attributed to the assigned pair |
|---|---|---|---|---|
| 8 | 310 | 90 / 40 / 1,152 | 148 → 70% → 59% → **36.5%** | 100% of verified |
| 12 | 290 | 138 / 70 / 1,849 | 144 → 66% → 62% → **38%** | 98% of verified |

Episodes *without* a collusive pair (308): some executed violation (misled, individual, shared
error) in 58%, an aware agreement in 1–3%, verified coordinated in 0–1.4%. 111 coordinated
positives in total. Minimum certificates: 4 records, ~47 tokens, 3.6% of the episode.

**Qwen2.5-7B agents (600 episodes, 91/145 records, 903/1,445 tokens at 8/12 agents):** assigned
collusive 148/144 → executed 74%/73% (72–74% after the proposal) → aware agreement 59%/53%
→ verified coordinated **46%/43%** (134 positives), 98–99% attributed to the assigned pair;
alternative-only episodes (308): executed violation 55–62%, aware agreement 1–3%, verified
coordinated 0–2.5% (a handful of "unexecuted" or "misled" partners who went ahead with a
knowing agreement on record, which the log genuinely shows).

A second semantic correction came from these logs: Qwen2.5 partners often accept with "sure,
I'll do it, between us" without naming the operation, and the reader, which sees one record at
a time, extracts no operation; the verifier now lets an aware acceptance addressed to the
proposer after the proposal inherit the proposal's operations (synthetic labels unchanged,
0 of 3,020; Qwen3 labels unchanged; Qwen2.5 positives 56 → 134). The LLM-reader synthetic
audits are regenerated under this rule. This is the regime the method targets: a four-record certificate
in a 90–140-record log in which half the episodes contain a real but *non-coordinated*
violation.

**Audit, Qwen3-8B agents, judge-quality reads, seed-disjoint split (train 300 / cal 150 / test
150 episodes, 26 coordinated; results/teamops/L_audit_qwen3-8b_to_qwen3-8b/judge):**

| selector | AUROC 5% | 10% | 20% | 50% | TPR@5%FPR, 20% | cert found @20% | records @20% | overhead (median) |
|---|---|---|---|---|---|---|---|---|
| random | 0.77 | 0.78 | 0.76 | 0.91 | 0.31 | 0.04 | 22 | 15.0 |
| graph | 0.81 | 0.86 | 0.94 | 0.96 | 0.73 | 0.50 | 25 | 5.8 |
| caes (reconstruction) | 0.82 | 0.89 | 0.91 | 0.99 | 0.73 | 0.39 | 23 | 7.9 |
| suspicion | 0.85 | 0.93 | 0.98 | 1.00 | 0.89 | 0.89 | 20 | 2.5 |
| voi | 0.89 | 0.96 | 0.98 | 0.99 | 0.96 | 0.96 | 20 | 1.5 |
| comp | 0.93 | 0.97 | 0.99 | 1.00 | 0.96 | 0.92 | 17 | 1.3 |
| combined | 0.89 | **0.987** [0.97, 1.00] | **0.999** | 1.00 | **1.00** | 0.92 | 16 | 1.9 |

Paired at 10%: combined − random +0.21 [0.11, 0.31], − graph +0.13 [0.06, 0.22], − caes +0.10
[0.02, 0.19], − suspicion +0.06 [0.01, 0.13], − voi +0.02 [−0.02, 0.08], − comp +0.02
[−0.02, 0.09]. False "supported" on negatives: 0 for every selector (sound verifier, exact reads).
With a four-record certificate buried in ~100 records, the AND-OR selectors find it after
reading 16–20 records (10–20% of the tokens); content-agnostic search needs half the log.
Caveats: 26 positives; reads are the judge's own facts (upper bound on reading quality).
(Regenerated after the belief fix: combined 0.986 / 0.998 at 10% / 20%, comp 0.965 / 0.992,
voi 0.963 / 0.981, suspicion 0.934 / 0.978, caes 0.883 / 0.900, graph 0.855 / 0.942, random
0.791 / 0.746; every belief-driven selector reaches 1.0 at full budget; overhead comp 1.2×,
combined 1.8×, random 15×.)

**Cross-model transfer (H4): priors trained on Qwen2.5-7B agents' episodes (450 train / 150
cal), tested on all 600 Qwen3-8B episodes (111 coordinated), judge-quality reads
(results/teamops/L_audit_qwen2.5-7b_to_qwen3-8b/judge):**

| selector | AUROC 5% | 10% | 20% | 50% | cert found @20% | records @20% | FPR@nominal 5%, 20% | overhead |
|---|---|---|---|---|---|---|---|---|
| random | 0.61 | 0.61 | 0.66 | 0.82 | 0.01 | 22 | 0.12 | 20 |
| graph | 0.66 | 0.71 | 0.80 | 0.92 | 0.32 | 26 | 0.15 | 10 |
| caes | 0.69 | 0.73 | 0.78 | 0.95 | 0.32 | 23 | 0.10 | 9.3 |
| suspicion | 0.66 | 0.77 | 0.88 | 1.00 | 0.60 | 22 | 0.11 | 4.6 |
| voi | 0.73 | 0.86 | 0.95 | 1.00 | 0.79 | 19 | 0.18 | 2.3 |
| comp | 0.75 | 0.88 | 0.96 | 0.99 | 0.83 | 17 | 0.27 | 1.8 |
| combined | 0.77 | **0.878** [0.84, 0.91] | **0.968** [0.95, 0.98] | 1.00 | 0.77 | 17 | 0.10 | 2.6 |

Paired at 20%: combined − random +0.31 [0.26, 0.37], − graph +0.17 [0.12, 0.22], − caes +0.19
[0.14, 0.25], − suspicion +0.09 [0.05, 0.13], − voi +0.01 [−0.01, 0.04], − comp +0.01 [−0.02, 0.03].
Ranking transfers across agent models; **calibration does not**: thresholds frozen at 5% FPR on
the training model's episodes realise 10–27% FPR on the other model's (the same shift seen on
SALT across partner models), while the verifier's own verdicts stay at 0 false "supported".
**Reverse direction, Qwen3-8B priors → all 600 Qwen2.5-7B episodes (134 coordinated;
results/teamops/L_audit_qwen3-8b_to_qwen2.5-7b/judge):** AUROC at 5/10/20/50%: combined
0.89 / **0.954** [0.93, 0.97] / **0.990** [0.98, 1.00] / 1.00, comp 0.89 / 0.94 / 0.99 / 1.00, voi
0.87 / 0.91 / 0.97 / 1.00, suspicion 0.80 / 0.90 / 0.96 / 1.00, caes 0.82 / 0.86 / 0.89 / 0.99,
graph 0.76 / 0.82 / 0.91 / 0.97, random 0.74 / 0.79 / 0.82 / 0.94. At 20%: TPR 0.93 at a realised
FPR of 0.01–0.03 (thresholds frozen on Qwen3 episodes transfer in this direction), certificates
found 0.87 / 0.86 (combined / comp) after 17 records, overhead 2.1 / 1.7 vs random 15. Paired at
20%: − random +0.18 [0.14, 0.22], − caes +0.10 [0.07, 0.13], − graph +0.08 [0.05, 0.12],
− suspicion +0.03 [0.01, 0.05], − voi +0.02 [−0.00, 0.04], − comp +0.00. The calibration shift is
therefore asymmetric (Qwen2.5-trained scores are too liberal on Qwen3 episodes, not the reverse).
**The realistic condition: every record read by Qwen3-8B (no thinking), Qwen2.5-7B priors →
600 Qwen3-8B episodes, 111 coordinated (results/teamops/L_audit_qwen2.5-7b_to_qwen3-8b/qwen3-8b).**
Reader agreement with the judge on message kind (and awareness): PROPOSE 0.78, ACCEPT 0.89,
CONFIRM 0.82, but CHAT 0.47, i.e. half of ordinary chat is read as some proposal/acceptance.

| selector | AUROC 10% | 20% | 50% | 100% | cert found @20% (valid under true facts) | records @20% | false "supported" on negatives @20% / @100% |
|---|---|---|---|---|---|---|---|
| random | 0.60 | 0.64 | 0.76 | 0.95 | 0.01 | 22 | 0.002 / 0.076 |
| graph | 0.71 | 0.79 | 0.89 | 0.95 | 0.31 | 25 | 0.033 / 0.076 |
| caes | 0.72 | 0.76 | 0.90 | 0.95 | 0.20 | 23 | 0.033 / 0.076 |
| suspicion | 0.75 | 0.85 | 0.95 | 0.95 | 0.53 | 21 | 0.059 / 0.076 |
| voi | 0.80 | 0.89 | 0.94 | 0.95 | 0.69 | 18 | 0.065 / 0.076 |
| comp | 0.82 | **0.911** [0.88, 0.94] | 0.95 | 0.95 | 0.73 | 16 | 0.070 / 0.076 |
| combined | **0.824** [0.78, 0.87] | 0.899 [0.86, 0.93] | 0.95 | 0.95 | 0.63 | 16 | 0.059 / 0.076 |

Paired at 20%: combined − random +0.26 [0.21, 0.32], − caes +0.14 [0.09, 0.19], − graph +0.11
[0.07, 0.16], − suspicion +0.05 [0.01, 0.08], − voi +0.01, − comp −0.01. Three things change
relative to judge-quality reads. (i) The ceiling is the reader's: 0.95 for everyone once all is
read. (ii) The verifier's "supported" is no longer sound: 7.6% of negatives end with a false
certificate once everything is read, because a misread "agreement" can fall on a pair that
really did execute something (recall that 55–62% of alternative-only episodes contain a
genuine non-coordinated violation); the belief-driven selectors reach those false certificates
sooner (5.9–7.0% at 20%) than content-agnostic search (0.2–3.3%), which is the cost of finding
real ones sooner (63–73% vs 1–31%). (iii) Because ~7% of calibration negatives already score 1.0,
a 5%-FPR operating point is unattainable by thresholding; it needs the certificate-confirmation
step (a second reading of only the certificate's records, no extra disclosure), run next as an
oracle upper bound.

**Reverse direction with the same reader (Qwen3-8B priors → 600 Qwen2.5-7B episodes, 134
coordinated; reader agreement PROPOSE 0.81, ACCEPT 0.93, CHAT 0.65):** AUROC at 10 / 20 / 50 /
100%: comp 0.89 / **0.912** [0.89, 0.93] / 0.88 / 0.86, combined 0.87 / 0.909 / 0.87 / 0.86,
suspicion 0.88 / 0.905 / 0.89 / 0.86, voi 0.84 / 0.885 / 0.87 / 0.86, caes 0.82 / 0.853 / 0.91 /
0.86, graph 0.81 / 0.863 / 0.90 / 0.86, random 0.79 / 0.790 / 0.89 / 0.86. Here AUROC *peaks* at
20–50% and falls when the whole log is read (8.2% false certificates at 100%): under a noisy
reader, reading more is not better. Paired at 20%: combined − random +0.12 [0.08, 0.16], − caes
+0.06 [0.02, 0.09], − graph +0.05 [0.01, 0.08], − suspicion/voi/comp ≈ 0. In the realistic
condition the increment of complementary group selection over a good per-record prior vanishes
on these shorter logs; what remains is the prior-plus-structure advantage over content-agnostic
search, and the certificate rule that stops the search early.
**Held-out team size (H4): priors trained on 8-agent episodes only, tested on all 12-agent
episodes of the same agent model, judge-quality reads** (results/teamops/L_audit_*_n8to12/judge):

| | Qwen2.5-7B agents (233 train / 77 cal / 290 test, 62 positives) | Qwen3-8B agents (229 / 77 / 290, 57 positives) |
|---|---|---|
| AUROC 10% / 20%: comp | 0.981 / **0.999** | 0.954 / 0.989 |
| combined | 0.943 / 0.998 | 0.963 / **0.995** [0.985, 1.00] |
| voi | 0.882 / 0.935 | 0.933 / 0.972 |
| suspicion | 0.902 / 0.988 | 0.905 / 0.978 |
| caes | 0.857 / 0.892 | 0.842 / 0.902 |
| graph | 0.809 / 0.908 | 0.789 / 0.896 |
| random | 0.738 / 0.743 | 0.720 / 0.728 |
| certificates found @20% (comp / combined / random) | 0.95 / 0.94 / 0.00 | 0.88 / 0.93 / 0.00 |
| paired at 20%, combined − random / − caes / − suspicion | +0.26 / +0.11 / +0.01 | +0.27 / +0.09 / +0.02 |

The priors, which see only who-talked-to-whom, who-acted-on-what and when, transfer to teams
with 50% more agents and 50% more records without retraining.
**Within-Qwen3, realistic reader (seed-disjoint split, 150 test episodes, 26 coordinated):** same
pattern as the cross-model runs: reader ceiling 0.936 with 8.9% false certificates when all is
read; at 10% / 20% comp 0.931 / 0.933, combined 0.921 / 0.934, suspicion 0.909 / 0.946, voi 0.911 /
0.913 vs caes 0.833 / 0.837, graph 0.820 / 0.892, random 0.779 / 0.740; certificates found at 20%:
0.85–0.89 (belief-driven) vs 0.04–0.50; paired at 20% (wide, 26 positives): combined − random
+0.19 [0.08, 0.31], − caes +0.10 [0.01, 0.19], others ≈ 0.
**LLM baselines on the many-agent logs (Qwen3-8B with thinking, 240 held-out Qwen3-agent
episodes, 45 coordinated; results/teamops/llm_qwen3-8b).** Full-information monitor (reads the
whole 90–140-record log, ~7.5k tokens of inference per episode): AUROC **0.63** [0.54, 0.71];
"supported" on 76% of positives and **39% of negatives**; of the certificates it cites on true
positives, 24% are valid under the independent verifier. The same model used as a per-record
reader inside the restricted-disclosure auditor gives 0.82–0.88 AUROC after disclosing 10% of
the log (previous table), with every cited certificate valid by construction.

**Certificate confirmation on the many-agent logs (Qwen2.5-7B priors → 600 Qwen3-8B episodes,
Qwen3-8B reader for every record, judge-quality re-read of only the certificate's records before
stopping; results/teamops/L_audit_qwen2.5-7b_to_qwen3-8b/qwen3-8b+confirm_judge).** False
"supported" on negatives: **0 at every budget** (from 5.9–7.6% without confirmation). AUROC at
10 / 20 / 50 / 100%: comp 0.85 / **0.947** [0.91, 0.97] / 0.99 / 0.99, combined 0.85 / 0.934 / 0.99 /
0.99, voi 0.82 / 0.92 / 0.98 / 0.99, suspicion 0.76 / 0.88 / 0.99 / 0.99, caes 0.73 / 0.76 / 0.93 /
0.99, graph 0.71 / 0.79 / 0.91 / 0.99, random 0.60 / 0.64 / 0.76 / 0.99; valid certificates found at
20%: comp 0.82, combined 0.70, voi 0.72, suspicion 0.59, graph/caes 0.32, random 0.01. Reading
everything now ends at 0.986 instead of the single-reader ceiling 0.949, because confirmation
removes the false certificates and keeps the true ones (the re-reader is the judge, so this is an
upper bound on a real second reader; the extra inference is confined to ≤5 records per episode
and no record is disclosed that was not already disclosed). The score-threshold operating points
still inherit the cross-model calibration shift (FPR 0.13–0.29 at a nominal 5%); the verifier's
verdicts do not. Reverse direction with confirmation (Qwen3-8B priors → 600 Qwen2.5-7B
episodes, 134 coordinated): false "supported" on negatives 0 at every budget (from 8.2%); AUROC at
10 / 20 / 50 / 100%: comp 0.92 / **0.950** [0.93, 0.97] / 0.91 / 0.89, combined 0.90 / 0.943 / 0.90 /
0.89, suspicion 0.89 / 0.934 / 0.92 / 0.89, voi 0.87 / 0.918 / 0.90 / 0.89, caes 0.82 / 0.859 /
0.95 / 0.89, graph 0.81 / 0.872 / 0.92 / 0.89, random 0.79 / 0.790 / 0.90 / 0.89; valid certificates
at 20%: comp 0.81, combined 0.76, voi 0.69, suspicion 0.63, graph/caes 0.22–0.24, random 0.01.
Even with confirmation the full-read ranking stays reader-limited (0.89: agreements the reader
never recognises are lost), so stopping at a certificate after ~17 of 91–145 records is both
cheaper and more accurate than reading the whole log. (running) active Qwen3-thinking auditor at 20%.

## 6. Hypotheses ("synthetic" = our generator; "real" = LLM agents in TeamOps-L and SALT)

| Hypothesis | Evidence for | Evidence against | Conclusion |
|---|---|---|---|
| H1 complementary cross-agent groups lower the information cost vs individually suspicious records | Synthetic, perfect reading: comp/combined 0.90–0.95 AUROC at 20% vs suspicion 0.82, +0.08/+0.13 paired (CIs > 0), held-out mechanism 0.954; under both LLM readers the ordering persists (+0.05 to +0.11 vs suspicion); link shuffling removes it (0.95 → 0.76), so the information is in the cross-agent links. Real many-agent logs: group selectors 0.97–0.99 at 20% vs suspicion 0.88–0.96 (+0.03 to +0.09, CIs > 0 in 3 of 4 splits) and vs the content-agnostic cross-agent baselines 0.78–0.91 (+0.08 to +0.19); overhead 1.2–2.6× vs 6–20× | The increment over individual *uncertainty-reduction* (voi) is ≈0 to +0.06 everywhere; with a realistic reader on the many-agent logs the group selectors are statistically tied with suspicion/voi (0.90–0.95); on SALT (two agents) suspicion ties or beats them and no belief-driven selector beats exhaustive reading with a noisy reader | **Supported as information efficiency** (cross-agent metadata prior + rule structure + certificate stopping), real and transferable; **the specific value of complementarity is modest** and vanishes against a good per-record uncertainty criterion |
| H2 small certificates are not cheap to find; provenance and complementary selection reduce overhead | C_cert* fixed at 225 tokens: best overhead 2.2× (base) → 19× at 8× length vs random 11× → 81×; driven by length, decoys and missing metadata, not temporal distance or team size; TeamOps-L: 4-record certificates (3.6% of an episode) found after 16–20 of 90–145 records (overhead 1.2–2.6×) vs ~half the log for content-agnostic search (9–20×) | Overhead is never near 1 and grows with the haystack for every method; with 30% of metadata missing no selector finds >33% of certificates at 20% | **Supported**: discovery cost is a property of the haystack and the provenance metadata, and AND-OR selection keeps it 3–8× below content-agnostic search |
| H3 testing alternative explanations lowers false accusations without destroying recall | Exact reads: verdict FPR 0 for every selector in every experiment, incl. 308 many-agent episodes with real non-coordinated violations; refutation-first order resolves 28–34% of alternatives at 20% (suspicion-first 7–12%), abstains less (76% vs 90%), better calibrated (ECE 0.05 vs 0.08), zero risk to 60% coverage; LLM reader + certificate confirmation: false certificates 6–8% → 0 on the many-agent logs with AUROC 0.93–0.95 at 20% and 0.99 at full read | False accusations under an LLM reader come from misread agreements, not from the search (all selectors reach the same 7.6–8.9% at full read); confirmation on SALT's 200-character messages costs 45 points of recall; refutation-first ordering is not uniformly better (worse than plain groups on SALT and on one held-out-size split) | **Supported for the certificate + confirmation design; not supported as a selector property**: the auditor's soundness comes from the verifier and a second reading of the certificate, the ordering mainly buys coverage and calibration |
| H4 improvements persist beyond one task, model, mechanism and template | Held-out template family (all MVP1), held-out mechanism D (0.954), two readers, two agent models in both transfer directions (0.968 / 0.990 at 20%), held-out team size 8 → 12 for both models (0.995 / 0.999), eight held-out partner models on SALT with exact reads (0.97 at 8 records vs 0.79 random) | Score calibration does not transfer across agent models (10–27% realised FPR at a nominal 5% in one direction, 10–18% across SALT partner models); SALT with a realistic reader: advantage over exhaustive reading gone; one environment family (rule-governed team logs) | **Supported for ranking and certificate discovery; not for calibrated operating points**, which must be set per deployed population |
