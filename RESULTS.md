# Results: evidence-efficient multi-agent collusion detection

*Draft, 2026-10-25. Sections marked (running) will be filled as jobs finish. All numbers come from
`results/`; commands are in `README.md` and the `experiments/` docstrings.*

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
| random | 0.53 | 0.57 | 0.59 | 0.74 | 0.01 | 10.2 |
| random_groups | 0.52 | 0.54 | 0.60 | 0.74 | 0.01 | 9.8 |
| graph | 0.53 | 0.54 | 0.64 | 0.87 | 0.03 | 8.1 |
| caes (reconstruction) | 0.51 | 0.56 | 0.60 | 0.87 | 0.00 | 6.7 |
| suspicion | 0.52 | 0.59 | 0.83 | 1.00 | 0.29 | 2.8 |
| voi | 0.62 | 0.76 | 0.91 | 0.99 | 0.58 | 2.1 |
| comp | 0.56 | 0.66 | 0.93 | 1.00 | 0.62 | 2.1 |
| combined | 0.60 | 0.75 | 0.96 | 1.00 | 0.53 | 2.3 |

TPR at a 5% FPR threshold frozen on the calibration family, 20% budget: random 0.14, graph 0.17,
caes 0.09, suspicion 0.46, voi 0.72, comp 0.81, combined 0.80 (realised FPR 0.03–0.08).

Paired bootstrap (combined minus other, AUROC, 20% budget): vs random +0.37 [0.31, 0.42];
vs caes +0.36 [0.31, 0.41]; vs graph +0.32 [0.27, 0.37]; vs suspicion +0.13 [0.09, 0.16];
vs voi +0.05 [0.02, 0.08]; vs comp +0.02 [−0.00, 0.05]. At 10%: vs comp +0.09 [0.04, 0.14],
vs voi −0.01 [−0.06, 0.04]. Reading of the 2×2: most of the gain over content-agnostic baselines
comes from the metadata prior plus the rule structure (suspicion alone: 0.83); selecting
hypothesis-level complementary groups adds +0.10 over individually-scored suspicion (comp vs
suspicion) and +0.05 over individual uncertainty-reduction (combined vs voi); the refutation-first
order adds little at 20% and +0.09 at 10%, where it stops spending on hopeless hypotheses sooner.

Ablations (AUROC at 20%, perfect reading):

| | none | shuffle links | no provenance | no comms | no actions | hide outcomes |
|---|---|---|---|---|---|---|
| combined | 0.956 | 0.750 | 0.754 | 0.50 | 0.56 | 0.973 |
| comp | 0.934 | 0.680 | 0.713 | 0.50 | 0.53 | 0.947 |
| voi | 0.909 | 0.711 | 0.816 | 0.50 | 0.55 | 0.939 |
| suspicion | 0.831 | 0.682 | 0.645 | 0.50 | 0.54 | 0.874 |
| caes | 0.596 | 0.630 | 0.618 | 0.50 | 0.53 | 0.621 |
| random | 0.591 | 0.603 | 0.637 | 0.50 | 0.50 | 0.617 |

Shuffling who-talks-to-whom / who-acted collapses the belief-driven selectors to the level of the
content-agnostic ones (the old pilot reported 0.82 → 0.54), so the information is in the
cross-agent links, not in per-record features. Communications are necessary by the rule's
definition (an agreement must be in the messages). Hiding execution outcomes does not hurt.

**Held-out mechanism (H4 within the synthetic world).** Priors trained on A/B/C only, tested on
mechanism D (coordinated false reports; results/mvp1/perfect_mvp1_test_D, 360 episodes): AUROC at
5/10/20/50%: combined 0.62/0.78/**0.955** [0.93, 0.98]/1.00, comp 0.55/0.73/0.94/1.00,
voi 0.64/0.82/0.93/1.00, suspicion 0.50/0.51/0.89/1.00, caes 0.57/0.58/0.61/0.90, graph
0.53/0.51/0.56/0.85, random 0.58/0.54/0.61/0.70. Certificates found at 20%: combined 0.70,
comp 0.68, voi 0.63, suspicion 0.30, others 0. The rule structure, not the mechanism, carries
the priors. Caveat: thresholds frozen on the D calibration family realise 8–15% FPR on this
smaller negative set (120 negatives + 60 nulls), so operating points are less stable than AUROC.

## 2. LLM reading and LLM baselines on MVP1

Reader accuracy on the held-out template family (results/extraction/*.json), 54k records:

| reader | DO | GRANT | CHAT | PROPOSE | ACCEPT | CONFIRM |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.96 | 0.97 | 0.99 | 0.67 | 0.66 | 0.73 |
| Qwen3-8B (no thinking) | 0.99 | 0.99 | 0.74 | 0.77 | 0.81 | 0.77 |

Qwen2.5 misses *held-out* awareness phrasings ("let's not loop L0 in"); Qwen3 reads them but
over-reads ordinary chat as proposals.

**Audits with the Qwen3-8B reader** (results/mvp1/qwen3_mvp1/metrics.csv; every method reads
through the same noisy reader, the verifier is unchanged):

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

**With the weaker Qwen2.5-7B reader** (results/mvp1/qwen25_mvp1): full reading tops out at 0.65
(it misses the held-out awareness phrasings), combined at 20% reaches 0.74 [0.69, 0.78] vs
suspicion 0.65, voi 0.66, comp 0.65, random/graph/caes 0.55–0.56; paired gains +0.09 [0.04, 0.14],
+0.07 [0.03, 0.12], +0.08 [0.03, 0.13], +0.17 [0.11, 0.24]. No selector produces a false
"supported" on negatives with either reader on this split (supported_on_neg ≤ 0.01).

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
setting, one knob varied at a time; priors trained on the whole grid, family 0). C_cert* = 225
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
| suspicion (A) | 0.46 | 0.89 | 0.07 / 0.11 / 0.13 | 0.60 | 0.082 | 0.042 |
| voi (B) | 0.48 | 0.85 | 0.23 / 0.08 / 0.08 | 0.68 | 0.065 | 0.012 |
| comp (C without refutation-first) | 0.63 | 0.89 | — | 0.81 | 0.063 | 0.006 |
| combined (C) | **0.63** | **0.76** | **0.35 / 0.28 / 0.30** | **0.80** | **0.048** | **0.000** |
| caes | 0.01 | 1.00 | 0.00 / 0.00 / 0.00 | 0.04 | 0.102 | 0.103 |

At 50% every belief-driven selector reaches TPR 1.0 with 38–52% abstention; the misled,
shared-error and individual conditions are almost never resolved at 20% (they need the waiver
closure or all messages) and are correctly left "insufficient". The false-accusation question is
decided under LLM reading (§3b, running), where misreads can satisfy the verifier.

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
**reader-limited**, not search-limited. (running) thinking-mode reader as first reader.

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
positives in total. Minimum certificates: 4 records, ~47 tokens, 3.6% of the episode. This is the regime the method targets: a four-record certificate
in a 90–140-record log in which half the episodes contain a real but *non-coordinated*
violation. (running) Qwen2.5-7B agents; cross-model audits (train on one agent model, test on
the other), held-out team size (train 8, test 12), Qwen3 reader, Qwen3-thinking auditor/monitor.

## 6. Hypotheses (provisional until §5 audits are in; "synthetic" = our generator, "real" = LLM agents)

| Hypothesis | Evidence for | Evidence against | Conclusion (provisional) |
|---|---|---|---|
| H1 complementary cross-agent groups lower the information cost vs individually suspicious records | Synthetic, perfect reading: comp/combined 0.93–0.96 AUROC at 20% vs suspicion 0.83, +0.10/+0.13 paired (CIs > 0); under Qwen3/Qwen2.5 reading the gain persists (+0.05 to +0.12); held-out mechanism D 0.955; link shuffling removes it (0.96 → 0.75), so it is the cross-agent structure, not per-record features | Increment over individual *uncertainty-reduction* (voi) is small (+0.02 to +0.06); on SALT with perfect reading plain suspicion ties or beats the group selectors (two agents: nothing to combine); with a noisy reader on SALT all belief-driven selectors lose to exhaustive reading | Supported in the synthetic world and across readers; the specific value of *complementarity* beyond a good per-record prior is modest; real many-agent verdict pending (§5) |
| H2 small certificates are not cheap to find; provenance and complementary selection reduce overhead | C_cert* fixed at 225 tokens: best overhead 2.2× (base) rising to 19× at 8× length vs random 11× → 81×; overhead driven by length, decoys and missing metadata, not by temporal distance or team size; TeamOps-L certificates are 4 records / 3.6% of an episode | Overhead is never near 1 and grows with the haystack for every method; no selector finds certificates in >33% of episodes at 20% when 30% of metadata is missing | Supported: discovery cost is a property of the haystack and the metadata, and the AND-OR selectors keep overhead 3–5× below content-agnostic search |
| H3 alternative-explanation testing lowers false accusations without destroying recall | Sound verifier: verdict FPR 0 for every selector under exact reads; refutation-first order resolves 25–35% of alternatives at 20% (suspicion-first 7–13%), abstains less, better calibrated, zero risk to 60% coverage; on TeamOps-L episodes with real non-coordinated violations the verifier flags 0–1.4% as coordinated | Under a noisy reader (SALT) false accusations come from misread agreements (23% of simultaneous-relaxation controls) and the belief-driven stop rule loses; second-reader confirmation removes them but recall drops to the reader's | Partly supported: the gain is in coverage/calibration, false accusations are a *reader* problem that certificate confirmation fixes at a recall cost; LLM-reader MVP3 and TeamOps audits pending |
| H4 improvements persist beyond one task, model, mechanism and template | Held-out template family (all MVP1 numbers), held-out mechanism D (0.955), two readers, real SALT episodes with eight held-out partner models (perfect reading: 0.97 at 8 records vs 0.79 random) | SALT with a realistic reader: advantage gone; calibration thresholds do not transfer across model families (10–18% realised FPR at a nominal 5%) | Inconclusive until the TeamOps-L cross-model and held-out team-size audits (§5) are in |
