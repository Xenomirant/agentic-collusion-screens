# Research directions: detecting collusion among AI agents in markets

*Memo, 2026-10-25. Built from a web-only literature check (search and fetch, no downloads), the team chat, and `FINDINGS.md`. Each reference was checked against arXiv, a DOI page or a publisher listing, unless the References section marks it otherwise.*

---

## 0. Bottom line

1. **The team's main intuition is already established in economics.** Shared algorithms change competition, and correlation can come from a common algorithm rather than an agreement: Hansen, Misra & Pai 2021; Banchio & Mantegazza; Harrington 2022; Assad et al. 2024 (margins rise only when all rivals adopt); Calder-Wang & Kim 2026 and Eschenbaum 2026 (structural evidence of shared-provider coordination); Keppo et al. 2026 (collusion among identical LLMs falls apart under heterogeneity). Monoculture in model errors is also documented (Kleinberg & Raghavan 2021; Bommasani et al. 2022; Kim et al. 2025; Goel et al. 2025). "Same weights ⇒ correlated behaviour ⇒ false positives" is not new.
2. **What is open is identification under partial observability.** Nobody has a tested way to tell three mechanisms apart using only market data, plus at most the regulator's own market interventions:
   - **(A) common-cause monoculture:** a shared prior, with independent policies;
   - **(B) identity-conditioned tacit coordination:** copies recognise each other or share focal points, with no channel;
   - **(C) channel collusion:** a provider-internal or steganographic channel.

   Our own results show the ingredients: same-model sellers in *different* markets coincide on 9–15 of 59 rounds (E8), and state conditioning is what beats correlation (Bonjour head-to-head). Structural IO work (Eschenbaum; Calder-Wang & Kim) estimates a conduct parameter from observational data. It does not separate B from C, and it does not use designed probes.
3. **LLM agents bring two things that ordinary pricing software does not.** First, self-recognition and focal-point coordination among copies (Panickssery et al. 2024; Aharon et al. 2026). Second, open weights hand the regulator an exact model of the "innocent" behaviour distribution. Both can be exploited.
4. **The best next step is Direction 1:** an identification ladder that separates A from B∪C using interventional probes and matched-replay controls, then C from A∪B using a no-signalling (information-set) test. It needs inference only, reuses the E8 harness, and fits two weeks.

---

## 1. What we already know (from FINDINGS.md) and must not redo

- No new multifractal or tensor detectors: they have no advantage over fairly configured rho_DCCA/MFDCCA. What carries detection is the method-agnostic pipeline (removing the shared-model common mode, scoring within markets, choosing lags and scales).
- Passive early-warning indicators were at chance: Hurst and MFDFA width in E13; coupling during Calvano learning in E1, where the AUC reversed. The Fish setup (E3) had no transition at all, since high prices were static and focal.
- Exact price coincidence catches real Qwen pairs (AUC 0.95), and warned pairs evade it (0.78). State conditioning matters (Bonjour head-to-head). The common mode is linear only in the simulator.

---

## 2. Literature map: which intuitions are covered, partly open, or open

| Team intuition | Status | Key prior work | What is still open |
|---|---|---|---|
| AI agents collude without explicit agreement | **Covered** | Calvano et al. 2020 (Q-learning); Fish, Gonczarowski & Shorrer 2024 (GPT-4 pricing, prompt sensitivity, off-path analysis); Lin et al. 2024 (market division); Agrawal et al. 2025 (double auction); Keppo et al. 2026; Lee & Park 2026 (CoT faithful yet collusive) | Little; replication only |
| Homogeneity (shared weights) raises correlated behaviour and collusion risk | **Covered in outline** | Kleinberg & Raghavan 2021; Bommasani et al. 2022; Kim et al. 2025; Goel et al. 2025; Assad et al. 2024; Keppo et al. 2026 (heterogeneity breaks collusion; 32B-vs-14B size asymmetry *stabilises* it through leader-follower dynamics); Song (ICLR 2026, shared retrieval → herding) | How to *separate* homogeneity from coordination from market data |
| Collusion through a common provider (hub and spoke) | **Covered in law and economics; partly open on detection** | Harrington 2018, 2022; Calder-Wang & Kim 2026 (joint-profit conduct among adopters of the same software); Eschenbaum 2026 (same-provider batteries, conduct ≈ 1 above 30% share); RealPage (DOJ settlement Nov 2025 bans nonpublic data and identical recommendations); Agri Stats (DOJ settlement May 2026); *Gibson v. Cendyn* (9th Cir. 2025, dismissal affirmed) | Structural tests need cost and demand models and cannot tell B from C. No interventional test exists |
| Steganographic channels between LLMs | **Covered for text channels** | Motwani et al. 2024; Mathew et al. 2024 (stego emerges from misspecified RL reward; paraphrasing is not enough); Zolkowski et al. 2025 (early, limited ability); Schroeder de Witt et al. 2023 (perfect security ⇔ coupling); Cachin 1998; Tailor 2025 (ColludeBench); Ghanem 2026 (trace-only steganalysis); Rose et al. 2026 (activation probes, NARCBench-Stego) | Channels carried by **market actions** (prices, timing) under a regulator who sees only market data; capacity versus detectability for that channel |
| Detectability of covert coordination in principle | **Theory exists; not applied to markets** | Cachin 1998 (KL-based security); Bash, Goeckel & Towsley 2013 (square-root law: O(√n) covert bits in n uses); Schroeder de Witt et al. 2023 | A no-signalling test for markets, and its power against a channel of known capacity |
| Model identity and provenance from behaviour | **Covered for text; open for actions** | LLMmap (Pasquini et al. 2025; 42 models from 8 queries); Sun et al. 2025 (idiosyncrasies, 97% from text); Nikolić et al. 2025 (black-box provenance test); Foley et al. 2023; strategy classifiers on repeated-game play (arXiv:2512.07462; framing effects as large as architecture effects) | Attributing the base model from **prices and orders alone**, robust to prompts and fine-tunes |
| Learning inside the system changes collusion over time | **Partly open** | Alignment Tipping Process (Han et al. 2025: self-evolution drift, imitative diffusion across agents); Shi et al. 2026 (cross-episode memory accelerates collusion; removing memory almost eliminates it); Piche et al. 2025 (RL-trained LLMs become opportunistic, LoRA agent buffer); Vallinder & Hughes 2024 (cultural evolution); parameter-sharing MARL (Terry et al. 2020; Christianos et al. 2021); PBT (Jaderberg et al. 2017); LOLA (Foerster et al. 2018) | **Group-structured** fine-tuning (shared-init groups in an N-agent market) and the resulting within-group vs cross-group coordination structure. Not found |
| Early-warning signals of collusion onset | **Theory of failure exists; not applied to learned collusion** | Scheffer et al. 2009; Hastings & Wysham 2010 (shifts with no warning); Ditlevsen & Johnsen 2010; Boettiger & Hastings 2012 (error rates, prosecutor's fallacy); Diks, Hommes & Wang 2019 (critical slowing down only before 1987); Bury et al. 2021 (deep-learning EWS) | **Active** EWS (recovery time after a designed perturbation) during training. Passive EWS are already at chance for us |
| Classical screens | **Covered for human cartels; untested under monoculture** | Abrantes-Metz et al. 2006 (variance screen); Bajari & Ye 2003 (conditional independence and exchangeability of bids); Chassang et al. 2022 (robust "missing bids" screen valid under any information structure); Huber & Imhof 2019 (ML screens); Bresnahan 1982 (conduct identified by demand rotation); Hartline, Long & Zhang 2024 (audit by calibrated regret) | Bajari–Ye conditional independence is *violated by construction* under a shared model, so the screen falsely rejects competition. That cuts both ways: it is a known hazard, and it means invariances that survive a shared prior are needed |

**Funding framing.** The Schmidt Sciences / Google DeepMind / ARIA / Cooperative AI Foundation / Google.org call "Scaling AI Safety for a Multi-Agent World" (deadline 8 Aug 2026, now closed; decisions in autumn 2026) has four priority areas:
- Sandboxes and Testbeds (including virtual marketplaces);
- Science of Agent Networks ("detect dangerous, unexpected population-level properties");
- Strengthening Agent Infrastructure (identity, reputation, commitments);
- Multi-Agent Oversight and Control ("monitor deployed agent populations").

The public pages I read do **not** use the words "collusion detection under partial observability". That wording is the team's paraphrase. It does match Hammond et al. 2025, which names collusion as one of three failure modes and lists information asymmetries and selection pressures among its seven risk factors. Directions below are mapped to these areas.

---

## 3. A sharper framing

The useful split is A / B / C (Section 0), not "monoculture vs collusion".
- **B is new with LLMs.** Copies of a model can recognise each other's outputs (self-recognition above 50%, and above 90% after 500 fine-tuning examples; Panickssery et al. 2024). They also share focal points (Aharon et al. 2026). Kin-conditioned strategies therefore need no channel.
- **The legal and policy relevance differs by class.** A is lawful parallel conduct. C is close to an agreement, and the RealPage and Agri Stats remedies target exactly the nonpublic data flows behind it. B sits in Harrington's liability gap.
- **Tests split along the same lines.** A detector that cannot tell A from B∪C fails the Gibson/Cendyn bar. One that cannot tell C from B cannot support a hub-and-spoke case.

There are three observability tiers, and each direction states which one it needs:
- **T1 (passive):** market data only.
- **T2 (market interventions):** a regulator-controlled entrant, test buyer or probe orders. Garra 2026 uses an "active random entrant" as a remedy; we use it as a probe.
- **T3 (sandbox query access):** the regulator can query a deployed agent with counterfactual histories. This is black-box audit access as in Casper et al. 2024 and Hartline et al. 2024.

---

## 4. Directions, ranked

Scoring is 1–5 on each axis. Feasibility assumes 3–4×H100, Qwen2.5-7B, Qwen3-8B and Olmo-3-7B, vLLM, and no paid APIs.

| # | Direction | Novelty | Feasibility | Relevance | Score |
|---|---|---|---|---|---|
| 1 | Identification ladder: A vs B∪C via probes and matched replay | 4 | 5 | 5 | 100 |
| 2 | Market invariants: a no-signalling test for channels, and robust-competition bounds | 4 | 4 | 5 | 80 |
| 3 | N/K groups × K same-type agents trained in-system (team idea, sharpened) | 4 | 3 | 5 | 60 |
| 4 | Active early warning during in-system learning | 3 | 4 | 4 | 48 |
| 5 | Base-model attribution from market actions (a "monoculture map") | 3 | 5 | 3 | 45 |
| 6 | Covert price channels when the cover model is open-weight | 3 | 3 | 4 | 36 |

### Direction 1: identification ladder for separating monoculture from coordination (covers item i)

**Question.** Given market data plus T2 or T3 probes, can we reject "pure common cause" (A) in favour of coordination (B∪C), with controlled false positives on same-model populations?

**Hypothesis H1.** Under A, an agent's response to a rival's deviation depends only on the public price path. Swapping the deviator's *identity* while holding the public path fixed (matched replay) therefore changes nothing. Under B or C the response depends on who deviated: punishment or accommodation is targeted at kin or at partners. The identity-interaction effect separates B∪C from A with AUC ≥ 0.8 at T3 and ≥ 0.65 at T2. Falsified if the effect under A is not ≈ 0 (the monoculture confound leaks through), or if it is not detectably non-zero under seeded C.

**Why it is not done.**
- Calvano et al. 2020 and Fish et al. 2024 use forced deviations and impulse responses to *characterise* collusion in their own simulations, with white-box access. They do not use them to *identify* it against a shared-model null.
- Eschenbaum 2026 and Calder-Wang & Kim 2026 estimate conduct structurally. That needs demand and cost models, and it cannot separate B from C.
- Hartline et al. 2024 audit a single algorithm's calibrated regret, not cross-agent identity dependence.
- Bonjour-style state conditioning removes public-state confounding but not a shared prior. Our E8 cross-market coincidences show that the shared prior is large.

**Minimal experiment.**
- **Environment:** the E8 posted-price market (4 sellers, 60 rounds), with Qwen2.5-7B, Qwen3-8B and Olmo-3-7B sellers.
- **Conditions:**
  - A: homogeneous, no channel;
  - A′: mixed providers, the control that fixes the null;
  - B: homogeneous with a prompt revealing that rivals may be copies, or later a jointly trained group from Direction 3;
  - C: the existing seeded private channel (`patches/...seeded-pair-channel`).
- **Probes:**
  - T3: at random rounds, re-query seller j with the history edited so that the deviation (a 10–20% price cut) is attributed to a same-model vs a different-model rival. The rival's price path is held fixed, and the rival's behavioural signature is swapped, for example by grafting a different-model rival's earlier price history. With open weights we read j's exact action distribution from logprobs, so no sampling is needed.
  - T2: a regulator entrant (a scripted seller) cuts price. We compare responses when the entrant mimics a Qwen price signature vs an Olmo one.
- **Metric:** the identity-interaction effect (the KL or Wasserstein shift in j's price distribution) and AUC across conditions, with a permutation null built from A′.
- **Size and cost:** 3 models × 4 conditions × 40 sessions ≈ 115k generations plus about 20k logprob probes, roughly 10–20 GPU-hours.

**Main failure modes.**
- 7B models may not condition on inferred identity at all, which would leave B empty and C indistinguishable from A at T2. That is still a publishable negative for T2, but it removes the method.
- Grafted histories can be off-distribution, so j reacts to oddness rather than identity. Controls: graft same-model histories from other sessions (a placebo), and measure perplexity.
- T3 access is a policy assumption. State it plainly as an audit design question in the sense of Casper et al.

### Direction 2: market invariants agents cannot change (covers item iii)

**Question.** Which constraints of the mechanism hold whatever the agents' weights are, and which tests built on them are robust to a shared prior?

Three candidates follow. The first is the core.

1. **No-signalling (information-set) invariance.** Absent a channel, j's action at time t is independent of i's *private* information given the public history and j's own private information. A shared prior does not break this: it changes the *function* both agents use, not what j can know. A violation means a channel (C), and it does not arise under A or B. This is Bajari–Ye conditional independence moved from costs to information sets, and it survives monoculture because it conditions on information rather than on outcomes.
2. **Timing and latency.** No response can precede the public arrival of the information it uses. A lead of j over a public event that only i observes is a channel signature.
3. **Robust competitive bounds.** Chassang et al. 2022 derive tests valid under *any* information structure: the missing mass of close losing bids. Hartline et al. 2024 audit calibrated regret. Neither depends on agents being heterogeneous.

**Hypothesis H2.** In the double auction with private seller costs, a conditional-independence test of j's ask on i's private cost shock (drawn i.i.d. and logged by the environment), given public history:
- keeps size at 5% under A and B, including same-model populations;
- reaches power ≥ 0.8 against a seeded channel at 60 rounds and 40 sessions;
- has power that scales with channel capacity in a way consistent with the square-root law: a channel carrying o(√n) bits stays undetectable.

**Why it is not done.**
- Classical screens test outcome patterns (variance, bid gaps) or conditional independence given *costs* (Bajari–Ye), which a shared model violates.
- Steganalysis work (Tailor 2025; Ghanem 2026; Rose et al. 2026) inspects text, traces or activations, not the dependence of prices on others' private information.
- The square-root law (Bash et al. 2013) has not been carried over to market actions.

**Minimal experiment.**
- **Environment:** the Agrawal et al. 2025 double auction (code is public), with logged private costs.
- **Channel:** a seeded channel whose capacity is controlled, by restricting messages to k ∈ {0, 1, 2, 4} bits per round.
- **Agents:** Qwen2.5-7B and Qwen3-8B.
- **Test:** a conditional permutation test within public-state strata (the calibrated `influence_z` machinery already exists), with private costs as the shuffled variable.
- **Size and cost:** 2 models × 5 capacities × 40 sessions, about 15 GPU-hours.

**Main failure modes.**
- A real regulator rarely sees private shocks with timestamps. The usable version needs natural experiments: station-level outages in FuelCheck, or firm-specific cost events. Otherwise it becomes a T2 design in which the regulator *creates* the private shock, for example by selling one firm a discounted input.
- Colluders can choose not to condition on private information and coordinate on public focal prices. That is class B, which this test is designed not to catch.

### Direction 3: N/K groups × K same-type agents trained inside the system (covers item ii)

**Question.** When agents keep learning in deployment and start from shared weights in groups, does collusion form within groups, across groups, or neither? Does its structure follow the initialisation?

**Sharpened design.** N = 8 sellers in a repeated Bertrand market with logit demand (the Calvano/Fish parameterisation). The groups:
- K ∈ {1, 2, 4, 8}, giving N/K groups;
- each group has either (a) **one shared LoRA adapter**, the "provider deploys one fine-tune" case, which is parameter sharing, or (b) **K separate adapters with the same initialisation**, the "continual per-instance learning" case;
- all groups start from the same base model (Qwen2.5-7B);
- a heterogeneous-base arm uses Qwen2.5 / Qwen3 / Olmo-3 per group.

Rewards come in two arms:
- **individual profit:** collusion, if it appears, is emergent;
- **group profit:** this models provider-internal coordination and gives a positive control.

Training uses REINFORCE/GRPO with per-agent advantage normalisation and fixed rollout length.

**What to measure every 10 iterations.**
- Collusion index Δ = (π − π_Nash)/(π_Monopoly − π_Nash), per group and overall.
- Residual coupling within vs across groups, using our pipeline: common mode removed, multi-lag rho_DCCA, and the exact-coincidence channel.
- Community detection on the residual-coupling graph, and its Adjusted Rand Index against the group labels.
- Probe responses from Direction 1 (identity interaction, and punishment after a forced deviation).
- A static-vs-reactive decomposition: price level, and rigidity vs lag-1 reactivity. The E3 lesson is that high prices can be static and focal.
- Weight-space divergence between adapters (LoRA subspace overlap) against behavioural coupling.

**Hypothesis H3.**
- Under individual reward with separate adapters, within-group coupling exceeds cross-group coupling (ARI > 0.5) by mid-training. Δ rises with K, because shared initialisation makes symmetric collusive equilibria easier to reach.
- The heterogeneous-base arm shows lower Δ, as Keppo et al. 2026 found for prompted agents.
- Falsified if ARI ≈ 0, or if Δ is the same across K.

**Why it is not done.**
- The Alignment Tipping Process (Han et al. 2025) shows drift and imitative diffusion in stylised games, without market structure or group-structured initialisation.
- Piche et al. 2025 train LLM agents in social dilemmas with an agent buffer, not shared-init groups.
- Shi et al. 2026 vary memory, not weights.
- Keppo et al. 2026 vary heterogeneity among *prompted*, not trained, agents.
- Parameter-sharing MARL (Terry et al. 2020; Christianos et al. 2021) studies performance, not collusion.

**Cost.**
- Per iteration: 256 episodes × 20 rounds × 8 agents ≈ 41k short generations, about 2–4 min on one H100 with vLLM, plus a LoRA step.
- 200 iterations is about 10–15 GPU-hours per run.
- 4 values of K × 2 sharing modes × 2 reward types × 2 seeds is about 32 runs, or 350–500 GPU-hours. That is roughly one week on 4 H100s.
- Do a pilot first on Qwen2.5-1.5B at about one-fifth the cost.

**Main failure modes.**
- Multi-agent RL with LLMs is unstable: shared-baseline variance and collapse.
- Agents may learn format hacks or a static high price rather than reactive collusion.
- With a shared adapter, "within-group coupling" is trivially high because it is one policy. Compare only across conditions and arms, never against zero.

### Direction 4: active early warning of learned collusion (covers item iv)

**Position.** Passive EWS (Hurst, MF width, variance, AR(1)) assume a slow parameter drifting towards a bifurcation in the observed series. Learned collusion is a change in the *policy*, so the known failure modes apply (Hastings & Wysham 2010; Boettiger & Hastings 2012), as our E1 and E13 results confirm. **Reject passive multiscale EWS as a primary line.**

**Measure the mechanism actively instead.** Critical slowing down means slow recovery from small perturbations (Scheffer et al. 2009). In Direction 3 runs, inject a small forced deviation every k iterations and measure recovery half-life, punishment amplitude and the Direction 1 identity interaction.

**Hypothesis H4.** These active indicators rise before Δ crosses 0.5 and give a lead-time AUC ≥ 0.75, while passive Hurst/MF width stays ≤ 0.6 on the same runs. Falsified if the active indicators move only with or after Δ.

**Closest work.** Calvano et al.'s impulse responses are measured after convergence, not as leading indicators. Bury et al. 2021 is passive. **Cost:** close to zero beyond Direction 3, about 10% extra rollouts. **Failure mode:** collusion in LLM agents can be immediate rather than learned. Fish agents start high (our E3), so there may be no onset to warn about. Direction 3's individual-reward arm must actually show a rise.

### Direction 5: base-model attribution from market actions

Can a regulator tell an agent's base model from prices and timing alone? If yes, mixed-provider controls (A′) and provider-share concentration measures need no disclosure; Eschenbaum 2026 shows ownership-based concentration misses same-provider effects.

**H5.** A sequence classifier over 50-round price paths identifies the base model among Qwen2.5-7B, Qwen3-8B and Olmo-3-7B at ≥ 80% across 10 held-out prompts and 3 LoRA fine-tunes per base. **Closest work:** LLMmap, Sun et al. 2025 and Nikolić et al. 2025 use text; arXiv:2512.07462 classifies strategies, not models, and finds framing matters as much as architecture. **Cost:** 20–40 GPU-hours. **Failure mode:** three bases is a toy problem and prompts may dominate, so report it as a lower bound.

### Direction 6: covert price channels when the cover model is open-weight

Monoculture cuts both ways. Colluders share a codebook through common focal points, while a regulator with open weights can evaluate the exact "innocent" action distribution and run likelihood-ratio steganalysis.

**H6.** Without a shared key, price encoding above O(√n) bits over n rounds is detected. With a key fine-tuned in, minimum-entropy-coupling schemes are undetectable (Schroeder de Witt et al. 2023), so the open problem is key distribution, which a provider controls.

**Why ranked low.** The theory largely exists (Cachin; Bash et al.), and the test needs each agent's private prompt. A bounded side study, about 30 GPU-hours.

**Rejected.** New multifractal detectors and passive EWS (refuted by our audits). Text-channel steganography detection (well covered by NARCBench, ColludeBench and Mathew et al., and outside market-only observability). "Show that homogeneity raises collusion" (covered by Keppo et al. 2026 and Assad et al. 2024; only identification is new).

---

## 5. Recommended next step: Direction 1, the identification ladder

**Why this one.** It answers the team's recurring question (monoculture or collusion?) at the open gap in the literature: structural IO cannot separate B from C, and screens fail under a shared prior. It needs inference only, reuses the E8 sellers and the seeded-channel patch, and supplies the probes that Directions 3 and 4 need as collusion meters. It maps to "Multi-Agent Oversight and Control" and "Science of Agent Networks".

The deliverable is a table: which tier (T1, T2, T3) identifies which class (A, B, C), with calibrated false-positive rates on same-model populations.

**Week 1: harness and nulls**
1. Day 1–2. Extend the E8 market runner to three base models: Qwen2.5-7B, Qwen3-8B and Olmo-3-7B, served through vLLM with logprobs. Add a mixed-provider configuration (A′) and a deterministic replay API: rebuild any agent's prompt at round t from the logged public history.
2. Day 3. Write the history-grafting probe. Swap the deviator's past price signature with one taken from a same-model or a different-model session, holding the deviation round fixed. Add placebo grafts (same model, other session) and a perplexity check.
3. Day 4. Write the T2 probe: a scripted entrant with configurable price signatures that makes one 15% cut at a random round.
4. Day 5. Run A and A′ only (3 models × 40 sessions) and fit the null distribution of the identity-interaction statistic. **Gate:** the statistic must be centred at 0 under A, with the placebo matching. If not, fix the grafting before going further.

**Week 2: signal and decision**
1. Day 6–7. Run B (kin-reveal prompt) and C (seeded channel). For C, use two channel strengths, unrestricted and 1 bit per round, so the same runs also pilot Direction 2.
2. Day 8. Compute AUCs for A vs B∪C at T3 and T2. Compare against the T1 baselines: our cleaned multi-lag rho_DCCA, exact coincidence, and state-conditioned influence.
3. Day 9. On the C runs, run the no-signalling test against logged private costs as the Direction 2 pilot.
4. Day 10. Write-up and decision:
   - **Go to Direction 3** if T3 AUC ≥ 0.8 and the A′ false-positive rate is ≤ 5%. The probes then become the collusion meter for in-system training.
   - **If T3 works but T2 does not**, frame the result as an argument for audit access (T3) in policy.
   - **If B is empty** (7B agents do not condition on identity), report that and move B to the trained agents from Direction 3.

**Budget.** About 30–50 GPU-hours on one or two H100s. No downloads beyond models already on the server, and no paid APIs.

---

## References

Every entry was checked on 2026-10-25 against arXiv, a DOI page or a publisher/proceedings listing found by search, except where marked.
- **[R]** = taken from `/workspace/projects/CertAudit/RELATED_WORK.md` (verified there the same day) and not re-fetched.
- **[sec]** = facts from secondary or legal summaries only.
- **[part]** = some bibliographic fields not confirmed.

**Algorithmic and LLM collusion**
- Calvano, Calzolari, Denicolò & Pastorello (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *AER* 110(10):3267–97. doi:10.1257/aer.20190623
- Fish, Gonczarowski & Shorrer (2024; v5 2026). Algorithmic Collusion by Large Language Models. arXiv:2404.00806
- Lin, Ojha, Cai & Chen (2024). Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions. arXiv:2410.00031
- Agrawal, Teo, Vazquez, Kunnavakkam, Srikanth & Liu (2025). Evaluating LLM Agent Collusion in Double Auctions. arXiv:2507.01413
- Keppo, Li, Tsoukalas & Yuan (2026). On the Fragility of AI Agent Collusion. arXiv:2603.20281
- Garra (2026). Mitigating Emergent Collusion in LLM Pricing Agents. arXiv:2609.13037
- Lee & Park (2026). Faithful yet Collusive. arXiv:2609.18346 [R]
- Shi, Zhang & Yang (2026). Emergent Collusion in Long-Horizon LLM Agent Interaction. arXiv:2609.24967 [R]
- Hansen, Misra & Pai (2021). Frontiers: Algorithmic Collusion: Supra-competitive Prices via Independent Algorithms. *Marketing Science* 40(1):1–12. doi:10.1287/mksc.2020.1276
- Banchio & Mantegazza. Artificial Intelligence and Spontaneous Collusion (earlier titles: "Adaptive Algorithms, Tacit Collusion, and Design for Competition"; "…Statistical Collusion"). arXiv:2202.05946
- Hartline, Long & Zhang (2024). Regulation of Algorithmic Collusion. CSLAW '24; arXiv:2401.15794
- Bonjour, Aggarwal & Bhargava (2022). Net-influence collusion detection, UAI 2022. Cited from FINDINGS.md, not re-verified.

**Common provider, hub and spoke, empirical IO**
- Harrington (2022). The Effect of Outsourcing Pricing Algorithms on Market Competition. *Management Science* 68(9):6889–6906. doi:10.1287/mnsc.2021.4241
- Harrington (2018). Developing Competition Law for Collusion by Autonomous Artificial Agents. *J. Competition Law & Economics* [part: the working paper is confirmed; volume and pages are not]
- Assad, Clark, Ershov & Xu (2024). Algorithmic Pricing and Competition: Empirical Evidence from the German Retail Gasoline Market. *JPE* 132(3):723–771. doi:10.1086/726906
- Calder-Wang & Kim (2026 version). Algorithmic Pricing in Multifamily Rentals: Efficiency Gains or Price Coordination? Working paper (FTC-hosted PDF) [part]
- Eschenbaum (2026). Shared Bidding Algorithms and Competition: Evidence from Electricity Markets. arXiv:2607.13002
- U.S. v. RealPage: proposed final judgment filed 24 Nov 2025 [sec]. U.S. v. Agri Stats: settlement May 2026 [sec]. *Gibson v. Cendyn Group*, 9th Cir. No. 24-3576, 15 Aug 2025 [sec].

**Screens and identification**
- Abrantes-Metz, Froeb, Geweke & Taylor (2006). A Variance Screen for Collusion. *IJIO* 24(3):467–486
- Bajari & Ye (2003). Deciding Between Competition and Collusion. *REStat* 85(4):971–989. doi:10.1162/003465303772815871
- Chassang, Kawai, Nakabayashi & Ortner (2022). Robust Screens for Noncompetitive Bidding in Procurement Auctions. *Econometrica* 90(1):315–346. doi:10.3982/ECTA17155
- Huber & Imhof (2019). Machine Learning with Screens for Detecting Bid-Rigging Cartels. *IJIO* 65:277–301
- Bresnahan (1982). The Oligopoly Solution Concept Is Identified. *Economics Letters* 10:87–92

**Monoculture and correlated behaviour**
- Kleinberg & Raghavan (2021). Algorithmic Monoculture and Social Welfare. *PNAS* 118(22); arXiv:2101.05853
- Bommasani, Creel, Kumar, Jurafsky & Liang (2022). Picking on the Same Person: Does Algorithmic Monoculture Lead to Outcome Homogenization? NeurIPS 2022; arXiv:2211.13972
- Kim, Garg, Peng & Garg (2025). Correlated Errors in Large Language Models. ICML 2025; arXiv:2506.07962
- Goel et al. (2025). Great Models Think Alike and This Undermines AI Oversight. ICML 2025; arXiv:2502.04313
- Song (2026). Information Homogenization Induces Herding in Retrieval-Augmented LLM Agent Markets. ICLR 2026 [part: conference listing only]
- Panickssery, Bowman & Feng (2024). LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS 2024; arXiv:2404.13076
- Aharon, La Malfa, Wooldridge & Kraus (2026). Tacit Coordination of Large Language Models. arXiv:2601.22184

**Steganography and covert channels**
- Cachin (1998). An Information-Theoretic Model for Steganography. Information Hiding, LNCS 1525:306–318. doi:10.1007/3-540-49380-8_21
- Bash, Goeckel & Towsley (2013). Limits of Reliable Communication with Low Probability of Detection on AWGN Channels. *IEEE JSAC*; arXiv:1202.6423
- Schroeder de Witt, Sokota, Kolter, Foerster & Strohmeier (2023). Perfectly Secure Steganography Using Minimum Entropy Coupling. ICLR 2023; arXiv:2210.14889
- Motwani, Baranchuk, Strohmeier, Bolina, Torr, Hammond & Schroeder de Witt (2024). Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. NeurIPS 2024; arXiv:2402.07510
- Mathew, Matthews, McCarthy, Velja, Schroeder de Witt, Cope & Schoots (2024). Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs. arXiv:2410.03768
- Zolkowski, Nishimura-Gasparian, McCarthy, Zimmermann & Lindner (2025). Early Signs of Steganographic Capabilities in Frontier LLMs. arXiv:2507.02737
- Tailor (2025). Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs. arXiv:2510.04303
- Ghanem (2026). Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations. arXiv:2608.02698 [R]
- Rose et al. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv:2604.01151 [R]

**Fingerprinting and provenance**
- Pasquini, Kornaropoulos & Ateniese (2025). LLMmap: Fingerprinting for Large Language Models. USENIX Security 2025; arXiv:2407.15847
- Sun, Yin, Xu, Kolter & Liu (2025). Idiosyncrasies in Large Language Models. ICML 2025; arXiv:2502.12150
- Nikolić, Baluta & Saxena (2025). Model Provenance Testing for Large Language Models. NeurIPS 2025; arXiv:2502.00706
- Foley et al. (2023). Matching Pairs: Attributing Fine-Tuned Models to their Pre-Trained Large Language Models. ACL 2023; arXiv:2306.09308
- Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics. arXiv:2512.07462 [part: authors not checked]

**Learning dynamics and MARL**
- Han et al. (2025). Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails. arXiv:2510.04860 [part: models used not confirmed from the abstract]
- Piche, Muqeeth, Aghajohari, Duque, Noukhovitch & Courville (2025). Learning Robust Social Strategies with Large Language Models. arXiv:2511.19405
- Vallinder & Hughes (2024). Cultural Evolution of Cooperation among LLM Agents. arXiv:2412.10270
- Terry et al. (2020). Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning. arXiv:2005.13625
- Christianos et al. (2021). Scaling Multi-Agent RL with Selective Parameter Sharing. ICML 2021; arXiv:2102.07475
- Jaderberg et al. (2017). Population Based Training of Neural Networks. arXiv:1711.09846
- Foerster et al. (2018). Learning with Opponent-Learning Awareness. arXiv:1709.04326

**Early warning**
- Scheffer et al. (2009). Early-Warning Signals for Critical Transitions. *Nature* 461:53–59. doi:10.1038/nature08227
- Hastings & Wysham (2010). Regime Shifts in Ecological Systems Can Occur with No Warning. *Ecology Letters* 13:464–472. doi:10.1111/j.1461-0248.2010.01439.x
- Ditlevsen & Johnsen (2010). Tipping Points: Early Warning and Wishful Thinking. *GRL* 37. doi:10.1029/2010GL044486 [part: full text not read]
- Boettiger & Hastings (2012). Quantifying Limits to Detection of Early Warning for Critical Transitions. *J. R. Soc. Interface* 9(75):2527–2539. doi:10.1098/rsif.2012.0125
- Diks, Hommes & Wang (2019). Critical Slowing Down as an Early Warning Signal for Financial Crises? *Empirical Economics* 57(4):1201–1228. doi:10.1007/s00181-018-1527-3
- Bury et al. (2021). Deep Learning for Early Warning Signals of Tipping Points. *PNAS* 118:e2106140118

**Programmes, agendas, testbeds, audit access**
- Schmidt Sciences et al. (2026). Scaling AI Safety for a Multi-Agent World (call page at cooperativeai.com/grants/multi-agent-safety; Google DeepMind blog, 11 Jun 2026)
- Hammond et al. (2025). Multi-Agent Risks from Advanced AI. Cooperative AI Foundation Tech. Rep. #1; arXiv:2502.14143
- Tomašev, Franklin, Jacobs, Krier & Osindero (2025). Distributional AGI Safety. arXiv:2512.16856
- Schroeder de Witt et al. (2025). Open Challenges in Multi-Agent Security. arXiv:2505.02077 [R]
- Casper et al. (2024). Black-Box Access is Insufficient for Rigorous AI Audits. arXiv:2401.14446
- Microsoft Research (2025). Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets. arXiv:2510.25779
