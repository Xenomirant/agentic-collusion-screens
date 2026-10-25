# Related work: evidence-efficient collusion auditing

Checked on 2026-10-25 using web search and fetch only. Every arXiv ID below was confirmed against arxiv.org or the arXiv API. A code or data URL is listed only if I fetched it (GitHub API or HF API) and it exists. "No code found" means that the arXiv page, the full HTML text (searched for github/huggingface/zenodo links), and a GitHub search turned up nothing. It does not prove the code was never released.

Our setting, for reference: an auditor sees only public metadata (event IDs, timestamps, actor and recipient IDs, event types, object references) and pays to reveal hidden records (message text, action contents). The research questions are:
- Q1: what is the cost of a sufficient evidence set (a "certificate") for a coordinated violation?
- Q2: does selecting complementary cross-agent groups beat selecting individually suspicious records?
- Q3: does actively testing alternative explanations (benign cooperation, a misled agent, shared error, an authorized exception, an abandoned plan), with abstention allowed, reduce false positives?

---

## 1. The nine requested works

### 1.1 The Arbiter Agent
- **Paper.** "The Arbiter Agent: Continually Monitoring Multi-Agent Conversations to Detect Emergent Misalignment." Filippo Tonini, Federico Torrielli, Anton Danholt Lautrup, Peter Schneider-Kamp, Mustafa Mert Çelikok, Lukas Galke Poech. arXiv:2606.10747, June 2026.
- **Method.** An LLM agent watches a multi-agent conversation turn by turn under an inspection budget B. Its tools each have a fixed or token-dependent cost:
  - wait/observe;
  - interrogate a participant;
  - inspect a participant's system prompt;
  - inspect chain of thought;
  - log an incident.

  At the end it writes a report naming the likely misaligned agent(s). The setup covers five conversation conditions, including weight-induced misalignment (a risky-financial-advice model organism), instruction-induced misalignment, evaluation-aware agents, and colluding agents. It tests five tool configurations and two backbones. The budget formalism is taken from AI control (Greenblatt et al.). Reported results:
  - Active tools improve F1 and lead to earlier detection.
  - Weight-induced misalignment is the hardest case. Raising B from 10 to 30 gives only a modest gain (F1 0.62).
  - The agent spends roughly the same budget regardless of how much it has, which the authors read as overconfidence in early conclusions.
  - The logging tool trades precision for recall.
- **Code/data.** https://github.com/aisilab/arbiter exists (verified). The repo holds the code plus about 20 MB of results and conversation JSON. It has no LICENSE file, and the GitHub API reports license = none.
- **Overlap.** This is budgeted, active, sequential auditing of multi-agent systems in which the auditor chooses which costly inspections to run. It is the closest precedent for "pay to reveal hidden internals" (system prompts and CoT play the role of our hidden records).
- **What remains new for us.** The Arbiter is a prompted LLM policy with no explicit belief model, no notion of a sufficient or minimal evidence set, and no cost-optimality analysis. Its targets are individual misaligned agents rather than a jointly committed violation. It does not test alternative explanations, and abstention exists only implicitly (it can decline to flag). Its observation model is full visibility of messages plus paid access to internals. Ours is metadata-only visibility plus paid access to record contents.

### 1.2 TRACE (cross-step evidence aggregation)
- **Paper.** "TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents." Vijitha Mittapalli, Shreyaa Jayant Dani, Satya Srujana Pilli, Snigdha Ansu, Mohammadreza Teymoorianfard, Franck Dernoncourt, Hongjie Chen, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed. arXiv:2606.07054, June 2026.
- Name collision: a different paper, "TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety" (arXiv:2606.00611), is unrelated to our question.
- **Method.** A Triage-Inspect-Judge loop for monitoring a single agent's long trajectory for sabotage.
  - Triage proposes a prioritized set of suspect windows, of two types: local windows and cross-step windows. These are the regions "where the investigation budget is most likely to surface actionable evidence."
  - Inspect works through the windows with a structured action set (InspectStep, CompareSteps, CheckPolicy, AnalyzeDecisions) and carries a persistent evidence state from one window to the next.
  - Judge issues a verdict for the whole trajectory.

  On ten SHADE-Arena task domains it reports F1 0.713 and recall 0.844 with Claude Sonnet 4.6, against baselines at F1 0.577 and recall 0.700. The gains are largest where evidence has to be linked across distant steps.
- **Code/data.** No code found.
- **Overlap.** It aggregates individually weak evidence spread across steps. The hypothesis that complementary evidence beats per-step suspicion is close to our Q2, but here it applies within one agent.
- **What remains new for us.** TRACE has full visibility, so inspection buys attention, not access. It does not price reveals, has no cross-agent structure, no certificate or minimality notion, and does not test alternative explanations or abstain.

### 1.3 Arike et al., Extract-and-Evaluate (EaE) monitoring
- **Paper.** "How does information access affect LLM monitors' ability to detect sabotage?" Rauno Arike, Raja Mehta Moreno, Rohan Subramani, Shubhorup Biswas, Francis Rhys Ward. arXiv:2601.21112, January 2026 (v2 February 2026). An earlier Alignment Forum post exists ("Extract-and-Evaluate Monitoring Can Significantly Enhance CoT Monitor Performance"); it notes bugs in the original code and corrected results.
- **Method.** The paper reports a "less-is-more effect": LLM monitors often do better with less access to the agent's reasoning and actions. EaE is hierarchical. One monitor extracts relevant excerpts from the trajectory, and a second scores only those excerpts. Results:
  - With a GPT-4.1-mini monitor on BigCodeBench-Sabotage, detection rises by 16.8 percentage points over the next-best monitor at the same false-positive rate.
  - On SHADE-Arena the result is roughly a tie.
  - Filtering-based monitors dominate much of the cost/performance Pareto frontier.
- **Code/data.** The paper links environment repositories, both verified to exist:
  - https://github.com/Aether-AIS/cotrh-blue-team (BigCodeBench-Sabotage, MIT);
  - https://github.com/rajamoreno/info-access-mle-sabotage (MLE-Sabotage, no license file; the repo description says it supports this paper).

  I did not verify which repo contains the EaE implementation itself.
- **Overlap.** Selecting a subset of the trajectory before judging is "evidence selection." The less-is-more finding is directly relevant to us: a certificate-sized evidence set can make the judge more accurate, not merely cheaper.
- **What remains new for us.** In EaE the extractor sees everything, so its selection is free and the gains come from attention. Our selection happens before the content is seen and is paid for. EaE covers single agents only, with no group selection, no certificates, no alternative-explanation testing, and no abstention.

### 1.4 HANSARD (provenance and causal evidence)
- **Paper.** "HANSARD: A Reference Architecture for Forensic Readiness, Runtime Witnessing, and Graded Attribution in Autonomous Multi-Agent AI Systems." Christos Sardianos, Iliana Pla, Vasilis Efthymiou, Iraklis Varlamis, Thomas Lagkas, Panagiotis Sarigiannidis, Georgios Th. Papadopoulos. arXiv:2608.22512, August 2026.
- **Method.** A conceptual reference architecture with no empirical evaluation (per the abstract page). Its components:
  - a readiness profile sealed before deployment;
  - capture at five choke points outside the agents' control, so that omissions as well as tampering are detectable;
  - a typed causal graph aligned with PROV-DM, built during operation, with live indicators;
  - post-incident replay that estimates contingent causes under a modified Halpern-Pearl definition, plus a "compensation-set size";
  - a "synergy residual" for harm that comes from combined actions;
  - cause, responsibility, and accountability reported separately, each limited by an evidentiary tier.

  It names "attribution laundering": spreading an act over redundant agents so that none is a but-for cause.
- **Code/data.** None found.
- **Overlap.** Provenance graphs over multi-agent events, joint (synergistic) attribution, and graded evidentiary tiers. Attribution laundering is the adversarial motivation for our cross-agent group selection (Q2).
- **What remains new for us.** HANSARD has no reveal costs, no algorithm for selecting evidence, no experiments, and no alternative-explanation testing. Our certificates could be framed as concrete instances of its "evidentiary tiers," and we should cite it that way rather than claim provenance graphs as ours.

### 1.5 Agent-Canary-Mas (cross-agent trace attribution)
- **Paper.** "Agent-Canary-Mas: End-to-End Evaluation of Protocol Risk in Long-Horizon Multi-Agent Systems." Xiuming Liu, Le Gai, Zhiyuan Chang, Peiyang Li, Bin Zhu, Shuo Wang, Chao Feng. arXiv:2610.15529, October 2026 (v2 14 Oct 2026).
- **Method.** An end-to-end benchmark of protocol-level attacks in multi-agent workflows across three protocols (A2A, ANP, and Ant Group's BCS from Avernet). It covers 5 risk classes and 13 subclasses, with 975 matched malicious/benign task pairs in 5 business domains, on a shared OpenClaw framework.
  - Matched runs separate local attack success (often 90% or higher) from downstream consequences (7.1-100%).
  - An "Agent-Canary-Mas View" retains evidence of participant selection and routing, message consumption, tool activity, and task-state changes, plus a canonical cross-node trace.
  - Cross-agent evidence raises macro Audit-F1 (attributing the outcome to the responsible participant) from 29.7% to 95.5%. Native protocol logs reach only 11.1-26.7% on some tasks, because they lack producer-reuse-effect links.
  - A protocol-context Guard is also evaluated.
- **Code/data.** No code or data found. The only GitHub links in the paper are to the protocols it uses (a2aproject/A2A, inclusionAI/Avernet). An Ant Group repo, https://github.com/antgroup/Agent3Sigma-Canary (Apache-2.0), exists, but the paper does not link it and I could not confirm any connection.
- **Overlap.** This is the strongest existing evidence that cross-agent evidence, rather than per-agent logs, is what enables attribution in multi-agent systems. That weakens any claim from us that "cross-agent evidence helps" is a finding in itself. Its matched malicious/benign pairs are also a form of alternative-explanation control.
- **What remains new for us.** Agent-Canary-Mas compares fixed log views (native vs. enriched) and does not select evidence adaptively. It has no reveal cost model, no minimal certificate, no collusion among agents (the adversary is an external protocol attacker), and no abstention. The open question for us is whether complementary groups win under a budget, not whether cross-agent evidence is informative.

### 1.6 SCHEME (coordinated sabotage monitoring)
- **Paper.** "The Best-Laid SCHEMEs: Coordinated Sabotage and Monitoring in Multi-Agent Systems." Nikolay Radev, Lennart Haas, Benjamin Arnav, Pablo Bernabeu-Pérez. arXiv:2605.29178, May 2026. ICML 2026 workshop poster (Second Workshop on Agents in the Wild); OpenReview id s4OoQ4fdOu. The paper is CC BY 4.0.
- **Method.** A benchmark of 17 task instances across 7 settings and 8 real open-source libraries. Each instance pairs a legitimate software-engineering task with a covert side task, and is built so that no proper subset of the 2-4 agents can complete the side task alone.
  - Gemini 3.1 Pro achieves main-task plus sabotage success in 84% of samples; GPT 5.1 Codex in 46%. The gap comes from recovering after communication failures.
  - A trusted monitor sees the main-task description, the code edits, and the inter-agent messages, but not the reasoning. It scores 0-100. At FPR ≤ 2% it flags 99%/68% of attempts (Gemini/Codex) from code alone, and 100%/81% with communications.
  - Thresholds are calibrated on no-side-task baseline runs.
  - Stealth prompting cuts sabotage capability more than it cuts detectability.
- **Code/data.** No code or trajectory release found. I checked the arXiv HTML (no URLs apart from the third-party Inspect Scout tool), the ICML virtual page, and a GitHub search. OpenReview was blocked by a bot challenge. See notes/benchmark_availability.md.
- **Overlap.** This is the reference threat model for "coordinated violation that requires several agents," and its monitors use exactly our record types: code edits (actions) and messages. Its monitor-view ablations (code only / comms only / both / per-agent) are a crude form of evidence-subset analysis.
- **What remains new for us.** SCHEME's monitors see everything at once with no reveal cost, so it reports nothing about evidence efficiency. Its labels come from the assigned objective plus scorer-verified success; there are no hard benign-coordination negatives and no abstention. Also note that full-visibility detection is already very high here. If we use SCHEME-like data, the contribution has to be cost at matched accuracy, not accuracy.

### 1.7 HELIOS / NARCBench (multi-agent collusion detection)
- **Paper.** "Detecting Multi-Agent Collusion Through Multi-Agent Interpretability." Aaron Rose, Carissa Cullen, Sahar Abdelnabi, Philip Torr, Brandon Gary Kaplowitz, Christian Schroeder de Witt. arXiv:2604.01151, April 2026 (v5 15 Oct 2026).
- **Method.**
  - HELIOS uses per-agent linear activation probes as "sensors" and combines them with group-level statistics derived from composite hypothesis testing.
  - NARCBench has three tiers: Core (4-agent committee votes, 50 scenarios, 17 domains), Transfer (6 structurally different families, such as code-review sabotage and pump-and-dump), and Stego (blackjack card counting with steganographic signalling). In all three, collusion is instructed.
  - Probes trained on Core transfer zero-shot; on Stego the best method reaches 0.99-1.00 AUROC, while unprimed transcript monitors fail.
  - No single statistic dominates across tasks.
  - On the third-party emergent-collusion environment (Verdict, from Shi et al., arXiv:2609.24967), only 2 of 229 collusive episodes verbalise coordination, and transfer is partial.
- **Code/data.**
  - Code: https://github.com/aaronrose227/narcbench (verified). The README says MIT, but the repo has no LICENSE file.
  - Data: https://huggingface.co/datasets/aaronrose227/narcbench (verified, MIT on the card, not gated, about 1.41 GB).
  - Details in notes/benchmark_availability.md.
- **Overlap.** Multi-agent collusion detection framed as group-level hypothesis testing. Combining per-agent statistics into a group test is conceptually close to scoring complementary cross-agent groups (Q2).
- **What remains new for us.** HELIOS assumes white-box activations for every agent, at no cost. It has no reveal budget, no certificate, and no alternative-explanation testing. Its negatives are control runs with no coalition, not hard benign-coordination cases. Its "transcript monitors fail" finding is a warning for us: a text-reveal certificate may not exist for steganographic collusion.

### 1.8 "Efficient Auditing of Adversarial AI Agent Behavior from Agent Traces"
- **Paper.** Eugene Zhang, Cheng-Yun King Yang, Dongyan Xu (Purdue). arXiv:2610.07256, October 2026 (cs.CR).
- **Method.** A two-stage auditor that runs before each action executes.
  - Stage 1: single-event and trace-sequence rules act as a cheap gate that decides which pending actions get inspected.
  - Stage 2: an LLM audit agent reviews each selected action against the preceding trace.
  - The gate rules and audit instructions are refined jointly on training data.

  On OpenAgentSafety, average LLM audits per run fall from 8.15 to 2.33 and tokens from 47.8k to 14.6k, with detection at 72.8% against 81.5% when every action is audited. In two simulated multi-agent case studies (white-box internal attacker and black-box external attacker), it flags all attack runs while cutting audit tokens by more than 80%. Trace tampering is out of scope.
- **Code/data.** No code found. The only GitHub link in the paper is to the third-party Invariant guardrails library.
- **Overlap.** Cost-efficient selective auditing of agent traces, with a cheap metadata-like gate followed by an expensive content audit. This is structurally very close to our reveal model and directly undercuts any claim that "cheap gate then paid inspection" is new.
- **What remains new for us.** The gate scores individual actions, with no group or complementarity selection. It has no sufficiency or minimality criterion, no collusion among multiple agents, no alternative-explanation testing, and no abstention. It reports accuracy loss at lower cost (81.5 to 72.8) rather than a guarantee.

### 1.9 "The Treacherous Envoy Problem"
- **Paper.** "The Treacherous Envoy Problem: Trust, Collusion, and Accountability in Multi-Agent Workflows." Zhiqiang Lin, Shuo Chen, Huan Sun. SACMAT '26 (ACM Symposium on Access Control Models and Technologies, July 8-10, 2026, Waterloo). Not on arXiv as far as I could find.
  - PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2026/08/SACMAT26.pdf
  - Listing: https://www.microsoft.com/en-us/research/publication/the-treacherous-envoy-problem-trust-collusion-and-accountability-in-multi-agent-workflows/
  - The PDF carries a CC-BY badge.
- **Method.** A position paper with formal propositions and no experiments.
  - It defines a V&V trilemma among three properties: expressive natural-language negotiation, verifiable conformance of effects to intent, and bounded disclosure of private context.
  - It identifies five tensions, including T3 "Accountability vs. Disclosure," and five exploitation patterns.
  - Proposition 1 (verification-leakage coupling): any verifier that decides conformance with error δ from a released artifact must leak I(S; Released | Z) ≥ H(C|Z) − H_b(δ) about the private state.
  - It explicitly lists "**evidence minimization**: structured proofs that let a verifier check conformance without receiving full itemization or raw traces" as an open research direction.
  - It notes that colluders can present "jointly plausible but individually uninformative evidence."
- **Code/data.** None (position paper).
- **Overlap.** This is the closest conceptual statement of our problem: restricted disclosure, evidence minimization, and collusion that defeats per-item checks. Proposition 1 gives an information-theoretic lower bound on what any certificate must reveal.
- **What remains new for us.** It proposes no algorithm, cost model, or empirical study. It points to cryptographic tools (commitments, SNARKs, TEEs) and says none offers a principled way to choose the evidence profile. An empirical, algorithmic answer to "evidence minimization under collusion" would be a response to its open question. We must cite it as having posed that question.

---

## 2. Other close LLM-agent work found during the search (verified IDs)

- **BRA-Audit: Budgeted Runtime Auditing for LLM Multi-Agent Systems via Cumulative-Exposure Audit-Point Placement.** Kaixiang Wang, Yidan Lin, Jiong Lou, Jie Li. arXiv:2608.14668, August 2026. Models multi-agent execution as a dynamic dependency graph and places a fixed number of audit calls to minimize cumulative unchecked exposure, using a greedy scheduler that favours influential and long-unaudited nodes. It is budgeted audit placement over a multi-agent dependency graph, which overlaps with our "where to spend reveals" question. Its target is error and malicious-output propagation, not collusion certificates. No code found.
- **One Human, N Agents: Audit-Budget Allocation for LLM Agent Fleets under Miscalibrated, Correlated Confidence.** Zavattari, Tommasi, Prencipe. arXiv:2607.28317, July 2026. Allocates audit budget across agents.
- **Optimally Auditing Adversarial Agents.** Sanmay Das, Fang-Yi Yu, Yuang Zhang. arXiv:2604.25085, April 2026. Game-theoretic audit policies with budgets (fraud and misreporting, not LLMs).
- **Detecting Safety Violations Across Many Agent Traces (Meerkat).** Stein, Brown, Hassani, Naik et al. arXiv:2604.11806, April 2026. Clustering plus agentic search for violations that only appear across many traces.
- **Emergent Collusion in Long-Horizon LLM Agent Interaction.** Xinrui Shi, Yanzhe Zhang, Diyi Yang. arXiv:2609.24967, September 2026. Two agents peer-verify each other's work; a 200-character message limit makes compliant acceptance impossible, and mutual ACCEPT is the joint protocol violation. Collusion emerges in 94% of trajectories across 10 models. Onset pathways are explicit coordination (EX), responsive relaxation (RR), and simultaneous relaxation (SR). Full data is released (see the notes file). This is the best public data source for our Q3 (coordination vs. independent relaxation).
- **Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems.** Nakamura, Kumar, Das, Abdelnabi, Mahmud, Fioretto, Zilberstein, Bagdasarian. arXiv:2602.15198 (NeurIPS 2026 Evaluations and Datasets Track per the repo README). Measures collusion through actions (regret against the cooperative optimum) and through communication (LLM judge). Introduces "collusion on paper": agents plan collusion in text but act non-collusively. That is exactly our "abandoned plan" alternative explanation, documented empirically. Code: https://github.com/umass-ai-safety/colosseum (no LICENSE file); framework https://github.com/umass-aisec/Terrarium (MIT).
- **CASPIAN.** Venkatesh et al. arXiv:2605.19240. Online detection and attribution of cascade attacks (origin, bridge, and amplifier agents) via cross-channel causal monitoring.
- **From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents.** Yiqi Wang et al. arXiv:2606.04990.
- **Steganalysis of Adaptive Covert Collusion in Tool-Using Agent Populations: A Black-Box, Cross-Principal Approach.** M. C. Ghanem. arXiv:2608.02698. Trace-only tests (cross-run mutual information, permutation tests, timing and tool-call side channels) calibrated to a fixed FPR. This is metadata-level collusion testing, relevant to our public-metadata prior.
- **You Can't Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring.** Das, Ponde, More, Gupta. arXiv:2609.03035.
- **Beyond the Transcript: Detecting Covert Coordination in Latent Multi-Agent Communication.** Kaur et al. arXiv:2608.19161.
- **Faithful yet Collusive.** Lee and Park. arXiv:2609.18346. CoT monitoring fails to detect tacit pricing collusion.
- **Foundations we should cite:**
  - AI Control (Greenblatt, Shlegeris, Sachan, Roger; arXiv:2312.06942): the audit budget and trusted monitoring.
  - Ctrl-Z (Bhatt et al.; arXiv:2504.10374): resampling as active evidence gathering.
  - SHADE-Arena (Kutasov et al.; arXiv:2506.15740).
  - Secret Collusion among AI Agents (Motwani et al.; arXiv:2402.07510).
  - Open Challenges in Multi-Agent Security (Schroeder de Witt et al.; arXiv:2505.02077).

---

## 3. Classical prior work that bounds our novelty

Entries with an arXiv ID were checked against the arXiv API. Entries without one (Heckerman et al. 1995, Turney 1995, Greiner et al. 2002, Saar-Tsechansky et al. 2009, Shih et al. 2018, Ribeiro et al. 2018, Buhrman & de Wolf 2002, Chow 1970, Chernoff 1959, Chaloner & Verdinelli 1995) are standard references cited from memory and not re-fetched. Check them before citing.

### Active feature acquisition and dynamic feature selection
Paying to observe features one at a time under a budget is a mature area. Examples:
- Saar-Tsechansky, Melville & Provost, "Active feature-value acquisition," Management Science 2009.
- Covert et al., "Learning to Maximize Mutual Information for Dynamic Feature Selection," arXiv:2301.00557 (ICML 2023).

Our "reveal a hidden record at a cost" is active feature acquisition in which the features are records and the prior comes from metadata. That framing is not new.

### Test-cost-sensitive diagnosis and troubleshooting
- Decision-theoretic troubleshooting: Heckerman, Breese & Rommelse, CACM 1995.
- Cost-sensitive classification with test costs: Turney, JAIR 1995 (ICET).
- Greiner, Grove & Roth, "Learning cost-sensitive active classifiers," AIJ 2002.

These already treat sequential, costly tests that discriminate among hypotheses (faults) and stop when the decision is certain enough. Testing our alternative explanations is structurally fault diagnosis.

### Bayesian active learning for decisions (most directly relevant)
- Golovin, Krause & Ray, "Near-Optimal Bayesian Active Learning with Noisy Observations" (EC², equivalence-class determination), arXiv:1010.3091 (NeurIPS 2010).
- Golovin & Krause, "Adaptive Submodularity," arXiv:1003.3967 (JAIR 2011).
- Javdani, Chen, Karbasi, Krause et al., "Near Optimal Bayesian Active Learning for Decision Making" (DRD), arXiv:1402.5886 (AISTATS 2014).

EC² and DRD select costly tests to separate groups of hypotheses, for example "violation" vs. the union of benign explanations, rather than to identify the exact hypothesis. They come with greedy near-optimality guarantees. This is close to the mathematical core of our Q1 and Q3. Any "near-optimal certificate acquisition" claim must be positioned against these papers.

### Bayesian experimental design and active hypothesis testing
- Chaloner & Verdinelli, Statistical Science 1995.
- Rainforth, Foster, Ivanova, Bickford Smith, "Modern Bayesian Experimental Design," arXiv:2302.14545 (Statistical Science 2024).
- Chernoff, "Sequential design of experiments," Ann. Math. Stat. 1959.
- Naghshvar & Javidi, "Active sequential hypothesis testing," arXiv:1203.4626 (Ann. Stat. 2013).

Expected-information-gain record selection is standard BED.

### Abductive, minimal, and sufficient explanations
- Ignatiev, Narodytska & Marques-Silva, "Abduction-Based Explanations for Machine Learning Models," arXiv:1811.10656 (AAAI 2019). Subset-minimal and cardinality-minimal sets of feature values that entail the prediction.
- Shih, Choi & Darwiche, prime-implicant explanations, IJCAI 2018.
- Darwiche & Hirth, "On the Reasons Behind Decisions," arXiv:2002.09284 (ECAI 2020). Sufficient reasons.
- Wang, Khosravi & Van den Broeck, "Probabilistic Sufficient Explanations," arXiv:2105.10118 (IJCAI 2021).
- Ribeiro et al., Anchors, AAAI 2018.
- Certificate complexity in query complexity (e.g., Buhrman & de Wolf, TCS 2002). The minimum number of input bits that fix f's value is literally a "minimum certificate."

Our "certificate" is an abductive or sufficient explanation over records. Minimal sufficient sets, and the NP-hardness and greedy approximations for computing them, are known.

### Audit games and selective classification
- Blocki, Christin, Datta, Procaccia et al., "Audit Games," arXiv:1303.0356 (IJCAI 2013). Budgeted strategic auditing.
- Chow 1970.
- Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks," arXiv:1705.08500 (NeurIPS 2017). Abstention and the risk-coverage trade-off.

---

## 4. What is not new (do not claim)

1. Selective or budgeted monitoring of agent traces: Arbiter, Efficient Auditing (2610.07256), BRA-Audit, One Human N Agents, AI Control.
2. Active evidence acquisition with costly inspections, including active interrogation of agents: Arbiter, Ctrl-Z, active feature acquisition, decision-theoretic troubleshooting.
3. A cheap gate followed by expensive content audit: 2610.07256, EaE.
4. Less (selected) evidence can make LLM judges more accurate: Arike et al.
5. Aggregating weak evidence across steps: TRACE. Cross-agent evidence is needed for attribution in multi-agent systems: Agent-Canary-Mas (Audit-F1 29.7% to 95.5%), HANSARD's synergy residual and attribution laundering.
6. Provenance or causal graphs over multi-agent events: HANSARD, the 2606.04990 survey, CASPIAN.
7. Minimum or sufficient evidence sets as explanations or certificates: abductive explanations, prime implicants, certificate complexity. "Evidence minimization" under restricted disclosure is posed as an open problem by the Treacherous Envoy paper.
8. Group-level hypothesis tests for collusion: HELIOS. Metadata or trace-only collusion statistics: 2608.02698.
9. Separating hypothesis groups with costly tests and near-optimal greedy policies: EC², DRD, adaptive submodularity.
10. Abstention and the risk-coverage trade-off: selective classification.
11. Benchmarks with assigned-coalition collusion and scorer-verified sabotage: SCHEME, NARCBench, Colosseum. "Collusion on paper" (planned but not executed) is already documented by Colosseum.

## 5. Candidate contribution

Each ingredient exists separately. What we could not find is their combination under one evaluated protocol:
- an auditor that sees only public event metadata (actor/recipient IDs, types, timestamps, object references) and pays per revealed record;
- a stopping rule defined by a *sufficient certificate* for a joint, multi-agent violation, rather than by a confidence score on one agent or one action;
- record selection that explicitly targets *complementary cross-agent groups*, where individually uninformative records become jointly sufficient (the "jointly plausible but individually uninformative evidence" that Treacherous Envoy identifies, and the attribution laundering HANSARD names);
- explicit acquisition of evidence that *discriminates against named benign explanations* (benign cooperation, a misled agent, shared error, an authorized exception, an abandoned plan / "collusion on paper"), with abstention, evaluated on hard negatives that share the violation's metadata signature.

The defensible claim is empirical and comparative. On public multi-agent collusion data (emergent: Shi et al.; instructed: NARCBench) and, where needed, synthetic or Colosseum/Terrarium-generated multi-agent episodes, we would measure:
- reveal cost to a valid certificate at matched precision;
- group-complementary vs. individually-suspicious selection at equal budget;
- false-positive reduction from alternative-explanation tests at a fixed abstention rate.

Analysis should be positioned as an application of EC²/DRD-style decision-region acquisition, not as new theory. A cost lower bound could be related to Treacherous Envoy's Proposition 1.

Main novelty risks:
- Agent-Canary-Mas already shows that cross-agent evidence matters for attribution.
- Efficient Auditing (2610.07256) and BRA-Audit already show budgeted selective auditing with multi-agent case studies.
- EC²/DRD already provide the algorithmic core. If complementary group selection reduces to greedy EC² over a joint hypothesis space, the algorithmic novelty is small, and the contribution rests on the problem formulation, the alternative-explanation hypothesis set, and the evaluation.
