# Agency read off the environment: a theory memo for collusion among weight-sharing agents

*Memo, 2026-10-25. Companion to `notes/research_directions_2026-10-25.md` (cited below as **RD**). I checked the literature on the web only: the arXiv API, Crossref, and publisher or Europe PMC full text where noted. Nothing was cloned. The References section shows how each entry was checked. Throughout, **[E]** marks an established result, **[C]** a conjecture of this memo, and **[S]** a synthesis that combines established pieces in a new way.*

---

## 0. Bottom line

1. **"Agency = lowering entropy in the environment" is necessary but not sufficient, and this is established.** Dissipative structures such as whirlpools, crystals and Bénard cells lower local entropy too. The literature adds two more conditions:
   - **counterfactual adaptivity:** the structure is restored, or the policy changes, when the mechanism is perturbed (Kenton et al. 2022; Kolchinsky & Wolpert 2018);
   - **a boundary:** a set of variables that carries its own future, i.e. is informationally closed (Krakauer et al. 2020; Bertschinger et al. 2008).

   The operational definition I propose below combines three estimable parts: improbability under a mechanism-only null, adaptivity under intervention, and closure. **[S]**
2. **Shared weights split the "agent boundary" into two different notions:**
   - the **dynamical boundary:** information flows that close a set of variables (Krakauer; Kenton);
   - the **logical boundary:** agents that run the same computation, so one agent's choice is evidence about another's (Lewis 1979; functional decision theory; Oesterheld et al. 2023; Meulemans et al. 2026).

   Communicating colluders cross the dynamical boundary. Copies that cooperate through similarity cross only the logical one. **[S]**
3. **The central consequence for detection [C, with an established core]:** copies with independent sampling noise and no channel are **conditionally independent given the public history and their parameters**. Passive dependence statistics (transfer entropy, synergy, the no-signalling test) therefore cannot separate (b) similarity-based cooperation from (a′) a merely common prior, *in principle*. They differ only in *which policy* is played. That is visible in levels (entropy deficit, collusion index), in the absence of retaliation, and in sensitivity to *believed* similarity, which only interventions reveal.

   This gives RD Direction 1 (identification by probes) a theoretical justification. It also explains why removing the common mode carried our detection while the passive multiscale indicators were at chance (FINDINGS.md).
4. **Most promising experiment:** X2 (§4). It is a 2×2 design that crosses *actual* weight sharing with *believed* similarity among Qwen/Olmo sellers in the E8 market. It measures the collusion index, the entropy deficit, transfer entropy conditional on the public history, and retaliation after a forced deviation. It is the cleanest test of whether class (b) exists for 7–8B models. It uses inference only.

---

## 1. What the literature establishes

I list only results that the framing uses. For the market-collusion, monoculture and steganography literature, see RD §2. I do not repeat it here.

### 1.1 Agency and individuality defined from dynamics

- **Krakauer, Bertschinger, Olbrich, Flack & Ay (2020)** split the system's predictive information as `I(S_{n+1}; S_n, E_n) = I(S_{n+1}; S_n) + I(S_{n+1}; E_n | S_n)`. They define three quantities:
  - organismal individuality `A* = I(S_{n+1}; S_n)`;
  - colonial individuality `A = I(S_{n+1}; S_n | E_n)`;
  - environmental determination (non-closure) `nC = I(S_{n+1}; E_n | S_n)`.

  In partial information decomposition (PID) terms, `A*` is shared plus unique-to-S information, and `A` is unique plus synergistic. They find boundaries by growing S while `A*` rises and `nC` falls. **[E]**

  **Correction to the brief:** the paper does *not* use the Game of Life. I checked the 2014 arXiv text and the 2020 open-access full text. Its examples are coupled binary stochastic processes. The Game-of-Life work is the next item.
- **Game of Life, cellular automata and Lenia** **[E]**:
  - Beer (2004, 2014, 2015) treats the glider as an autopoietic unity, with a "cognitive domain" of perturbations it survives.
  - Biehl, Ikegami & Polani (2016) propose "integrated spatiotemporal patterns" as the building block of an agent definition and test them in the Game of Life.
  - Local transfer entropy picks out gliders as information transfer in elementary CA (Lizier et al. 2008). Local statistical complexity filters out coherent structures (Shalizi et al. 2006), and so does PID (Flecker et al. 2011).
  - Hamon et al. (2025) find Lenia patterns with *sensorimotor agency*: they move, keep their integrity and get past obstacles, found by diversity search.
- **Causal and Bayesian definitions** **[E]**:
  - Kenton et al. (2022): agents are "systems that would adapt their policy if their actions influenced the world in a different way". They give a discovery algorithm that needs interventional data on the mechanism.
  - Orseau, McGill & Legg (2018): Bayes' rule gives the posterior that a system is an *agent* (described by what it optimises) rather than a *device* (described by its input-output map), from behaviour alone. This is the "intentional stance from data".
  - Biehl & Virgo (2022): an agent is a system *plus* an interpretation of it as a POMDP solution, and such interpretations are not unique.
  - Abel et al. (2025) argue that agency is frame-dependent.
  - MacDermott et al. (2024) define maximum-entropy goal-directedness (MEG) on top of maximum causal entropy, the formal relative of Yudkowsky's (2008, blog) "optimisation power": `-log2` of the probability, under a baseline, of an outcome at least this good.
- **Empowerment and entropic forces** **[E]**: empowerment is the action→future-sensor channel capacity (Klyubin et al. 2005; Salge et al. 2014). Causal entropic forces maximise future path entropy (Wissner-Gross & Freer 2013, contested by Kappen and by Canessa). Both say agents *keep their options open* while *structuring* what they care about. So "lowering entropy" must be stated per variable: low in goal-relevant environment variables, high in the agent's reachable futures. **[S]**
- **Free energy and thermodynamics** **[E]**:
  - The free-energy principle (Friston 2010, 2019) and Markov blankets (Kirchhoff et al. 2018) are contested (Biehl, Pollock & Kanai 2021; Bruineberg et al. 2022). I use blankets only as a modelling device.
  - England (2013) and Perunov et al. (2016) tie adaptation to dissipation history. Still et al. (2012) bound dissipation by non-predictive memory.
  - Kolchinsky & Wolpert (2018) define *semantic information* as the system–environment information that is causally necessary, tested by scrambling correlations, for the system to stay in a low-entropy state. They define agency from it.
  - Ortega & Braun (2013): bounded-rational choice is free-energy minimisation, `p(a) ∝ p0(a) exp(βU(a))`.
- **Autonomy** **[E]**: Bertschinger et al. (2008); Seth (2010, Granger autonomy); Barandiaran, Di Paolo & Rohde (2009: individuality, interactional asymmetry, normativity).

### 1.2 Recovering agents, shared latents and coordination from observations

- **Number of shared latents** **[E]**: factor counts are identified (Bai & Ng 2002). Tetrad and rank constraints cluster variables by the latent they share (Silva et al. 2006; Kummerfeld & Ramsey 2016; Huang et al. 2022). ICA and LiNGAM allow hidden confounders (Comon 1994; Hoyer et al. 2008). FCI is complete under latent confounding (Zhang 2008), and Tetrad implements these methods (Ramsey & Andrews 2023). Our common-mode removal is the linear-factor special case.
- **Directed flow** **[E]**: transfer entropy (Schreiber 2000) equals Granger causality for Gaussian variables (Barnett et al. 2009). See also directed information (Massey 1990; Quinn et al. 2015), PCMCI (Runge et al. 2019) and causal flow (Ay & Polani 2008). A known hazard: a hidden common driver creates spurious transfer entropy. This is exactly the shared-latent confound.
- **Synergy and integration** **[E]**:
  - PID (Williams & Beer 2010; Bertschinger et al. 2014), O-information (Rosas et al. 2019), ΦID (Mediano et al. 2021), causal emergence (Hoel et al. 2013; Rosas et al. 2024).
  - **Riedl (2025, ICLR 2026)** applies PID of time-delayed mutual information to groups of LLM agents in a guessing game with no direct communication but with group-level feedback. Personas plus a "think about what others will do" instruction produce goal-directed cross-agent synergy. This is the closest existing empirical use of these measures on LLM collectives.

### 1.3 Decision theory for correlated agents

- **Classical decision theory** **[E]**:
  - Causal decision theory (CDT) defects in a one-shot prisoner's dilemma (PD) against an exact copy. Evidential decision theory (EDT) cooperates. Lewis (1979) showed that the PD against a copy *is* a Newcomb problem (Nozick 1969; Gibbard & Harper 1978).
  - Hofstadter's superrationality (1983 columns, collected 1985) and functional decision theory (Yudkowsky & Soares 2017) recommend cooperating with agents that run the same algorithm.
  - The corresponding equilibrium notions are dependency equilibria (Spohn 2007; Treutlein 2023), translucent players (Halpern & Pass 2018; Capraro & Halpern 2019) and the embedded equilibrium (below).
- **Program equilibrium and open-source game theory** **[E]**:
  - Revealing source code supports cooperative equilibria in the one-shot PD (Tennenholtz 2004).
  - Cooperation stays robust beyond syntactic identity through provability logic (Barasz et al. 2014; Critch 2019) or through simulation (Oesterheld 2019; Cooper, Oesterheld & Conitzer 2024). Institution design has its own surprises (Critch, Dennis & Russell 2022).
  - **Oesterheld, Treutlein, Grosse, Conitzer & Foerster (NeurIPS 2023)**: a *single similarity number* supports the same set of cooperative outcomes as full transparency, and simple learners can learn it.
- **LLM evidence (2025–2026)** **[E, each one study]**:
  - Meulemans et al. (2026): foundation-model agents that plan optimally converge to cooperation in social dilemmas. Their "embedded Bayesian agent" treats its own deliberation as evidence about a similar partner, and they propose the *embedded equilibrium* in place of Nash.
  - Kundu et al. (2026): LLMs differ sharply in how they respond to graded similarity signals. They self-identify as highly similar when judging another model's chain of thought, and the dataset behind the similarity score barely matters.
  - Tewolde et al. (2026, CoopEval): recent models *defect* in one-shot dilemmas.
  - Sistla & Kleiman-Weiner (2025): LLMs write similarity-based programs in open-source games.
  - Oesterheld et al. (2024): a benchmark of Newcomb-like questions on LLMs' decision-theoretic reasoning.

---

## 2. Framing

### 2.1 Making "environment-structuring that lowers entropy" operational

**Setting.** We observe `Y_t = g(X_t)`: a partial, possibly coarse-grained view of an environment `X`, such as prices and quantities, or logs with metadata. Agents are not labelled. We know the *mechanism* `M` (market clearing, demand family, accounting identities) up to parameters. The only structural assumption about agents is that their policies are drawn from a few families with shared parameters.

I define the **agency signature** of a set of observed variables `S ⊆ Y` relative to the rest `E` by three estimable parts. **[S]**

1. **Improbability under a mechanism-only null (structure).** The *entropy deficit* is `ΔH(S) = h_null(S) - h_obs(S)`. Here `h` is an entropy rate, and the null replaces the decision makers with MaxEnt policies constrained only by `M` and the observed marginals (Jaynes 1957), for example zero-intelligence traders (Gode & Sunder 1993). The goal-directed version is optimisation power, `OP(S) = -log2 P_null(U(S) ≥ U_obs)`, for a utility class `U` such as profit (Yudkowsky 2008; MEG gives the principled form). Dissipative structures already pass this test, so it is not enough on its own. **[E]**
2. **Adaptivity under intervention (agency rather than device).** Change the mechanism: the demand slope, a rival's price, a local rule. Then measure whether `S` re-optimises, which is Kenton's criterion, and whether its target is restored, which is Beer's and Kolchinsky–Wolpert's notion of viability. Passively, the Orseau–McGill–Legg posterior "agent vs device" over a hypothesis class is the fallback, but it is identified only up to the choice of class (Biehl & Virgo). In markets the regulator's tier-2 interventions in RD (a scripted entrant, probe orders) are exactly these mechanism changes.
3. **Closure (the boundary).** Choose the partition of the observed variables that maximises `A*` and minimises `nC` (Krakauer et al.). This is computed *after* conditioning on the estimated shared latents (§2.2). Otherwise every same-model agent looks environmentally determined by the common factor.

**What partial observation does.** Two established facts limit what can be read off `Y`.
- Mutual information is non-increasing under processing of *one* argument. Transfer entropy and synergy are not monotone under marginalisation, and a hidden common driver creates them. **[E]**
- Entropy deficits measured in `Y` are relative to a null that is itself fitted on `Y`.

Every quantity below is therefore stated *conditional on an estimated latent model*. The quality of that model is the main risk. FINDINGS.md says the same thing empirically: removing the shared common mode mattered more than any choice of statistic.

### 2.2 What shared latents add: one distributed agent, or many?

**Model [S].** Agent `i` in market `μ` acts as `a_i(t) ~ π(· | h_μ(t), o_i(t); θ_i)`. The terms are:
- `h_μ` is the public history;
- `o_i` is private information;
- `θ_i = θ_{g(i)} + δ_i` combines the base model `g` and a fine-tune or prompt `δ_i`.

For KL-regularised fine-tuning the optimal policy has the Ortega–Braun form `π_i ∝ π_{g} exp(r_i/β)` (Rafailov et al. 2023, for RLHF). The shared "latent" is literally the shared prior policy `π_g`, and its contribution shrinks as optimisation strength `1/β` grows. **[E for the functional form; C for its use as a model of monoculture strength]**

**Three ways copies can be dependent**, which correspond to RD's A/B/C:

| | Mechanism | Where the dependence lives |
|---|---|---|
| (a′) common prior | the same `π_g` evaluated on similar public states | a common factor across markets and agents of the same base |
| (b) similarity-based / superrational | the policy itself conditions on "my rivals are like me", with no information flow | the *policy* (equilibrium selection) |
| (c) channel | private messages or encoded actions | flows from `o_j` to `a_i` |

**Proposition (conditional independence of copies) [E, an elementary consequence of the model].** Suppose there is no channel and sampling noise is independent across agents. Then `a_i(t) ⊥ a_j(t) | h_μ(t), θ_i, θ_j` and `a_i(t) ⊥ o_j(·) | h_μ(t), θ_i`. This holds under both (a′) and (b): superrational cooperation changes the *function* π, not what each agent can know. Two consequences follow.
- Transfer entropy conditional on the public history, synergy among correctly conditioned innovations, and the no-signalling test of RD Direction 2 are all ≈ 0 under (a′) and under (b). They can only flag (c). **[E given the model]**
- (b) differs from (a′) only through π. It shows up as a different *level* (supra-competitive price, lower entropy rate) and a different *counterfactual*: π changes when the agent's belief about the rivals' similarity changes. It does *not* show up as a different dependence pattern. **[S]**

An exception: with greedy decoding, or a shared seed and prompt, copies are deterministic functions of the same inputs, so dependence is total and degenerate. That is still common cause, not coordination.

**Where is the agent boundary? [S]**
- *Dynamically* (Krakauer, Kenton), a coalition under (c) merges into one individual. Adding the partner to `S` internalises flow, which shows as a drop in `nC` beyond what the public state explains, i.e. `I(S_{n+1}; ΔS_n | S_n, E_n) > 0`.
- Under (b) each copy is already closed relative to the public environment, so there is *no* dynamical merger.
- *Logically* (Lewis, FDT, Oesterheld, Meulemans), the copies are one decision procedure instantiated N times. "One distributed agent" is therefore true of (b) only in the logical sense.

This maps onto law and welfare. Agreement-based antitrust tracks the dynamical boundary (channels and nonpublic data, as in the RealPage remedies in RD §2), while consumer harm can come from the logical boundary alone. This is Harrington's liability gap stated as two notions of individuality. Abel et al.'s frame-dependence result says that neither notion is privileged a priori.

**Decision-theoretic signature of (b) [C].** Learned tacit collusion in repeated games (Calvano et al.) is sustained by *punishment*: forced deviations trigger retaliation and then a return. Similarity-based cooperation needs no punishment, because cooperation is chosen for its evidential value, even one-shot. The predicted signature is therefore:
- supra-competitive levels in single-shot or short-horizon markets;
- no targeted retaliation after a forced deviation;
- dependence on believed similarity.

Our E3 finding that Fish-style LLM sellers price high *statically*, with no transition, fits (a′) or (b) and does not fit learned collusion. That is a reason to take (b) seriously.

### 2.3 Predictions that separate the classes

Below, (a) means independent heterogeneous agents. "Cond." means conditional on the public history and the estimated base-model latents. E/C marks the status of each prediction.

| Quantity (estimable from environment data unless noted) | (a) independent | (a′) common prior | (b) similarity coop | (c) channel colluders | Status |
|---|---|---|---|---|---|
| Entropy deficit ΔH / optimisation power vs a MaxEnt mechanism null | ≈ the competitive benchmark | small–moderate (focal points) | large, rigid | large, adaptive | C |
| Cross-market factor rank (Bai–Ng) beyond public drivers | 0 | = number of base models | = number of base models | = number of base models; the coalition residual is *within*-market only | E (a, a′); C (b, c) |
| Cross-market coincidence of same-model agents (E8) | ≈ chance | > chance | > chance | > chance | E (measured: 9–15/59 vs ≈1) |
| Cond. transfer entropy j→i; no-signalling test on `o_j` | 0 | 0 | 0 | > 0 if the channel carries private information | E given the proposition |
| Cond. synergy / O-information of innovations | ≈ 0 | ≈ 0 *only if the latent is removed correctly* | ≈ 0 (same caveat) | < 0 (synergy-dominated) for parity-type coalitions | E (c, planted); C (leakage) |
| Retaliation after a forced deviation (tier 2) | none | none (non-targeted) | none or weak | targeted | C (b); E (learned collusion, Calvano) |
| Policy shift when *believed* similarity is changed with the history fixed (tier 3 / logprobs) | 0 | 0 | > 0 | ≥ 0 | C |
| Collusion index drop when one rival is swapped to a different base | 0 | small | large | depends on whether the channel survives | C; heterogeneity breaks prompted LLM collusion (Keppo et al., E) |
| Market invariants (clearing, accounting, the conduct parameter under demand rotation; Bresnahan 1982) | competitive conduct | ≈ competitive conduct | collusive conduct | collusive conduct | E (identification of conduct) |

The table also shows *why* passive early-warning indicators failed (FINDINGS): none of the rows that separate (b) from (a′) is a passive property of the multiscale structure of a single series.

---

## 3. Status summary

- **Established:** the individuality and closure measures; agents need interventions to be discovered; similarity-based and program equilibria; the CDT/EDT split on copies; factor-count identification; spurious transfer entropy under hidden drivers; synergy among LLM agents without direct messages.
- **Synthesis:** the three-part agency signature; dynamical vs logical boundary; the KL-prior model of monoculture.
- **Conjecture:**
  - (b) exists at 7–8B and shows as "high level, no retaliation, belief-sensitive";
  - linear cleaning leaks synergy under nonlinear shared latents;
  - the boundary search recovers coalitions under (c) and not under (b).

---

## 4. Experiments

Everything below runs on local Qwen2.5-7B, Qwen3-8B and Olmo-3-7B through vLLM, or on small RL policies, on 1–2 H100s with no paid APIs. Each experiment states what it adds to RD, so the two memos do not overlap.

### X1. Agent boundaries in a Life-like commons, from the grid alone

- **Question.** Can the number of agents, their territories and their coalitions be recovered from the environment's state history alone? Does a channel merge boundaries while weight sharing does not?
- **Design.**
  - The environment is a 64×64 Game-of-Life grid with a 1–5% flip rate. K = 8 agents each own a hidden 4×4 patch. Each agent reads its 8×8 neighbourhood, writes cells in its patch, and is rewarded for holding a target pattern against the background dynamics.
  - Four arms: independent policies; parameter-shared (one network); parameter-shared with a "partner is a copy" input bit; pairs with a private 1-bit channel.
  - Policies are small PPO CNNs. An optional arm uses Qwen2.5-7B as the policy for 4 patches over 200 steps, with the neighbourhood serialised as text.
  - The observer sees only grid states and runs three things:
    1. local transfer entropy and local statistical complexity maps;
    2. a greedy Krakauer search (grow S while `A*`↑ and `nC`↓) over candidate patches, after removing a fitted Life-rule null;
    3. a Kenton-style intervention: change the rule locally (e.g. B36/S23) and test which regions re-adapt.
- **Predictions.**
  - Agent patches are separated from gliders by adaptivity, not by entropy deficit. Gliders also have high local transfer entropy and low entropy, but do not re-adapt.
  - Channel pairs merge into one closed unit: `nC` drops when the partner is added. Parameter-shared pairs do not merge, but they load on one cross-patch factor.
  - **Falsified if** the boundary search merges parameter-shared pairs as often as channel pairs, or cannot beat the glider confound even with interventions.
- **Closest prior work.** Krakauer et al. 2020 (binary toy processes only); Biehl, Ikegami & Polani 2016 (Game of Life, no agents with goals); Beer 2004/2014; Lizier et al. 2008; Hamon et al. 2025 (Lenia agency found by search, not by observers); Kenton et al. 2022 (small causal games).
- **How it could fail.** Conditional mutual information estimators are biased at patch dimensionality. Mitigation: count summaries of the binary patches, shuffle nulls, and bias-corrected plug-in estimators. The agents may also be too weak to hold a target against the noise, which would leave nothing to detect.
- **Cost.** Mostly CPU. The PPO runs take a few GPU-hours. The LLM arm is about 40k generations, roughly 2 GPU-hours.

### X2. Similarity-based cooperation among weight-sharing LLM sellers (2×2: actual × believed)

- **Question.** Does class (b) exist for 7–8B models, and can it be separated from (a′) and (c) using environment data plus logprob probes?
- **Design.** The E8 posted-price market (4 sellers, logit demand, 60 rounds), plus a one-shot variant (single-round markets, 400 per cell).
  - **Factor 1, actual composition:**
    - homogeneous Qwen2.5-7B;
    - homogeneous Olmo-3-7B;
    - mixed (Qwen2.5 / Qwen3 / Olmo-3 / Qwen2.5-LoRA).
  - **Factor 2, believed similarity:**
    - none;
    - "your rivals run the same model and prompt as you";
    - a graded score in Kundu et al.'s format;
    - a placebo of equal salience about an irrelevant shared attribute ("same data centre").
  - The "same model" statement is shown in both homogeneous and mixed markets, so actual and believed similarity are crossed: a *false-belief* arm.
  - Positive control for (c): the existing seeded private channel.
- **Measurements.**
  - Environment-only:
    - the collusion index Δ = (π − π_Nash)/(π_M − π_Nash);
    - the entropy deficit against a MaxEnt seller null;
    - cross-market coincidence;
    - transfer entropy and the no-signalling test conditional on the public history.
  - Tier 2: retaliation after a forced 15% cut by one seller at a random round.
  - Tier 3: the logprob shift in a seller's price distribution when only the similarity statement is edited in the replayed prompt.
- **Predictions under (b).**
  - Δ rises with *believed* similarity at fixed composition, and with *actual* homogeneity at fixed belief.
  - The placebo does nothing.
  - Conditional transfer entropy stays ≈ 0. There is no targeted retaliation.
  - One-shot cooperation rises with similarity (as in Meulemans et al. and Kundu et al.).
- **Predictions under (a′) only.** Belief has no effect, cross-market coincidence stays high, and Δ does not depend on belief.
- **Falsified (no class b at 7–8B) if** the belief and placebo effects are equal, or both ≈ 0.
- **Closest prior work.**
  - Oesterheld et al. 2023 (theory, small learners); Kundu et al. 2026 (one-shot PD, frontier models, no market); Meulemans et al. 2026 (stylised dilemmas).
  - Keppo et al. 2026 (heterogeneity breaks collusion, belief not manipulated); Fish et al. 2024 (prompt sensitivity).
  - RD Direction 1 (identity probes). X2 is its B arm made explicit and tied to decision theory, and it should share its harness and its null.
- **How it could fail.** The similarity statement may act as an instruction to collude (a demand effect). The placebo and false-belief cells catch part of this, but not all of it. The models may defect regardless (CoopEval), or prompt variance may swamp the effects (Fish et al.). The shared prior inflates Δ in homogeneous cells, so Δ is compared across belief levels, never against zero.
- **Cost.** 3 compositions × 4 belief arms × (40 repeated + 400 one-shot markets), about 0.3M short generations plus 20k probes. Roughly 20–30 GPU-hours.

### X3. Optimisation power and closure as a boundary detector on market worlds

- **Question.** Does a boundary search by optimisation power and closure, run on seller price series only, recover planted coalitions? Does it distinguish (c) from (b), where the conjecture predicts super-additive optimisation power without closure?
- **Design.** Data: E14 mixed-threat worlds, the E7 seeded double auction and the X2 logs. Two nulls: MaxEnt sellers constrained to the observed price support and demand, and the same base model with market states shuffled across sessions (RD's matched replay). Compute `OP(S)`, the super-additivity `OP(S∪T) − OP(S) − OP(T)`, and the Krakauer greedy search after removing the latent.
- **Prediction.** Super-additive optimisation power appears in (b) and (c). A closure drop on merging (`nC`↓) appears only in (c). Coalition recovery is within 0.03 AUC of the cleaned multi-lag rho_DCCA. It is not expected to beat it (FINDINGS); the value is the (b)/(c) split. **Falsified if** closure drops equally for (b) and (c), or optimisation power is never super-additive.
- **Closest prior work.** Krakauer et al. 2020; MacDermott et al. 2024 (MEG, small MDPs); Bonjour-style net influence and our head-to-head with it; Bresnahan 1982 (conduct).
- **How it could fail.** Optimisation power is only as good as the null (a misspecified demand model gives spurious bits). With 4 sellers the boundary search space is tiny, so the result might be trivially obtained by other means.
- **Cost.** Mainly CPU, plus the X2 generations already budgeted.

### X4. Does linear cleaning leak spurious synergy under nonlinear shared latents?

- **Question.** Is synergy among innovations a channel-specific signature, as the table claims, or does a shared latent acting through nonlinear per-agent responses fake it once removed linearly?
- **Design.** Use the uncommitted `experiments/sims/highorder.py` with `nonlin=True`, with and without parity coalitions. Compare O-information, Riedl's PID synergy of time-delayed mutual information, and `fathon.tensor.hoi` Fourier z-scores on innovations under three cleanings: a linear common mode, the learned null (`fathon/tensor/learned.py`), and the oracle latent. Repeat on homogeneous Qwen sellers from X2.
- **Prediction.** With no coalitions, the false-positive rate of synergy flags is above nominal under linear cleaning and nominal under the learned and oracle nulls. With parity coalitions, power is retained under all three.
- **Falsified if** linear cleaning is already nominal (synergy is robust), or if the learned null also leaks (synergy is not identifiable without an oracle).
- **Closest prior work.** Rosas et al. 2019; Riedl 2025 (shuffle baselines, no latent confound); Flecker et al. 2011; the team's hoi/learned modules (work in progress).
- **How it could fail.** Synergy estimators are biased at small sample sizes, which interacts with the leakage. Use matched shuffle nulls per condition.
- **Cost.** CPU and GPU for minutes. The LLM part reuses the X2 logs.

---

## 5. How this becomes the spine of a paper

The empirical collusion work has a negative headline: no tensor or multifractal advantage, and the population pipeline is what matters. It also has a positive one: removing the shared common mode and conditioning on state carry detection. This framing explains both. Detection works exactly where the agent boundary is *dynamical*, i.e. channels, followers and coalitions that close information flow beyond the public state and the shared prior. It must fail, *in principle and for any passive statistic*, where the boundary is only *logical*, i.e. weight-sharing agents that cooperate because they reason alike.

A paper could be organised in three parts:
1. **Theory:** the agency signature (structure, adaptivity, closure), the conditional-independence proposition, and the dynamical vs logical boundary split.
2. **Detection results:** the existing scale results, recast as estimators of closure after latent removal, including the honest negatives.
3. **Identification:** X2 (and RD Direction 1) showing what interventions add, i.e. which tier of regulator access identifies which class.

The claim "monoculture makes collusion a question of where the agent boundary is" then becomes a testable, partly established thesis rather than a slogan. It also leads to a policy result: oversight that watches only market data can police channels but not superrational cooperation among copies.

---

## References

Codes for how each entry was checked:
- **[ax]** confirmed via the arXiv API (title, authors, date; journal reference where arXiv lists one);
- **[cr]** confirmed via Crossref (title, authors, venue, DOI);
- **[ft]** full text read for the specific claim made here;
- **[part]** some fields from memory or secondary sources, not confirmed;
- **[blog]** not peer-reviewed.

Where an author list is truncated, it is truncated in the source listing I saw.

**Agency, individuality, autonomy**
- Krakauer, Bertschinger, Olbrich, Flack & Ay (2020). The information theory of individuality. *Theory in Biosciences* 139:209–223. doi:10.1007/s12064-020-00313-7; arXiv:1412.2447 [ax][cr][ft: no Game of Life in either version]
- Bertschinger, Olbrich, Ay & Jost (2008). Autonomy: An information theoretic perspective. *BioSystems* 91:331–345. doi:10.1016/j.biosystems.2007.05.018 [cr]
- Seth (2010). Measuring Autonomy and Emergence via Granger Causality. *Artificial Life* 16(2):179–196. doi:10.1162/artl.2010.16.2.16204 [cr]
- Barandiaran, Di Paolo & Rohde (2009). Defining Agency: Individuality, Normativity, Asymmetry, and Spatio-temporality in Action. *Adaptive Behavior* 17:367–386. doi:10.1177/1059712309343819 [cr]
- Kenton, Kumar, Farquhar, Richens, MacDermott & Everitt (2022). Discovering Agents. arXiv:2208.08345 (later in *Artificial Intelligence*, 2023 [part]) [ax]
- Orseau, McGill & Legg (2018). Agents and Devices: A Relative Definition of Agency. arXiv:1805.12387 [ax]
- Biehl & Virgo (2022). Interpreting systems as solving POMDPs: a step towards a formal understanding of agency. IWAI 2022, CCIS 1721. arXiv:2209.01619; doi:10.1007/978-3-031-28719-0_2 [ax]
- Virgo, Biehl & McGregor (2021). Interpreting Dynamical Systems as Bayesian Reasoners. arXiv:2112.13523 [ax]
- Abel, Barreto, Bowling, Dabney, Dong, Hansen, Harutyunyan, Khetarpal et al. (2025). Agency Is Frame-Dependent. arXiv:2502.04403 [ax]
- MacDermott, Fox, Belardinelli & Everitt (2024). Measuring Goal-Directedness. arXiv:2412.04758 (NeurIPS 2024 [part]) [ax]
- Yudkowsky (2008). Measuring Optimization Power. LessWrong, 27 Oct 2008 [blog]

**Empowerment, entropic forces, free energy, thermodynamics**
- Klyubin, Polani & Nehaniv (2005). Empowerment: A Universal Agent-Centric Measure of Control. IEEE CEC 2005, 1:128–135. doi:10.1109/CEC.2005.1554676 [cr]
- Salge, Glackin & Polani (2014). Empowerment — an Introduction. arXiv:1310.1863 [ax]
- Wissner-Gross & Freer (2013). Causal Entropic Forces. *PRL* 110:168702. doi:10.1103/PhysRevLett.110.168702 [cr]; comments by Kappen, arXiv:1312.4185, and by Canessa, arXiv:1308.4375 [ax]
- Friston (2010). The free-energy principle: a unified brain theory? *Nat. Rev. Neurosci.* 11:127–138. doi:10.1038/nrn2787 [cr]
- Friston (2019). A free energy principle for a particular physics. arXiv:1906.10184 [ax]
- Kirchhoff, Parr, Palacios, Friston & Kiverstein (2018). The Markov blankets of life. *J. R. Soc. Interface* 15:20170792. doi:10.1098/rsif.2017.0792 [cr]
- Biehl, Pollock & Kanai (2021). A Technical Critique of Some Parts of the Free Energy Principle. arXiv:2001.06408 (*Entropy* 23:293 [part]) [ax]
- Bruineberg, Dołęga, Dewhurst & Baltieri (2022). The Emperor's New Markov Blankets. *Behav. Brain Sci.* 45. doi:10.1017/S0140525X21002351 [cr]
- England (2013). Statistical physics of self-replication. *J. Chem. Phys.* 139:121923. arXiv:1209.1179; doi:10.1063/1.4818538 [ax]
- Perunov, Marsland & England (2016). Statistical Physics of Adaptation. *PRX* 6:021036. arXiv:1412.1875 [ax]
- Kolchinsky & Wolpert (2018). Semantic information, autonomous agency and non-equilibrium statistical physics. *Interface Focus* 8:20180041. arXiv:1806.08053 [ax]
- Still, Sivak, Bell & Crooks (2012). The thermodynamics of prediction. *PRL* 109:120604. arXiv:1203.3271 [ax]
- Ortega & Braun (2013). Thermodynamics as a theory of decision-making with information-processing costs. *Proc. R. Soc. A* 469:20120683. doi:10.1098/rspa.2012.0683 [cr]
- Jaynes (1957). Information Theory and Statistical Mechanics. *Phys. Rev.* 106:620–630. doi:10.1103/PhysRev.106.620 [cr]
- Rafailov, Sharma, Mitchell, Manning, Ermon & Finn (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023. doi:10.52202/075280-2338 [cr]

**Cellular automata, Game of Life, Lenia**
- Beer (2004). Autopoiesis and Cognition in the Game of Life. *Artificial Life* 10:309–326. doi:10.1162/1064546041255539 [cr]
- Beer (2014). The Cognitive Domain of a Glider in the Game of Life. *Artificial Life* 20:183–206. doi:10.1162/artl_a_00125 [cr]
- Beer (2015). Characterizing Autopoiesis in the Game of Life. *Artificial Life* 21:1–19. doi:10.1162/artl_a_00143 [cr]
- Biehl, Ikegami & Polani (2016). Towards information based spatiotemporal patterns as a foundation for agent representation in dynamical systems. ALIFE 2016. arXiv:1605.05676 [ax][ft: Game-of-Life experiments]
- Lizier, Prokopenko & Zomaya (2008). Local information transfer as a spatiotemporal filter for complex systems. *PRE* 77:026110. arXiv:0809.3275 [ax]
- Shalizi, Haslinger, Rouquier, Klinkner & Moore (2006). Automatic filters for the detection of coherent structure in spatiotemporal systems. *PRE* 73:036104. arXiv:nlin/0508001 [ax]
- Flecker, Alford, Beggs, Williams & Beer (2011). Partial information decomposition as a spatiotemporal filter. *Chaos* 21:037104 [part: article number]. doi:10.1063/1.3638449 [cr]
- Chan (2019). Lenia — Biology of Artificial Life. *Complex Systems* 28(3):251–286. arXiv:1812.05433 [ax]
- Hamon, Etcheverry, Chan, Moulin-Frier & Oudeyer (2025). Discovering sensorimotor agency in cellular automata using diversity search. *Sci. Adv.* 11(44):eadp0834. arXiv:2402.10236 [ax]

**Latent structure, flow and synergy**
- Bai & Ng (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica* 70:191–221. doi:10.1111/1468-0262.00273 [cr]
- Comon (1994). Independent component analysis, a new concept? *Signal Processing* 36:287–314. doi:10.1016/0165-1684(94)90029-9 [cr]
- Shimizu, Hoyer, Hyvärinen & Kerminen (2006). A Linear Non-Gaussian Acyclic Model for Causal Discovery. *JMLR* 7:2003–2030 [part]
- Hoyer, Shimizu, Kerminen & Palviainen (2008). Estimation of causal effects using linear non-Gaussian causal models with hidden variables. *IJAR* 49:362–378. doi:10.1016/j.ijar.2008.02.006 [cr]
- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search*, 2nd ed. MIT Press [part]
- Zhang (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. *Artificial Intelligence* 172:1873–1896. doi:10.1016/j.artint.2008.08.001 [cr]
- Silva, Scheines, Glymour & Spirtes (2006). Learning the Structure of Linear Latent Variable Models. *JMLR* 7:191–246 [part]
- Kummerfeld & Ramsey (2016). Causal Clustering for 1-Factor Measurement Models. KDD 2016:1655–1664. doi:10.1145/2939672.2939838 [cr]
- Huang, Low, Xie, Glymour & Zhang (2022). Latent Hierarchical Causal Structure Discovery with Rank Constraints. NeurIPS 2022. doi:10.52202/068431-0401 [cr]
- Ramsey & Andrews (2023). Py-Tetrad and RPy-Tetrad. arXiv:2308.07346 [ax]
- Schreiber (2000). Measuring Information Transfer. *PRL* 85:461–464. doi:10.1103/PhysRevLett.85.461 [cr]
- Barnett, Barrett & Seth (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *PRL* 103:238701. arXiv:0910.4514 [ax]
- Massey (1990). Causality, Feedback and Directed Information. Proc. ISITA 1990 [part]
- Quinn, Kiyavash & Coleman (2015). Directed Information Graphs. *IEEE Trans. Inf. Theory* 61:6887–6909. doi:10.1109/TIT.2015.2478440 [cr]
- Runge, Nowack, Kretschmer, Flaxman & Sejdinovic (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Sci. Adv.* 5:eaau4996. doi:10.1126/sciadv.aau4996 [cr]
- Ay & Polani (2008). Information Flows in Causal Networks. *Adv. Complex Syst.* 11:17–41. doi:10.1142/S0219525908001465 [cr]
- Jaques, Lazaridou, Hughes, Gulcehre, Ortega, Strouse, Leibo & de Freitas (2019). Social Influence as Intrinsic Motivation for Multi-Agent Deep RL. arXiv:1810.08647 [ax] (counterfactual influence between agents; background for X2 probes)
- Williams & Beer (2010). Nonnegative Decomposition of Multivariate Information. arXiv:1004.2515 [ax]
- Bertschinger, Rauh, Olbrich, Jost & Ay (2014). Quantifying unique information. *Entropy* 16:2161–2183. arXiv:1311.2852 [ax]
- Rosas, Mediano, Gastpar & Jensen (2019). Quantifying high-order interdependencies via multivariate extensions of the mutual information. *PRE* 100:032305. arXiv:1902.11239 [ax]
- Mediano, Rosas, Luppi, Carhart-Harris, Bor, Seth & Barrett (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition. arXiv:2109.13186 [ax]
- Rosas, Geiger, Luppi, Seth, Polani, Gastpar & Mediano (2024). Software in the natural world: A computational approach to hierarchical emergence. arXiv:2402.09090 [ax]
- Hoel, Albantakis & Tononi (2013). Quantifying causal emergence shows that macro can beat micro. *PNAS* 110:19790–19795. doi:10.1073/pnas.1314922110 [cr]
- Riedl (2025). Emergent Coordination in Multi-Agent Language Models. ICLR 2026. arXiv:2510.05174 [ax]

**Decision theory, program equilibrium, similarity**
- Nozick (1969). Newcomb's Problem and Two Principles of Choice. In *Essays in Honor of Carl G. Hempel*, 114–146. doi:10.1007/978-94-017-1466-2_7 [cr]
- Gibbard & Harper (1978). Counterfactuals and Two Kinds of Expected Utility. doi:10.1007/978-94-009-9117-0_8 [cr]
- Lewis (1979). Prisoners' Dilemma Is a Newcomb Problem. *Philosophy & Public Affairs* 8(3):235–240 [part: Crossref lists only the 1985 and 1987 reprints, e.g. doi:10.1093/0195036468.003.0011]
- Hofstadter (1985). *Metamagical Themas*. Basic Books; collects the 1983 *Scientific American* columns on superrationality [part: chapter and column titles not confirmed]
- Yudkowsky & Soares (2017). Functional Decision Theory. arXiv:1710.05060 [ax]
- Spohn (2007). Dependency Equilibria. *Philosophy of Science* 74:775–789. doi:10.1086/525621 [cr]
- Halpern & Pass (2018). Game theory with translucent players. *IJGT* 47:949–976. doi:10.1007/s00182-018-0626-x [cr]
- Capraro & Halpern (2019). Translucent players: Explaining cooperative behavior in social dilemmas. *Rationality and Society* 31:371–408. doi:10.1177/1043463119885102 [cr]
- Tennenholtz (2004). Program equilibrium. *Games Econ. Behav.* 49:363–373. doi:10.1016/j.geb.2004.02.002 [cr]
- Barasz, Christiano, Fallenstein, Herreshoff, LaVictoire & Yudkowsky (2014). Robust Cooperation in the Prisoner's Dilemma: Program Equilibrium via Provability Logic. arXiv:1401.5577 [ax]
- Critch (2019). A parametric, resource-bounded generalization of Löb's theorem, and a robust cooperation criterion for open-source game theory. *J. Symbolic Logic* 84:1368–1381. doi:10.1017/jsl.2017.42 [cr]
- Critch, Dennis & Russell (2022). Cooperative and uncooperative institution designs: Surprises and problems in open-source game theory. arXiv:2208.07006 [ax]
- Oesterheld (2019). Robust program equilibrium. *Theory and Decision* 86:143–159. doi:10.1007/s11238-018-9679-3 [cr]
- Cooper, Oesterheld & Conitzer (2024). Characterising Simulation-Based Program Equilibria. arXiv:2412.14570 [ax]
- Oesterheld, Treutlein, Grosse, Conitzer & Foerster (2023). Similarity-based cooperative equilibrium. NeurIPS 2023. arXiv:2211.14468 [ax]
- Treutlein (2023). Modeling evidential cooperation in large worlds. arXiv:2307.04879 [ax]
- Oesterheld, Cooper, Kodama, Nguyen & Perez (2024). A dataset of questions on decision-theoretic reasoning in Newcomb-like problems. arXiv:2411.10588 [ax]
- Meulemans, Wołczyk, Weis, Nasser, Rocca, Kobayashi, Lajoie, Steger et al. (2026). A game theory for foundation models shows new paths to rational cooperation through similarity inference. arXiv:2608.03958 [ax]
- Kundu, Tewolde, Berker, Brown & Conitzer (2026). Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation. arXiv:2608.12125 (ICML 2026 listing [part]) [ax]
- Tewolde, Zhang, Guzman Piedrahita, Conitzer & Jin (2026). CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas. arXiv:2604.15267 [ax]
- Sistla & Kleiman-Weiner (2025). Evaluating LLMs in Open-Source Games. arXiv:2512.00371 [ax]
- Tonini & Galke (2025). Super-additive Cooperation in Language Model Agents. arXiv:2508.15510 [ax]
- Rubinstein (1998). *Modeling Bounded Rationality*. MIT Press. doi:10.7551/mitpress/4702.001.0001 [cr]

**Markets (only those not already in RD)**
- Gode & Sunder (1993). Allocative Efficiency of Markets with Zero-Intelligence Traders. *JPE* 101:119–137. doi:10.1086/261868 [cr]
- Calvano et al. 2020; Fish et al. 2024; Keppo et al. 2026; Bresnahan 1982; Aharon et al. 2026; Panickssery et al. 2024: see RD References.
