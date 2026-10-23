# Source materials (summarised; originals removed from the project)

The original PDFs were deleted on 2026-10-24 to slim the project for transfer. This note keeps
what the project uses from them.

## 1. "Abstractions" — ILIAD Intensive 2026 (Rosas, Boyd, Ehrlich; XOR Labs), 62 slides

**Sections I–II: transducers and prediction.**
- An agent is a causal (non-anticipatory) *interface* Pr(Y_{0:L} | X_{0:L}), from perceptions to
  actions. Every causal interface has a *transducer* presentation, i.e. an input-conditioned,
  edge-emitting HMM with transitions T^{(y|x)}_{r→r'}. Each interface has infinitely many
  presentations.
- The minimal predictive presentation is the **ε-transducer**. Its states are the causal states, the
  equivalence classes of pasts with equal conditional futures. It generalises the ε-machine.
  - Barnett & Crutchfield, J. Stat. Phys. 161 (2015).
  - Rosas, Boyd & Baltieri, "AI in a vat", arXiv:2504.04608.
- **Linear form.** Write the state as |ρ_t⟩ and use operators T̂^{(y|x)}. Then
  Pr(y_{0:L} | x_{0:L}) = ⟨1| T̂^{(y_{0:L}|x_{0:L})} |ρ_0⟩.
- **Belief states and the mixed-state presentation (MSP).** Belief states
  |ρ(y_{0:t}|x_{0:t})⟩ = T̂|ρ_0⟩ / ⟨1|T̂|ρ_0⟩ are sufficient predictors. They coarse-grain to the causal
  states, and their geometry in the simplex is often fractal.

**Section III: composition and factored belief states (the part this project builds on).**
- **Two agents in feedback.** Two agents T and U in a loop form one joint HMM with
  V^{(xy)}_{rs→r's'} = T^{(y|x)}_{r→r'} U^{(x'|y)}_{s→s'}.
  - *Any* joint process can be written as a feedback process, so viewing data as interacting agents
    loses no generality.
- **Networks of agents.** For a network, the joint operator is a **tensor product**
  V̂ = ⊗_i Â^{(i)} (with routing).
  - Beliefs about the parts are then conditionally independent, and a factored representation is
    possible: dimension Σ_i d_i instead of Π_i d_i, an exponential saving.
  - Coarse-graining the interface destroys factorization ("complexification").
- **Transformers.** Transformers trained on such processes represent each agent's belief geometry
  in orthogonal subspaces, and their dimension grows linearly in the number of agents.
  - Shai et al., NeurIPS 37 (2024).
  - Shai et al., "Transformers learn factored representations", arXiv:2602.02385 (2026).
- **Hypotheses.**
  - MABGH: each agent's belief geometry appears in the activations.
  - MABDH: latent dimension ≤ D_fact, the factored bound.

**How this project uses it.**
- Non-interacting agents ⇒ the joint transfer operator is a Kronecker product ⇒ the cut matrix of
  joint action windows has rank one across any cut of the agents.
- Collusion is measured as the *factorization defect* (`fathon.tensor.factorization`).
- The state-conditioned version of the same idea is Bonjour et al.'s influence
  (`fathon.tensor.influence`).

## 2. Course "Early-warning signals of critical transitions in complex systems"

A. V. Dmitriev, HSE MIEM, 2022–23, in Russian. Folder `crit 17.14.00`, 11 lectures plus references.

| # | Lecture | Content kept |
|---|---|---|
| 1 | EWS of critical transitions | Self-organised criticality (open, dissipative, non-equilibrium systems). Two approaches: *structural* (spatio-temporal) measures vs *temporal* measures on observed series. **Expanding window with a fixed left boundary**: indicator m_t computed on [0, t] for t = n₀..N, rather than a sliding window. |
| 2 | Critical phenomena and phase transitions | First- vs second-order transitions; sandpile cellular automata as generators of test series with known transitions; critical slowing down measured by correlation-theory statistics. |
| 3 | Correlation theory of EWS measures | Distribution moments and lag-1 autocorrelation in the window [0, t] as EWS. |
| 4 | Random processes and realisations | — |
| 5–6 | EWS in stationary series | R/S Hurst exponent; PSD power-law exponent (FFT); STFT for piecewise-stationary series. |
| 7 | EWS in non-stationary series: DFA | Profile → detrend per window → F(s) ~ s^H. H_G < 0.5 anti-persistent, 0.5 white, (0.5, 1) persistent, 1 pink, 1.5 Brownian. |
| 8 | Multifractal formalism | τ(q), h(q), Legendre spectrum. |
| 9 | EWS in non-stationary series: MF-DFA | **Spectrum-based indicators**: position h* = argmax D(h); support width Δh = h_max − h_min; quadratic fit D(h) = A(h−h*)² + B(h−h*) + C, with **B as asymmetry**. Model critical transitions (cellular automata) change these parameters. |
| 10 | Wavelet transforms | CWT-based EWS. |
| 11 | WTMM | Wavelet-transform modulus maxima spectrum; wavelet leaders. |
| 13 | Chaos theory (pptx) | Correlation dimension, entropy, Lyapunov exponent, recurrence quantification analysis (RQA). |

Method menu for non-stationary series, from lectures 7 and 9:
- moments via correlation theory;
- DFA generalised Hurst exponent;
- MF-DFA spectrum;
- WTMM;
- wavelet leaders;
- correlation dimension, entropy, Lyapunov exponent, RQA.

**Not yet implemented here:**
- spectrum-shape indicators (h*, Δh, B) as rolling or expanding-window EWS;
- the expanding-window protocol itself.

Both are candidates for the server-side early-warning study (see `FINDINGS.md`, next steps).

**Other references that were in the folder:**
- *Nonlinear Time Series Analysis* (Kantz & Schreiber; 387 pp).
- Iudin & Koposov, *Фракталы: от простого к сложному* (185 pp).
- Reservoir-computing papers:
  - "Next generation reservoir computing";
  - "Reservoir Computing Universality With Stochastic Inputs";
  - "Connecting RC to statistical modelling and DL".
