# Novelty audit: multifractal formalism ⇄ tensor networks

*Compiled 2026-10-23 from two prior-art search rounds (nine parallel search agents, ≈300 web queries, plus an earlier deep-research pass with 3-vote adversarial verification). Companion to `REPORT.md` v0.4.*

**Verification tags:** **[FT]** full text read · **[Abs]** abstract or record page only · **[2nd]** known only through citing papers · **[unverified]** lead not confirmed.

---

## 1. Per-claim verdicts

| # | Claim | Verdict | Key prior art |
|---|---|---|---|
| A | Moment tensor $\mathbb E[\mu(w^1)\cdots\mu(w^q)]$ of a (matrix) Mandelbrot cascade = exact uniform MPS / finite automaton, with set-partition states, ≤ Bell(q)·r^q bond (attained) and block-triangular refinement structure | **Formulation new; ingredients known** | Greiner–Eggers–Lipa 1998; Harris–Roberts 2017; Falconer 2013; Bertini–Klobas–Kos–Malz 2024 (same state space, other model) |
| A2 | MRW Gaussian layer as boundary vector → even-block partitions | mechanism standard, count not found | Bacry–Delour–Muzy 2001 |
| A3 | Kahane–Peyrière moment criterion = spectral gap in non-singleton sectors | spectral reading not found | Kahane–Peyrière 1976; Huang 2014 |
| B1 | Integer moments $\mathbb E Z_n(q)^m$ = max over replica groupings | **partly anticipated** (non-rigorous on trees; rigorous for REM); matrix version new | Goldschmidt 1998; Saakian 2012; Ogure–Kabashima 2009 |
| — | *Correction:* integer-$m$ sector crossings are $L^m$-boundedness thresholds, **not** freezing. Freezing = fractional ($s\le1$) / $m\to0$ crossing. | — | flagged in round 2 |
| L | Additive martingale with $P_t$-eigenfunction, nondegenerate iff $t\Phi'<\Phi$ | **KNOWN** | Grama–Mentemeier–Xiao arXiv:2507.09737 Thm 3.2 |
| — | Speed of max = $\inf\Phi(t)/t$ | **KNOWN** for $GL(V)$; follows for nonnegative | Grama–Mentemeier–Xiao arXiv:2206.04941 |
| B3 | Quenched $L^q$ spectrum with freezing for matrix cascades: $\tau=-\inf_{0<s\le1}\Phi(qs)/s$ | **new for matrices** (scalar classical); a theorem via GMX | scalar: Molchan, Ossiander–Waymire, Barral, Attia–Barral; multitype unfrozen: Attia–Balegh 2026 |
| O | Overlap law $\mathbb E\sum G_q^2\to1-t_c/q$ for matrix cascades | **new for matrices** | scalar: Derrida–Spohn; Jagannath; Mallein |
| F | $f(\alpha)$ of matrix cascades as Legendre transform | **partly known** (multitype special case, unfrozen branch) | Attia–Balegh 2026 |
| C5a | $\min(s,t)$ interleaved QTT rank 4 | **KNOWN, retracted** | Kazeev–Khoromskij 2012; Khoromskij survey Prop. 2.9 |
| C5b | Path unfolding SVD decay $j^{-(H+1/2)}$ | folklore (KL asymptotics) | Ali–Nouy |
| C5c | Unfolding spectra (almost) blind to multifractality | not found (priority fragile) | — |
| D | Depth- and dimension-free bond of moment tensors in $d$ dims | **not found** | Gelß–Schütte 2018 (deterministic analogue) |
| C7 | Continuum matrix chaos: GL(r) Brownian motion in log-scale, replica generator $\mathcal L_q$ | **concept anticipated; construction not found** | Schertzer–Lovejoy 1995; Schertzer–Tchiguirinskaia 2015; Chevillard–Rhodes–Vargas 2013; Baverez 2026 |

---

## 2. Bibliography by theme

### 2.1 Deterministic multifractal formalism ↔ matrix products / MPS (relabelling, known)
- Feng & Lau, "The pressure function for products of non-negative matrices", Math. Res. Lett. 9 (2002), arXiv:math/0205028. [FT]
- Feng, "Lyapunov exponents for products of matrices and multifractal analysis. Part I", Israel J. Math. 138 (2003). [Abs]
- Feng & Käenmäki, arXiv:1009.3129. [Abs]
- Kronecker / p-radius identity $\rho_p^p=\rho(\sum A_i^{\otimes p})$:
  - Crisanti–Paladin–Vulpiani (1993); Protasov (1997); Zhou (1998). [2nd]
  - Morris, SIAM J. Matrix Anal. Appl., arXiv:1905.00749 (real $p$ via a trace-class operator; Vanneste earlier). [FT]
- Adhikary et al., AISTATS 2021 (uMPS ≡ HMM ≡ WFA, with caveats). [FT]
- Li–Precup–Rabusseau, Mach. Learn. 2022. [Abs]
- Culik & Kari, Acta Inform. 34 (1997): WFA gray-tone fractal dimension. [Abs]

### 2.2 Stochastic processes / random fields with TT/MPS structure (the user's suspicion: they exist; none covers our claims)
- *Laws along time as MPS:*
  - Temme & Verstraete, "Stochastic matrix product states", arXiv:1003.2545. [Abs]
  - Johnson, Clark & Jaksch, arXiv:1006.2639. [Abs]
  - Garrahan, classical stochastic dynamics and cMPS, arXiv:1602.07966. [Abs]
  - Yang, Binder, Narasimhachar & Gu, MPS for quantum stochastic modelling, arXiv:1803.08220. [Abs]
- *Random MPS with i.i.d. cores:*
  - Garnerone, de Oliveira & Zanardi, arXiv:0908.3877. [FT]
  - Haferkamp, Bertoni, Roth & Eisert, arXiv:2103.02634. [Abs]
  - Movassagh & Schenker, arXiv:1909.11769. [Abs]
  - Lami, De Nardis & Turkeshi, PRL 134 (2025), arXiv:2409.13023: $S_k$ replica transfer matrix, annealed IPR. [FT]
  - Magni, Christopoulos, De Luca & Turkeshi, arXiv:2502.20455. [FT]
- *Moment/correlation tensors of random fields in TT:*
  - Kressner, Kumar, Nobile & Tobler, SIAM/ASA JUQ 3 (2015), doi:10.1137/140968938. [Abs]
  - Bonizzoni, Nobile & Kressner, MATHICSE 39.2014: rank $\binom{N+p-1}{p}$ in KL coordinates. [FT]
  - Bu, Zhao & Wang, arXiv:1907.06304. [FT]
- *TT for SDE / master equations / densities:*
  - Kazeev, Khammash, Nip & Schwab, PLoS CB (2014).
  - Dolgov et al., arXiv:1810.01212. [Abs]
- *Free (noncommutative) cascades:* McKeague, "Noncommutative probability and multiplicative cascades", Stat. Sci. 36 (2021). [FT]
- *MERA fBm:* Descamps, arXiv:1602.00924. [FT]

### 2.3 Cascade multipoint moments, partitions, replicas (ingredients of A, B1)
- Greiner, Eggers & Lipa, "Analytic multivariate generating function for random multiplicative cascade processes", PRL 80 (1998) 5333, chao-dyn/9804024. [FT]
- Greiner, Schmiegel, Eickemeyer, Lipa & Eggers, PRE 58 (1998) 554, chao-dyn/9804028. [FT]
- Eggers, Greiner & Lipa, hep-ph/9811204. [FT]
- Schmiegel & Greiner, cond-mat/0111364. [FT]
- Harris & Roberts, "The many-to-few lemma and multiple spines", AIHP 53 (2017), arXiv:1106.4761. [FT]
- Falconer, "Higher moments for random multiplicative measures", arXiv:1311.0160. [FT]
- **Bertini, Klobas, Kos & Malz, "Quantum and classical dynamics with random permutation circuits", arXiv:2407.11960**: set-partition replica states counted by Bell numbers. **Cite.** [FT]
- Saakian, J. Stat. Mech. (2012) P04007, arXiv:1212.1536: tree-REM integer-moment recursion via groupings. [FT]
- Goldschmidt, J. Phys. A 31 (1998), cond-mat/9809330: replica bundles on Cayley trees (non-rigorous). [FT]
- Ogure & Kabashima, J. Stat. Mech. (2009) P03010, arXiv:0812.4654: REM moments via partition types; rigorous 1RSB. [FT]
- Gardner & Derrida, J. Phys. A 22 (1989) 1975. [FT]
- Medina & Kardar, J. Stat. Phys. 71 (1993) 967. [Abs]
- Medina-Espinosa & Moreno Flores, arXiv:2310.02209: $q=2$ sector crossing for complex tree polymers. [FT]
- Derrida, Evans & Speer, CMP 156 (1993). [2nd]
- Buffet, Patrick & Pulé, J. Phys. A 26 (1993) 1823. [2nd]
- Cook & Derrida (1989; 1991). [Abs/FT]
- Monthus & Garel, arXiv:1004.1537 (travelling waves for cascades). [Abs]
- Fyodorov, arXiv:0911.2765. [FT]
- No theorem found that moments of random weighted automata / random substitutions are rational or k-regular. Michaliszyn–Otop (arXiv:1908.04625) shows irrationality in the nondeterministic case.

### 2.4 Matrix / vector / complex / multitype cascades (B3, O, F, L)
- **Grama, Mentemeier & Xiao, "Spinal decomposition, martingale convergence and the Seneta–Heyde scaling for matrix branching random walks", arXiv:2507.09737 (2025).** Nonnegative matrices; Thm 3.2 = Lemma L. [FT]
- **Grama, Mentemeier & Xiao, "The extremal position of a branching random walk on the general linear group", AIHP 61 (2025), arXiv:2206.04941.** [FT/Abs]
- Bui, Grama & Liu, CLT and precise large deviations for BRW with products of random matrices, hal-02911860. [Abs via talks]
- Buraczewski, Damek, Guivarc'h & Mentemeier, "On multidimensional Mandelbrot cascades", JDEA 20 (2014), arXiv:1109.1845. [FT]
- Menshikov, Petritis & Popov, math/0312298. [FT]
- Huang, "Moments for multi-dimensional Mandelbrot's cascades", arXiv:1405.2681. [FT]
- Kolesko & Mentemeier, arXiv:1409.7220 (critical case). [Abs]
- Mentemeier, PTRF 2016 (fixed points of the multivariate smoothing transform). [2nd]
- Mei & Liu, arXiv:2412.21173; J. Theor. Probab. 38 (2025). [Abs]
- Barral, "Generalized vector multiplicative cascades", Adv. Appl. Probab. 33 (2001) 874–895: Banach-algebra martingales, no matrix $L^q$ spectrum. [Abs]
- Barral & Jin, "Multifractal analysis of complex random cascades", CMP 2010, arXiv:0906.1501 (scalar in effect). [FT]
- Attia & Barral, CMP 2014, arXiv:1305.2034 (random environment, scalar). [Abs]
- Waymire & Williams, "Markov cascades" (1997); TAMS 348 (1996); math/9410222. Carrying dimension only. [Abs/FT]
- Kyprianou & Rahimzadeh Sani, Bernoulli 7 (2001); Biggins & Kyprianou, AAP 2004; Biggins, AAP 2012 (arXiv:1003.4716). [FT/Abs]
- **Attia & Balegh, "Branching random walks on multi-type Galton–Watson trees: a multifractal analysis", Fractal Fract. 10(10):731 (online 20 Oct 2026).** Closest prior art; multitype ⊂ matrix cascades; unfrozen branch only. **Check the full text.** [Abs]
- Attia & Mahjoub, *Vectorial Multifractal Measures*, SpringerBriefs 2026 (paywalled). [unverified]
- Scalar overlap: Jagannath, arXiv:1509.07527. [Abs]
- Quansheng Liu, talk "On Mandelbrot's multiplicative cascades" (CSUST, 30 Jul 2026): vector/matrix-valued cases without positivity; no preprint found. [Abs]

### 2.5 QTT ranks, rough functions, turbulence (C5, D)
- Kazeev & Khoromskij, "Low-rank explicit QTT representation of the Laplace operator and its inverse", SIMAX 33 (2012) 742–758, doi:10.1137/100820479; MPI MIS preprint 75/2010. Exact ranks unread (PDF host blocked). [Abs]
- Khoromskij, survey arXiv:1408.4053, Prop. 2.9 ($\mathrm{rank}(\Delta^{-1})\le5$, Dirichlet). [FT]
- Kazeev, Khoromskij & Tyrtyshnikov, multilevel Toeplitz QTT (SISC 2013). [Abs]
- Matveev & Smirnov, arXiv:2404.12230. [FT]
- Ali & Nouy, arXiv:2007.00118 and arXiv:2007.00128 (Besov ⊂ TN approximation classes). [FT]
- Gourianov et al., Nat. Comput. Sci. 2022, arXiv:2106.05782 (turbulence MPS; 3D ranks grow; no multifractality). [FT]
- Gourianov, Givi, Jaksch & Pope, arXiv:2407.09169. [Abs]
- Pisoni et al., arXiv:2506.05477 (sample bond dim ∝ number of scales). [FT]
- Guzman, Tiunov & Aolita, arXiv:2601.03885. [FT]
- Hölscher et al., arXiv:2406.17823. [Abs]
- Ritter et al., QTCI, PRL 132 (2024), arXiv:2303.11819. [FT]
- Shinaoka et al., PRX 13 (2023), arXiv:2210.12984. [Abs]
- Lindsey, arXiv:2311.12554. [Abs]
- Jobst et al., Quantum 8 (2024), arXiv:2311.07666. [FT]
- Gelß & Schütte, "Tensor-generated fractals", arXiv:1812.00814 (TT rank 2 independent of $d$; deterministic). [FT]
- Sabiu, arXiv:2608.24897 (tree transfer operator, 2D cascades). [FT]
- Palla, Lovász & Vicsek, arXiv:1004.5225. [FT]
- Jaffard, Seuret, Wendt, Leonarduzzi, Roux & Abry, "Multivariate multifractal analysis", ACHA 2019. [FT]
- Seuret, arXiv:2411.13959. [Abs]

### 2.6 Continuum / non-abelian chaos (C7)
- Schertzer & Lovejoy, "From scalar cascades to Lie cascades: joint multifractal analysis of rain and cloud processes" (CUP 1995), doi:10.1017/CBO9780511564475.014. [Abs]
- Schertzer & Tchiguirinskaia, "Multifractal vector fields and stochastic Clifford algebra", Chaos 25 (2015) 123127. [FT §I–VII]
- Chevillard, Rhodes & Vargas, "Gaussian multiplicative chaos for symmetric isotropic matrices", J. Stat. Phys. 150 (2013), arXiv:1207.1582 (parabolic ζ). [FT]
- **Baverez, "Non-abelian multiplicative chaos on the circle", arXiv:2607.18824 (2026)**: Casimir on $V\otimes V$; heuristically non-polynomial spectrum. **Cite.** [FT]

---

## 3. Open items (ranked by risk to novelty)
1. **Attia–Balegh 2026 full text.** Does it reach general matrices, or the frozen/linear branch?
2. **Q. Liu group.** Watch arXiv/HAL for the matrix-cascade work behind the July 2026 talk.
3. **Attia–Mahjoub 2026 book** (paywalled).
4. **Buffet–Patrick–Pulé 1993 and Derrida–Evans–Speer 1993 first-hand.** Integer moments on trees via groupings?
5. **Moment Lyapunov exponents of linear SDEs** (Arnold et al.). These likely give the single-branch top eigenvalue of $\mathcal L_q$.
6. **Kazeev–Khoromskij exact ranks** (open MPI MIS preprint 75/2010 manually).
7. **Schertzer–Tchiguirinskaia 2015 §§VIII–XI**, and their 2020 "multifractal operators" paper.

## 4. Positioning advice (for a paper)
- Lead with the **structural object**: the moment tensors of random cascades are exact uniform MPS on the set-partition lattice, depth- and dimension-free. Then derive the spectral consequences.
- Present the matrix-cascade quenched spectrum and overlap law as **new theorems built on GMX 2025**, not as independent martingale theory.
- Cite **Bertini et al. 2024** (partition replica states) and **Lami et al. 2025** (permutation replica states). Contrast node-wise randomness (partitions, refinement order, freezing) with level-wise Haar randomness (permutations, no freezing).
- Do not claim: $\min(s,t)$ rank, Lemma L, the max-speed, or the factorisation itself.
