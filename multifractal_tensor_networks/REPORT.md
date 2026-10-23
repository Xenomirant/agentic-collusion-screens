# Multifractal formalism ⇄ tensor networks: replica tensor trains for random cascades

*Working report v0.4 (2026-10-23).*

*Changes in v0.4:*
* *second targeted search round integrated into the audit;*
* *correction to how freezing relates to sector crossings.*

*Changes in v0.3:*
* *deeper novelty audit (six parallel searches, ~200 queries, key items checked against abstracts or full texts); corrections below;*
* *arbitrary-dimension test (§9).*

## Novelty audit v0.4 (read this first)

*Two search rounds, about 300 queries in total, nine search agents. Key items were checked against abstracts or full texts. Round 2 targeted the items left open in v0.3.*

**Bottom line**

*Known or partly known (no longer claimed as new):*
* $\min(s,t)$ having QTT rank 4;
* Lemma L (Grama–Mentemeier–Xiao 2025);
* the coalescence-tree factorisation under Theorem A;
* the integer-$m$ "max over replica groupings", which is known non-rigorously on trees (Goldschmidt 1998) and for the REM (Ogure–Kabashima).

*Survives as new, as far as ~300 queries reach:*
1. The **uniform-MPS / finite-automaton form** of cascade moment tensors: set-partition states, the Bell($q$)·$r^q$ bound attained, and block-triangular refinement structure with diagonal blocks $\bigotimes T_{|\beta|}$.
2. The **quenched $L^q$ spectrum with freezing**, the **overlap law $1-t_c/q$**, and the **integer-moment formula** for **matrix** Mandelbrot cascades. All of these are now theorems via GMX 2025, apart from the heuristic overlap law.
3. **Depth- and dimension-free bond dimension** of moment tensors (§9).
4. **Sample-vs-law blindness** of unfolding spectra.

*Partly anticipated:* the continuum matrix-chaos direction (§5). See the Lie-cascade and non-abelian-chaos rows of the table.

**Correction (v0.4) to Theorem B's wording.** For *integer* $m$, a crossing between partition sectors of $\mathbb E[Z_n(q)^m]$ (e.g. $2\Phi(q)$ vs $\Phi(2q)$) is an **$L^m$-boundedness threshold** of Kahane–Peyrière type, *not* the freezing point. Freezing at $t_c$ corresponds to the crossing between the $s=1$ and $s<1$ "blocks" in the **fractional** ($s\in(0,1]$) or $m\to0$ continuation. Rigorously:
* the upper bound uses exactly the fractional moment $\mathbb E Z^s\le\mathbb E\sum\mu^{qs}$, a single block of size $s$;
* the lower bound uses GMX.

So "freezing = sector crossing" is a correct *heuristic* (Derrida–Spohn) reading. The theorem is B3, not B1. *(Flagged by a round-2 search agent.)*

**Prior art for "stochastic processes with TT/MPS structure":** these exist, but none is a node-wise random cascade in quantized format with partition states.
* *Laws of Markov/HMM processes as MPS along time:*
  * Temme–Verstraete (arXiv:1003.2545);
  * Johnson–Clark–Jaksch (arXiv:1006.2639);
  * Garrahan (arXiv:1602.07966);
  * Yang–Binder–Narasimhachar–Gu (arXiv:1803.08220).
* *Random MPS with i.i.d. cores:*
  * Garnerone–de Oliveira–Zanardi (arXiv:0908.3877);
  * Haferkamp–Bertoni–Roth–Eisert (arXiv:2103.02634);
  * Movassagh–Schenker (arXiv:1909.11769);
  * Lami–De Nardis–Turkeshi (arXiv:2409.13023);
  * Magni–Christopoulos–De Luca–Turkeshi (arXiv:2502.20455).
* *Moment/correlation tensors of random fields in TT:*
  * Kressner–Kumar–Nobile–Tobler, SIAM/ASA JUQ 3 (2015);
  * Bonizzoni–Nobile–Kressner (rank $\binom{N+p-1}{p}$ in KL coordinates, growing with $N$);
  * Bu–Zhao–Wang (arXiv:1907.06304).
* *TT for SDE / master-equation / densities:* Kazeev–Khammash–Nip–Schwab (2014); Dolgov et al. (arXiv:1810.01212).
* *Free cascades:* McKeague, Stat. Sci. 36 (2021).

**Claim-by-claim status**

| Claim | Status | Closest prior art | Residual risk |
|---|---|---|---|
| A: moment tensor = uniform MPS / automaton, set-partition states, ≤ Bell(q)·r^q, attained, block-triangular | **Formulation not found; ingredients known** | Coalescence-tree factorisation: Greiner–Eggers–Lipa PRL 80 (1998); Greiner et al. PRE 58 (1998). Refining partitions of marks: Harris–Roberts (arXiv:1106.4761). Join points: Falconer (arXiv:1311.0160). **Set-partition replica states, counted by Bell numbers, in another model:** Bertini–Klobas–Kos–Malz, random permutation circuits (arXiv:2407.11960); must be cited. Grouping recursion for tree-REM moments: Saakian (arXiv:1212.1536). | A referee may say "immediate from the factorisation", so present the MPS/spectral structure as the contribution. No theorem was found on rationality of moments of random weighted automata (searched). |
| A2: MRW Gaussian layer restricted to even partitions | mechanism standard (Wick; Bacry–Delour–Muzy 2001) | — | low novelty value |
| A3: Kahane–Peyrière = spectral gap of the replica transfer matrix (integer q) | spectral reading not found; the criterion is classical | Kahane–Peyrière; Huang (arXiv:1405.2681) has non-matching sufficient/necessary matrix conditions | our sharp tensor-power criterion may add something for matrices; compare with Huang in detail |
| B1: integer moments E[Z^m] = max over groupings | **partly anticipated** (non-rigorous on trees; rigorous for REM). The matrix version was not found. | Goldschmidt (cond-mat/9809330); Ogure–Kabashima (arXiv:0812.4654); Saakian (arXiv:1212.1536); Medina-Espinosa–Moreno Flores (arXiv:2310.02209) | Buffet–Patrick–Pulé (1993) and Derrida–Evans–Speer (1993) not read first-hand |
| Lemma L | **KNOWN**: Grama–Mentemeier–Xiao (arXiv:2507.09737), Thm 3.2 | — | — |
| Max speed | **KNOWN** for $GL(V)$ (GMX, AIHP 2025, arXiv:2206.04941); follows for nonnegative matrices | — | — |
| **B3: quenched $L^q$ spectrum with freezing, matrix cascades** | **New for matrices** (scalar case classical) | Scalar: Molchan; Ossiander–Waymire; Barral (math/0503421); Attia–Barral (CMP 2014). Multitype special case, unfrozen only: Attia–Balegh, Fractal Fract. 10(10):731 (online 20 Oct 2026; abstract only). | Q. Liu talk (July 2026): no preprint found. Liu-group papers found (Mei–Liu arXiv:2412.21173, JTP 2025) do no dimension or spectrum work. Barral AAP 33 (2001) checked: Banach-algebra martingales, no matrix L^q spectrum. A referee may call the matrix extension routine. |
| Overlap law $1-t_c/q$, matrix cascades | **new for matrices**; scalar known | Derrida–Spohn; Jagannath (arXiv:1509.07527); Mallein | — |
| f(α) for MMC | **partly known**: multitype special case, unfrozen branch (Attia–Balegh 2026) | also Waymire–Williams, TAMS 348 (1996) (carrying dimension only) | **check the Attia–Balegh full text when posted** |
| $\min(s,t)$ QTT rank 4 | **KNOWN, retracted** | Kazeev–Khoromskij, SIMAX 33 (2012). Khoromskij survey (arXiv:1408.4053), Prop. 2.9 (bridge ≤ 5). | exact KK ranks unread (PDF host blocked) |
| path SVD decay $j^{-(H+1/2)}$ | folklore (KL asymptotics) | Ali–Nouy | — |
| unfolding spectrum (almost) blind to multifractality | not found | Gourianov et al. 2022; Guzman–Tiunov–Aolita (arXiv:2601.03885); Pisoni et al. (arXiv:2506.05477): sample ranks grow with the number of scales | the argument is simple, so priority is fragile |
| depth- and dimension-free bond of moment tensors (§9) | **not found** | Gelß–Schütte (arXiv:1812.00814), deterministic fractals; Sabiu (arXiv:2608.24897); existing random-field TT ranks grow with KL terms or scales | low–moderate |
| Continuum matrix chaos (§5) | **concept anticipated; specific construction not found** | Schertzer–Lovejoy "Lie cascades" (1995); Schertzer–Tchiguirinskaia, Chaos 25 (2015) (Clifford generators; no K(q) for non-commuting stochastic generators). Chevillard–Rhodes–Vargas, J. Stat. Phys. 150 (2013) (matrix GMC, pointwise exponential, parabolic). **Baverez, "Non-abelian multiplicative chaos on the circle" (arXiv:2607.18824)** (Casimir on tensor powers, heuristically non-polynomial spectrum); closest, must be cited. | the moment Lyapunov exponents of linear SDEs (Arnold et al.) likely contain the single-branch top-eigenvalue part |

**Still open:**
* the Attia–Balegh full text;
* the Attia–Mahjoub book *Vectorial Multifractal Measures* (2026, paywalled);
* Buffet–Patrick–Pulé and Derrida–Evans–Speer first-hand;
* the exact Kazeev–Khoromskij ranks (MIS preprint 75/2010; open the PDF manually);
* the Schertzer–Tchiguirinskaia 2015 §§VIII–XI;
* any unpublished work from Liu's group.

---

*Changes in v0.2:*
* *verified literature pass integrated (§6, with novelty table);*
* *frozen-phase formula for matrix cascades confirmed to 3·10⁻⁴ by a noise-free F-KPP recursion;*
* *lower-bound proof sketched;*
* *1RSB overlap law confirmed;*
* *Kahane–Peyrière criterion recast as a spectral gap.*

*Claim tags:* **[proved]** short proof given here · **[verified-num]** checked numerically by the scripts in `code/` · **[conj]** conjecture · **[lit]** verified prior work (3-vote adversarial check) · **[lit?]** prior work not yet verified.

## Verdict on the question "has this been studied?"

* **Deterministic connection: yes, known under other names.** For measures given by products of nonnegative matrices, $\tau(q)$ is the *matrix pressure* and the Legendre formalism holds where it is differentiable: Feng–Lau 2002 [arXiv:math/0205028], Feng 2003 (Israel J. Math. 138), Feng–Käenmäki [arXiv:1009.3129].
  * For integer $q$ and cone-preserving matrices the pressure is $\log\rho(\sum_cA_c^{\otimes q})$. This was found independently by Crisanti–Paladin–Vulpiani 1993, Protasov 1997 and Zhou 1998. Morris [arXiv:1905.00749] gives a trace-class operator for real $q$.
  * The objects themselves are known to coincide: nonnegative uniform MPS = HMM = probabilistic weighted automata (Adhikary et al., AISTATS 2021; Li–Precup–Rabusseau, Mach. Learn. 2022).
  * So "multifractal formalism = spectral theory of the replica transfer matrix of a uniform tensor train" is a **relabelling** **[lit]**, not new.
* **Stochastic connection: only fragments exist; no unified theory was found** **[lit]**.
  * Matrix-valued Mandelbrot cascades and the multivariate smoothing transform: Buraczewski–Damek–Guivarc'h–Mentemeier [arXiv:1109.1845]; Menshikov–Petritis–Popov [math/0312298]; Huang [arXiv:1405.2681]; Kolesko–Mentemeier [arXiv:1409.7220]. These cover existence, moments and tails only, **with no multifractal spectrum**.
  * Random tree tensor networks map to directed polymers on trees: Nahum–Roy–Skinner–Ruhman [arXiv:2009.11311], Lopez-Piqueres–Ware–Vasseur [arXiv:2003.01138], Feng–Skinner–Nahum [arXiv:2210.07264]. These treat entanglement phases, not $\tau(q)$.
  * Haar random MPS: an exact *annealed* IPR via an $S_k$ (permutation) replica transfer matrix (Lami–De Nardis–Turkeshi, PRL 2025 [arXiv:2409.13023]). It uses no multifractal language and has no freezing.
  * fBm via MERA: Descamps [arXiv:1602.00924]. It uses no $\tau(q)$.
* **What this report adds** (the gap the literature pass identified):
  * the set-partition replica MPS for node-wise random cascades (Theorem A);
  * the Brownian/MRW layer as a boundary vector over even partitions;
  * freezing as a sector crossing on the partition lattice (Theorem B);
  * **the quenched multifractal spectrum of matrix Mandelbrot cascades**: rigorous upper bound, proof sketch of the lower bound, numerics to 3·10⁻⁴;
  * the sample-vs-law rank dichotomy (Proposition C).
* **Caveat.** The web-search budget ran out before three targeted novelty checks:
  * Bui–Grama–Liu-type results on branching products of random matrices, which may already contain Lemma L / the max-speed;
  * Cates–Deutsch-style multipoint cascade correlations, which contain the factorisation behind Theorem A but likely not the MPS packaging;
  * 2024–26 random-circuit multifractality (Sierant, Turkeshi et al.).

  Re-check these before claiming priority.

---

## 0. TL;DR

1. **Deterministic side (mostly known, reframed).** Encode a measure at resolution $b^{-n}$ as an order-$n$ tensor $\mu_n\in(\mathbb R^b)^{\otimes n}$ (quantized-tensor-train / QTT indexing by $b$-adic digits). Then:
   * multinomial (binomial) cascades are exactly the **QTT rank-1** tensors;
   * bounded-rank QTT with stationary cores, i.e. uniform MPS, i.e. weighted finite automata, are exactly the **matrix-product measures**. These include Feng-type measures from products of nonnegative matrices, hidden-Markov measures, and self-similar measures with finite-type overlaps;
   * the multifractal partition function is a **contraction of $q$ replicas**: $Z_n(q)=\sum_w\mu_n(w)^q=\langle u^{\otimes q}|T_q^{\,n}|v^{\otimes q}\rangle$, with $T_q=\sum_c A_c^{\otimes q}$. So $\tau(q)=-\log_b\rho(T_q)$ for integer $q$, and for real $q$ it is the matrix pressure. **[lit]** Feng–Lau; Feng; Crisanti–Paladin–Vulpiani / Protasov / Zhou; Morris. The identity needs even $q$, or cone-preserving / nonnegative matrices.
2. **Stochastic side (the gap; contributions of this report).** Define the **matrix Mandelbrot cascade (MMC)**: an independent random tuple of nonnegative $r\times r$ matrices at every node of the $b$-ary tree. It contains Mandelbrot/Kahane–Peyrière cascades ($r=1$), deterministic matrix-product measures (zero randomness), and, with a Gaussian leaf layer, the dyadic multifractal random walk (MRW). Main results:
   * **Theorem A (replica MPS) [proved, verified-num].** A single MMC sample is generically QTT-full-rank. But its **$q$-point moment tensor** $\mathbb E[\mu_n(w^1)\cdots\mu_n(w^q)]$ is an **exact uniform MPS** in interleaved replica digits, with bond dimension $\le B_q\,r^q$ ($B_q$ = Bell number), independent of depth $n$. The bond state is the *set partition of replicas that still share a tree node*. Numerically the bound is saturated: ranks 1, 2, 5, 15 for $q=1..4$.
   * **Brownian layer [proved, verified-num].** For MRW increments $\delta X(w)=\sqrt{\mu(w)}\,\xi_w$, the Gaussian layer is only a *right boundary vector* $R(\pi)=\prod_{\beta\in\pi}(|\beta|-1)!!$. It prunes the automaton to **partitions into even blocks**, so bond dims are 1, 4, 31, 379 (OEIS A005046) for $q=2,4,6,8$. Brownian covariance $\min(s,t)$ has interleaved QTT rank exactly **4**. *(Known: Kazeev–Khoromskij 2012, see the audit.)*
   * **Theorem B (moments and freezing on the partition lattice).** The replica transfer matrix is **block-triangular over the refinement lattice of set partitions**. Consequences (see the v0.4 correction: integer-$m$ crossings are $L^m$ thresholds; freezing is the fractional / $m\to0$ crossing):
     * **[proved]** exact growth of integer moments $\mathbb E[Z_n(q)^m]$ as a max over compositions of $m$;
     * **[proved]** a rigorous a.s. upper bound $\limsup\frac1n\log_b Z_n(q)\le\inf_{0<s\le1}\Phi(qs)/s$ for all $r$;
     * the replica limit $m\to0$ gives the **quenched spectrum** $\tau_{\rm qu}(q)=-\inf_{0<s\le1}\Phi(qs)/s$, with $\Phi(t)=\log_b\rho\big(\sum_c\mathbb E A_c^{\otimes t}\big)$. This recovers Derrida–Spohn / Biggins for $r=1$.
     * For $r>1$ it is **a theorem**: the one lemma it needs is published (Grama–Mentemeier–Xiao 2025; see the audit).
     * It is **confirmed numerically**. In the unfrozen phase the match is to 3 decimals. In the frozen phase a noise-free F-KPP recursion gives front speed −0.11835 vs predicted −0.11863, the same +3·10⁻⁴ offset as the rigorous $r=1$ control.
     * The 1RSB overlap law $\mathbb E\sum_w G_q(w)^2\to1-t_c/q$ also holds for $r=2$.
   * **Proposition C (sample vs law) [verified-num].** Unfolding singular values of sample paths decay as $j^{-(H+1/2)}$ (fBm). For MRW they are *almost blind to intermittency* (exponent ≈ −1 like Brownian motion, while $\zeta(4)$ moves from 2.00 to 1.64). Pathwise tensor ranks see only 2nd-order statistics. **Multifractality lives in the law, so the right tensor object is the replica (moment) tensor, not the sample.**
3. **Continuum direction [conj].** Let the scale-time $\ell=-\log_b|I|$ drive a multiplicative Brownian motion on the bond space (a "Lie-group BM" in log-scale, branching at every split). This gives a *matrix-valued Gaussian multiplicative chaos*; for $r=1$ it reduces to GMC/lognormal cascades. This is the precise sense in which "the stochastic component similar to Brownian motion" enters tensor networks: **the cores perform a random walk / Brownian motion along the scale axis, and the multifractal spectrum is the spectral theory of the replica generator**.

---

## 1. Setting and dictionary

Fix the alphabet $\Sigma=\{0,\dots,b-1\}$. A word $w=w_1\cdots w_n$ indexes the $b$-adic interval $I_w$. A measure at resolution $n$ is a tensor $\mu_n\in(\mathbb R_{\ge0}^b)^{\otimes n}$, and projectivity is $\mu_n(w)=\sum_c\mu_{n+1}(wc)$.

**Matrix-product measure (deterministic).** $\mu_n(w)=u^\top A_{w_1}\cdots A_{w_n}v$, with $A_c\ge0$ of size $r\times r$, $\big(\sum_cA_c\big)v=v$ and $u^\top v=1$. This is a uniform MPS / WFA with QTT rank $\le r$ at every cut. For $r=1$ it is the multinomial cascade, and $\mu_n=a^{\otimes n}$ is rank-1 **[verified-num: binomial cascade QTT ranks all 1]**.

**Replica identity.** $\mu_n(w)^q=(u^{\otimes q})^\top A_{w_1}^{\otimes q}\cdots A_{w_n}^{\otimes q}v^{\otimes q}$, hence
$$Z_n(q)=\langle u^{\otimes q}|T_q^n|v^{\otimes q}\rangle,\qquad T_q=\sum_cA_c^{\otimes q},\qquad \tau(q)=-\log_b\rho(T_q)\ (q\in\mathbb N).$$
For real $q$, replace $\rho(T_q)$ by the spectral radius of the projective (Ruelle-type) transfer operator $P_qf(\bar x)=\sum_c\|xA_c\|^qf(\overline{xA_c})$ on the positive cone. **[verified-num: projective operator vs. Kronecker formula agree to 5 decimals at q=2,3]**.

*Reading:* the multifractal formalism is the spectral theory of the **replica transfer operator** of a uniform MPS. The QTT rank $r$ is a complexity parameter. Rank 1 gives the Bernoulli/binomial parabola-like spectra. Rank $r>1$ gives the matrix-product spectra, which can be non-concave or show phase transitions in the reducible case.

## 2. Matrix Mandelbrot cascades (MMC)

**Definition.** To every node $x$ of the $b$-ary tree attach an independent copy $(A^{(x)}_0,\dots,A^{(x)}_{b-1})$ of a random tuple of nonnegative $r\times r$ matrices, with $\mathbb E\big[\sum_cA_c\big]v=v$. Set
$$\mu_n(w)=u^\top A^{(\varnothing)}_{w_1}A^{(w_1)}_{w_2}\cdots A^{(w_1\cdots w_{n-1})}_{w_n}v .$$
* $r=1$: Mandelbrot / Kahane–Peyrière cascade.
* Deterministic tuple: matrix-product measure.
* Total mass $Y_n=\sum_w\mu_n(w)$ is a nonnegative martingale. The vector of limiting masses satisfies $Y\overset d=\sum_cA_cY^{(c)}$, i.e. it is a fixed point of the **multivariate smoothing transform**.
* **[lit]** BDGM [1109.1845] show that $\mathbb E[N]\kappa(s)$, with $\kappa(s)=r(P^s)$ the spectral radius of the transfer operator, plays the role of the scalar $\theta(s)=\mathbb E\sum W^s$. A nontrivial fixed point exists iff $\kappa'(1^-)<0$. In our notation $b^{\Phi(s)}=b\,\kappa(s)$.
* **[lit]** Huang [1405.2681] gives moments; Kolesko–Mentemeier [1409.7220] treat the critical case and a derivative martingale.
* **None of these computes $\tau(q)$ or $f(\alpha)$ of the cascade measure.**

### Theorem A (replica MPS) [proved]
For integer $q$, the moment tensor $M^{(q)}_n(w^1,\dots,w^q)=\mathbb E\prod_a\mu_n(w^a)$, written in interleaved digits $(w^1_k,\dots,w^q_k)_{k=1}^n$ (site dimension $b^q$), is a uniform MPS with:
* bond space spanned by pairs $(\pi,\mathbf i)$, where $\pi\in\Pi_q$ is a set partition and $\mathbf i\in[r]^q$; dimension $\le B_q r^q$;
* core $G[(\pi,\mathbf i),\mathbf c,(\pi',\mathbf j)]=\mathbf 1[\pi'=\pi\wedge\ker\mathbf c]\prod_{\beta\in\pi}\mathbb E\Big[\prod_{a\in\beta}(A_{c_a})_{i_aj_a}\Big]$;
* left boundary $(\{[q]\},u^{\otimes q})$ and right boundary $\sum_\pi(\pi,v^{\otimes q})$.

*Proof.* Expand $\prod_a\mu(w^a)$ as $(u^{\otimes q})^\top\prod_k\big(\bigotimes_aA^{(w^a_{<k})}_{w^a_k}\big)v^{\otimes q}$. At level $k$, replicas whose prefixes coincide use the *same* node tuple. These groups are the blocks of $\pi_{k-1}$, the prefix-equality partition. Distinct nodes are independent, so the expectation factorises over levels and over blocks. Moreover $\pi_k=\pi_{k-1}\wedge\ker(\mathbf c_k)$ is computed by a deterministic finite automaton reading the digits. ∎

**[verified-num]** Exact moment tensors of a scalar cascade with a random 4-atom law have interleaved TT ranks:
* $q=2$: [2,2,2,2,2,2]
* $q=3$: [4,5,5,5]
* $q=4$: [8,15,13], where the last cut is boundary-limited.

So the Bell bound is attained. By contrast, a *sample* has QTT ranks [2,4,8,16,32,64,32,…] (maximal).

**Symmetry reduction.** For contractions invariant under replica permutations (e.g. $\mathbb E Z^m$), the bond reduces to partition *types* (integer partitions, $p(q)$).

**Contrast with Haar random MPS** (Lami–De Nardis–Turkeshi). Randomness drawn *once per level* and Haar-averaged gives replica states in $S_k$ (permutations, Weingarten calculus). Randomness drawn *independently at every tree node* (cascades) gives replica states in the **set-partition lattice** $\Pi_q$.
* The partition lattice is ordered, and blocks only refine. This order is what makes the transfer matrix triangular.
* It is also what produces freezing and ultrametric overlaps (§3). Neither appears in the Haar-MPS setting.

**Corollary A3 (Kahane–Peyrière as a spectral gap) [proved].** Let $q\ge2$ be an integer. Then $\mathbb E[Y_n^q]=\langle u^{\otimes q}|\cdots|\mathbf 1\otimes v^{\otimes q}\rangle$ is the full contraction of the replica MPS. The diagonal blocks are $D_\pi=\bigotimes_{\beta\in\pi}T_{|\beta|}$, and $\rho(T_1)=1$.
* The all-singletons sector therefore has eigenvalue 1.
* Hence $\sup_n\mathbb EY_n^q<\infty$ iff $\rho(T_k)<1$ for $2\le k\le q$.
* By convexity of $\Phi$ with $\Phi(1)=0$, this is iff $\Phi(q)<0$, i.e. $\tau_{\rm ann}(q)>0$.

For $r=1$ this is exactly the Kahane–Peyrière moment criterion $b\,\mathbb E W^q<1$. For $r>1$ it gives the integer-moment criterion for matrix cascades, which should agree with Huang [1405.2681] **[lit?: not checked in detail]**.

**Corollary A1 (annealed spectrum) [proved, verified-num].** On the diagonal $w^1=\dots=w^q$ the replicas never split, so
$$\mathbb EZ_n(q)=\langle u^{\otimes q}|\Big(\textstyle\sum_c\mathbb E A_c^{\otimes q}\Big)^n|v^{\otimes q}\rangle,\qquad \tau_{\rm ann}(q)=-\Phi(q),\quad \Phi(q)=\log_b\rho\Big(\sum_c\mathbb EA_c^{\otimes q}\Big).$$
This unifies Mandelbrot ($\Phi=\log_b\sum_c\mathbb EW_c^q$) and Feng ($\Phi=\log_b\rho(\sum A_c^{\otimes q})$). Brute-force enumeration over all node assignments ($n=3$, $r=2$) matches the formula to 10 digits.

**Corollary A2 (Brownian layer).** Take $r=1$ and $\delta X(w)=\sqrt{\mu_n(w)}\,\xi_w$ with $\xi_w$ iid $N(0,1)$ (dyadic MRW / Brownian motion in multifractal time).
* The cores use half-moments $\mathbb E\prod_cW_c^{m_c/2}$.
* The right boundary becomes $R(\pi)=\prod_{\beta}\mathbb E\xi^{|\beta|}$.
* A block of odd size can only refine into a partition that still has an odd block, so odd states are dead. The live bond space is therefore the even set partitions: dims 1, 4, 31, 379, …
* **[verified-num]** $q=2$ gives ranks [1,…,1]; $q=4$ gives [4,4,4].
* Path moments (cumulative sums) follow by applying the QTT summation operator (rank 2 per replica). **[verified-num]** For Brownian motion itself, the covariance $\min(s,t)$ has interleaved QTT rank 4 at every cut. fBm covariances are not exactly low rank, but have small ε-ranks (≈12 at ε=1e-6, ≈22 at 1e-10, n=10).

## 3. Freezing as a sector crossing in the replica transfer matrix

The $q$-replica transfer matrix is **block upper-triangular with respect to refinement** of $\pi$, because blocks can split but never merge. Hence its spectrum is the union of the spectra of the diagonal blocks.
* In sector $\pi$ every block keeps its replicas together, so the diagonal block is $D_\pi=\bigotimes_{\beta\in\pi}T_{|\beta|}$, with $T_t=\sum_c\mathbb EA_c^{\otimes t}$.
* Its spectral radius is $\prod_\beta\rho(T_{|\beta|})$.

**Theorem B1 (integer moments) [proved, modulo routine positivity/irreducibility].** For $q,m\in\mathbb N$,
$$\lim_n\frac1n\log_b\mathbb E\big[Z_n(q)^m\big]=\max_{s_1+\dots+s_k=m,\ s_j\ge1}\ \sum_{j}\Phi(qs_j).$$
The $m$ groups of $q$ replicas are forced to be diagonal inside each group. The sectors are partitions of the groups, and at most $m-1$ refinements occur, which only contributes polynomial factors in $n$.

**Caution (v0.4).** The integer-$m$ sector crossings in B1 are $L^m$-boundedness thresholds, not the freezing point.

**Replica limit (non-rigorous, Derrida–Spohn style).** Restrict to equal blocks of size $s$ ($m/s$ blocks) and continue $m\to0$. The max over $s\in[1,m]$ becomes a min over $s\in[m,1]$:
$$\lim_n\frac1n\log_bZ_n(q)=\inf_{0<s\le1}\frac{\Phi(qs)}{s}\quad\text{a.s.},\qquad \tau_{\rm qu}(q)=-\inf_{0<s\le1}\frac{\Phi(qs)}{s}.$$
* $s^*=1$ is the unfrozen phase, where quenched equals annealed.
* $s^*<1$ is the frozen phase: $\tau$ becomes linear beyond $t_c=\arg\min_t\Phi(t)/t$.
* For $r=1$ this is the known Biggins / Derrida–Spohn / Molchan / Barral result.

**Proposition B2 (rigorous upper bound, all $r$) [proved].** For $s\in(0,1]$, subadditivity gives $(\sum x_i)^s\le\sum x_i^s$, so $\mathbb EZ_n(q)^s\le\mathbb E\sum_w\mu_n(w)^{qs}=b^{n(\Phi(qs)+o(1))}$. Markov's inequality plus Borel–Cantelli then give a.s.
$$\limsup_n\tfrac1n\log_bZ_n(q)\le\inf_{s\in(0,1]}\Phi(qs)/s .$$

**Theorem B3 (quenched spectrum of MMC, $q>0$) [theorem: Lemma L is Grama–Mentemeier–Xiao 2025, Thm 3.2].**

*Assumptions.* The matrices are a.s. strictly positive, with entry ratios bounded by $\kappa$ (Birkhoff contraction is uniform), and $\mathbb E\|A\|^{t}<\infty$ near the relevant $t$.

*Claim.* Almost surely $\frac1n\log_bZ_n(q)\to\inf_{0<s\le1}\Phi(qs)/s=q\inf_{0<u\le q}\Phi(u)/u$. Equivalently, with $t_c=\arg\min_{t>0}\Phi(t)/t$:
* $\tau_{\rm qu}(q)=-\Phi(q)$ for $q\le t_c$;
* $\tau_{\rm qu}(q)=-q\,\Phi(t_c)/t_c$ for $q\ge t_c$ (linear: frozen).

*Proof sketch.* The upper bound is B2.
1. **RPF.** $P_tf(\bar x)=\mathbb E\sum_c\|xA_c\|^tf(\overline{xA_c})$ on $C(\Delta)$ has a simple leading eigenvalue $b^{\Phi(t)}$ with positive eigenfunction $h_t$, by Birkhoff contraction plus Ruelle–Perron–Frobenius. $\Phi$ is convex and differentiable. Since $v>0$ and $h_t\asymp1$, we have $\mu_n(w)\asymp\|x_w\|$.
2. **Additive martingale.** $W_n(t)=b^{-n\Phi(t)}\sum_{|w|=n}\|x_w\|^th_t(\bar x_w)$ is a positive martingale.
   * *Lemma L* (Biggins–Lyons criterion, Markov-modulated). $W_\infty(t)>0$ a.s. iff $t\Phi'(t)<\Phi(t)$ (⇔ $t<t_c$) and $\mathbb E W_1\log^+W_1<\infty$.
   * Route: spine decomposition, where the tilted spine is a Markov-additive process on the simplex with uniformly ergodic direction chain. Its log-norm has LLN speed $\Phi'(t)$ under the tilt. A zero-one law follows from $q(\bar x)=\mathbb E\prod_cq(\overline{xA_c})$ and uniform contraction.
   * Hence for $q<t_c$: $Z_n(q)\asymp b^{n\Phi(q)}W_n(q)$, and the lower bound follows.
3. **Frozen phase.** For $t<t_c$ the size-biased law $Q_t$ is equivalent to $P$, and under $Q_t$ the spine has speed $\Phi'(t)$. So $P$-a.s. some path satisfies $\log_b\mu_n(w)\ge n(\Phi'(t)-\varepsilon)$. Letting $t\uparrow t_c$, where $t_c\Phi'(t_c)=\Phi(t_c)$, gives $\liminf\frac1n\max_w\log_b\mu_n(w)\ge\Phi(t_c)/t_c$. Then $Z_n(q)\ge\max_w\mu_n(w)^q$ matches the upper bound. ∎

*Literature for Lemma L (updated v0.3).*
* **Lemma L is a published theorem.** Grama–Mentemeier–Xiao, "Spinal decomposition, martingale convergence and the Seneta–Heyde scaling for matrix branching random walks", arXiv:2507.09737 (2025). Their setting is nonnegative (allowable) matrices, and Thm 3.2 gives $W_\infty(s)>0\iff M(s)>sM'(s)$ plus $\mathbb E W_1\log^+W_1<\infty$, with the same eigenfunction-weighted martingale.
* The speed of the maximum, $\inf_sM(s)/s$, is Grama–Mentemeier–Xiao, AIHP 61 (2025), arXiv:2206.04941, stated for $GL(V)$ with a density. For nonnegative matrices it follows from the 2025 martingale theorem by the spine argument of step 3.
* **Hence Theorem B3 is a theorem** under the GMX hypotheses. What appears new is the statement and use of the *quenched $L^q$ spectrum with freezing for the cascade measure*, plus the overlap law.
* At $t=1$ this is the BDGM nondegeneracy theorem; the critical case is the Kolesko–Mentemeier derivative martingale.

**1RSB corollary [heuristic, verified-num].** Two replicas drawn from the Gibbs measure $G_q=\mu_n^q/Z_n(q)$ either split near the root or share (almost) their whole path. The latter happens with probability $\mathbb E\,Y_n=\mathbb E\,Z_n(2q)/Z_n(q)^2\to(1-t_c/q)_+$, the Derrida–Spohn law, now with the matrix $t_c$. In the replica MPS the "split time" of two replicas *is* the sector transition, so the ultrametric overlap structure is literally the refinement lattice.

**[verified-num]**

*Unfrozen phase* (`t2_matrix_cascade.py`, $r=2$). The simulated slope of $\log_2Z_n(q)$ matches $\Phi(q)$:

| q | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|---|
| predicted | .4825 | 0 | −.4513 | −.8756 | −1.6592 | −2.3780 | −3.6908 |
| simulated (n≤21) | .4827 | .0007 | −.4500 | −.8736 | −1.6554 | −2.3728 | −3.709 |

*Frozen phase.*
* Fixed-$N$ beam (N-BRW, depth 300, `t5_beam.py`): the simulated velocity sits below the predicted $\gamma^*$ by 0.012 for $r=1$ (a rigorous case) and by 0.013 for $r=2$. Equal gaps mean the shift is the Brunet–Derrida selection bias, not a failure of the formula.
* Extrapolation $N\to\infty$ in $1/\ln^2N$ (`t8_beam_extrap.py`): $r=1$ gives −0.0522 vs −0.0502 predicted. $r=2$ gives −0.027 vs −0.035, but noisy (2 reps, non-monotone in $N$), so it needs more reps.
* Full-tree simulations at $n\le22$ (`t3`, `t4`) are dominated by $\log n$ corrections and are inconclusive in the frozen phase. This includes the $r=1$ control.
* **Decisive, noise-free test** (`t9b_fkpp_tail.py`). The law of the maximum, $H_n(x,\bar x)=P(\max_w\log_2\|x A_w\|>x)$, obeys a type-dependent F-KPP recursion: $1-H_{n+1}(x,\bar x)=\mathbb E\prod_c\big(1-H_n(x-\log_2\|xA_c\|,\overline{xA_c})\big)$. It is iterated deterministically for 1000 steps on a tail-accurate grid (the grid reaches $H\sim10^{-100}$ to avoid a Brunet–Derrida cutoff), and the Bramson $-\frac{3}{2t_c}\log_2n$ term is removed:

  | | $t_c$ | predicted $\gamma^*$ | front speed | diff |
  |---|---|---|---|---|
  | $r=1$ (rigorous control) | 1.540 | −0.10824 | −0.10797 | +0.00027 |
  | $r=2$ (matrix cascade) | 1.550 | −0.11863 | −0.11835 | +0.00028 |

  *The first version, `t9_fkpp.py`, truncated the tail at ~1e-8. That acts as a cutoff and biased even the $r=1$ control by −0.015, so it is kept only as a cautionary record.*
* **Overlap / 1RSB** (`t10_overlap.py`, $n=20$, 12 reps; heavy-tailed $Y$, so noisy):

  | | q=0.8 | q=1.6 | q=2 | q=3 | q=4 |
  |---|---|---|---|---|---|
  | $r=2$: $\mathbb E Y$ | .001 | .174 | .305 | .519 | .632 |
  | pred $1-t_c/q$ ($t_c$=1.442) | 0 | .099 | .279 | .519 | .640 |
  | $r=1$ control: $\mathbb E Y$ | .002 | .180 | .304 | .495 | .594 |
  | pred ($t_c$=1.290) | 0 | .194 | .355 | .570 | .678 |

## 4. Sample vs law (why "stochastic" changes what "rank" means)

* A sample of a nondegenerate random cascade (or of a Brownian / fBm path) is **QTT-full-rank** (incompressible). Its unfolding singular values decay algebraically, $\sigma_j\sim j^{-(H+1/2)}$ (Karhunen–Loève). **[verified-num]** Fitted exponents: −0.77, −1.02, −1.24 for $H=0.3,0.5,0.7$; predicted −0.8, −1.0, −1.2. So ε-ranks grow like $\varepsilon^{-1/(H+1/2)}$.
* Unfolding spectra are 2nd-order objects: $\mathbb E[\text{Gram}]$ depends only on the 2-point tensor. For MRW, $\mathbb E\mu(I)=|I|$ forces $\zeta(2)=1$, so the expected Gram equals Brownian motion's. **[verified-num]** As $\lambda^2$ goes 0 → 0.3:
  * SVD decay exponent moves only from −1.02 to −1.09;
  * $\zeta(4)$ moves from 2.00 to 1.64 (theory 1.70).
* **Moral.** The tensor-network analogue of *statistical* self-similarity is **bounded rank of the replica/moment tensors with scale-stationary cores**, not low rank of samples. Call a random tensor family *$(r_q)$-replica-low-rank* if $\mathrm{rank}_{\rm TT}\,\mathbb E[X^{\otimes q}]\le r_q$ uniformly in $n$. Random cascades, MMC, MRW, Brownian motion and GMC (dyadic) are all in this class.

## 5. Continuum limit: Brownian motion along the scale axis [conj / programme]

Make the cascade steps infinitesimal in scale-time $\ell=-\log_b|I|$. Along one branch the bond vector then obeys a matrix SDE
$$dx_\ell=x_\ell\Big(G\,d\ell+\sum_iK_i\circ dB_i(\ell)\Big),$$
with branches sharing noise up to their split point (a log-correlated / tree-indexed field). This is a **multiplicative Brownian motion on $GL(r)$ in log-scale**, and the measure is a **matrix-valued GMC**.
* The $q$-replica generator is $\mathcal L_q=\sum_aG^{(a)}+\tfrac12\sum_i\big(\sum_aK_i^{(a)}\big)^2$, plus the splitting operator on partitions.
* $\tau_{\rm ann}(q)$ is $-\lambda_{\max}(\mathcal L_q|_{\rm diag})/\ln b$, up to normalisation.
* For $r=1$ this reduces to the lognormal spectrum $\tau(q)=(q-1)(1-\lambda^2q/2)$.
* For non-commuting $K_i$ the spectrum is non-parabolic and genuinely new (a "Casimir"-type replica operator).
* Open issue: positivity. Restrict to cone-preserving semigroups, or move to amplitudes, $|\psi|^{2q}$ and MPS wavefunction multifractality.

## 6. Prior work and novelty map

*Verified by the deep-research pass: 22 sources fetched, 25 claims put to a 3-vote adversarial check, 21 confirmed, 4 refuted. Full record in `code/research_result.json`.*

| Topic | What exists | Source(s) | Relation to this report |
|---|---|---|---|
| Matrix-product multifractal formalism | $\tau(q)$ = matrix pressure; Gibbs measures; Legendre formula where $P$ is differentiable; phase transitions without irreducibility; $q<0$ differentiability open | Feng–Lau 2002 [math/0205028]; Feng 2003 Israel J. Math. 138; Feng–Käenmäki [1009.3129] | §1 is a relabelling of this |
| Replica/Kronecker identity | $\rho_p^p=\rho(\sum A_i^{\otimes p})$ for even $p$, or integer $p$ with a cone; real $p$ via a trace-class operator (Vanneste earlier) | Crisanti–Paladin–Vulpiani 1993; Protasov 1997; Zhou 1998; Morris [1905.00749] | §1 replica identity is known |
| uMPS ≡ HMM ≡ WFA | equivalences (non-terminating limit; caveats) | Adhikary et al. AISTATS 2021; Li–Precup–Rabusseau 2022 | dictionary |
| WFA fractal images | gray-tone fractal dimension from a WFA (monofractal) | Culik–Kari, Acta Inform. 34 (1997) | early deterministic precedent |
| Matrix Mandelbrot cascades | fixed points, nondegeneracy $\kappa'(1^-)<0$, moments, tails, critical case | BDGM [1109.1845]; MPP [math/0312298]; Huang [1405.2681]; Kolesko–Mentemeier [1409.7220] | our MMC; **their $\tau,f(\alpha)$ were never computed** → Thm B3 |
| Random tree TN ↔ directed polymers | entanglement transitions via smoothing-transform recursions; replica cavity method on a Cayley tree | Nahum–Roy–Skinner–Ruhman [2009.11311]; Feng–Skinner–Nahum [2210.07264]; Lopez-Piqueres–Ware–Vasseur [2003.01138] | same polymer physics, different observable |
| Log-correlated multifractal measures | disorder-induced multifractality ≡ polymers on trees, freezing ↔ RSB | Fyodorov [0911.2765] | the $r=1$ statistical-mechanics side |
| Random MPS replica transfer matrix | exact Haar-averaged IPR via an $S_k$ transfer matrix (annealed; no "multifractal" wording) | Lami–De Nardis–Turkeshi PRL 134 (2025) [2409.13023] | closest analogue; permutations vs our partitions; no freezing |
| fBm / multifractal-ish processes via MERA | circuit construction; moment scaling only | Descamps [1602.00924] | only paper with both keywords; no formalism |
| TN approximation of rough functions | Besov spaces embed into tensorised-TN approximation classes | Ali–Nouy [2007.00128] | consistent with algebraic ε-ranks of paths (§4) |
| Multivariate multifractal formalism | bivariate Legendre spectrum fails as an upper bound | Seuret [2411.13959] | relevant to future multivariate/tensor-data work |

**Novel as far as the verified search reaches.** The synthesis agent of the research run independently re-derived the Bell($q$) MPS and also flagged it as unverified-novel.
1. Theorem A: the set-partition replica MPS, the Bell bound, and its saturation.
2. Corollary A2: the even-partition Brownian layer.
3. Corollary A3: the Kahane–Peyrière criterion recast as a spectral gap.
4. Theorem B: integer moments via the refinement lattice, and freezing as a sector crossing.
5. **Theorem B3: the quenched multifractal spectrum of matrix Mandelbrot cascades**, with a numerical confirmation.
6. Proposition C: the sample-vs-law dichotomy, and the near-blindness of the SVD spectrum to intermittency.

**Not yet checked** (search budget exhausted):
* Bui–Grama–Liu and coauthors (branching products of random matrices);
* Cates–Deutsch, O'Neil–Meneveau (multipoint cascade correlations);
* Barral (random/matrix cascades);
* Sierant–Turkeshi et al. (random-circuit multifractality, 2024–26);
* Kazeev–Khoromskij (explicit QTT of Laplacian-type operators, relevant to the rank-4 $\min(s,t)$).

## 7. Next steps
1. Close the unchecked priority items above (targeted search).
2. Write Lemma L in full: spine decomposition with a Markov-modulated spine; Guivarc'h–Le Page spectral gap; zero-one law. Then the $f(\alpha)$ formalism for MMC: Mandelbrot measures $\mu_t$ built from $W_\infty(t)$ carry the level sets $\alpha=-\Phi'(t)$.
3. Negative and real $q$ in the stochastic setting: randomise Morris's trace-class operator, or use a Fredholm determinant over the partition lattice.
4. Estimation. Fit MMC cores to data by matching empirical replica tensors (TT-cross on structure-function tensors). This would be a tensor-network alternative to MF-DFA / wavelet leaders, and connects to `fathon` in this folder. A rank/sector-based early-warning indicator is speculative.
5. Continuum: rigorous matrix GMC in the cone-preserving case; $\tau$ from the replica generator $\mathcal L_q$; non-parabolic spectra from non-commuting noise.

## 8. Code (`code/`, numpy only; t2/t3/t5/t8 need a few minutes each)
| file | what it checks |
|---|---|
| `t1_replica_rank.py` | Bell-number TT ranks of cascade moment tensors; sample = full rank; binomial = rank 1 |
| `t2_matrix_cascade.py` | MMC: brute force vs replica formula; projective pressure; quenched slopes (unfrozen) |
| `t3_freezing.py`, `t4_max.py` | full-tree frozen-phase tests (finite-size dominated; kept for the record) |
| `t5_beam.py`, `t8_beam_extrap.py` | frozen-phase speed via N-BRW beam, Brunet–Derrida extrapolation |
| `t6_brownian.py` | QTT rank of min(s,t)=4; fBm ε-ranks; path SVD decay $j^{-(H+1/2)}$; MRW even-partition ranks |
| `t7_mrw_svd.py` | SVD spectrum of MRW paths is (almost) blind to intermittency |
| `t9_fkpp.py` | first F-KPP attempt (tail cut at 1e-8: biased; cautionary) |
| `t9b_fkpp_tail.py` | **noise-free frozen-phase test**: type-dependent F-KPP for the max, tail-accurate; r=1 and r=2 agree with theory to 3e-4 (r=2 run ≈13 min) |
| `t10_overlap.py` | 1RSB participation ratio $\mathbb E Y\to 1-t_c/q$ for r=1, r=2 (≈10 min) |
| `results_log.txt` | raw outputs of all runs above |
| `research_result.json` | full verified literature pass (claims, votes, evidence, refuted claims, sources) |
| `t11_ddim.py` | d-dimensional cascades: replica-MPS ranks at every cut (interleaved and fused) for iid / exchangeable / general child laws, d=2,3, q=2..4 |

## 9. Scalability to arbitrary dimension (v0.3)

Setting: a cascade on $[0,1]^d$ with $2^d$ children per node. The quantized format puts one site per (level, coordinate) bit, carrying that bit for all $q$ replicas, so the site dimension is $2^q$ and there are $nd$ sites. The "fused" format uses one site per level, with site dimension $2^{dq}$.

**Proposition D (dimension-free bond dimension) [proved, verified-num].**
1. *Any child law, fused format.* The bond dimension is $\le B_q r^q$, independent of $n$ **and** $d$. This is Theorem A with alphabet $2^d$.
2. *i.i.d. children (the standard model), interleaved format.* The child weights are independent given the parent. The level weight then factorises over the blocks of the *new* partition, $\prod_{\beta'}\mathbb EW^{|\beta'|}$, and can be applied after the last coordinate bit. So the state is just the current partition, and the bond is $\le B_q r^q$ at **every** cut, including cuts inside a level.
3. *Exchangeable child laws.* The weight depends on (start partition, end partition). So the bond is $\le\#\{(\pi,\pi'):\pi'\le\pi\}\cdot r^q$: 3, 12, 60 for $q=2,3,4$, still independent of $d$.
4. *General (non-exchangeable, non-product) child laws, interleaved format.* Cuts inside a level must remember partial child labels, so the rank there grows with $d$. At level boundaries it stays $B_q$.

**[verified-num]** Maximum TT rank over all cuts, interleaved format (`t11_ddim.py`):

| (d, n, q) | iid | exchangeable | general (interleaved) | general (fused) | $B_q$ |
|---|---|---|---|---|---|
| (2,4,2) | 2 | 2 | 6 | 2 | 2 |
| (3,3,2) | 2 | 2 | 8 | 2 | 2 |
| (2,3,3) | 5 | 8 | 22 | 5 | 5 |
| (3,2,3) | 5 | 5 | 40 | 5 | 5 |
| (2,2,4) | 15 | 15 | 16 | 15 | 15 |

**Cost scaling (honest).**
* **Depth $n$ and dimension $d$:** linear (bond fixed).
* **Matrix size $r$:** the bond grows as $r^q$, reducible to $\binom{r+q-1}{q}$ on the symmetric subspace for the spectrum. Real $q$ needs the projective operator on an $(r-1)$-simplex, which suffers a curse in $r$. Remedies: TT-compress that operator, or use population dynamics.
* **Moment order $q$:** Bell numbers are super-exponential. Exchangeable contractions reduce to integer partitions $p(q)\sim e^{\pi\sqrt{2q/3}}$, and $\tau(q)$ itself needs only the one-block sector.
* **Not addressed:** single samples. 3D turbulence snapshots have growing MPS ranks (Gourianov et al. 2022), and cascade samples are full rank. The framework scales at the level of the *law*, not of realisations.
