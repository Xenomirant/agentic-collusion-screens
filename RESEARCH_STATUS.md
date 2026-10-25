# Research status: evidence-efficient multi-agent collusion detection

*Last updated: 2026-10-25 (Phase 2 in progress)*

## 1. Recovery of prior work (Phase 1)

The brief describes an earlier pilot: 192 synthetic two-agent trajectories, cross-agent evidence selection with AUROC 0.823 at a 20% budget, a link-shuffle ablation, and so on. **None of it could be found.** Searched:

- every project under `/workspace/projects` (TenDeFlAn, AdaClip, CaONext, DFT, HyperscaleES, context-language-models), including all git branches and remotes;
- the collusion-sprint starter repo `TenDeFlAn/external/collusion_hackathon` (17 sub-projects: trace-guard, colosseum, steganographic-collusion, ...);
- `/workspace/data`, `/workspace/logs`, `/workspace/scripts`;
- all 45 GitHub repositories of the account.

The nearest existing work is TenDeFlAn, a market-data collusion detector (tensorized DFA/DCCA) with a different observation model. Nothing from it is reused here except the local vLLM environment.

**Decision (user, 2026-10-25):** rebuild from scratch in this new repository. The pilot numbers are treated as an independent prior observation to compare against, not as a baseline this code reproduces. The "existing cross-agent selector" (`caes`) is a **reconstruction from the textual description** and is labelled that way everywhere.

## 2. Environment

| Resource | State |
|---|---|
| GPUs | 4× H100 80 GB, shared with other sessions' jobs (GRPO on GPU 0/1, a vLLM Qwen3-8B on GPU 3). This project uses **GPU 2** (vLLM Qwen2.5-7B-Instruct, port 8101, 50% memory). |
| Local models | Qwen2.5-7B-Instruct, Qwen3-8B, Olmo-3-7B-Instruct-SFT, Qwen2.5-3B (HF cache `/workspace/caches/hf`). No paid APIs are used. |
| Python | `.venv` (3.13, numpy 2.5, sklearn 1.9); vLLM 0.32 from `TenDeFlAn/.venv-vllm`. |
| Disk | 74 TB free. |

## 3. What exists now

- `certaudit/world.py`: episode data model (public metadata vs hidden text and facts).
- `certaudit/verifier.py`: independent three-valued verifier V(E, R) with closed-world waiver exclusion, plus the exact minimum certificate. It is cross-checked against brute-force subset search, and monotonicity and minimality are tested.
- `certaudit/generator.py`, `textgen.py`: mechanisms A (sequential), B (joint action), C (delayed reciprocity), D (coordinated false report); negative controls benign_coop / individual / misled / shared_error / authorized / unexecuted / null; decoy interactions; MVP2 difficulty knobs; three disjoint text-template families used as split keys. Token cost is drawn independently of content (tested).
- `certaudit/oracle.py`: the disclosure boundary. Every reveal is logged and charged; ablations transform the view.
- `certaudit/belief.py`: metadata-only record priors (an MLP trained with random reveal masks on the train split) combined through the public rule's AND-OR structure.
- `certaudit/selectors.py`: random, random connected groups, graph traversal, caes (reconstruction), and the 2×2 factorial suspicion (G0R0) / voi (G0R1) / comp (G1R0) / combined (G1R1).
- `certaudit/llm.py`: LLM extraction reader (Extract-and-Evaluate style), active LLM auditor, full-information LLM monitor.
- `RELATED_WORK.md`, `notes/benchmark_availability.md`: verified literature and dataset availability.

## 4. Plan and status

| Phase | Content | Status |
|---|---|---|
| 1 | Recovery, environment, leakage controls | done (pilot not found; `tests/test_leakage.py` perturbs unrevealed content and checks that every selector's request sequence is unchanged) |
| 2 | MVP1 complementarity, A/B/C, 8 selectors, budgets 5/10/20/50/100%, ablations | running (`experiments/run_mvp1.py --ablations`) |
| 2b | MVP1 with LLM reading (Qwen2.5-7B done: 54k records; Qwen3-8B running), active LLM auditor + full monitor (Qwen2.5-7B running, Qwen3-8B smoke) | running |
| 2c | Held-out mechanism D (coordinated false reports), priors never trained on it | split defined (`mvp1_test_D`), todo |
| 3 | MVP2 search complexity at fixed C_cert* (`generate_controlled`: C_cert* verified invariant under length 60–480, 0–8 decoys, gap, agents; only metadata quality changes it) | driver written, waiting for CPU |
| 4 | MVP3 matched alternative explanations (same skeleton metadata under 7 conditions), abstention, risk-coverage, calibration | running (`experiments/run_mvp3.py`) |
| 5 | MVP4 real LLM trajectories: SALT-NLP/agent-collusion mapped onto the verifier (verifier label == dataset label on 738/738 sampled episodes; 8 held-out model families) | running (`experiments/run_salt.py`); LLM-reader pass next |
| 5b | Richer real setting (3+ agents with actions and objects): Colosseum/Terrarium or own local Qwen multi-agent runs | todo, decide after SALT results |
| 6 | Analysis, RESULTS.md, hypothesis table | todo |

**Emphasis (user, 2026-10-25):** synthetic gains are provisional; the selectors are built around the same rule structure that generates the synthetic data, so the evidence that counts is (i) the LLM-reading condition, (ii) held-out mechanisms/templates, and (iii) real trajectories (MVP4). **Later the same day:** two-agent data (SALT) is not enough; the user wants large agentic-scale experiments. MVP4 therefore centres on **TeamOps-L**: 8–12 LLM agents per episode, several concurrent private pairs (at most one collusive, the others live alternative explanations), 10 rounds, ~600 episodes per agent model (Qwen2.5-7B and Qwen3-8B), judged by Qwen3-8B with thinking, audited across models. SALT stays as a two-agent sanity check with the caveat that its certificates are small.

**Container restarts.** The container is recreated roughly daily (all processes die, `/tmp` is wiped). `experiments/relaunch.sh` is the single idempotent entry point: it starts the two vLLM servers if needed and (re)starts the CPU, GPU and TeamOps chains, all of which skip finished stages through cached outputs; generation is chunk-checkpointed. The raw SALT download lived in `/tmp` and is gone; the derived splits (`cache/splits/salt.pkl`) and manifests survive, so SALT audits rerun without it (re-download via the HF API through the proxy only if the splits must be rebuilt). The first restart (12:50 UTC) killed MVP3 mid-prior, the TeamOps judge, the Qwen3 4-agent generation and the SALT thinking extraction; all are relaunched.

## 5. Known limitations so far

- The synthetic world is rule-structured and the auditor knows the rules. The proposed selectors exploit that structure; whether this transfers is exactly what H4/MVP4 must test.
- With perfect reading, V is exact, so 100%-budget accuracy is perfect by construction. Detection-quality claims therefore rest on the budgeted regime and on the LLM-reading condition.
- Priors are trained on the generator's distribution. Held-out mechanisms and templates are used to test robustness.
