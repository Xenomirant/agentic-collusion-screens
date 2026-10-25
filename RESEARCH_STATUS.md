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
| 1 | Recovery, environment, leakage controls | done (pilot not found; leakage test in `tests/`) |
| 2 | MVP1 complementarity, A/B/C, 8 selectors, budgets 5/10/20/50/100%, ablations | running |
| 2b | MVP1 with LLM reading, active LLM auditor, full-information LLM monitor | extraction running |
| 3 | MVP2 search complexity at fixed C_cert* | todo |
| 4 | MVP3 matched alternative-explanation pairs, abstention, risk-coverage | todo |
| 5 | MVP4 real LLM trajectories (SALT-NLP/agent-collusion and/or local multi-agent runs) | todo |
| 6 | Analysis, RESULTS.md, hypothesis table | todo |

## 5. Known limitations so far

- The synthetic world is rule-structured and the auditor knows the rules. The proposed selectors exploit that structure; whether this transfers is exactly what H4/MVP4 must test.
- With perfect reading, V is exact, so 100%-budget accuracy is perfect by construction. Detection-quality claims therefore rest on the budgeted regime and on the LLM-reading condition.
- Priors are trained on the generator's distribution. Held-out mechanisms and templates are used to test robustness.
