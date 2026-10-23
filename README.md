# TenDeFlAn
Tensorized methods for multifractal fluctuation analysis for billions of agents

Detecting collusion between AI agents from their observable actions. This started at the Apart AI
Collusion Research Sprint (Oct 2026), Markets and Collusion track. **Results and next steps:
[`FINDINGS.md`](FINDINGS.md).**

## Layout

```
FINDINGS.md                        results summary and next steps (start here)
fathon/                            fork of stfbnc/fathon (full history; our work = the tensor-ext commits)
  fathon/tensor/                   pure-numpy extension: detrend, mvdfa, decomp, factorization, replica, ews, influence
  tests/test_tensor.py             44 tests (70 incl. upstream)
experiments/
  run_e0_validation.py             estimator checks on processes with known answers
  run_e1_calvano.py                Calvano Q-learning duopoly (low priority now)
  run_e2_coupled.py                synthetic coupled agents, XOR coalitions
  run_e3_llm_pricing.py, run_e3_power.py   LLM pricing data (Fish et al.)
  run_e4_double_auction.py         Agrawal double auction, shipped data
  run_e5_rps.py                    Bonjour et al. replication, 3-player RPS
  run_e6_four_agent.py             4-agent RPS, one seeded pair, common-state confound
  run_e6_llm_judge.py              LLM judge on the E6 transcripts (needs API key)
  run_e7_da_seeded.py              seeded-pair double auction analyzer (target experiment)
  run_e7_llm_judge.py              LLM judge on public auction logs (needs API key)
  make_figures.py, figures/        report figures from saved results
  sims/                            calvano_q.py, coupled_agents.py, rps.py
  results/e0..e6/                  every reported number (JSON / CSV)
patches/                           seeded-pair harness patch for poseidon-research/collusion_hackathon @ d585a7b
multifractal_tensor_networks/      companion theory note (replica tensor trains for cascades) + code
notes/source_materials.md          summaries of the source materials (Abstractions deck, EWS course)
transfer-exclude.txt               rsync excludes if copying a working tree between machines
```

`external/` is not in this repository. E3, E4 and E7 need the sprint starter repo. Clone it into
`external/` and apply the patch:

```bash
git clone https://github.com/poseidon-research/collusion_hackathon external/collusion_hackathon
git -C external/collusion_hackathon checkout -b seeded-pair-channel d585a7b
git -C external/collusion_hackathon am ../../patches/collusion_hackathon-seeded-pair-channel.patch
(cd external/collusion_hackathon/llm-agent-collusion && uv sync)       # harness environment (E7)
```

`git am` needs a git identity, set via `user.name` / `user.email`.

## Setup (Linux)

```bash
sudo apt-get install -y libgsl-dev build-essential          # fathon's C extensions need GSL + OpenMP
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r experiments/requirements.txt cython
pip install -e ./fathon                                      # builds the C modules; installs fathon.tensor
pip install -e external/collusion_hackathon/util             # poseidon_compute: model routing for judges
python -m pytest fathon/tests -q                             # expect 70 passed
```

`fathon.tensor` is pure numpy. If the C build fails, it still imports, because the compiled modules
are optional imports. The parity tests against the C code will then fail.

### API keys (judge arms and E7 only)

Copy `external/collusion_hackathon/compute.env.example` to `compute.env`, then set either:
- `OPENROUTER_API_KEY`, for closed models; the harness's default models are closed;
- or `NEBIUS_API_KEY`, for open models.

Leave `LLM_MODEL` unset: it would collapse per-seat model choices.

## Reproduce

All commands run from `experiments/`. Everything is free (no API calls) unless marked.

| Experiment | Command | Output |
|---|---|---|
| E0 | `python run_e0_validation.py` | `results/e0/summary.json` |
| E1 | `python run_e1_calvano.py --seeds 20 --procs 8` (raw runs are regenerated, not stored) | `results/e1/` |
| E2 | `python run_e2_coupled.py` | `results/e2/` |
| E3 | `python run_e3_llm_pricing.py && python run_e3_power.py` | `results/e3/` |
| E4 | `python run_e4_double_auction.py` | `results/e4/` |
| E5 | `python run_e5_rps.py` | `results/e5/` |
| E6 | `python run_e6_four_agent.py` (also writes the judge transcripts) | `results/e6/` |
| E6 judge (**API**) | `python run_e6_llm_judge.py --model <id>` (dry run), then add `--run` | `results/e6/judge_<model>.json` |
| E7 runs (**API**) | in `external/collusion_hackathon/llm-agent-collusion`: `uv run python src/continuous_double_auction/simulation.py --sellers 4 --buyer_models zip --rounds 60 --seller_channel_pair 0 1 --seeds 0 1 ... --log_dir <dir>`, plus controls without `--seller_channel_pair` | run folders |
| E7 analysis | `python run_e7_da_seeded.py <dir> [<dir> ...]` | `results/e7/` |
| E7 judge (**API**) | `python run_e7_llm_judge.py --model <id> [--run] <dir>` | `results/e7/judge_<model>.json` |
| Figures | `python make_figures.py` | `figures/` |

## License

| Path | License |
|---|---|
| `fathon/` (fork of [stfbnc/fathon](https://github.com/stfbnc/fathon)), including `fathon/fathon/tensor/` | GPLv3: see `fathon/LICENSE` and the file headers |
| `patches/` | modifies the sprint starter repo; that repo's own license applies |
| everything else | MIT: see `LICENSE` |
