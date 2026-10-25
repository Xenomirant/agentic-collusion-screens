# CertAudit — evidence-efficient multi-agent collusion detection

Restricted-disclosure auditing: an auditor sees public metadata of a multi-agent
log (who messaged whom, who acted on which object, when, and how long each record is)
and pays tokens to reveal hidden records. Questions: how much disclosure is needed
to establish a coordinated violation with an independently checkable certificate,
and which selection strategies find such certificates cheaply without raising false
accusations.

See `RESEARCH_STATUS.md` (status, plan), `RESULTS.md` (findings), `RELATED_WORK.md`.

## Setup
```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python numpy scipy pandas scikit-learn matplotlib openai pytest
# local LLM (GPU 2): vLLM OpenAI server on :8101
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b --port 8101 --gpu-memory-utilization 0.5
```

## Run
```bash
experiments/relaunch.sh                                   # after a container restart: servers + all chains (idempotent)
.venv/bin/python -m pytest -q tests                      # verifier, leakage, model parity
.venv/bin/python experiments/run_mvp1.py --ablations      # MVP1, perfect reading
.venv/bin/python experiments/extract.py --splits mvp1_cal mvp1_test --name qwen25_mvp1
.venv/bin/python experiments/run_mvp1.py --reader qwen25_mvp1
.venv/bin/python experiments/run_mvp2.py                  # search complexity at fixed C_cert*
.venv/bin/python experiments/run_mvp3.py                  # matched alternative explanations, abstention
# real LLM agents at scale (TeamOps-L): generate (resumable), judge (incremental), audit across models
.venv/bin/python experiments/run_teamops_gen.py --model qwen2.5-7b --tag L --n-episodes 600 --agents 8 12
.venv/bin/python experiments/run_teamops_judge.py --model qwen2.5-7b --tag L
.venv/bin/python experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen3-8b --reader qwen3-8b
.venv/bin/python experiments/run_salt.py --root <salt download> --reader qwen3-8b   # two-agent sanity check
.venv/bin/python experiments/make_figures.py
```
Splits are regenerated deterministically from seeds (`experiments/common.py:SPLITS`); TeamOps and
SALT splits are cached under `cache/splits/`; manifests are in `results/manifests/`. Chains
(`experiments/queue_*.sh`) skip finished stages, so they can be relaunched at any time.
