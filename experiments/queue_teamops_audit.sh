#!/bin/bash
# TeamOps-L audits after both judge passes: cross-model transfer, held-out team size, readers, LLM baselines.
cd "$(dirname "$0")/.." || exit 1
exec 9>runs/queue_teamops_audit.lock; flock -n 9 || exit 0
P=.venv/bin/python
wait_file() { until [ -f "$1" ]; do sleep 60; done; }
wait_file cache/splits/teamops_L_qwen2.5-7b_judged.pkl; wait_file cache/splits/teamops_L_qwen3-8b_judged.pkl
run() { [ -f "results/teamops/$1/metrics.csv" ] || { echo "$(date) $1"; shift; "$@"; }; }
run L_audit_qwen2.5-7b_to_qwen3-8b/judge     $P experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen3-8b --reader judge --procs 16
run L_audit_qwen3-8b_to_qwen2.5-7b/judge     $P experiments/run_teamops_audit.py --train-model qwen3-8b --test-model qwen2.5-7b --reader judge --procs 16
run L_audit_qwen2.5-7b_to_qwen3-8b/qwen3-8b  $P experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen3-8b --reader qwen3-8b --procs 16
run L_audit_qwen3-8b_to_qwen2.5-7b/qwen3-8b  $P experiments/run_teamops_audit.py --train-model qwen3-8b --test-model qwen2.5-7b --reader qwen3-8b --procs 16
run L_audit_qwen2.5-7b_to_qwen2.5-7b_n8to12/judge $P experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen2.5-7b --train-agents 8 --test-agents 12 --reader judge --procs 16
run L_audit_qwen3-8b_to_qwen3-8b_n8to12/judge $P experiments/run_teamops_audit.py --train-model qwen3-8b --test-model qwen3-8b --train-agents 8 --test-agents 12 --reader judge --procs 16
[ -f results/teamops/llm_qwen3-8b/teamops_monitor.jsonl.gz ] || { echo "$(date) llm baselines on TeamOps-L (qwen3 agents)"; $P experiments/run_llm_baselines.py --model qwen3-8b --pickle cache/splits/teamops_L_qwen3-8b_judged.pkl --splits teamops --budgets 0.2 0.5 --limit 240 --concurrency 64 --out teamops/llm_qwen3-8b; }
echo "$(date) teamops audit chain done"
