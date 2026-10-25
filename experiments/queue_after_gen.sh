#!/bin/bash
# After the resumed Qwen3 generation and the first judge pass: judge the remaining episodes
# incrementally, then start the TeamOps audit chain and the paused GPU chain.
cd "$(dirname "$0")/.." || exit 1
exec 9>runs/queue_after_gen.lock; flock -n 9 || exit 0
P=.venv/bin/python
while pgrep -f "^[.]venv/bin/python experiments/run_teamops_gen.py --model qwen3" >/dev/null; do sleep 60; done
until grep -q "teamops chain done for qwen3-8b" runs/queue_teamops_qwen3-8b.log 2>/dev/null; do sleep 60; done
echo "$(date) incremental judge qwen3-8b"; $P experiments/run_teamops_judge.py --model qwen3-8b --tag L --concurrency 96
rm -f runs/queue_teamops_audit.lock runs/queue_gpu.lock
(setsid nohup experiments/queue_teamops_audit.sh >> runs/queue_teamops_audit.log 2>&1 &)
(setsid nohup experiments/queue_gpu.sh >> runs/queue_gpu.log 2>&1 &)
echo "$(date) audit and gpu chains launched"
