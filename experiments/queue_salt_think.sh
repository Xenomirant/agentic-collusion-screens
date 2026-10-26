#!/bin/bash
# SALT with the thinking-mode Qwen3 reader: extraction (cached), then as first reader, then as confirmation reader.
cd "$(dirname "$0")/.." || exit 1
exec 9>runs/queue_salt_think.lock; flock -n 9 || exit 0
P=.venv/bin/python; S=/nonexistent
# CPU-heavy / secondary work runs after the many-agent audit chain (vLLM needs the 16-CPU quota while judging)
[ "$SKIP_GUARD" = 1 ] || until grep -q "teamops audit chain done" runs/queue_teamops_audit.log 2>/dev/null; do sleep 120; done
   # raw data not needed: splits are cached in cache/splits/salt.pkl
[ -f results/salt/perfect/metrics.csv ] || { echo "$(date) salt perfect"; $P experiments/run_salt.py --root $S --reader perfect --procs 8; }
[ -f results/salt/qwen3-8b/metrics.csv ] || { echo "$(date) salt qwen3 reader"; $P experiments/run_salt.py --root $S --reader qwen3-8b --procs 8; }
[ -f "results/salt/qwen3-8b+confirm_perfect/metrics.csv" ] || { echo "$(date) salt confirm perfect"; $P experiments/run_salt.py --root $S --reader qwen3-8b --confirm perfect --procs 8; }
[ -f cache/extract/salt_qwen3-8b-think.pkl ] || { echo "$(date) salt thinking extraction"; $P experiments/extract_salt_thinking.py; }
[ -f "results/salt/qwen3-8b-think/metrics.csv" ] || { echo "$(date) salt think reader"; $P experiments/run_salt.py --root $S --reader qwen3-8b-think --procs 4; }
[ -f "results/salt/qwen3-8b+confirm_salt_qwen3-8b-think/metrics.csv" ] || { echo "$(date) salt confirm think"; $P experiments/run_salt.py --root $S --reader qwen3-8b --confirm salt_qwen3-8b-think --procs 4; }
echo "$(date) salt think chain done"
