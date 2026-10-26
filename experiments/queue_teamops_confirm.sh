#!/bin/bash
# After the audit chain: realistic-reader audits with oracle certificate confirmation (upper bound).
cd "$(dirname "$0")/.." || exit 1
exec 9>runs/queue_teamops_confirm.lock; flock -n 9 || exit 0
P=.venv/bin/python
[ "$SKIP_GUARD" = 1 ] || until grep -q "teamops audit chain done" runs/queue_teamops_audit.log 2>/dev/null; do sleep 120; done
run() { [ -f "results/teamops/$1/metrics.csv" ] || { echo "$(date) $1"; shift; "$@"; }; }
run L_audit_qwen2.5-7b_to_qwen3-8b/qwen3-8b+confirm_judge $P experiments/run_teamops_audit.py --train-model qwen2.5-7b --test-model qwen3-8b --reader qwen3-8b --confirm judge --procs 16
run L_audit_qwen3-8b_to_qwen2.5-7b/qwen3-8b+confirm_judge $P experiments/run_teamops_audit.py --train-model qwen3-8b --test-model qwen2.5-7b --reader qwen3-8b --confirm judge --procs 16
echo "$(date) teamops confirm chain done"
