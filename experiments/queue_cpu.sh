#!/bin/bash
# CPU chain, strictly sequential (16-CPU cgroup): each stage waits for its inputs.
exec 9>runs/queue_cpu.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
P=.venv/bin/python
# CPU-heavy / secondary work runs after the many-agent audit chain (vLLM needs the 16-CPU quota while judging)
[ "$SKIP_GUARD" = 1 ] || until grep -q "teamops audit chain done" runs/queue_teamops_audit.log 2>/dev/null; do sleep 120; done
wait_file() { until [ -f "$1" ]; do sleep 30; done; }
while pgrep -f "^[.]venv/bin/python experiments/run_mvp1" >/dev/null; do sleep 30; done
echo "$(date) mvp1 perfect";       $P experiments/run_mvp1.py --ablations --procs 16
echo "$(date) mvp3 perfect";       $P experiments/run_mvp3.py --procs 16 > runs/mvp3_perfect.log 2>&1
echo "$(date) mvp1 qwen25 reader"; $P experiments/run_mvp1.py --reader qwen25_mvp1 --ablations --procs 16
echo "$(date) mvp1 held-out D perfect"; $P experiments/run_mvp1.py --test mvp1_test_D --cal mvp1_cal_D --procs 16
echo "$(date) mvp2";               $P experiments/run_mvp2.py --procs 16 > runs/mvp2.log 2>&1
wait_file results/extraction/qwen3_mvp1.json
echo "$(date) mvp1 qwen3 reader";  $P experiments/run_mvp1.py --reader qwen3_mvp1 --ablations --procs 16
wait_file results/extraction/qwen3_mvp1D.json
echo "$(date) mvp1 D qwen3 reader"; $P experiments/run_mvp1.py --reader qwen3_mvp1D --test mvp1_test_D --cal mvp1_cal_D --procs 16
wait_file results/extraction/qwen3_mvp3.json
echo "$(date) mvp3 qwen3 reader";  $P experiments/run_mvp3.py --reader qwen3_mvp3 --procs 16
echo "$(date) cpu chain done"
