#!/bin/bash
# CPU chain: runs after the perfect-reader MVP1 finishes; each stage waits for its inputs.
exec 9>runs/queue_cpu.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
P=.venv/bin/python
wait_file() { until [ -f "$1" ]; do sleep 30; done; }
while pgrep -f "^[.]venv/bin/python experiments/run_mvp1" >/dev/null; do sleep 30; done
echo "$(date) mvp1 qwen25 reader" ; $P experiments/run_mvp1.py --reader qwen25_mvp1 --ablations --procs 16
echo "$(date) mvp1 held-out D perfect"; $P experiments/run_mvp1.py --test mvp1_test_D --cal mvp1_cal_D --procs 16
wait_file cache/models/q_mvp2.pkl
while pgrep -f "^[.]venv/bin/python experiments/run_mvp3" >/dev/null; do sleep 30; done
echo "$(date) mvp2"; $P experiments/run_mvp2.py --procs 16
wait_file cache/extract/qwen3_mvp1.pkl; wait_file results/extraction/qwen3_mvp1.json
echo "$(date) mvp1 qwen3 reader"; $P experiments/run_mvp1.py --reader qwen3_mvp1 --ablations --procs 16
wait_file results/extraction/qwen3_mvp1D.json
echo "$(date) mvp1 D qwen3 reader"; $P experiments/run_mvp1.py --reader qwen3_mvp1D --test mvp1_test_D --cal mvp1_cal_D --procs 16
wait_file results/extraction/qwen3_mvp3.json
echo "$(date) mvp3 qwen3 reader"; $P experiments/run_mvp3.py --reader qwen3_mvp3 --procs 16
echo "$(date) cpu chain done"
