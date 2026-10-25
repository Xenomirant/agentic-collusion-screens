#!/bin/bash
# GPU chain (vLLM servers on :8101 qwen2.5-7b, :8102 qwen3-8b); each stage waits for its inputs.
exec 9>runs/queue_gpu.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
P=.venv/bin/python
S=/tmp/claude-0/-workspace-projects-TenDeFlAn/6de85ac9-150b-4f09-b2d8-4d650af188dd/scratchpad/salt_data
wait_file() { until [ -f "$1" ]; do sleep 30; done; }
wait_file results/extraction/qwen3_mvp1.json
echo "$(date) extract D qwen3"; $P experiments/extract.py --splits mvp1_cal_D mvp1_test_D --model qwen3-8b --name qwen3_mvp1D
wait_file results/salt/perfect/metrics.csv
echo "$(date) salt qwen3 reader"; $P experiments/run_salt.py --root $S --reader qwen3-8b --procs 8
# synthetic GPU stages run after the many-agent audits (the user's priority) have finished
until grep -q "teamops audit chain done" runs/queue_teamops_audit.log 2>/dev/null; do sleep 120; done
wait_file cache/splits/mvp3_test.pkl; wait_file cache/splits/mvp3_cal.pkl
echo "$(date) extract mvp3 qwen3"; $P experiments/extract.py --splits mvp3_cal mvp3_test --model qwen3-8b --name qwen3_mvp3
echo "$(date) extract D qwen2.5"; $P experiments/extract.py --splits mvp1_cal_D mvp1_test_D --model qwen2.5-7b --name qwen25_mvp1D
echo "$(date) llm baselines qwen3 (test, 0.2/0.5)"; $P experiments/run_llm_baselines.py --model qwen3-8b --splits mvp1_test --budgets 0.2 0.5 --concurrency 96
echo "$(date) gpu chain done"
