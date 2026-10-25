#!/bin/bash
# TeamOps-at-scale chain for one agent model: generation (resumable) then the judge pass.
# Usage: experiments/queue_teamops.sh <model> [n_episodes]
model=${1:?model}; n=${2:-600}
cd "$(dirname "$0")/.." || exit 1
exec 9>"runs/queue_teamops_${model//./}.lock"; flock -n 9 || exit 0
P=.venv/bin/python
echo "$(date) gen $model";   $P experiments/run_teamops_gen.py --model "$model" --tag L --n-episodes "$n" --agents 8 12 --rounds 10 --concurrency 128
[ -f "cache/splits/teamops_L_${model}_judged.pkl" ] || { echo "$(date) judge $model"; $P experiments/run_teamops_judge.py --model "$model" --tag L --concurrency 96; }
echo "$(date) teamops chain done for $model"
