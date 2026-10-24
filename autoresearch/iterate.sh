#!/usr/bin/env bash
# usage: autoresearch/iterate.sh "<description>"   (after editing source files)
# commits the working tree, runs the harness and the regression gate, prints a summary line
set -u
P=/workspace/projects/TenDeFlAn
DESC="$1"
cd $P && git add -A fathon/fathon/tensor experiments/sims && git commit -q -m "experiment: $DESC

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || { echo "nothing to commit"; }
C=$(git rev-parse --short HEAD)
OUT=$(cd $P/fathon && CUDA_VISIBLE_DEVICES=${AR_GPU:-0} timeout 2400 $P/.venv/bin/python $P/benchmark.py 2>$P/autoresearch/last_cases.txt)
RC=$?
METRIC=$(echo "$OUT" | awk '/^metric:/{print $2}')
LAT=$(echo "$OUT" | awk '/^avg_latency_ms:/{print $2}')
TESTS=$(cd $P/fathon && $P/.venv/bin/python -m pytest tests -q -x 2>&1 | tail -1)
PASSED=$(echo "$TESTS" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
echo "commit=$C rc=$RC metric=${METRIC:-crash} latency_ms=${LAT:-} tests_passed=${PASSED:-0} (gate $(cat $P/autoresearch/.regression-baseline))"
