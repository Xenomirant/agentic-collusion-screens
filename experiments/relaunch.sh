#!/bin/bash
# Idempotent relaunch after a container restart: starts the two vLLM servers if they are not
# serving, waits for them, then (re)starts the CPU and GPU chains, which skip finished stages
# through their cached outputs.  Usage: experiments/relaunch.sh [--no-chains]
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
V=/workspace/projects/TenDeFlAn/.venv-vllm/bin/vllm
export HF_HOME=/workspace/caches/hf

serve() {  # gpu port model served_name extra...
  local gpu=$1 port=$2 model=$3 name=$4; shift 4
  if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then echo "$name already serving on :$port"; return; fi
  pgrep -f "served-model-name $name " >/dev/null && { echo "$name starting"; return; }
  echo "starting $name on GPU $gpu :$port"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $V serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.6 --enable-prefix-caching --max-num-seqs 256 "$@" > "runs/vllm_${name//./}.log" 2>&1 &
}
serve 2 8101 Qwen/Qwen2.5-7B-Instruct qwen2.5-7b --max-model-len 32768
serve 3 8102 Qwen/Qwen3-8B qwen3-8b --max-model-len 40960 --reasoning-parser qwen3

for port in 8101 8102; do
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && echo ":$port ready" || echo ":$port NOT ready"
done

[ "$1" = "--no-chains" ] && exit 0
rm -f runs/queue_cpu.lock runs/queue_gpu.lock
pgrep -f "experiments/queue_cp[u].sh" >/dev/null || (setsid nohup experiments/queue_cpu.sh >> runs/queue_cpu.log 2>&1 &)
pgrep -f "experiments/queue_gp[u].sh" >/dev/null || (setsid nohup experiments/queue_gpu.sh >> runs/queue_gpu.log 2>&1 &)
echo "chains running: $(pgrep -fc 'experiments/queue_[cg]p[u].sh')"
for m in qwen2.5-7b qwen3-8b; do
  pgrep -f "queue_teamops.sh $m" >/dev/null || (setsid nohup experiments/queue_teamops.sh $m 600 >> "runs/queue_teamops_${m//./}.log" 2>&1 &)
done
echo "teamops chains: $(pgrep -fc 'queue_teamops.sh')"
pgrep -f "queue_salt_thin[k].sh" >/dev/null || (setsid nohup experiments/queue_salt_think.sh >> runs/queue_salt_think.log 2>&1 &)
pgrep -f "queue_teamops_audi[t].sh" >/dev/null || (setsid nohup experiments/queue_teamops_audit.sh >> runs/queue_teamops_audit.log 2>&1 &)
