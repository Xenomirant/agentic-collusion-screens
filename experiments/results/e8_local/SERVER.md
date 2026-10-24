# Local vLLM server for E8

Model: `Qwen/Qwen2.5-7B-Instruct` (weights from the HF cache, `HF_HOME=/workspace/caches/hf`).
Port: **8011** (OpenAI-compatible API at `http://127.0.0.1:8011/v1`). GPU 2 only.
The server is left running after the E8 runs; anyone can use it.

Exact launch command (run from this directory; the log is `server.log`):

```bash
cd /workspace/projects/TenDeFlAn/experiments/results/e8_local
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 nohup /workspace/projects/TenDeFlAn/.venv-vllm/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8011 \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 256 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  > server.log 2>&1 &
```

Readiness check: `curl -s localhost:8011/v1/models` lists `Qwen/Qwen2.5-7B-Instruct`.
Startup takes about 5-6 minutes on this machine (CUDA-graph capture dominates).

Why these settings
- `HF_HUB_OFFLINE=1`: the weights and tokenizer are already in the HF cache. Without it vLLM lists the
  repo files on the Hub at startup; on 2026-10-24 the machine's HTTP proxy answered `503 Forwarding
  failure` and the server died before loading (log kept as `server_failed_proxy_20261024.log`).
  Offline mode skips that lookup. Nothing is downloaded either way.
- `--max-model-len 16384`: the seller prompt grows with the memory manager (one line per hour, 60
  hours), the last-5-hours order book, up to 50 past trades and a 250-word scratchpad; by hour 60 a
  prompt is ~3-4k tokens, and the harness asks for `max_tokens=4000`. vLLM rejects a request whose
  prompt + max_tokens exceeds the model length (a 400 the harness turns into a crashed run), so 8192
  would be unsafe; 16384 leaves headroom. KV cache at this length: 1,025,552 tokens, i.e. ~62
  concurrent 16k requests, far more than the ~60-100 concurrent requests the batch launcher creates.
- `--served-model-name Qwen/Qwen2.5-7B-Instruct`: the id the sellers send must match exactly
  (`--seller_models Qwen/Qwen2.5-7B-Instruct`); see MANIFEST.md for how the harness routes.
- `--max-num-seqs 256`: allows 16-24 markets x 4 sellers in flight at once.

How the harness reaches it: `external/collusion_hackathon/compute.env` contains only

```
LLM_BASE_URL=http://127.0.0.1:8011/v1
LLM_API_KEY=EMPTY
```

`poseidon_compute.llm_config` treats `LLM_BASE_URL` as a "custom" endpoint, uses `LLM_API_KEY`
as the bearer (placeholder; vLLM runs without auth) and never touches Nebius/OpenRouter. No
provider key is needed or set anywhere.

Session log: the server first ran on 2026-10-23 22:46 (log kept as `server_session1_20261023.log`);
the Claude Code session restart that night killed it, and it was relaunched on 2026-10-24 05:00 with the
identical command plus `HF_HUB_OFFLINE=1` (`server.log`).
