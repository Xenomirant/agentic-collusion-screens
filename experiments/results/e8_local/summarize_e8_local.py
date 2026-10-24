#!/usr/bin/env python3
"""Per-run and aggregate statistics for the E8 datasets (feeds MANIFEST.md).

    HF_HUB_OFFLINE=1 /workspace/projects/TenDeFlAn/.venv-vllm/bin/python summarize_e8_local.py

Writes summary.json and per_run.csv next to this file. Token counts re-tokenize the logged prompts
(through the Qwen chat template, as vLLM applies it) and the logged JSON responses with the
Qwen2.5 tokenizer; they cover the calls that produced a logged response (retries after a JSON decode
failure are not logged by the harness, so the vLLM counters in SERVER/MANIFEST are the upper bound).
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from load_e8_local import load_e8_local, _BLOCK  # noqa: E402

PROMPT = re.compile(r"## Prompt: Round (\d+) - [^\n]*\n``````\n(.*?)\n``````\n", re.S)
RESPONSE = re.compile(r"## Response: Round (\d+) - [^\n]*\n````json\n(.*?)\n````\n", re.S)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def n_prompt_tokens(text):
        rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           tokenize=False)
        return len(tok(rendered, add_special_tokens=False)["input_ids"])

    runs_meta = {}
    if (HERE / "runs.jsonl").exists():
        for line in open(HERE / "runs.jsonl"):
            r = json.loads(line)
            runs_meta[r["tag"]] = r
    for f in (HERE / "logs").glob("timing_*.log"):  # the single timed run, launched by hand
        m = re.search(r"WALL=(\d+)s", f.read_text())
        if m:
            runs_meta.setdefault(f.stem[len("timing_"):], {})["wall_s"] = int(m.group(1))
    data = load_e8_local(str(HERE))
    rows = []
    for dataset, runs in data.items():
        for r in runs:
            d = HERE / dataset / r["run"]
            tag = r["run"].split("_", 2)[2]
            calls = p_tok = c_tok = 0
            lens = []
            for s in r["sellers"]:
                txt = open(d / f"{s}.md", encoding="utf-8").read()
                prompts = PROMPT.findall(txt)
                resps = RESPONSE.findall(txt)
                calls += len(resps)
                p_tok += sum(n_prompt_tokens(p) for _, p in prompts)
                c_tok += sum(len(tok(j)["input_ids"]) for _, j in resps)
                lens.append([n_prompt_tokens(p) for _, p in prompts])
            last_prompt = max(l[-1] for l in lens if l)
            wall = runs_meta.get(tag, {}).get("wall_s")
            asks = r["raw_asks"]
            rows.append(dict(
                dataset=dataset, run=r["run"], seed=r["seed"], pair=r["true"], rounds=r["rounds"],
                logged_calls=calls, prompt_tokens=p_tok, completion_tokens=c_tok,
                max_prompt_tokens=int(np.max([max(l) for l in lens])), last_round_prompt_tokens=int(last_prompt),
                wall_s=wall, trade_rounds=int(r["trade"].sum()), n_trades=int(r["n_trades"].sum()),
                missing_asks=int(np.isnan(asks).sum()), mean_ask=float(np.nanmean(asks)),
                mean_ask_last20=float(np.nanmean(asks[:, -20:])),
                min_ask=float(np.nanmin(asks)), max_ask=float(np.nanmax(asks)),
            ))
    with open(HERE / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for dataset in ("controls", "seeded", "seeded_monitored"):
        rs = [x for x in rows if x["dataset"] == dataset]
        if not rs:
            summary[dataset] = {"n_runs": 0}
            continue
        walls = [x["wall_s"] for x in rs if x["wall_s"]]
        summary[dataset] = dict(
            n_runs=len(rs), seeds=sorted(x["seed"] for x in rs),
            pairs={str(k): sum(1 for x in rs if x["pair"] == k) for k in sorted({x["pair"] for x in rs if x["pair"]}, key=str)},
            rounds=sorted({x["rounds"] for x in rs}),
            calls_per_run=float(np.mean([x["logged_calls"] for x in rs])),
            prompt_tokens_per_run=float(np.mean([x["prompt_tokens"] for x in rs])),
            completion_tokens_per_run=float(np.mean([x["completion_tokens"] for x in rs])),
            max_prompt_tokens=max(x["max_prompt_tokens"] for x in rs),
            wall_s_mean=float(np.mean(walls)) if walls else None,
            wall_s_min=float(np.min(walls)) if walls else None,
            wall_s_max=float(np.max(walls)) if walls else None,
            trade_rounds_mean=float(np.mean([x["trade_rounds"] for x in rs])),
            missing_asks_total=int(sum(x["missing_asks"] for x in rs)),
            mean_ask=float(np.mean([x["mean_ask"] for x in rs])),
            mean_ask_last20=float(np.mean([x["mean_ask_last20"] for x in rs])),
        )
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
