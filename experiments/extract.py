"""Run LLM extraction over splits and cache {(tid, eid): Fact} for the CachedReader.

    python experiments/extract.py --splits mvp1_cal mvp1_test --model qwen2.5-7b --name qwen25_mvp1
"""
import argparse
import asyncio
import json
import pickle
import time

from common import CACHE, RESULTS, get_split
from certaudit.llm import extract_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = CACHE / "extract" / f"{a.name}.pkl"
    table = pickle.load(open(out, "rb")) if out.exists() else {}
    usage_all = []
    for s in a.splits:
        trajs = [t for t in get_split(s) if (t.tid, t.events[0].eid) not in table]
        t0 = time.time()
        tab, usage = asyncio.run(extract_all(trajs, model=a.model))
        table.update(tab)
        usage_all += usage
        print(s, len(tab), f"{time.time() - t0:.0f}s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(table, open(out, "wb"))
    # field accuracy against the hidden truth (offline evaluation only)
    acc = {}
    for s in a.splits:
        for t in get_split(s):
            for e in t.events:
                g, f = table[(t.tid, e.eid)], e.fact
                nv = lambda x: "NONVALID" if x.kind == "DENY" or (x.kind == "GRANT" and not x.valid) else x.kind
                ok = nv(g) == nv(f) and (f.kind in ("CHAT", "SOURCE", "DENY") or set(g.ops) == set(f.ops)) \
                    and (f.aware is None or g.aware == f.aware)
                d = acc.setdefault(f.kind, [0, 0])
                d[0] += ok
                d[1] += 1
    rep = dict(model=a.model, n_records=len(table), accuracy={k: v[0] / v[1] for k, v in acc.items()},
               counts={k: v[1] for k, v in acc.items()},
               mean_prompt_tokens=sum(p for p, _ in usage_all) / max(1, len(usage_all)),
               mean_completion_tokens=sum(c for _, c in usage_all) / max(1, len(usage_all)))
    (RESULTS / "extraction").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(RESULTS / "extraction" / f"{a.name}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
