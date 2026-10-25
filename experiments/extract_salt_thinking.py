"""Second, stronger reading of SALT messages (Qwen3-8B with thinking), cached for the
certificate-confirmation step.  python experiments/extract_salt_thinking.py"""
import asyncio, json, pickle, time
import numpy as np
from common import CACHE, RESULTS
from certaudit.salt import extract_salt

splits = pickle.load(open(CACHE / "splits" / "salt.pkl", "rb"))
trajs = splits["cal"] + splits["test"]
t0 = time.time()
tab, usage = asyncio.run(extract_salt(trajs, model="qwen3-8b", concurrency=96, thinking=True))
pickle.dump(tab, open(CACHE / "extract" / "salt_qwen3-8b-think.pkl", "wb"))
acc = {}
for t in trajs:
    for e in t.events:
        if e.pub.etype == "msg":
            d = acc.setdefault(e.fact.kind, [0, 0]); d[0] += tab[(t.tid, e.eid)].kind == e.fact.kind; d[1] += 1
rep = dict(model="qwen3-8b-think", seconds=time.time() - t0, n=len(usage), accuracy={k: v[0] / v[1] for k, v in acc.items()},
           counts={k: v[1] for k, v in acc.items()}, mean_completion_tokens=float(np.mean([c for _, c in usage])))
(RESULTS / "salt").mkdir(exist_ok=True, parents=True)
json.dump(rep, open(RESULTS / "salt" / "extraction_qwen3-8b-think.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
