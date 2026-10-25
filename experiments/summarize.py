"""Print the headline tables of every finished experiment from results/*.csv (for RESULTS.md).

    python experiments/summarize.py
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

from common import RESULTS
from certaudit.metrics import auroc

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
ORDER = ["random", "random_groups", "graph", "caes", "suspicion", "voi", "comp", "combined"]


def mvp1(path, label):
    f = RESULTS / path / "metrics.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    print(f"\n### {label} ({path}) — AUROC by budget, ablation=none")
    n = d[d.ablation == "none"] if "ablation" in d else d
    print(n.pivot_table(index="selector", columns="budget", values="auroc").reindex(ORDER).dropna(how="all").round(3).to_string())
    print(n[n.budget == 0.2].set_index("selector").reindex(ORDER).dropna(how="all")[
        ["auroc_lo", "auroc_hi", "tpr@5", "fpr@5", "cert_found", "records", "supported_on_neg", "overhead_median"]].round(3).to_string())
    if "ablation" in d and d.ablation.nunique() > 1:
        print("AUROC at 20% by ablation:")
        print(d[d.budget == 0.2].pivot_table(index="selector", columns="ablation", values="auroc").reindex(ORDER).dropna(how="all").round(3).to_string())
    p = RESULTS / path / "paired.csv"
    if p.exists():
        q = pd.read_csv(p)
        q = q[q.ablation == "none"] if "ablation" in q else q
        print("paired (combined minus other) at 10%/20%:")
        print(q[q.budget.isin([0.1, 0.2])].pivot_table(index="other", columns="budget", values=["diff", "lo", "hi"]).round(3).to_string())


def mvp2():
    f = RESULTS / "mvp2" / "metrics.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["setting"] = d.setting.str.replace("=", "")
    print("\n### MVP2 — median overhead C_read/C_cert* and AUROC at 20%")
    for m in ["overhead_median", "auroc@0.2", "disc@0.2"]:
        print(m)
        print(d.pivot_table(index="setting", columns="selector", values=m)[[s for s in ORDER if s in set(d.selector)]].round(2).to_string())


def mvp3(reader):
    f = RESULTS / "mvp3" / reader / "operating.csv"
    if not f.exists():
        return
    o = pd.read_csv(f)
    print(f"\n### MVP3 ({reader}) — operating points")
    print(o[o.budget.isin([0.1, 0.2, 0.5, 1.0])][["selector", "budget", "auroc", "tpr_verdict", "fpr_verdict", "abstain", "tpr_at_5", "fpr_at_5", "ece", "brier"]].round(3).to_string())
    v = pd.read_csv(RESULTS / "mvp3" / reader / "verdicts.csv")
    print("verdict shares at 20% (supported / refuted / insufficient) by condition:")
    print(v[v.budget == 0.2].pivot_table(index="condition", columns="selector", values=["supported", "refuted"]).round(2).to_string())


def salt_by_records(reader):
    man = RESULTS / "manifests" / "salt.json"
    d = RESULTS / "salt" / reader / "runs"
    if not man.exists() or not d.exists():
        return
    lab = {x["tid"]: x["coord"] for x in json.load(open(man))["test"]}
    cond = {x["tid"]: x["condition"] for x in json.load(open(man))["test"]}
    print(f"\n### SALT ({reader}) — AUROC by records revealed; final 'supported' by condition")
    for s in ORDER:
        f = d / f"test_{s}.jsonl.gz"
        if not f.exists():
            continue
        runs = [json.loads(l) for l in gzip.open(f, "rt")]
        y = np.array([lab[r["tid"]] for r in runs])

        def at_k(r, k):
            best = r["trace"][0]
            for st in r["trace"]:
                if st[0] <= k:
                    best = st
            return best[2]
        fin = [r["trace"][-1] for r in runs]
        byc = {c: round(float(np.mean([t[3] == "supported" for t, r in zip(fin, runs) if cond[r["tid"]] == c])), 3)
               for c in ["coord", "sim_relax", "abandoned", "authorized", "none"]}
        print(f"{s:10s}", " ".join(f"k={k}:{auroc(y, np.array([at_k(r, k) for r in runs])):.3f}" for k in (2, 4, 6, 8, 12)),
              "| final", byc, f"records={np.mean([len(r['log']) for r in runs]):.1f}")


def teamops():
    base = RESULTS / "teamops"
    if not base.exists():
        return
    for tag in sorted(os.listdir(base)):
        if not tag.startswith("L_audit"):
            continue
        for reader in sorted(os.listdir(base / tag)):
            f = base / tag / reader / "metrics.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            print(f"\n### TeamOps {tag} / reader={reader}")
            print(d.pivot_table(index="selector", columns="budget", values="auroc").reindex(ORDER).dropna(how="all").round(3).to_string())
            print(d[d.budget.isin([0.1, 0.2])].set_index(["selector", "budget"])[["auroc_lo", "auroc_hi", "tpr@5", "fpr@5", "cert_found", "records", "supported_on_neg", "overhead_median"]].round(3).to_string())
            p = base / tag / reader / "paired.csv"
            if p.exists():
                q = pd.read_csv(p)
                print("paired (combined minus other) at 10%/20%:")
                print(q[q.budget.isin([0.1, 0.2])].pivot_table(index="other", columns="budget", values=["diff", "lo", "hi"]).round(3).to_string())
            e = base / tag / reader / "extraction.json"
            if e.exists():
                print("reader agreement with judge:", json.load(open(e)).get("agreement_with_judge"))
    for f in sorted(base.glob("llm_*/summary_*.json")):
        print(f"\n### TeamOps LLM baselines {f}")
        print(json.dumps(json.load(open(f)), indent=1)[:3000])


if __name__ == "__main__":
    mvp1("mvp1/perfect", "MVP1 perfect reading")
    mvp1("mvp1/qwen3_mvp1", "MVP1 Qwen3-8B reader")
    mvp1("mvp1/qwen25_mvp1", "MVP1 Qwen2.5-7B reader")
    mvp1("mvp1/perfect_mvp1_test_D", "Held-out mechanism D, perfect")
    mvp1("mvp1/qwen3_mvp1D_mvp1_test_D", "Held-out mechanism D, Qwen3 reader")
    mvp2()
    for r in ["perfect", "qwen3_mvp3"]:
        mvp3(r)
    for r in ["perfect", "qwen3-8b", "qwen3-8b-think", "qwen3-8b+confirm_salt_qwen3-8b-think", "qwen3-8b+confirm_perfect"]:
        salt_by_records(r)
    teamops()
