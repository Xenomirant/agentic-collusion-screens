"""Figures from results/*.csv (skips what is missing).  python experiments/make_figures.py"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, ROOT  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
ORDER = ["random", "random_groups", "graph", "caes", "suspicion", "voi", "comp", "combined"]
COL = {"random": "#999999", "random_groups": "#bbbbbb", "graph": "#7f7f7f", "caes": "#8c564b",
       "suspicion": "#1f77b4", "voi": "#2ca02c", "comp": "#ff7f0e", "combined": "#d62728"}


def budget_curves(csv, title, out, metric="auroc", ablation="none"):
    if not os.path.exists(csv):
        return
    d = pd.read_csv(csv)
    if "ablation" in d:
        d = d[d.ablation == ablation]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in ORDER:
        g = d[d.selector == s].sort_values("budget")
        if len(g):
            ax.plot(g.budget, g[metric], "o-", color=COL[s], label=s)
            if metric == "auroc" and "auroc_lo" in g:
                ax.fill_between(g.budget, g.auroc_lo, g.auroc_hi, color=COL[s], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
    ax.set_xlabel("disclosure budget (fraction of tokens)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def ablation_bars(csv, out, budget=0.2):
    if not os.path.exists(csv):
        return
    d = pd.read_csv(csv)
    d = d[d.budget == budget]
    abl = ["none", "shuffle_links", "no_provenance", "no_comms", "no_actions", "hide_outcomes"]
    sels = [s for s in ["caes", "suspicion", "voi", "comp", "combined"] if s in set(d.selector)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / len(sels)
    for i, s in enumerate(sels):
        vals = [d[(d.selector == s) & (d.ablation == a)].auroc.mean() for a in abl]
        ax.bar(np.arange(len(abl)) + i * w, vals, w, color=COL[s], label=s)
    ax.set_xticks(np.arange(len(abl)) + 0.4 - w / 2)
    ax.set_xticklabels(abl, rotation=20, fontsize=8)
    ax.set_ylabel(f"AUROC at {int(budget * 100)}% budget")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def salt_records(readers, out):
    man = RESULTS / "manifests" / "salt.json"
    if not man.exists():
        return
    lab = {x["tid"]: x["coord"] for x in json.load(open(man))["test"]}
    from sklearn.metrics import roc_auc_score
    fig, axes = plt.subplots(1, len(readers), figsize=(4.5 * len(readers), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, reader in zip(axes, readers):
        for s in ORDER:
            f = RESULTS / "salt" / reader / "runs" / f"test_{s}.jsonl.gz"
            if not f.exists():
                continue
            runs = [json.loads(l) for l in gzip.open(f, "rt")]
            y = np.array([lab[r["tid"]] for r in runs])
            ks = range(1, 13)
            au = []
            for k in ks:
                sc = []
                for r in runs:
                    best = r["trace"][0]
                    for st in r["trace"]:
                        if st[0] <= k:
                            best = st
                    sc.append(best[2])
                au.append(roc_auc_score(y, sc))
            ax.plot(list(ks), au, "o-", ms=3, color=COL[s], label=s)
        ax.set_title(f"SALT, reader = {reader}", fontsize=9)
        ax.set_xlabel("records revealed")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def mvp2(csv, out):
    if not os.path.exists(csv):
        return
    d = pd.read_csv(csv)
    knobs = [k for k in ["length", "n_decoys", "gap", "n_agents", "q_meta"] if k in set(d.knob)]
    fig, axes = plt.subplots(2, len(knobs), figsize=(3.4 * len(knobs), 6), squeeze=False)
    base = d[d.knob == "base"]
    for j, k in enumerate(knobs):
        g = d[d.knob == k]
        for s in ORDER:
            gs = pd.concat([base[base.selector == s].assign(value=None), g[g.selector == s]])
            if not len(gs):
                continue
            gs = gs.sort_values("value", na_position="first")
            x = gs.value.fillna({"length": 60, "n_decoys": 2, "gap": 1.0, "n_agents": 4, "q_meta": 1.0}[k])
            axes[0, j].plot(x, gs.overhead_median, "o-", color=COL[s], label=s)
            axes[1, j].plot(x, gs["auroc@0.2"], "o-", color=COL[s])
        axes[0, j].set_title(k)
        axes[0, j].set_yscale("log")
        axes[1, j].set_xlabel(k)
        for a in axes[:, j]:
            a.grid(alpha=0.3)
    axes[0, 0].set_ylabel("search overhead C_read/C_cert* (median)")
    axes[1, 0].set_ylabel("AUROC at 20% budget")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


def mvp3(d_dir, out):
    f = Path(d_dir) / "risk_coverage.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    op = pd.read_csv(Path(d_dir) / "operating.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for s in ["suspicion", "voi", "comp", "combined", "caes", "random"]:
        g = d[(d.selector == s) & (d.budget == 0.2)].sort_values("coverage")
        if len(g):
            axes[0].plot(g.coverage, g.risk, "o-", color=COL[s], label=s)
        h = op[op.selector == s].sort_values("budget")
        if len(h):
            axes[1].plot(h.budget, h.fpr_verdict, "o-", color=COL[s], label=s)
            axes[1].plot(h.budget, h.tpr_verdict, "s--", color=COL[s], alpha=0.6)
    axes[0].set_xlabel("coverage (fraction classified by score confidence)")
    axes[0].set_ylabel("risk (error rate among classified), 20% budget")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("budget")
    axes[1].set_ylabel("verdict FPR (solid) / TPR (dashed)")
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    budget_curves(RESULTS / "mvp1/perfect/metrics.csv", "MVP1, perfect reading", "mvp1_perfect_auroc.png")
    budget_curves(RESULTS / "mvp1/qwen3_mvp1/metrics.csv", "MVP1, Qwen3-8B reader", "mvp1_qwen3_auroc.png")
    budget_curves(RESULTS / "mvp1/qwen25_mvp1/metrics.csv", "MVP1, Qwen2.5-7B reader", "mvp1_qwen25_auroc.png")
    budget_curves(RESULTS / "mvp1/perfect_mvp1_test_D/metrics.csv", "Held-out mechanism D, perfect reading", "mvp1_D_perfect_auroc.png")
    budget_curves(RESULTS / "mvp1/qwen3_mvp1D_mvp1_test_D/metrics.csv", "Held-out mechanism D, Qwen3-8B reader", "mvp1_D_qwen3_auroc.png")
    budget_curves(RESULTS / "mvp1/perfect/metrics.csv", "MVP1, perfect reading", "mvp1_perfect_certfound.png", metric="cert_found")
    ablation_bars(RESULTS / "mvp1/perfect/metrics.csv", "mvp1_ablations_perfect.png")
    ablation_bars(RESULTS / "mvp1/qwen3_mvp1/metrics.csv", "mvp1_ablations_qwen3.png")
    salt_records([r for r in ["perfect", "qwen3-8b", "qwen3-8b-think", "qwen3-8b+confirm_salt_qwen3-8b-think"]
                  if (RESULTS / "salt" / r / "runs").exists()], "salt_auroc_by_records.png")
    mvp2(RESULTS / "mvp2/metrics.csv", "mvp2_overhead.png")
    mvp3(RESULTS / "mvp3/perfect", "mvp3_perfect.png")
    mvp3(RESULTS / "mvp3/qwen3_mvp3", "mvp3_qwen3.png")
    for tag in ["L_audit_qwen2.5-7b_to_qwen3-8b", "L_audit_qwen3-8b_to_qwen2.5-7b", "L_audit_qwen3-8b_to_qwen3-8b",
                "L_audit_qwen3-8b_to_qwen3-8b_n8to12", "L_audit_qwen2.5-7b_to_qwen2.5-7b_n8to12"]:
        for reader in ["judge", "qwen3-8b", "qwen3-8b+confirm_judge"]:
            budget_curves(RESULTS / "teamops" / tag / reader / "metrics.csv", f"TeamOps {tag}, reader={reader}",
                          f"teamops_{tag}_{reader}.png")
    print("figures:", sorted(os.listdir(FIG)))
