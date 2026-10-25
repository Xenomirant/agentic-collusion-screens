"""Shared experiment plumbing: cached splits, cached prior models, parallel audit runs."""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from certaudit.belief import QModels, train_qmodels  # noqa: E402
from certaudit.generator import Knobs, make_split  # noqa: E402
from certaudit.oracle import CachedReader, PerfectReader  # noqa: E402
from certaudit.selectors import SELECTORS, run_audit  # noqa: E402
from certaudit.verifier import min_certificate  # noqa: E402

CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

# split name -> (family, n_per_cell, mechanisms, extra kwargs)
SPLITS = {
    "mvp1_train": dict(family=0, n_per_cell=10, mechs=("A", "B", "C")),
    "mvp1_cal": dict(family=1, n_per_cell=10, mechs=("A", "B", "C")),
    "mvp1_test": dict(family=2, n_per_cell=20, mechs=("A", "B", "C")),
    # held-out mechanism (never seen by the priors): coordinated false reports
    "mvp1_test_D": dict(family=2, n_per_cell=20, mechs=("D",), seed0=77),
    "mvp1_cal_D": dict(family=1, n_per_cell=10, mechs=("D",), seed0=77),
}


def get_split(name, knobs: Knobs | None = None, **override):
    path = CACHE / "splits" / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    spec = dict(SPLITS.get(name, {}))
    spec.update(override)
    trajs = make_split(spec["family"], spec["n_per_cell"], mechs=spec["mechs"], knobs=knobs or Knobs(),
                       seed0=spec.get("seed0", 0), conds=spec.get("conds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(trajs, fh)
    manifest = [dict(tid=t.tid, mechanism=t.mechanism, condition=t.condition, family=t.family,
                     coord=t.labels["coord"], n_events=len(t.events), total_cost=t.total_cost) for t in trajs]
    (RESULTS / "manifests").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "manifests" / f"{name}.json", "w") as fh:
        json.dump(manifest, fh)
    return trajs


def get_qmodels(train_name, ablation=None, n_masks=6, seed=0) -> Path:
    path = CACHE / "models" / f"q_{train_name}_{ablation or 'none'}_m{n_masks}_s{seed}.pkl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        qm = train_qmodels(get_split(train_name), n_masks=n_masks, seed=seed, ablation=ablation)
        qm.save(path)
    return path


def get_reader(spec):
    if spec == "perfect":
        return PerfectReader()
    with open(CACHE / "extract" / f"{spec}.pkl", "rb") as fh:
        return CachedReader(spec, pickle.load(fh))


_W = {}


def _init(qm_path, reader_spec, confirm_spec=None):
    _W["qm"] = QModels.load(qm_path)
    _W["reader"] = get_reader(reader_spec)
    _W["confirm"] = get_reader(confirm_spec) if confirm_spec else None


def _one(args):
    traj, sel, ablation, seed = args
    r = run_audit(traj, SELECTORS[sel](), _W["qm"], _W["reader"], ablation=ablation, seed=seed,
                  confirm_reader=_W["confirm"])
    r["reader"] = _W["reader"].name
    r["confirm"] = _W["confirm"].name if _W["confirm"] else None
    return r


def run_many(trajs, sel, qm_path, reader_spec="perfect", ablation=None, out=None, procs=48, seed=0, confirm_spec=None):
    if out is not None and Path(out).exists():
        return load_runs(out)
    with get_context("fork").Pool(procs, initializer=_init, initargs=(str(qm_path), reader_spec, confirm_spec)) as pool:
        runs = pool.map(_one, [(t, sel, ablation, seed) for t in trajs], chunksize=2)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wt") as fh:
            for r in runs:
                fh.write(json.dumps(r) + "\n")
    return runs


def load_runs(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh]


def cert_costs(trajs):
    out = {}
    for t in trajs:
        mc = min_certificate(t) if t.labels["coord"] else None
        out[t.tid] = mc[0] if mc else None
    return out
