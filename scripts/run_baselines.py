"""Run every classical baseline on a dataset's 5-fold splits.

Persists per-fold metrics under ``results/baselines/{dataset}.json`` so
``scripts/run_comparison.py`` can fold them together with the LLM runs
without re-fitting anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evofeat.baselines import all_classification_baselines, all_regression_baselines
from evofeat.data import load_banking77, load_dataset
from evofeat.evaluate import (
    evaluate_classification, evaluate_regression, identity_builder, primary_score,
)


log = logging.getLogger("evofeat.baselines")


def _load_dataset_obj(name: str):
    return load_banking77() if name == "banking77" else load_dataset(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/experiments/banking77_full.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out-dir", default="results/baselines")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    with open(args.experiment) as f:
        cfg = yaml.safe_load(f)

    dataset_name = args.dataset or cfg["dataset"]
    k = args.k or cfg.get("baselines", {}).get("k", 30)
    requested = set(cfg.get("baselines", {}).get("methods", []) or [])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ds = _load_dataset_obj(dataset_name)
    splits = ds.splits or None
    if splits is None:
        from evofeat.data import stratified_or_kfold_splits
        splits = stratified_or_kfold_splits(ds.y, ds.is_regression)

    if ds.is_regression:
        all_b = all_regression_baselines(k=k)
        eval_fn = evaluate_regression
    else:
        all_b = all_classification_baselines(k=k)
        eval_fn = evaluate_classification

    builders = {"base": identity_builder}
    for name, builder in all_b.items():
        if requested and name not in requested:
            continue
        builders[name] = builder

    results: dict[str, dict] = {}
    for name, builder in builders.items():
        log.info("[baselines] running %s on %s …", name, dataset_name)
        t0 = time.time()
        model_results = eval_fn(ds.X, ds.y, splits, builder=builder, method_name=name)
        elapsed = time.time() - t0
        per_fold = {}
        agg = {}
        for r in model_results:
            per_fold[r.model] = [f.metrics for f in r.folds]
            agg[r.model] = r.aggregate()
        results[name] = {
            "primary_score": primary_score(model_results, ds.is_regression),
            "per_fold": per_fold,
            "aggregate": agg,
            "n_features_in": model_results[0].folds[0].n_features_in,
            "n_features_out": model_results[0].folds[0].n_features_out,
            "elapsed_s": round(elapsed, 1),
        }

    out_path = os.path.join(out_dir, f"{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("wrote %s (%d methods)", out_path, len(results))


if __name__ == "__main__":
    main()
