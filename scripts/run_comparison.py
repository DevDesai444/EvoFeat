"""Combine evolutionary-search results + classical baselines into the
headline report.

Reads:

  results/runs/{backbone}/best.json   — produced by run_evofeat.py
  results/baselines/{dataset}.json    — produced by run_baselines.py

Writes:

  results/tables/headline.{csv,md,tex}
  results/tables/pairwise_stats.{csv,md,tex}
  results/figures/comparison_bars.png
  results/figures/boxplot.png
  results/figures/pareto.png
  results/shap/<backbone>/…         (SHAP run on the best program per backbone)

Run after both upstream scripts complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evofeat.cost import cloud_estimate, gpu_estimate, pareto_table
from evofeat.data import load_banking77, load_dataset, stratified_or_kfold_splits
from evofeat.evaluate import evaluate_classification, evaluate_regression, primary_score
from evofeat.llm_builder import llm_program_builder
from evofeat.reporting import (
    boxplot_per_method, comparison_bars, headline_comparison_table,
    pareto_plot, pairwise_significance_table,
)
from evofeat.shap_analysis import fit_and_explain
from evofeat.stats import all_pairs_table, summarize_scores, vs_reference_table


log = logging.getLogger("evofeat.compare")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_dataset_obj(name: str):
    return load_banking77() if name == "banking77" else load_dataset(name)


def _accumulate_llm_methods(
    ds, splits, runs_dir: str, dataset_name: str, prefer_metric: str = "accuracy",
):
    """Re-evaluate each saved LLM program through the canonical 5-fold harness."""
    method_scores: Dict[str, List[float]] = {}
    method_aggregates: Dict[str, dict] = {}
    backbone_for_method: Dict[str, str] = {}

    for entry in sorted(os.listdir(runs_dir)):
        bb_dir = os.path.join(runs_dir, entry)
        best_path = os.path.join(bb_dir, "best.json")
        if not os.path.isdir(bb_dir) or not os.path.exists(best_path):
            continue
        with open(best_path) as f:
            best = json.load(f)
        if best.get("dataset") != dataset_name:
            continue
        program = best.get("best_program")
        if not program:
            continue
        builder = llm_program_builder(program)
        method_name = f"llm:{entry}"
        log.info("[llm] re-evaluating %s on canonical splits …", method_name)
        eval_fn = evaluate_regression if ds.is_regression else evaluate_classification
        results = eval_fn(ds.X, ds.y, splits, builder=builder, method_name=method_name)
        # xgb scores per fold
        xgb_res = next((r for r in results if r.model == "xgb"), results[0])
        method_scores[method_name] = [f.metrics[prefer_metric if not ds.is_regression else "rmse"] for f in xgb_res.folds]
        method_aggregates[method_name] = xgb_res.aggregate()
        backbone_for_method[method_name] = entry
    return method_scores, method_aggregates, backbone_for_method


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/experiments/banking77_full.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--metric", default="accuracy",
                   help="classification metric to compare on (accuracy / macro_f1 / auc)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--skip-shap", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")
    cfg = _load_yaml(args.experiment)
    dataset_name = args.dataset or cfg["dataset"]
    tables_dir = os.path.join(args.out_dir, "tables")
    figures_dir = os.path.join(args.out_dir, "figures")
    shap_dir = os.path.join(args.out_dir, "shap")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    ds = _load_dataset_obj(dataset_name)
    splits = ds.splits or stratified_or_kfold_splits(ds.y, ds.is_regression)

    # 1) load baselines
    base_path = os.path.join(args.out_dir, "baselines", f"{dataset_name}.json")
    if not os.path.exists(base_path):
        raise SystemExit(f"missing {base_path} — run scripts/run_baselines.py first")
    with open(base_path) as f:
        baseline_results = json.load(f)

    metric = args.metric if not ds.is_regression else "rmse"
    method_scores: Dict[str, List[float]] = {}
    aggregates: Dict[str, dict] = {}
    for name, blob in baseline_results.items():
        # xgb is the primary; pull per-fold scores from there and flatten
        # the nested aggregate dict down to the xgb row so the LLM and
        # classical entries share a shape
        xgb_folds = blob["per_fold"].get("xgb") or blob["per_fold"][next(iter(blob["per_fold"]))]
        method_scores[name] = [f[metric] for f in xgb_folds]
        agg_full = blob["aggregate"]
        aggregates[name] = agg_full.get("xgb", next(iter(agg_full.values())))

    # 2) load + re-evaluate LLM runs
    runs_dir = os.path.join(args.out_dir, "runs")
    llm_scores, llm_agg, bb_for_method = (
        ({}, {}, {})
        if not os.path.isdir(runs_dir)
        else _accumulate_llm_methods(ds, splits, runs_dir, dataset_name, prefer_metric=metric)
    )
    method_scores.update(llm_scores)
    aggregates.update(llm_agg)

    # 3) summary + pairwise stats tables
    summary = summarize_scores(method_scores)
    headline_paths = headline_comparison_table(summary, tables_dir, name="headline")
    log.info("headline tables: %s", headline_paths)

    if len(method_scores) >= 2:
        pairs = all_pairs_table(method_scores)
        pairwise_significance_table(pairs, tables_dir, name="pairwise_stats")
        if "base" in method_scores:
            vs_base = vs_reference_table(method_scores, "base")
            pairwise_significance_table(vs_base, tables_dir, name="vs_base")

    # 4) plots
    comparison_bars(summary, metric, os.path.join(figures_dir, "comparison_bars.png"),
                    title=f"feature-method comparison on {dataset_name}")
    boxplot_per_method(method_scores, os.path.join(figures_dir, "boxplot.png"), metric=metric)

    # 5) cost / pareto — only the LLM methods enter the cost table
    estimates = []
    accs_for_pareto: Dict[str, float] = {}
    for backbone_label, method_name in [(v, k) for k, v in bb_for_method.items()]:
        bb_dir = os.path.join(runs_dir, backbone_label)
        stats_path = os.path.join(bb_dir, "stats.json")
        best_path = os.path.join(bb_dir, "best.json")
        if not (os.path.exists(stats_path) and os.path.exists(best_path)):
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        with open(best_path) as f:
            best = json.load(f)
        backend = "vllm" if best.get("family") in ("qwen", "mistral", "llama") and "vllm" in best.get("model_id", "") else "groq"
        if best.get("model_id", "").startswith(("Qwen/", "mistralai/", "meta-llama/")):
            backend = "vllm"
        if backend == "groq":
            est = cloud_estimate(backbone_label, best["model_id"], stats)
        else:
            est = gpu_estimate(backbone_label, stats, gpu="T4")
        estimates.append(est)
        accs_for_pareto[backbone_label] = (
            float(aggregates[method_name].get(metric, [float("nan")])[0])
            if isinstance(aggregates[method_name].get(metric), (list, tuple))
            else float("nan")
        )
    if estimates:
        pareto = pareto_table(estimates, accs_for_pareto)
        pareto.to_csv(os.path.join(tables_dir, "pareto.csv"), index=False)
        with open(os.path.join(tables_dir, "pareto.md"), "w") as f:
            f.write(pareto.to_markdown(index=False))
        pareto_plot(pareto, os.path.join(figures_dir, "pareto.png"))

    # 6) SHAP per LLM backbone
    if not args.skip_shap:
        first_split = splits[0]
        tr_idx, te_idx = first_split
        X_tr = ds.X.iloc[tr_idx].reset_index(drop=True)
        X_te = ds.X.iloc[te_idx].reset_index(drop=True)
        y_tr = ds.y[tr_idx]
        y_te = ds.y[te_idx]
        for method_name, backbone_label in bb_for_method.items():
            bb_dir = os.path.join(runs_dir, backbone_label)
            best_path = os.path.join(bb_dir, "best.json")
            with open(best_path) as f:
                best = json.load(f)
            program = best.get("best_program")
            if not program:
                continue
            builder = llm_program_builder(program)
            try:
                X_tr_t, X_te_t = builder(X_tr, y_tr, X_te)
            except Exception as e:
                log.warning("SHAP skipped for %s: %s", backbone_label, e)
                continue
            out_dir = os.path.join(shap_dir, backbone_label)
            try:
                fit_and_explain(X_tr_t, y_tr, X_te_t, y_te,
                                backbone=backbone_label, out_dir=out_dir)
            except Exception as e:
                log.exception("SHAP failed for %s: %s", backbone_label, e)

    # 7) write a small index json the README + notebooks can pick up
    index = {
        "dataset": dataset_name,
        "metric": metric,
        "n_methods": len(method_scores),
        "n_folds": len(next(iter(method_scores.values()), [])),
        "headline_table": headline_paths,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "tables", "index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
