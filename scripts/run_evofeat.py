"""Run the evolutionary feature-search loop for one (backbone, dataset)
pair. Outputs go under ``results/runs/{backbone}/``:

    curve.jsonl    — per-iteration metrics
    samples/       — one json per LLM candidate
    best.json      — best-of-run summary
    stats.json     — total tokens / latency / failures

Use ``scripts/run_comparison.py`` to fold these together with the
classical baselines into the headline report.
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

# allow `python scripts/run_evofeat.py …` directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evofeat.buffer import BufferConfig
from evofeat.data import load_banking77, load_dataset
from evofeat.llm import make_client
from evofeat.logger import RunLogger
from evofeat.search import SearchConfig, run_search


log = logging.getLogger("evofeat.run")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_dataset_obj(name: str):
    if name == "banking77":
        return load_banking77()
    return load_dataset(name)


def _build_search_config(s: dict) -> SearchConfig:
    spp = s.get("samples_per_prompt", 3)
    n_iter = s.get("num_iterations", 20)
    buf = s.get("buffer", {}) or {}
    buf_cfg = BufferConfig(
        functions_per_prompt=buf.get("functions_per_prompt", 2),
        num_islands=buf.get("num_islands", 3),
        reset_period_s=buf.get("reset_period_s", 4 * 60 * 60),
        sampling_t_init=buf.get("sampling_t_init", 0.1),
        sampling_t_period=buf.get("sampling_t_period", 30000),
    )
    return SearchConfig(
        samples_per_prompt=spp,
        evaluate_timeout_s=s.get("evaluate_timeout_s", 30),
        max_samples=n_iter * spp,
        early_stop_patience=s.get("early_stop_patience", 20),
        buffer=buf_cfg,
    )


def run_one(
    backbone_cfg_path: str, dataset_name: str, search_cfg: SearchConfig,
    out_root: str = "results/runs",
) -> dict:
    backbone_cfg = _load_yaml(backbone_cfg_path)
    label = backbone_cfg.get("backbone_label") or backbone_cfg["model"]
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)

    # the spec to evolve under
    if dataset_name == "banking77":
        spec_path = "specs/specification_banking77.txt"
    else:
        spec_path = f"specs/specification_{dataset_name}.txt"
    with open(spec_path) as f:
        spec = f.read()

    ds = _load_dataset_obj(dataset_name)
    meta = ds.meta.get("feature_descriptions", {}) if isinstance(ds.meta, dict) else {}
    inputs = {"data": {
        "inputs": ds.X,
        "outputs": ds.y,
        "is_cat": ds.is_cat,
        "is_regression": ds.is_regression,
    }}

    client = make_client(backbone_cfg)
    # quick reachability probe so a missing vLLM server doesn't burn an hour
    # cycling retries before we even start the loop
    try:
        client._session.get(
            client._endpoint().replace("/chat/completions", "/models"),
            timeout=5,
        )
    except Exception as e:
        raise RuntimeError(
            f"backbone {label!r} unreachable at {client._endpoint()}: {e}"
        ) from e
    log.info("[%s] %s/%s, max_samples=%d", label, dataset_name, client.model_id, search_cfg.max_samples)

    with RunLogger(run_name=f"{dataset_name}__{label}", config=backbone_cfg) as logger:
        t0 = time.time()
        profiler = run_search(
            spec=spec, inputs=inputs, meta=meta, client=client,
            config=search_cfg, log_dir=out_dir, logger=logger,
        )
        elapsed = time.time() - t0
        best_score = profiler.best_score if profiler.best_score != float("-inf") else None
        summary = {
            "backbone": label,
            "model_id": client.model_id,
            "family": client.family,
            "dataset": dataset_name,
            "elapsed_s": round(elapsed, 1),
            "best_score": best_score,
            "llm_stats": client.stats.snapshot(),
        }
        logger.summary(summary)

    with open(os.path.join(out_dir, "best.json"), "w") as f:
        json.dump({**summary, "best_program": profiler.best_program}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(client.stats.snapshot(), f, indent=2)
    log.info("[%s] done in %.1fs, best=%.4f", label, elapsed, best_score or 0.0)
    return summary


def main():
    p = argparse.ArgumentParser(description="Run EvoFeat evolutionary search for one backbone.")
    p.add_argument("--experiment", default="configs/experiments/banking77_full.yaml")
    p.add_argument("--backbone", default=None, help="path to a backbone YAML (overrides experiment list)")
    p.add_argument("--dataset", default=None, help="override dataset name (else read from experiment)")
    p.add_argument("--quick", action="store_true", help="use the smoke-test experiment config")
    p.add_argument("--only", default=None, help="restrict to a specific backbone label")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s | %(message)s")

    cfg_path = "configs/experiments/banking77_quick.yaml" if args.quick else args.experiment
    cfg = _load_yaml(cfg_path)
    dataset_name = args.dataset or cfg["dataset"]
    search_cfg = _build_search_config(cfg.get("search", {}))
    out_root = cfg.get("reporting", {}).get("out_dir", "results") + "/runs"

    if args.backbone:
        run_one(args.backbone, dataset_name, search_cfg, out_root=out_root)
        return

    for bb_path in cfg["backbones"]:
        bb_cfg = _load_yaml(bb_path)
        label = bb_cfg.get("backbone_label") or bb_cfg["model"]
        if args.only and label != args.only:
            continue
        try:
            run_one(bb_path, dataset_name, search_cfg, out_root=out_root)
        except Exception as e:
            log.exception("[%s] failed: %s", label, e)


if __name__ == "__main__":
    main()
