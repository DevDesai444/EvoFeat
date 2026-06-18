"""Token / cost accounting per backbone.

Two cost models live here:

  * ``cloud``  — dollars per million tokens. Numbers are loaded from a
    YAML file so they're easy to update without touching code; if a model
    isn't listed we default to USD 0 with a warning.
  * ``gpu``    — GPU-hour estimate for self-hosted vLLM runs. We don't
    try to be clever — wall-clock seconds × (USD/hr per GPU type from
    the same YAML).

Outputs feed the Pareto table (accuracy vs cost), produced as a CSV in
the reporting module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


log = logging.getLogger("evofeat.cost")


# tokens are billed per million; numbers below are illustrative defaults
# pulled from the public price-pages on 2026-04-21 — override via the
# `--price-file` flag on the comparison runner if they drift.
DEFAULT_PRICES_USD_PER_MTOKEN = {
    # groq cloud
    "groq:llama-3.1-8b-instant":   {"in": 0.05, "out": 0.08},
    "groq:llama-3.3-70b-versatile":{"in": 0.59, "out": 0.79},
    "groq:mistral-saba-24b":       {"in": 0.79, "out": 0.79},
    "groq:qwen/qwen3-32b":         {"in": 0.29, "out": 0.59},
    "groq:gemma2-9b-it":           {"in": 0.20, "out": 0.20},
    "groq:openai/gpt-oss-20b":     {"in": 0.10, "out": 0.30},
}

DEFAULT_GPU_HOURLY_USD = {
    "T4":         0.35,
    "P100":       0.60,
    "L4":         0.80,
    "A10":        1.00,
    "A100-40gb":  2.20,
    "A100-80gb":  3.20,
    "H100":       5.50,
}


@dataclass
class CostEstimate:
    backbone: str
    backend: str
    in_tokens: int
    out_tokens: int
    requests: int
    latency_s: float
    usd: float
    notes: str = ""


def cloud_estimate(
    backbone: str, model_id: str, stats: dict,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> CostEstimate:
    prices = prices or DEFAULT_PRICES_USD_PER_MTOKEN
    key = f"groq:{model_id}"
    if key not in prices:
        log.warning("no price entry for %s — defaulting to $0", key)
        usd = 0.0
        notes = "price unknown"
    else:
        p = prices[key]
        usd = (
            stats["prompt_tokens"]      / 1e6 * p["in"]
            + stats["completion_tokens"]/ 1e6 * p["out"]
        )
        notes = ""
    return CostEstimate(
        backbone=backbone, backend="groq",
        in_tokens=stats["prompt_tokens"],
        out_tokens=stats["completion_tokens"],
        requests=stats["requests"],
        latency_s=stats["latency_s"],
        usd=round(usd, 4),
        notes=notes,
    )


def gpu_estimate(
    backbone: str, stats: dict, gpu: str = "T4",
    rates: Optional[Dict[str, float]] = None,
) -> CostEstimate:
    rates = rates or DEFAULT_GPU_HOURLY_USD
    if gpu not in rates:
        log.warning("unknown gpu %s — defaulting to $0/hr", gpu)
        rate = 0.0
    else:
        rate = rates[gpu]
    hours = stats["latency_s"] / 3600.0
    usd = rate * hours
    return CostEstimate(
        backbone=backbone, backend="vllm",
        in_tokens=stats["prompt_tokens"],
        out_tokens=stats["completion_tokens"],
        requests=stats["requests"],
        latency_s=stats["latency_s"],
        usd=round(usd, 4),
        notes=f"{gpu} @ ${rate:.2f}/hr",
    )


def pareto_table(
    estimates: List[CostEstimate], accuracies: Dict[str, float],
) -> pd.DataFrame:
    """Build the headline trade-off table.

    ``accuracies`` is a {backbone: held-out accuracy} dict — kept separate
    so callers can swap the metric (accuracy / f1 / -rmse) without
    rebuilding the cost objects.
    """
    rows = []
    for e in estimates:
        rows.append({
            "backbone": e.backbone,
            "backend": e.backend,
            "in_tokens":  e.in_tokens,
            "out_tokens": e.out_tokens,
            "requests":   e.requests,
            "latency_s":  round(e.latency_s, 1),
            "usd":        e.usd,
            "accuracy":   round(accuracies.get(e.backbone, float("nan")), 4),
            "acc_per_usd": (
                round(accuracies[e.backbone] / e.usd, 4)
                if e.backbone in accuracies and e.usd > 0
                else float("inf")
            ),
            "notes": e.notes,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df
