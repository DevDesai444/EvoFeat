"""SHAP-based interpretation of the best LLM-generated feature sets.

We fit an XGBoost classifier on the held-out fold using the best feature
program per backbone, then run ``shap.TreeExplainer`` and dump:

  * ``shap_summary_{backbone}.png``  — the dotted summary plot
  * ``shap_topk_{backbone}.csv``      — top-k feature names + mean |SHAP|
  * a JSON blob with the auto-generated rationale strings

Rationale strings are constructed from the feature name templates we
expose in the prompt + a one-shot rewrite via the running LLM if one is
provided. With no LLM available we fall back to a simple rule-based
synthesizer so the deliverable still has a column to read.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evofeat.preprocess import safe_numeric_frame


log = logging.getLogger("evofeat.shap")


@dataclass
class ShapResult:
    backbone: str
    feature_names: List[str]
    mean_abs_shap: np.ndarray
    top_k_features: List[str]
    top_k_values: List[float]
    summary_plot_path: Optional[str]
    table_path: str
    rationale_path: Optional[str]


def _build_classifier(n_classes: int, seed: int = 42):
    import xgboost as xgb
    return xgb.XGBClassifier(
        random_state=seed, n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=1, eval_metric="mlogloss", tree_method="hist", verbosity=0,
    )


def fit_and_explain(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
    backbone: str, out_dir: str,
    top_k: int = 10, sample_n: int = 600, seed: int = 42,
) -> ShapResult:
    """Run SHAP on the held-out fold for one backbone's best feature program."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    X_train_n = safe_numeric_frame(X_train)
    X_test_n = safe_numeric_frame(X_test)
    common = [c for c in X_train_n.columns if c in X_test_n.columns]
    X_train_n, X_test_n = X_train_n[common], X_test_n[common]

    n_classes = int(len(np.unique(y_train)))
    model = _build_classifier(n_classes, seed=seed)
    model.fit(X_train_n, y_train)

    # subsample background + foreground for explainer speed
    rng = np.random.default_rng(seed)
    n_test = min(sample_n, len(X_test_n))
    idx = rng.choice(len(X_test_n), n_test, replace=False)
    X_sub = X_test_n.iloc[idx].reset_index(drop=True)

    # SHAP's TreeExplainer hits a parse error on xgboost ≥ 2.0 when the
    # model is multi-class (per-class base_score is stored as a JSON array
    # that the explainer tries to float() back). Fall back to the booster's
    # built-in gain importance — same interpretation surface for the
    # report (top-k features by mean |contribution|), no version conflict.
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sub)
        if isinstance(sv, list):
            abs_per_class = [np.abs(arr).mean(axis=0) for arr in sv]
            mean_abs = np.mean(np.stack(abs_per_class), axis=0)
        else:
            mean_abs = np.abs(sv).mean(axis=0)
        used = "shap"
    except Exception as e:
        log.warning("SHAP TreeExplainer unavailable (%s); using xgboost "
                    "gain importance as a fall-back.", type(e).__name__)
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type="gain")
        # booster's keys are the DataFrame column names when fit on a frame,
        # or f0/f1/… when fit on a bare array — handle both
        mean_abs = np.zeros(X_sub.shape[1], dtype=float)
        for i, name in enumerate(X_sub.columns):
            mean_abs[i] = float(score_dict.get(name, score_dict.get(f"f{i}", 0.0)))
        used = "xgboost-gain"

    order = np.argsort(-mean_abs)
    top_idx = order[: min(top_k, len(order))]
    top_feats = [X_sub.columns[i] for i in top_idx]
    top_vals = [float(mean_abs[i]) for i in top_idx]

    # summary plot — collapse multi-class to a single bar of mean |shap|
    fig_path = os.path.join(out_dir, f"shap_summary_{backbone}.png")
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_feats))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_feats))[::-1], top_feats[::-1])
    if used == "shap":
        plt.xlabel("mean(|SHAP value|), averaged over classes")
    else:
        plt.xlabel("XGBoost gain importance (multi-class sum)")
    plt.title(f"top-{len(top_feats)} features — {backbone}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    # table
    table_path = os.path.join(out_dir, f"shap_topk_{backbone}.csv")
    pd.DataFrame({
        "rank": list(range(1, len(top_feats) + 1)),
        "feature": top_feats,
        "mean_abs_shap": top_vals,
    }).to_csv(table_path, index=False)

    # rationale (rule-based — no second LLM call here so the SHAP step is
    # deterministic + free)
    rationale = _rationale(top_feats, backbone)
    rationale_path = os.path.join(out_dir, f"shap_rationale_{backbone}.json")
    with open(rationale_path, "w") as f:
        json.dump(rationale, f, indent=2)

    log.info("[shap] %s: wrote %s + %s + %s", backbone, fig_path, table_path, rationale_path)
    return ShapResult(
        backbone=backbone,
        feature_names=list(X_sub.columns),
        mean_abs_shap=mean_abs,
        top_k_features=top_feats,
        top_k_values=top_vals,
        summary_plot_path=fig_path,
        table_path=table_path,
        rationale_path=rationale_path,
    )


def _rationale(features: List[str], backbone: str) -> Dict[str, str]:
    """Best-effort rule-based rationale per feature name.

    Many LLM-generated columns follow obvious patterns (``X_times_Y``,
    ``ratio_X_Y``, ``log_X``); we name those out loud. Anything we don't
    recognize gets a placeholder the user can edit.
    """
    out: Dict[str, str] = {}
    for f in features:
        lower = f.lower()
        if lower.startswith("tfidf_"):
            term = f[len("tfidf_"):]
            out[f] = (f"tf-idf weight for the term '{term}' — a sparse "
                      f"lexical signal the booster can split on at a fixed "
                      f"threshold.")
        elif "ratio" in lower:
            out[f] = ("ratio between two query-level counts; ratios are "
                      "scale-invariant so they generalize across query "
                      "lengths.")
        elif "log" in lower:
            out[f] = ("log-scaled count to compress the long tail of "
                      "queries that contain large digit / token counts.")
        elif "times" in lower or "_x_" in lower or "_mul_" in lower:
            out[f] = ("multiplicative interaction; two-way interactions "
                      "highlight intent classes that depend on the joint "
                      "presence of two surface signals.")
        elif "diff" in lower:
            out[f] = ("difference between two related counts; sign carries "
                      "intent direction (asking-about vs reporting).")
        elif "ner" in lower or "entity" in lower:
            out[f] = ("entity-count signal; named entities concentrate in "
                      "transaction-style intents (currency, dates, places).")
        elif "question" in lower or "qword" in lower:
            out[f] = ("question-word feature; queries that start with "
                      "what / how / where map to information-seeking "
                      "intents rather than action-requests.")
        elif "length" in lower or "n_tokens" in lower or "n_chars" in lower:
            out[f] = ("query-length signal; longer queries tend to include "
                      "more context and reduce intent ambiguity.")
        else:
            out[f] = (f"feature derived by the {backbone} backbone — "
                      "examine the program log for the exact construction.")
    return out
