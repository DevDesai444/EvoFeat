"""Tables + plots.

This module is the only place we write CSV/Markdown/LaTeX output files
and create matplotlib figures. The rest of the codebase produces in-memory
DataFrames and structured dicts; this is where they land on disk.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


log = logging.getLogger("evofeat.reporting")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def headline_comparison_table(
    summary: pd.DataFrame, out_dir: str, name: str = "headline",
) -> Dict[str, str]:
    """Three formats of the same table — csv (machine), markdown (README),
    latex (paper-ready)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "csv": os.path.join(out_dir, f"{name}.csv"),
        "md":  os.path.join(out_dir, f"{name}.md"),
        "tex": os.path.join(out_dir, f"{name}.tex"),
    }
    summary.to_csv(paths["csv"], index=False)
    with open(paths["md"], "w") as f:
        f.write(summary.to_markdown(index=False))
    with open(paths["tex"], "w") as f:
        f.write(summary.to_latex(index=False, escape=False, float_format="%.4f"))
    return paths


def pairwise_significance_table(
    df: pd.DataFrame, out_dir: str, name: str = "pairwise_stats",
) -> Dict[str, str]:
    return headline_comparison_table(df, out_dir, name=name)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


def convergence_curves(
    curves: Dict[str, List[Tuple[int, float]]],
    out_path: str, title: str = "Best validation score vs iteration",
) -> str:
    plt = _setup_mpl()
    plt.figure(figsize=(7, 4.5))
    for name, pts in curves.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=name)
    plt.xlabel("evaluations")
    plt.ylabel("best validation score")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower right")
    plt.savefig(out_path)
    plt.close()
    return out_path


def comparison_bars(
    summary: pd.DataFrame, metric: str, out_path: str,
    title: Optional[str] = None,
) -> str:
    plt = _setup_mpl()
    df = summary.sort_values("mean", ascending=True)
    plt.figure(figsize=(7, max(3, 0.35 * len(df))))
    plt.barh(df["method"], df["mean"], xerr=df["std"], color="#4a7dbf")
    plt.xlabel(f"{metric} (mean ± std, 5-fold CV)")
    plt.title(title or f"feature method comparison — {metric}")
    plt.grid(axis="x", alpha=0.3)
    plt.savefig(out_path)
    plt.close()
    return out_path


def boxplot_per_method(
    method_scores: Dict[str, List[float]], out_path: str, metric: str = "accuracy",
) -> str:
    plt = _setup_mpl()
    items = sorted(method_scores.items(), key=lambda kv: -np.mean(kv[1]))
    names = [k for k, _ in items]
    data = [v for _, v in items]
    plt.figure(figsize=(8, max(3, 0.35 * len(names))))
    plt.boxplot(data, vert=False, labels=names, showmeans=True, meanline=True)
    plt.xlabel(f"{metric} across 5 folds")
    plt.title(f"distribution of {metric} per method")
    plt.grid(axis="x", alpha=0.3)
    plt.savefig(out_path)
    plt.close()
    return out_path


def pareto_plot(
    pareto: pd.DataFrame, out_path: str, x_col: str = "usd", y_col: str = "accuracy",
) -> str:
    plt = _setup_mpl()
    plt.figure(figsize=(6, 5))
    for _, row in pareto.iterrows():
        plt.scatter(row[x_col], row[y_col], s=70)
        plt.annotate(row["backbone"], (row[x_col], row[y_col]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.xlabel(f"{x_col} (USD)")
    plt.ylabel(y_col)
    plt.title("cost / quality frontier")
    plt.grid(alpha=0.3)
    plt.savefig(out_path)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# index file the README links to
# ---------------------------------------------------------------------------

def write_run_index(out_dir: str, index: Dict[str, object]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    return path
