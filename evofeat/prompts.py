"""Prompt construction.

The prompt has three blobs we splice in:
  * the program template (carrying the previous best ``modify_features``
    functions, renamed to ``_v0`` / ``_v1`` / ...);
  * the column-level feature description; and
  * a handful of serialized sample rows for grounding.

We alternate between two prompt 'flavors' on a coinflip — a generic
"domain reasoning" head/tail and a more constrained "advanced operators"
head/tail. The mix gives us a wider behavior cloud across iterations.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import pandas as pd

from evofeat.data import is_categorical


SYSTEM_PROMPT = (
    "You are a feature engineer. Given a tabular dataset description, you "
    "write a single Python function `modify_features(df_input)` that "
    "derives new features from the given DataFrame. Reply with ONLY the "
    "function definition — starting with `def modify_features(`, ending "
    "with `return df_output`. No prose before or after. No markdown "
    "fences. Keep all existing columns, append 3-6 new numeric columns. "
    "Use only pandas + numpy + scipy.stats + sklearn.preprocessing. "
    "Handle NaN with explicit fillna; do not depend on external state."
)


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _serialize_row(row: pd.Series) -> str:
    parts = []
    cols = list(row.index)
    for i, name in enumerate(cols):
        val = str(row[name]).strip(" .'").strip('"').strip()
        if i == 0:
            parts.append(f"If {name} is {val}")
        elif i < len(cols) - 1:
            parts.append(f"{name} is {val}")
        else:
            if len(name.strip()) < 2:
                continue
            parts.append(f"Then {name} is {val}.")
    return ", ".join(parts)


def feature_block(X: pd.DataFrame, meta: dict) -> str:
    lines: List[str] = []
    for col in X.columns:
        cat = is_categorical(X[col])
        desc = meta.get(col, col.replace("_", " "))
        if cat:
            vals = [str(v) for v in X[col].dropna().unique().tolist()[:8]]
            lines.append(f"- {col}: {desc} (categorical, values in [{', '.join(vals)}])")
        else:
            mn, mx = X[col].min(), X[col].max()
            lines.append(f"- {col}: {desc} (numerical, range [{mn}, {mx}])")
    return "\n".join(lines)


def example_block(X: pd.DataFrame, y, k: int = 4, rng: random.Random | None = None) -> str:
    """Serialize a handful of rows to ground the LLM in the data.

    For wide frames we trim to a representative subset of columns —
    ``k_cols`` non-sparse columns picked by absolute variance — so we
    don't blow past the per-request token cap on banking77-scale inputs
    (67 base cols + dozens generated downstream).
    """
    if rng is None:
        rng = random.Random()
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    df = df.reset_index(drop=True)

    if df.shape[1] > 20:
        # drop tf-idf columns (mostly zero) and trim by variance
        non_sparse = [c for c in df.columns if not c.startswith("tfidf_")]
        if len(non_sparse) > 20:
            try:
                sub = df[non_sparse].select_dtypes(include=["number"])
                top = sub.var().sort_values(ascending=False).head(20).index.tolist()
                non_sparse = top + [c for c in non_sparse if c not in top][: 20 - len(top)]
            except Exception:
                non_sparse = non_sparse[:20]
        df = df[non_sparse]

    n = min(k, len(df))
    df_out = pd.DataFrame(np.asarray(y), columns=["Result"]).reset_index(drop=True)
    joined = df.join(df_out)
    sampled = joined.sample(n=n, random_state=rng.randint(0, 10**6))
    return "\n".join(_serialize_row(r) for _, r in sampled.iterrows())


def load_prefix_suffix(kind: str, prompts_dir: str = "prompts") -> tuple[str, str]:
    if kind == "domain":
        return _read(os.path.join(prompts_dir, "domain_head.txt")), \
               _read(os.path.join(prompts_dir, "domain_tail.txt"))
    return _read(os.path.join(prompts_dir, "operations_head.txt")), \
           _read(os.path.join(prompts_dir, "operations_tail.txt"))
