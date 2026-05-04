"""Dataset loaders + small helpers.

Two paths into a ``Dataset``:

  * ``load_dataset(name)`` — UCI-style CSV + metadata-json pair under
    ``data/{name}.csv``. Target is the last column.
  * ``load_banking77()``   — the parquet + splits produced by
    ``evofeat.datasets.banking77.build()``.

Both return the same in-memory shape so downstream code (the search loop,
the baselines, the stat-test module) doesn't branch on dataset origin.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


REGRESSION_DATASETS = {
    "bike", "crab", "forest-fires", "housing", "insurance", "wine",
}


def is_categorical(s: pd.Series) -> bool:
    s = s.convert_dtypes()
    if is_string_dtype(s):
        return True
    vals = set(s.dropna().unique().tolist())
    if vals.issubset({0, 1}):
        return True
    if s.dtype in (int, float, "Int64", "Float64"):
        return False
    return True


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    is_regression: bool
    is_cat: List[bool]
    target_name: str
    meta: Dict
    label_names: Optional[List[str]] = None
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def n_cols(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return 0 if self.is_regression else int(len(np.unique(self.y)))


def load_dataset(name: str, data_dir: str = "data") -> Dataset:
    """UCI-style CSV loader (target = last column)."""
    csv_path = os.path.join(data_dir, f"{name}.csv")
    meta_path = os.path.join(data_dir, f"{name}-metadata.json")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    target_col = df.columns[-1]
    is_reg = name in REGRESSION_DATASETS

    X = df.drop(columns=[target_col]).convert_dtypes()
    y_raw = df[target_col].to_numpy()

    is_cat = [is_categorical(X[c]) for c in X.columns]

    label_names: Optional[List[str]] = None
    if not is_reg and y_raw.dtype.kind in ("O", "U", "S"):
        labels = pd.Series(y_raw).astype(str)
        label_names = sorted(labels.unique().tolist())
        idx = {v: i for i, v in enumerate(label_names)}
        y = np.array([idx[v] for v in labels.tolist()], dtype=np.int64)
    else:
        y = np.asarray(y_raw, dtype=float if is_reg else np.int64)

    meta: Dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                meta = {}

    return Dataset(
        name=name, X=X, y=y, is_regression=is_reg, is_cat=is_cat,
        target_name=target_col, meta=meta, label_names=label_names,
    )


def load_banking77(data_dir: str = "data/banking77") -> Dataset:
    """Load the parquet + splits file produced by the prep script."""
    parquet = os.path.join(data_dir, "features.parquet")
    splits_path = os.path.join(data_dir, "splits.npz")
    meta_path = os.path.join(data_dir, "metadata.json")
    for p in (parquet, splits_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing — run `python -m evofeat.datasets.banking77 --build`"
            )

    feats = pd.read_parquet(parquet)
    with open(meta_path) as f:
        meta = json.load(f)
    raw = np.load(splits_path)
    splits = [(raw[f"fold_{i}_train"], raw[f"fold_{i}_test"]) for i in range(meta["n_splits"])]

    feature_cols = meta["feature_columns"]
    X = feats[feature_cols].copy()
    y = feats[meta["target_column"]].to_numpy()
    label_names = sorted(feats[meta["label_column"]].unique().tolist()) if meta.get("label_column") else None
    is_cat = [is_categorical(X[c]) for c in X.columns]
    return Dataset(
        name="banking77", X=X, y=y, is_regression=False, is_cat=is_cat,
        target_name=meta["target_column"], meta=meta,
        label_names=label_names, splits=splits,
    )


def stratified_or_kfold_splits(
    y: np.ndarray, is_regression: bool, n_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold, StratifiedKFold
    if is_regression:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.zeros(len(y))))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(len(y)), y))
