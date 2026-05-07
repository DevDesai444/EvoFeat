"""Train/test consistent encoding for categorical columns.

Maps each categorical column to integer codes using a mapping fit on train.
Test rows with unseen categories get -1 (a sentinel xgboost handles fine).
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


Mapping = Dict[str, Dict[object, int]]


def _build_mappings(df: pd.DataFrame) -> Mapping:
    mappings: Mapping = {}
    for col in df.columns:
        name = df[col].dtype.name
        if name == "category" or name == "object" or name == "string":
            cats = pd.Series(df[col]).astype("category").cat.categories
            mappings[col] = {v: i for i, v in enumerate(cats)}
    return mappings


def _apply_column(col: pd.Series, mapping: Optional[Dict[object, int]]) -> pd.Series:
    if mapping is None:
        return col
    if col.dtype.name == "category":
        col = col.astype(object)
    return col.map(mapping).fillna(-1).astype(int)


def encode_categoricals(
    df_train: pd.DataFrame, df_test: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Mapping]:
    df_train = copy.deepcopy(df_train).infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test).infer_objects()

    mappings = _build_mappings(df_train)

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        out = df.apply(lambda c: _apply_column(c, mappings.get(c.name)), axis=0)
        return out.astype(float)

    df_train = _apply(df_train)
    if df_test is not None:
        df_test = _apply(df_test)

    return df_train, df_test, mappings


def safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce everything to float, fill NaNs with 0, replace inf."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if out[c].dtype == object or out[c].dtype.name == "category":
            out[c] = pd.factorize(out[c])[0]
    out = out.fillna(0.0).astype(float)
    return out
