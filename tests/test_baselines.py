"""Smoke tests on every classical baseline.

We don't try to verify accuracy here — just that every builder accepts a
small frame and returns a non-empty (X_train', X_test') with consistent
columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evofeat.baselines import all_classification_baselines


def _toy_data(n: int = 80, p: int = 12, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    # carve out a signal: 4 informative features
    coef = rng.normal(size=k)
    z = X.iloc[:, :k].to_numpy() @ coef
    y = (z + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X.iloc[:60], X.iloc[60:].reset_index(drop=True), y[:60], y[60:]


def test_all_classification_baselines_smoke():
    X_tr, X_te, y_tr, _ = _toy_data()
    builders = all_classification_baselines(k=6)
    assert builders
    for name, builder in builders.items():
        X_tr_new, X_te_new = builder(X_tr, y_tr, X_te)
        assert X_tr_new.shape[0] == X_tr.shape[0], name
        assert X_te_new.shape[0] == X_te.shape[0], name
        assert list(X_tr_new.columns) == list(X_te_new.columns), name
        assert X_tr_new.shape[1] > 0, name
