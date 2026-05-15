"""Downstream evaluation harness.

5-fold CV across XGBoost / LogReg / RandomForest for classification and
XGBoost / Ridge / RandomForest for regression. The harness takes a
"feature builder" — a callable ``(X_train, y_train, X_test) -> (X_train',
X_test')`` — so the same code paths back the raw baseline, the classical
baselines, and the LLM-generated transforms.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, median_absolute_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from evofeat.preprocess import encode_categoricals, safe_numeric_frame


FeatureBuilder = Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame],
]


@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]
    n_features_in: int
    n_features_out: int
    fit_seconds: float


@dataclass
class ModelResult:
    model: str
    feature_method: str
    folds: List[FoldResult] = field(default_factory=list)

    def aggregate(self) -> Dict[str, Tuple[float, float]]:
        keys = self.folds[0].metrics.keys()
        out = {}
        for k in keys:
            vs = [f.metrics[k] for f in self.folds]
            out[k] = (float(np.mean(vs)), float(np.std(vs)))
        return out


def identity_builder(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame):
    return X_tr.copy(), X_te.copy()


def _classifier(name: str, n_classes: int, seed: int = 42):
    # multi-class with n_classes ≥ 50 is dominated by the per-class fit cost
    # (logreg) and per-tree work (rf); the multipliers below kick in only
    # then to keep runtime reasonable on cpu.
    big = n_classes >= 50
    if name == "xgb":
        return xgb.XGBClassifier(
            random_state=seed,
            n_estimators=120 if big else 200,
            max_depth=6, learning_rate=0.1, n_jobs=2,
            eval_metric="mlogloss", tree_method="hist", verbosity=0,
        )
    if name == "logreg":
        return LogisticRegression(
            max_iter=800 if big else 2000,
            n_jobs=2, random_state=seed,
            solver="lbfgs", tol=1e-3 if big else 1e-4,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=150 if big else 300,
            random_state=seed, n_jobs=2, min_samples_leaf=2,
        )
    raise ValueError(name)


def _regressor(name: str, seed: int = 42):
    if name == "xgb":
        return xgb.XGBRegressor(
            random_state=seed, n_estimators=300, max_depth=6,
            learning_rate=0.1, n_jobs=1, tree_method="hist", verbosity=0,
        )
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=1, min_samples_leaf=2,
        )
    raise ValueError(name)


CLF_MODELS = ("xgb", "logreg", "rf")
REG_MODELS = ("xgb", "ridge", "rf")


def _safe_auc(y_true, proba, n_classes) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _scale_if_needed(model_name: str, X_tr: np.ndarray, X_te: np.ndarray):
    if model_name in ("logreg", "ridge"):
        sc = StandardScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te)
    return X_tr, X_te


def evaluate_classification(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    builder: FeatureBuilder = identity_builder,
    method_name: str = "base",
    models: Tuple[str, ...] = CLF_MODELS,
    seed: int = 42,
) -> List[ModelResult]:
    results = [ModelResult(model=m, feature_method=method_name) for m in models]
    n_classes = int(len(np.unique(y)))

    for fold_idx, (tr, te) in enumerate(splits):
        X_tr_raw = X.iloc[tr].reset_index(drop=True)
        X_te_raw = X.iloc[te].reset_index(drop=True)
        y_tr, y_te = y[tr], y[te]

        X_tr_new, X_te_new = builder(X_tr_raw, y_tr, X_te_raw)
        X_tr_enc, X_te_enc, _ = encode_categoricals(X_tr_new, X_te_new)
        X_tr_enc = safe_numeric_frame(X_tr_enc)
        X_te_enc = safe_numeric_frame(X_te_enc)
        # align columns (LLM-built features can shuffle)
        common = [c for c in X_tr_enc.columns if c in X_te_enc.columns]
        X_tr_enc = X_tr_enc[common]
        X_te_enc = X_te_enc[common]

        X_tr_np = X_tr_enc.to_numpy()
        X_te_np = X_te_enc.to_numpy()

        for r in results:
            X_tr_m, X_te_m = _scale_if_needed(r.model, X_tr_np, X_te_np)
            t0 = time.time()
            model = _classifier(r.model, n_classes, seed=seed + fold_idx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_m, y_tr)
            fit_s = time.time() - t0
            y_pred = model.predict(X_te_m)
            try:
                proba = model.predict_proba(X_te_m)
            except Exception:
                proba = np.zeros((len(y_te), n_classes))

            metrics = {
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
                "auc":      _safe_auc(y_te, proba, n_classes),
            }
            try:
                metrics["log_loss"] = float(log_loss(y_te, proba, labels=list(range(n_classes))))
            except Exception:
                metrics["log_loss"] = float("nan")

            r.folds.append(FoldResult(
                fold=fold_idx, metrics=metrics,
                n_features_in=X_tr_raw.shape[1], n_features_out=X_tr_enc.shape[1],
                fit_seconds=fit_s,
            ))
    return results


def evaluate_regression(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    builder: FeatureBuilder = identity_builder,
    method_name: str = "base",
    models: Tuple[str, ...] = REG_MODELS,
    seed: int = 42,
) -> List[ModelResult]:
    results = [ModelResult(model=m, feature_method=method_name) for m in models]
    for fold_idx, (tr, te) in enumerate(splits):
        X_tr_raw = X.iloc[tr].reset_index(drop=True)
        X_te_raw = X.iloc[te].reset_index(drop=True)
        y_tr, y_te = y[tr], y[te]

        X_tr_new, X_te_new = builder(X_tr_raw, y_tr, X_te_raw)
        X_tr_enc, X_te_enc, _ = encode_categoricals(X_tr_new, X_te_new)
        X_tr_enc = safe_numeric_frame(X_tr_enc)
        X_te_enc = safe_numeric_frame(X_te_enc)
        common = [c for c in X_tr_enc.columns if c in X_te_enc.columns]
        X_tr_enc = X_tr_enc[common]
        X_te_enc = X_te_enc[common]

        X_tr_np = X_tr_enc.to_numpy()
        X_te_np = X_te_enc.to_numpy()

        for r in results:
            X_tr_m, X_te_m = _scale_if_needed(r.model, X_tr_np, X_te_np)
            t0 = time.time()
            model = _regressor(r.model, seed=seed + fold_idx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_m, y_tr)
            fit_s = time.time() - t0
            y_pred = model.predict(X_te_m)

            metrics = {
                "rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred))),
                "mae":   float(mean_absolute_error(y_te, y_pred)),
                "r2":    float(r2_score(y_te, y_pred)),
                "medae": float(median_absolute_error(y_te, y_pred)),
            }
            r.folds.append(FoldResult(
                fold=fold_idx, metrics=metrics,
                n_features_in=X_tr_raw.shape[1], n_features_out=X_tr_enc.shape[1],
                fit_seconds=fit_s,
            ))
    return results


def primary_score(results: List[ModelResult], is_regression: bool, prefer_model: str = "xgb") -> float:
    """The score we report as 'the' number for the method.

    XGBoost on accuracy (clf) or RMSE-negated (reg) by default.
    """
    for r in results:
        if r.model == prefer_model:
            agg = r.aggregate()
            if is_regression:
                return -agg["rmse"][0]
            return agg["accuracy"][0]
    return float("nan")
