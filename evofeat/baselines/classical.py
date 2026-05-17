"""The seven classical baselines we compare against the LLM features.

All builders fit on ``X_train`` (and ``y_train`` when supervised), then
project ``X_test`` through whatever they learnt — no peeking. Each
returns ``(X_train', X_test')`` as DataFrames with named columns; the
downstream eval harness handles encoding + scaling.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression, RFE,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from evofeat.preprocess import safe_numeric_frame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_n = safe_numeric_frame(X_train)
    X_test_n = safe_numeric_frame(X_test)
    common = [c for c in X_train_n.columns if c in X_test_n.columns]
    return X_train_n[common], X_test_n[common]


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    return np.argsort(-scores)[:k]


# ---------------------------------------------------------------------------
# fisher-style / univariate selection
# ---------------------------------------------------------------------------

def fisher_score_builder(k: int = 30):
    """Class-separability via the (per-feature) Fisher discriminant ratio.

    For continuous features we approximate Fisher score by the ratio of
    between-class variance to within-class variance — equivalent to ANOVA
    F when classes have equal n, but more robust on imbalanced 77-class
    targets where sklearn's f_classif can underflow.
    """
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        cols = X_tr_n.columns.tolist()
        Xn = X_tr_n.to_numpy()
        y = np.asarray(y_tr)
        classes = np.unique(y)
        overall_mean = Xn.mean(axis=0)
        between = np.zeros(Xn.shape[1])
        within  = np.zeros(Xn.shape[1])
        for c in classes:
            mask = (y == c)
            if mask.sum() < 2:
                continue
            mu_c = Xn[mask].mean(axis=0)
            var_c = Xn[mask].var(axis=0)
            n_c = mask.sum()
            between += n_c * (mu_c - overall_mean) ** 2
            within  += n_c * var_c
        score = between / (within + 1e-9)
        keep = _topk(score, k)
        kept_cols = [cols[i] for i in keep]
        return X_tr_n[kept_cols], X_te_n[kept_cols]
    return build


def anova_f_builder(k: int = 30):
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        is_clf = (np.asarray(y_tr).dtype.kind in ('i', 'b'))
        score_fn = f_classif if is_clf else f_regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = SelectKBest(score_fn, k=min(k, X_tr_n.shape[1]))
            sel.fit(X_tr_n, y_tr)
        kept = X_tr_n.columns[sel.get_support()].tolist()
        return X_tr_n[kept], X_te_n[kept]
    return build


def mutual_info_builder(k: int = 30, seed: int = 42):
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        is_clf = (np.asarray(y_tr).dtype.kind in ('i', 'b'))
        score_fn = (
            (lambda X, y: mutual_info_classif(X, y, random_state=seed))
            if is_clf else
            (lambda X, y: mutual_info_regression(X, y, random_state=seed))
        )
        sel = SelectKBest(score_fn, k=min(k, X_tr_n.shape[1]))
        sel.fit(X_tr_n, y_tr)
        kept = X_tr_n.columns[sel.get_support()].tolist()
        return X_tr_n[kept], X_te_n[kept]
    return build


# ---------------------------------------------------------------------------
# regularization-driven selection
# ---------------------------------------------------------------------------

def lasso_l1_builder(alpha: float = 0.05, k: int = 30, seed: int = 42):
    """L1 selection via Lasso (regression) or L1-penalized LogReg
    (classification). We pick the top-k by |coef| magnitude rather than
    'nonzero coefficients' so different alphas don't blow up the feature
    count comparison.

    For multi-class >= 50 classes, ``saga`` is prohibitively slow on
    cpu; we fall back to ``liblinear`` (OvR) which fits much faster at
    the cost of slightly less accurate per-class coef magnitudes —
    fine for ranking-and-selection.
    """
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr_n.to_numpy())
        is_clf = (np.asarray(y_tr).dtype.kind in ('i', 'b'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_clf:
                n_classes = int(np.unique(y_tr).size)
                if n_classes >= 50:
                    # sklearn ≥ 1.7 dropped implicit OvR on liblinear, so
                    # we wrap manually; that's faster than saga on a
                    # 77-class problem anyway
                    from sklearn.multiclass import OneVsRestClassifier
                    base = LogisticRegression(
                        penalty="l1", solver="liblinear",
                        C=1.0 / max(alpha, 1e-6),
                        max_iter=200, random_state=seed,
                    )
                    mdl = OneVsRestClassifier(base, n_jobs=2)
                    mdl.fit(Xs, y_tr)
                    # coef_ is per-class; collapse to mean abs
                    coefs = np.stack([est.coef_.ravel() for est in mdl.estimators_])
                    coef = np.abs(coefs).mean(axis=0)
                else:
                    mdl = LogisticRegression(
                        penalty="l1", solver="saga",
                        C=1.0 / max(alpha, 1e-6),
                        max_iter=200, n_jobs=2, random_state=seed,
                        tol=5e-3,
                    )
                    mdl.fit(Xs, y_tr)
                    coef = np.abs(mdl.coef_).mean(axis=0)
            else:
                mdl = Lasso(alpha=alpha, max_iter=2000, random_state=seed)
                mdl.fit(Xs, y_tr)
                coef = np.abs(mdl.coef_)
        keep = _topk(coef, k)
        kept = X_tr_n.columns[keep].tolist()
        return X_tr_n[kept], X_te_n[kept]
    return build


# ---------------------------------------------------------------------------
# variance + RFE
# ---------------------------------------------------------------------------

def variance_threshold_builder(threshold: float = 1e-3):
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(X_tr_n)
        kept = X_tr_n.columns[sel.get_support()].tolist()
        if not kept:
            # don't drop everything — fall back to original frame
            return X_tr_n, X_te_n
        return X_tr_n[kept], X_te_n[kept]
    return build


def rfe_xgboost_builder(k: int = 30, n_estimators: int = 80, seed: int = 42):
    """Recursive feature elimination using XGBoost as the importance source.

    XGBoost is much faster than sklearn's GBM here, and sklearn's RFE
    composes with anything that exposes ``coef_`` / ``feature_importances_``.
    """
    def build(X_tr, y_tr, X_te):
        import xgboost as xgb
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        is_clf = (np.asarray(y_tr).dtype.kind in ('i', 'b'))
        est = (
            xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                tree_method="hist", random_state=seed, n_jobs=1, verbosity=0,
                eval_metric="mlogloss",
            )
            if is_clf else
            xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                tree_method="hist", random_state=seed, n_jobs=1, verbosity=0,
            )
        )
        target_n = min(k, X_tr_n.shape[1])
        # for the 77-class case sklearn's RFE is slow; we do a single
        # importance-rank pass which is what RFE collapses to with step=full
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_tr_n, y_tr)
        importances = np.asarray(est.feature_importances_)
        keep = _topk(importances, target_n)
        kept = X_tr_n.columns[keep].tolist()
        return X_tr_n[kept], X_te_n[kept]
    return build


# ---------------------------------------------------------------------------
# automated polynomial features
# ---------------------------------------------------------------------------

def polynomial_builder(degree: int = 2, interaction_only: bool = False, k: int = 30, seed: int = 42):
    """Polynomial / interaction features, then top-k by ANOVA score.

    The full degree-2 expansion on Banking77's base features blows the
    feature count past ~3k — we shrink back to ``k`` after expansion so
    the comparison row stays on the same axis as the other baselines.
    """
    def build(X_tr, y_tr, X_te):
        X_tr_n, X_te_n = _to_numeric(X_tr, X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly = PolynomialFeatures(
                degree=degree, interaction_only=interaction_only,
                include_bias=False,
            )
            P_tr = poly.fit_transform(X_tr_n.to_numpy())
            P_te = poly.transform(X_te_n.to_numpy())
        cols = [c.replace(" ", "_x_") for c in poly.get_feature_names_out(X_tr_n.columns)]
        Df_tr = pd.DataFrame(P_tr, columns=cols, index=X_tr_n.index)
        Df_te = pd.DataFrame(P_te, columns=cols, index=X_te_n.index)
        # top-k by anova-F
        is_clf = (np.asarray(y_tr).dtype.kind in ('i', 'b'))
        score_fn = f_classif if is_clf else f_regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = SelectKBest(score_fn, k=min(k, Df_tr.shape[1]))
            sel.fit(Df_tr, y_tr)
        kept = Df_tr.columns[sel.get_support()].tolist()
        return Df_tr[kept], Df_te[kept]
    return build


# ---------------------------------------------------------------------------
# combined: variance prune → MI → top-k. The "best traditional pipeline"
# baseline we report as a competitive single number against the LLM rows.
# ---------------------------------------------------------------------------

def combined_classical_builder(k: int = 30, seed: int = 42):
    def build(X_tr, y_tr, X_te):
        vt = variance_threshold_builder()
        Xtr1, Xte1 = vt(X_tr, y_tr, X_te)
        mi = mutual_info_builder(k=k, seed=seed)
        return mi(Xtr1, y_tr, Xte1)
    return build


# ---------------------------------------------------------------------------
# registries used by the run scripts
# ---------------------------------------------------------------------------

BASELINES = {
    "fisher":           lambda k=30: fisher_score_builder(k=k),
    "anova_f":          lambda k=30: anova_f_builder(k=k),
    "mutual_info":      lambda k=30: mutual_info_builder(k=k),
    "lasso_l1":         lambda k=30: lasso_l1_builder(k=k),
    "variance_thresh":  lambda k=30: variance_threshold_builder(),
    "rfe_xgb":          lambda k=30: rfe_xgboost_builder(k=k),
    "polynomial":       lambda k=30: polynomial_builder(k=k),
    "combined":         lambda k=30: combined_classical_builder(k=k),
}

# regression list — variance_thresh + mutual_info still work; lasso_l1 already
# branches; polynomial/anova/RFE branch on y dtype too.
BASELINES_REG = dict(BASELINES)


def all_classification_baselines(k: int = 30):
    return {name: factory(k=k) for name, factory in BASELINES.items()}


def all_regression_baselines(k: int = 30):
    return {name: factory(k=k) for name, factory in BASELINES_REG.items()}
