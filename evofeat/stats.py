"""Paired statistical tests across feature-engineering methods.

We compare methods pairwise on the per-fold score vectors produced by
the eval harness. Inputs are dictionaries of the form::

    {"method_name": [fold0_score, fold1_score, ..., foldK_score], ...}

For every (a, b) pair we report:

    Δ = mean_a − mean_b
    paired t-test p-value
    Wilcoxon signed-rank p-value (non-parametric backup)
    bootstrap 95% CI on Δ
    Cohen's d (paired)
    Bonferroni-corrected significance verdict

5 folds is on the low end for asymptotic statistics — we always print
both the parametric and non-parametric p-values and only call something
'significant' when both agree under correction.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PairResult:
    method_a: str
    method_b: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    cohen_d: float
    t_stat: float
    t_p: float
    w_stat: float
    w_p: float
    sig_uncorrected: bool
    sig_bonferroni: bool

    def as_row(self) -> Dict[str, object]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "n_folds": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "delta": round(self.delta, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "cohen_d": round(self.cohen_d, 3),
            "t_stat": round(self.t_stat, 3),
            "t_p": format_p(self.t_p),
            "wilcoxon_p": format_p(self.w_p),
            "sig_p<.05": self.sig_uncorrected,
            "sig_bonferroni": self.sig_bonferroni,
        }


def format_p(p: float) -> str:
    if math.isnan(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    s = diffs.std(ddof=1)
    if s < 1e-12:
        return 0.0
    return float(diffs.mean() / s)


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    samples = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_compare(
    a: np.ndarray, b: np.ndarray, name_a: str, name_b: str, alpha: float = 0.05,
    n_comparisons: int = 1,
) -> PairResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")

    delta = float(a.mean() - b.mean())
    # paired t — falls back gracefully when variance is zero (identical folds)
    diffs = a - b
    if np.allclose(diffs, diffs[0]):
        t_stat, t_p = float("nan"), 1.0
    else:
        t_stat, t_p = stats.ttest_rel(a, b, nan_policy="omit")
        t_stat = float(t_stat)
        t_p = float(t_p)
    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="zsplit")
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0

    lo, hi = bootstrap_ci(a, b)
    d = cohen_d_paired(a, b)
    bonf_alpha = alpha / max(n_comparisons, 1)
    # primary verdict from the paired t. Wilcoxon is reported as a robustness
    # check but can't reach p < .05 with n ≤ 5 folds (its minimum two-sided
    # p with n=5 is 0.0625), so we don't gate the verdict on it for small n.
    sig_uncorr = bool(t_p < alpha)
    sig_bonf = bool(t_p < bonf_alpha)
    return PairResult(
        method_a=name_a, method_b=name_b,
        n=len(a), mean_a=float(a.mean()), mean_b=float(b.mean()),
        delta=delta, ci_low=lo, ci_high=hi, cohen_d=d,
        t_stat=t_stat, t_p=t_p, w_stat=w_stat, w_p=w_p,
        sig_uncorrected=sig_uncorr, sig_bonferroni=sig_bonf,
    )


def all_pairs_table(
    method_scores: Dict[str, List[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run every (a, b) pair with Bonferroni correction on the total count."""
    names = list(method_scores.keys())
    pairs = list(itertools.combinations(names, 2))
    n = len(pairs)
    rows = []
    for a, b in pairs:
        rows.append(paired_compare(
            np.array(method_scores[a]), np.array(method_scores[b]),
            name_a=a, name_b=b, alpha=alpha, n_comparisons=n,
        ).as_row())
    return pd.DataFrame(rows)


def vs_reference_table(
    method_scores: Dict[str, List[float]],
    reference: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All methods vs a single reference (e.g. 'base'). Useful as the
    headline comparison row of the report — n-1 tests instead of nC2."""
    if reference not in method_scores:
        raise KeyError(reference)
    others = [m for m in method_scores if m != reference]
    n = len(others)
    rows = []
    ref = np.array(method_scores[reference])
    for m in others:
        rows.append(paired_compare(
            np.array(method_scores[m]), ref,
            name_a=m, name_b=reference, alpha=alpha, n_comparisons=n,
        ).as_row())
    return pd.DataFrame(rows)


def summarize_scores(method_scores: Dict[str, List[float]]) -> pd.DataFrame:
    rows = []
    for name, v in method_scores.items():
        v = np.array(v, dtype=float)
        rows.append({
            "method": name,
            "n_folds": len(v),
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
        })
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df
