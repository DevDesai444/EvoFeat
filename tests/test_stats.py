"""Statistical-tests sanity checks."""

from __future__ import annotations

import numpy as np

from evofeat.stats import (
    all_pairs_table, bootstrap_ci, cohen_d_paired, paired_compare,
    summarize_scores, vs_reference_table,
)


def test_paired_compare_recovers_effect_direction():
    rng = np.random.default_rng(0)
    a = rng.normal(0.85, 0.01, size=5)
    b = rng.normal(0.80, 0.01, size=5)
    r = paired_compare(a, b, "a", "b")
    assert r.delta > 0
    assert r.sig_uncorrected is True


def test_paired_compare_identical_inputs_yields_no_effect():
    a = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    r = paired_compare(a, a, "a", "a")
    assert r.delta == 0.0
    assert r.cohen_d == 0.0
    assert r.sig_uncorrected is False


def test_cohen_d_zero_variance_is_zero():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert cohen_d_paired(a, b) == 0.0


def test_bootstrap_ci_brackets_zero_when_identical():
    a = np.linspace(0, 1, 5)
    lo, hi = bootstrap_ci(a, a, n_boot=500, seed=1)
    assert lo == 0.0 and hi == 0.0


def test_all_pairs_and_vs_reference():
    scores = {
        "base": [0.80, 0.81, 0.79, 0.82, 0.80],
        "mi":   [0.83, 0.82, 0.83, 0.84, 0.82],
        "llm":  [0.88, 0.87, 0.89, 0.88, 0.87],
    }
    summary = summarize_scores(scores)
    assert list(summary["method"]) == ["llm", "mi", "base"]
    pairs = all_pairs_table(scores)
    assert len(pairs) == 3
    vs = vs_reference_table(scores, "base")
    assert set(vs["method_a"]) == {"mi", "llm"}
    # mi vs base should be significant uncorrected
    mi_row = vs.set_index("method_a").loc["mi"]
    assert bool(mi_row["sig_p<.05"]) is True
