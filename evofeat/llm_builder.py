"""Wrap a saved LLM-generated feature program as a ``FeatureBuilder``.

The search loop persists the best program text as a string. To put that
program on the same axis as the classical baselines we need to apply
it consistently on (X_train, X_test) inside the canonical eval harness.

This module exec's the program in an isolated namespace, locates the
``modify_features`` function, and returns a builder that calls it on
both halves of every fold.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import numpy as np
import pandas as pd


log = logging.getLogger("evofeat.llm_builder")


def _load_program(program_text: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    ns: dict = {}
    # the spec wraps modify_features in a fully-formed module; exec'ing
    # the program string is enough to materialize all of it
    exec(program_text, ns)
    # the program file may carry a versioned name (modify_features_v0…) —
    # always look for the canonical name first, otherwise pick the most
    # recent suffix
    if "modify_features" in ns:
        return ns["modify_features"]
    candidates = [k for k in ns if k.startswith("modify_features")]
    if not candidates:
        raise ValueError("no modify_features function in saved program")
    candidates.sort()
    return ns[candidates[-1]]


def llm_program_builder(program_text: str):
    """Return a FeatureBuilder that applies the saved program to both halves."""
    fn = _load_program(program_text)

    def build(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X_tr_new = fn(X_tr.copy())
            X_te_new = fn(X_te.copy())
        except Exception as e:
            log.warning("LLM program failed during application: %s — falling back to input", e)
            return X_tr.copy(), X_te.copy()

        # coerce + align columns: programs occasionally add cols that depend
        # on train-only state (rare, but happens with mean-encoded LLM
        # outputs); intersect column sets so the harness gets a square frame
        common = [c for c in X_tr_new.columns if c in X_te_new.columns]
        if len(common) < X_tr_new.shape[1]:
            log.debug("LLM produced %d train-only cols; dropping",
                      X_tr_new.shape[1] - len(common))
        X_tr_new = X_tr_new[common]
        X_te_new = X_te_new[common]
        return X_tr_new, X_te_new

    return build
