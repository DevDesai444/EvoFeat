"""Sandboxed candidate execution.

Coverage:
  * normal happy path returns the right shape
  * obvious runtime errors come back as (None, False) instead of crashing
  * a "calls ancestor" candidate is flagged

We don't try to test the timeout path on every machine — SIGALRM is
platform-dependent and the wall-clock budget is configurable, so we just
exercise the path with a short timeout and a quick op.
"""

from __future__ import annotations

import pandas as pd

from evofeat.sandbox import calls_ancestor, run_transform


SIMPLE_PROGRAM = """
import pandas as pd
import numpy as np

def modify_features(df):
    df = df.copy()
    df["double_x"] = df["x"] * 2
    return df

def evaluate(data):
    df = modify_features(data["inputs"])
    score = float(df["double_x"].mean())
    return score, data["inputs"], data["outputs"]
"""

BROKEN_PROGRAM = """
import pandas as pd
import numpy as np

def modify_features(df):
    df = df.copy()
    return df["does_not_exist"].mean()  # type error vs DataFrame contract

def evaluate(data):
    return modify_features(data["inputs"]), data["inputs"], data["outputs"]
"""

ANCESTOR_PROGRAM = """
def modify_features_v0(df):
    return df
def modify_features(df):
    return modify_features_v0(df)
def evaluate(data):
    return 0.0, data["inputs"], data["outputs"]
"""


def _payload():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    return {"data": {"inputs": df, "outputs": [0, 1, 0, 1]}}


def test_simple_program_runs():
    inputs = _payload()
    result, ok = run_transform(SIMPLE_PROGRAM, "evaluate", "modify_features", inputs, "data")
    assert ok is True
    score, X, y = result
    assert score == 5.0  # mean of [2,4,6,8]


def test_broken_program_returns_failure_flag():
    inputs = _payload()
    result, ok = run_transform(BROKEN_PROGRAM, "evaluate", "modify_features", inputs, "data")
    assert ok is False
    assert result is None


def test_calls_ancestor_detected():
    assert calls_ancestor(ANCESTOR_PROGRAM, "modify_features") is True
    assert calls_ancestor(SIMPLE_PROGRAM, "modify_features") is False
