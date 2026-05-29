from __future__ import annotations

import numpy as np
import pandas as pd

from evofeat.llm_builder import llm_program_builder


PROGRAM_TEXT = """
import pandas as pd
import numpy as np

def modify_features(df):
    df = df.copy()
    df['x_sq'] = df['x'] ** 2
    df['x_plus_y'] = df['x'] + df['y']
    return df
"""


def test_builder_round_trip():
    builder = llm_program_builder(PROGRAM_TEXT)
    X_tr = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, 0.2, 0.1]})
    X_te = pd.DataFrame({"x": [4.0, 5.0], "y": [0.3, 0.0]})
    out_tr, out_te = builder(X_tr, np.array([0, 1, 0]), X_te)
    assert "x_sq" in out_tr.columns and "x_plus_y" in out_tr.columns
    assert out_tr.loc[0, "x_sq"] == 1.0
    assert out_te.loc[0, "x_plus_y"] == 4.3
