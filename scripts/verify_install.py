"""Quick sanity check that the pipeline can run end-to-end.

  python scripts/verify_install.py

Walks through:

  1. import every evofeat module
  2. confirm GROQ_API_KEY is set and the cloud endpoint is reachable
  3. load banking77 features (or print build instructions if missing)
  4. run one classical baseline on a small fold
  5. exec one LLM-generated program through the sandbox

Useful as a quick smoke after a fresh clone before launching the full run.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def step(name: str):
    def deco(fn):
        def inner(*a, **kw):
            t0 = time.time()
            print(f"[{name}] …", flush=True)
            try:
                fn(*a, **kw)
            except Exception as e:
                print(f"[{name}] FAIL ({time.time() - t0:.1f}s): {e}")
                traceback.print_exc()
                return False
            print(f"[{name}] ok ({time.time() - t0:.1f}s)")
            return True
        return inner
    return deco


@step("import")
def _imports():
    import evofeat  # noqa
    import evofeat.baselines  # noqa
    import evofeat.data, evofeat.evaluate, evofeat.stats, evofeat.search  # noqa
    import evofeat.shap_analysis, evofeat.cost, evofeat.logger, evofeat.reporting  # noqa
    import evofeat.llm_builder  # noqa


@step("env")
def _env():
    if not os.environ.get("GROQ_API_KEY"):
        # try .env
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip()
    assert os.environ.get("GROQ_API_KEY"), \
        "GROQ_API_KEY not set; copy .env.example to .env and fill it in"


@step("groq-reach")
def _groq_reach():
    from evofeat.llm import GroqClient
    c = GroqClient(model="gpt-oss-20b", max_tokens=16)
    out = c.chat("You are a helper.", "Reply with the single word: ok")
    assert out.strip().lower().startswith("ok"), f"unexpected response: {out!r}"


@step("banking77-load")
def _banking77():
    from evofeat.data import load_banking77
    ds = load_banking77()
    assert ds.n_rows > 10_000 and ds.n_classes == 77, "banking77 shape unexpected"
    print(f"      {ds.n_rows} rows × {ds.n_cols} cols, {ds.n_classes} classes, {len(ds.splits)} folds")


@step("baseline-smoke")
def _baseline_smoke():
    from evofeat.baselines import all_classification_baselines
    from evofeat.data import load_banking77
    from evofeat.evaluate import evaluate_classification
    ds = load_banking77()
    # one fold, one method, one model
    splits = [ds.splits[0]]
    builder = all_classification_baselines(k=15)["fisher"]
    res = evaluate_classification(ds.X, ds.y, splits, builder=builder, method_name="fisher", models=("xgb",))
    fold = res[0].folds[0]
    print(f"      fisher → xgb fold-0 accuracy={fold.metrics['accuracy']:.4f}")


@step("sandbox-exec")
def _sandbox():
    from evofeat.sandbox import run_transform
    program = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def modify_features(df):\n"
        "    df = df.copy()\n"
        "    df['n_tokens_sq'] = df['n_tokens'] ** 2\n"
        "    return df\n"
        "def evaluate(data):\n"
        "    X = modify_features(data['inputs'])\n"
        "    return float(X['n_tokens_sq'].mean()), X, data['outputs']\n"
    )
    from evofeat.data import load_banking77
    ds = load_banking77()
    payload = {"data": {"inputs": ds.X.iloc[:100], "outputs": ds.y[:100]}}
    res, ok = run_transform(program, "evaluate", "modify_features", payload, "data")
    assert ok, "sandbox said the trivial program failed"
    print(f"      mean(n_tokens²) on first 100 rows = {res[0]:.2f}")


def main():
    ok = True
    for fn in (_imports, _env, _groq_reach, _banking77, _baseline_smoke, _sandbox):
        ok = fn() and ok
    if not ok:
        sys.exit(1)
    print("\nverify_install: all checks passed")


if __name__ == "__main__":
    main()
