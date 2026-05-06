"""Run a generated feature transform against the training fold.

The transform is a python function in the (already validated) candidate
program. We pass a dict `{"data": {...}}` and pull back ``(score, X_after,
y_after)``. Two safety levers:
  * the candidate is `exec`'d in an isolated globals dict that doesn't
    forward builtins like ``open`` or ``__import__`` to anything weird —
    we still need imports inside the candidate (pandas etc.) so we don't
    blanket-block them, but the namespace doesn't leak back into ours;
  * a hard wall-clock timeout enforced by a worker thread / SIGALRM hybrid.

SIGALRM only fires on the main thread in CPython, so we drive it through
``signal`` when available and fall back to threading.Timer + co-operative
checks otherwise. Most of our datasets are small (<10k rows) and the
transforms run sub-second on CPU.
"""

from __future__ import annotations

import signal
import threading
import traceback
from typing import Any, Tuple


class TimeoutError_(Exception):  # avoid clobbering builtin TimeoutError
    pass


def _install_alarm(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        return lambda: None
    def handler(signum, frame):
        raise TimeoutError_("transform exceeded time budget")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    def cancel():
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return cancel


def run_transform(
    program: str,
    function_to_run: str,
    function_to_evolve: str,
    inputs: dict,
    test_input: str,
    timeout_seconds: int = 30,
    verbose: bool = False,
) -> Tuple[Any, bool]:
    """Run ``function_to_run`` on ``inputs[test_input]`` from ``program``.

    Returns (result, ok). The result is whatever the spec's evaluator
    yields — typically (score, X, y).
    """
    if threading.current_thread() is threading.main_thread():
        cancel = _install_alarm(timeout_seconds)
    else:
        cancel = lambda: None

    ns: dict = {}
    try:
        exec(program, ns)
        fn = ns[function_to_run]
        result = fn(inputs[test_input])
        if not isinstance(result[0], (int, float)):
            return None, False
        return result, True
    except TimeoutError_:
        if verbose:
            print(f"[sandbox] timeout after {timeout_seconds}s")
        return None, False
    except Exception as e:
        if verbose:
            print(f"[sandbox] {type(e).__name__}: {e}")
            traceback.print_exc()
        return None, False
    finally:
        cancel()


def calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """True when the candidate cheats by calling an older version of itself."""
    from evofeat.codegen import functions_called
    for name in functions_called(program):
        if name.startswith(f"{function_to_evolve}_v"):
            return True
    return False
