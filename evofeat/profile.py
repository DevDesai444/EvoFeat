"""Tiny logger — writes one JSON line per evaluated candidate and tracks
the best-so-far curve. We deliberately avoid tensorboard / wandb to keep
the dependency footprint small; matplotlib reads the JSONL directly when
we want plots.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from evofeat import codegen


class Profiler:
    def __init__(self, log_dir: str):
        self._log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
        self._best_score: float = -float("inf")
        self._best_id: Optional[int] = None
        self._best_str: Optional[str] = None
        self._n_success = 0
        self._n_fail = 0
        self._tot_sample_s = 0.0
        self._tot_eval_s = 0.0
        self._seen: set[int] = set()
        self._curve_path = os.path.join(log_dir, "curve.jsonl")
        # truncate the curve file on construction so repeated runs don't
        # silently concatenate
        open(self._curve_path, "w").close()

    def register(self, prog: codegen.Function) -> None:
        idx = prog.global_sample_nums or 0
        if idx in self._seen:
            return
        self._seen.add(idx)

        score = prog.score
        text = str(prog).strip("\n")

        with open(os.path.join(self._log_dir, "samples", f"sample_{idx}.json"), "w") as f:
            json.dump({"sample_order": idx, "function": text, "score": score}, f)

        if score is not None:
            self._n_success += 1
            if score > self._best_score:
                self._best_score = score
                self._best_id = idx
                self._best_str = text
        else:
            self._n_fail += 1

        if prog.sample_time:
            self._tot_sample_s += prog.sample_time
        if prog.evaluate_time:
            self._tot_eval_s += prog.evaluate_time

        with open(self._curve_path, "a") as f:
            f.write(json.dumps({
                "i": idx,
                "score": score,
                "best": self._best_score,
                "ok": self._n_success,
                "fail": self._n_fail,
                "t_sample": self._tot_sample_s,
                "t_eval": self._tot_eval_s,
            }) + "\n")

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_program(self) -> Optional[str]:
        return self._best_str
