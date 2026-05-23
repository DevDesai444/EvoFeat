"""W&B logger with an offline JSONL fallback.

We try to use wandb if it's installed AND the user hasn't forced
``WANDB_MODE=disabled``. Otherwise (and during CI / Kaggle no-net runs)
we log one JSON line per iteration to ``results/runs/{run_name}.jsonl``.

The two paths expose the same ``RunLogger`` interface so the search loop
calls ``logger.log({...})`` either way.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


log = logging.getLogger("evofeat.logger")


def _wandb_available() -> bool:
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline"):
        return False
    try:
        import wandb  # noqa: F401
        return bool(os.environ.get("WANDB_API_KEY"))
    except ImportError:
        return False


@dataclass
class RunLogger:
    run_name: str
    project: str = "evofeat"
    config: Dict[str, Any] = field(default_factory=dict)
    out_dir: str = "results/runs"
    _wb: Any = None
    _records: List[Dict[str, Any]] = field(default_factory=list)
    _fh: Any = None

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._path = os.path.join(self.out_dir, f"{self.run_name}.jsonl")
        # truncate so reruns of the same name overwrite cleanly
        self._fh = open(self._path, "w")

        if _wandb_available():
            try:
                import wandb
                self._wb = wandb.init(
                    project=self.project, name=self.run_name,
                    config=self.config, reinit=True,
                )
                log.info("wandb run started: %s", self._wb.get_url() if self._wb else "?")
            except Exception as e:
                log.warning("wandb init failed (%s); falling back to local jsonl", e)
                self._wb = None

    def log(self, record: Dict[str, Any], step: Optional[int] = None) -> None:
        self._records.append(record)
        json.dump(record, self._fh, default=str)
        self._fh.write("\n")
        self._fh.flush()
        if self._wb is not None:
            try:
                self._wb.log(record, step=step)
            except Exception as e:
                log.warning("wandb log failed: %s", e)

    def summary(self, summary: Dict[str, Any]) -> None:
        # write summary as a final marker line
        out = {"__summary__": True, **summary}
        json.dump(out, self._fh, default=str)
        self._fh.write("\n")
        self._fh.flush()
        if self._wb is not None:
            try:
                self._wb.summary.update(summary)
            except Exception as e:
                log.warning("wandb summary update failed: %s", e)

    def finish(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self._wb is not None:
            try:
                self._wb.finish()
            except Exception:
                pass
            self._wb = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()


def replay(path: str) -> List[Dict[str, Any]]:
    """Read a jsonl run-log back into memory — useful for plotting."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
