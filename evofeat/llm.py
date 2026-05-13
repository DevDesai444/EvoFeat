"""LLM client layer for the evolutionary loop.

We support two transport backends:

* ``GroqClient``  — talks to api.groq.com, which exposes an
  OpenAI-compatible chat-completions endpoint. One client per logical model.
* ``VLLMClient``  — talks to a local vLLM ``--api-server`` (also
  OpenAI-compatible). Use this when running Qwen / Mistral / Llama on a
  GPU box (Kaggle, on-prem, whatever).

Both clients return strings. Both record per-call token usage so the
Pareto / cost analysis at the end reflects what actually happened, not
what was budgeted in the config.

The two transports share `BaseClient` so the search loop never sees the
difference. Adding a third transport later (Together, Anyscale, your own
server) is a 30-line subclass.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# Display name (used in configs and reports) -> (server-side model id, family).
# Pinned in code so reruns produce comparable rows in the cross-backbone table —
# Groq has retired a handful of these in the past and `latest` aliases drift.
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # cloud
    "llama-3.1-8b":     ("llama-3.1-8b-instant",    "llama"),
    "llama-3.3-70b":    ("llama-3.3-70b-versatile", "llama"),
    "mistral-saba-24b": ("mistral-saba-24b",        "mistral"),
    "qwen-3-32b":       ("qwen/qwen3-32b",          "qwen"),
    "gemma-2-9b":       ("gemma2-9b-it",            "gemma"),
    "gpt-oss-20b":      ("openai/gpt-oss-20b",      "gpt-oss"),
    # vllm-served (the model id is whatever you launched vllm with — these
    # are just the labels we use in tables)
    "qwen-2.5-7b":      ("Qwen/Qwen2.5-7B-Instruct",          "qwen"),
    "mistral-7b-v0.3":  ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    "llama-3.1-8b-instruct": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama"),
}


def resolve_model(name: str) -> Tuple[str, str]:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # let callers pass a raw model id too — useful when vLLM is launched
    # with a non-registered checkpoint
    return name, name.split("/")[0].split("-")[0].lower()


log = logging.getLogger("evofeat.llm")


@dataclass
class CallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    failures: int = 0
    latency_s: float = 0.0
    per_call: List[Dict] = field(default_factory=list)

    def add(self, usage: Dict, latency: float) -> None:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += pt + ct
        self.requests += 1
        self.latency_s += latency
        self.per_call.append(
            {"prompt": pt, "completion": ct, "latency_s": round(latency, 3)}
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "failures": self.failures,
            "latency_s": round(self.latency_s, 2),
        }


class BaseClient:
    """Common chat-completions interface.

    Subclasses implement ``_endpoint()`` (a full URL) and ``_headers()``;
    the retry / parsing logic is identical across providers thanks to
    OpenAI-compatible request/response shapes.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        max_retries: int = 6,
        timeout_s: int = 60,
    ):
        self.model_label = model
        self.model_id, self.family = resolve_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.stats = CallStats()
        self._session = requests.Session()
        self._session.headers.update(self._headers())

    # subclass hooks ---------------------------------------------------------
    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    # core path --------------------------------------------------------------
    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                r = self._session.post(self._endpoint(), data=json.dumps(payload), timeout=self.timeout_s)
                latency = time.time() - t0
                if r.status_code == 429 or r.status_code >= 500:
                    # honor retry-after when present, otherwise expo backoff
                    delay = float(r.headers.get("retry-after", 0)) or (2 ** attempt)
                    log.debug("backing off %.1fs after %s", delay, r.status_code)
                    time.sleep(delay + random.random() * 0.5)
                    continue
                if 400 <= r.status_code < 500:
                    # 4xx is not retriable (auth, payload-too-large, bad
                    # request). bail fast so the caller can downsize / fix.
                    raise RuntimeError(f"{r.status_code} from groq: {r.text[:300]}")
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                self.stats.add(data.get("usage", {}), latency)
                return text
            except Exception as e:
                last_err = e
                self.stats.failures += 1
                time.sleep(min(2 ** attempt, 16))
                continue
        raise RuntimeError(f"{self.__class__.__name__} call failed after {self.max_retries} retries: {last_err}")

    def samples(self, system: str, user: str, n: int) -> List[str]:
        """Draw ``n`` independent samples. Failures within a draw are
        swallowed so a one-off 429 doesn't tank an entire iteration."""
        out: List[str] = []
        for _ in range(n):
            try:
                out.append(self.chat(system, user))
            except Exception as e:
                log.warning("sample dropped: %s", e)
                continue
        return out


class GroqClient(BaseClient):
    """OpenAI-compatible POSTs against api.groq.com."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kw):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY missing. Copy .env.example to .env and fill it in, "
                "or pass api_key= explicitly."
            )
        super().__init__(model=model, **kw)

    def _endpoint(self) -> str:
        return GROQ_URL

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class VLLMClient(BaseClient):
    """OpenAI-compatible POSTs against a vLLM server running locally.

    Launch vLLM with the OpenAI-compatible API for this to work:

        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen2.5-7B-Instruct --port 8000

    The ``model`` passed here must match what vLLM was started with — vLLM
    will 404 otherwise.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = None,
        **kw,
    ):
        self.base_url = base_url.rstrip("/")
        # vLLM doesn't enforce an API key by default but accepts a bearer
        # token; use a placeholder so the header is well-formed
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        super().__init__(model=model, **kw)

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


_CLIENT_KWARGS = {"temperature", "max_tokens", "max_retries", "timeout_s"}


def make_client(spec: Dict[str, Any]) -> BaseClient:
    """Factory used by the YAML-driven runners.

    ``spec`` looks like::

        {"backend": "groq", "model": "llama-3.1-8b", "temperature": 0.8}
        {"backend": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct",
         "base_url": "http://127.0.0.1:8000/v1"}

    Any other key in ``spec`` (e.g. ``backbone_label``, comments) is
    metadata for the runners — silently dropped here.
    """
    backend = spec.get("backend", "groq").lower()
    kw = {k: v for k, v in spec.items() if k in _CLIENT_KWARGS}
    if backend == "groq":
        return GroqClient(model=spec["model"], api_key=spec.get("api_key"), **kw)
    if backend == "vllm":
        return VLLMClient(
            model=spec["model"],
            base_url=spec.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key=spec.get("api_key"),
            **kw,
        )
    raise ValueError(f"unknown backend {backend!r}")
