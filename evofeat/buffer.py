"""Island experience buffer.

Programs live in clusters (a cluster is a set of programs that all hit the
same score signature across the held-out folds). Each island holds many
clusters; sampling picks a cluster softmax-weighted by its score, then
picks a program inside the cluster weighted toward shorter implementations
(shorter ≈ less likely to overfit the prompt). Every so often we wipe the
weakest islands and reseed them from the survivors — the FunSearch trick.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import random
import time
from collections.abc import Mapping
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.special

from evofeat import codegen
from evofeat.data import is_categorical
from evofeat.prompts import (
    SYSTEM_PROMPT,
    example_block,
    feature_block,
    load_prefix_suffix,
)


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, t: float) -> np.ndarray:
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    p = scipy.special.softmax(logits / t, axis=-1)
    # numerical fix-up so np.random.choice doesn't complain about p summing
    # to 1+eps
    i = int(np.argmax(p))
    p[i] = 1.0 - np.sum(p[:i]) - np.sum(p[i + 1:])
    return p


def _reduce_score(scores: ScoresPerTest) -> float:
    vs = list(scores.values())
    return sum(vs) / len(vs) if vs else float("-inf")


def _signature(scores: ScoresPerTest) -> Signature:
    return tuple(scores[k] for k in sorted(scores.keys()))


@dataclasses.dataclass
class BufferConfig:
    functions_per_prompt: int = 2
    num_islands: int = 3
    reset_period_s: int = 4 * 60 * 60
    sampling_t_init: float = 0.1
    sampling_t_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Prompt:
    code: str
    version_generated: int
    island_id: int
    data_input: pd.DataFrame
    data_output: Any


class Cluster:
    def __init__(self, score: float, prog: codegen.Function, data_in, data_out):
        self._score = score
        self._programs = [prog]
        self._lengths = [len(str(prog))]
        self._data_in = [data_in]
        self._data_out = [data_out]

    @property
    def score(self) -> float:
        return self._score

    def add(self, prog: codegen.Function, data_in, data_out) -> None:
        self._programs.append(prog)
        self._lengths.append(len(str(prog)))
        self._data_in.append(data_in)
        self._data_out.append(data_out)

    def sample(self) -> codegen.Function:
        normed = (np.array(self._lengths) - min(self._lengths)) / (max(self._lengths) + 1e-6)
        p = _softmax(-normed, t=1.0)
        idx = np.random.choice(len(self._programs), p=p)
        f = self._programs[idx]
        # attach the data carried with this program so the next prompt can
        # condition on the right slice
        f.data_input = self._data_in[idx]
        f.data_output = self._data_out[idx]
        return f


class Island:
    def __init__(
        self,
        template: codegen.Program,
        function_to_evolve: str,
        functions_per_prompt: int,
        meta: dict,
        t_init: float,
        t_period: int,
        prompts_dir: str = "prompts",
    ):
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._meta = meta
        self._t_init = t_init
        self._t_period = t_period
        self._clusters: dict[Signature, Cluster] = {}
        self._n_progs: int = 0
        self._prompts_dir = prompts_dir

    def register(self, prog: codegen.Function, scores: ScoresPerTest, data_in, data_out) -> None:
        sig = _signature(scores)
        if sig not in self._clusters:
            self._clusters[sig] = Cluster(_reduce_score(scores), prog, data_in, data_out)
        else:
            self._clusters[sig].add(prog, data_in, data_out)
        self._n_progs += 1

    def get_prompt(self) -> tuple[str, int, pd.DataFrame, Any]:
        sigs = list(self._clusters.keys())
        scores = np.array([self._clusters[s].score for s in sigs])
        t = self._t_init * (1 - (self._n_progs % self._t_period) / self._t_period)
        p = _softmax(scores, t=max(t, 1e-3))
        k = min(len(sigs), self._functions_per_prompt)
        idx = np.random.choice(len(sigs), size=k, p=p)
        chosen = [sigs[i] for i in idx]
        progs = [self._clusters[s].sample() for s in chosen]
        scores_for_sort = [self._clusters[s].score for s in chosen]
        order = np.argsort(scores_for_sort)
        sorted_progs = [progs[i] for i in order]
        version = len(sorted_progs) + 1
        prompt = self._render(sorted_progs)
        return prompt, version, sorted_progs[-1].data_input, sorted_progs[-1].data_output

    def _render(self, implementations: list[codegen.Function]) -> str:
        implementations = copy.deepcopy(implementations)
        versioned: list[codegen.Function] = []
        last_input: Optional[pd.DataFrame] = None
        last_output = None
        for i, impl in enumerate(implementations):
            new_name = f"{self._function_to_evolve}_v{i}"
            impl.name = new_name
            if i >= 1:
                impl.docstring = f"Improved version of `{self._function_to_evolve}_v{i-1}`."
            last_input = impl.data_input
            last_output = impl.data_output
            text = codegen.rename_calls(str(impl), self._function_to_evolve, new_name)
            versioned.append(codegen.text_to_function(text))

        next_v = len(implementations)
        next_name = f"{self._function_to_evolve}_v{next_v}"
        header = dataclasses.replace(
            implementations[-1],
            name=next_name,
            body="",
            docstring=f"Improved version of `{self._function_to_evolve}_v{next_v-1}`. Propose stronger features.",
        )
        versioned.append(header)
        prog = dataclasses.replace(self._template, functions=versioned)

        text = str(prog)
        kind = "operations" if random.randint(0, 1) == 0 else "domain"
        prefix, suffix = load_prefix_suffix(kind, prompts_dir=self._prompts_dir)
        text = text.replace("[PREFIX]", prefix).replace("[SUFFIX]", suffix)

        if last_input is not None:
            features = feature_block(last_input, self._meta)
            examples = example_block(last_input, last_output)
            text = text.replace("[FEATURES]", features).replace("[EXAMPLES]", examples)
        return text


class ExperienceBuffer:
    def __init__(
        self,
        config: BufferConfig,
        template: codegen.Program,
        function_to_evolve: str,
        meta: dict,
        prompts_dir: str = "prompts",
    ):
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._meta = meta
        self._prompts_dir = prompts_dir

        self._islands = [
            Island(
                template, function_to_evolve, config.functions_per_prompt, meta,
                config.sampling_t_init, config.sampling_t_period,
                prompts_dir=prompts_dir,
            )
            for _ in range(config.num_islands)
        ]
        self._best_score = [-float("inf")] * config.num_islands
        self._best_program = [None] * config.num_islands
        self._best_scores_per_test = [None] * config.num_islands
        self._last_reset = time.time()

    def get_prompt(self) -> Prompt:
        i = np.random.randint(len(self._islands))
        code, version, data_in, data_out = self._islands[i].get_prompt()
        return Prompt(code, version, i, data_in, data_out)

    def register_program(
        self,
        prog: codegen.Function,
        island_id: Optional[int],
        scores: ScoresPerTest,
        data_in: pd.DataFrame,
        data_out: Any,
        **kw,
    ) -> None:
        if island_id is None:
            for i in range(len(self._islands)):
                self._register(prog, i, scores, data_in, data_out, **kw)
        else:
            self._register(prog, island_id, scores, data_in, data_out, **kw)
        if time.time() - self._last_reset > self._config.reset_period_s:
            self._last_reset = time.time()
            self._reset_weakest_half()

    def _register(self, prog, island_id, scores, data_in, data_out, **kw):
        self._islands[island_id].register(prog, scores, data_in, data_out)
        score = _reduce_score(scores)
        if score > self._best_score[island_id]:
            self._best_score[island_id] = score
            self._best_program[island_id] = prog
            self._best_scores_per_test[island_id] = scores
        profiler = kw.get("profiler")
        if profiler:
            prog.score = score
            prog.global_sample_nums = kw.get("global_sample_nums")
            prog.sample_time = kw.get("sample_time")
            prog.evaluate_time = kw.get("evaluate_time")
            prog.data_input = data_in
            prog.data_output = data_out
            profiler.register(prog)

    def _reset_weakest_half(self) -> None:
        order = np.argsort(
            np.array(self._best_score) + np.random.randn(len(self._best_score)) * 1e-6
        )
        n_reset = self._config.num_islands // 2
        to_reset = order[:n_reset]
        keep = order[n_reset:]
        for i in to_reset:
            self._islands[i] = Island(
                self._template, self._function_to_evolve,
                self._config.functions_per_prompt, self._meta,
                self._config.sampling_t_init, self._config.sampling_t_period,
                prompts_dir=self._prompts_dir,
            )
            self._best_score[i] = -float("inf")
            seed = int(np.random.choice(keep))
            founder = self._best_program[seed]
            founder_scores = self._best_scores_per_test[seed]
            self._islands[i].register(founder, founder_scores, None, None)

    def best(self) -> tuple[Optional[codegen.Function], float]:
        i = int(np.argmax(self._best_score))
        return self._best_program[i], self._best_score[i]
