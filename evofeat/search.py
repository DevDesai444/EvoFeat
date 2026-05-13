"""End-to-end search loop: sample → evaluate → register, repeated.

Parameters live in ``SearchConfig`` so the runner can construct one from
the CLI / YAML and the same object is what the buffer / evaluator inspect
later. The loop talks to any subclass of ``BaseClient`` (Groq, vLLM, …)
so the search is identical across backbones.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import time
from typing import Any, Optional

import pandas as pd

from evofeat import codegen
from evofeat.buffer import BufferConfig, ExperienceBuffer
from evofeat.llm import BaseClient
from evofeat.logger import RunLogger
from evofeat.profile import Profiler
from evofeat.prompts import SYSTEM_PROMPT
from evofeat.sandbox import calls_ancestor, run_transform


log = logging.getLogger("evofeat.search")


@dataclasses.dataclass
class SearchConfig:
    samples_per_prompt: int = 3
    evaluate_timeout_s: int = 30
    max_samples: int = 60          # total candidate evaluations (≈ iters * spp)
    early_stop_patience: int = 20  # number of consecutive non-improvements
    buffer: BufferConfig = dataclasses.field(default_factory=BufferConfig)


def _extract_function_names(spec: str) -> tuple[str, str]:
    runs = list(codegen.yield_decorated(spec, "evaluate", "run"))
    if len(runs) != 1:
        raise ValueError("spec must have exactly one @evaluate.run function")
    evolves = list(codegen.yield_decorated(spec, "equation", "evolve"))
    if len(evolves) != 1:
        raise ValueError("spec must have exactly one @equation.evolve function")
    return evolves[0], runs[0]


def _candidate_to_program(
    body: str,
    version: Optional[int],
    template: codegen.Program,
    function_to_evolve: str,
) -> tuple[codegen.Function, str]:
    body = codegen.trim_function_body(body)
    if version is not None:
        body = codegen.rename_calls(body, f"{function_to_evolve}_v{version}", function_to_evolve)
    prog = copy.deepcopy(template)
    fn = prog.get_function(function_to_evolve)
    fn.body = body
    return fn, str(prog)


class Evaluator:
    def __init__(
        self,
        buffer: ExperienceBuffer,
        template: codegen.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: dict,
        timeout_seconds: int,
    ):
        self._buffer = buffer
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout = timeout_seconds

    def analyse(
        self,
        candidate: str,
        island_id: Optional[int],
        data_input: pd.DataFrame,
        data_output: Any,
        version_generated: Optional[int],
        logger: Optional[RunLogger] = None,
        **kw,
    ) -> tuple[Optional[float], bool]:
        new_fn, program = _candidate_to_program(
            candidate, version_generated, self._template, self._function_to_evolve,
        )
        scores: dict[str, float] = {}
        t0 = time.time()
        self._inputs["data"]["inputs"] = data_input
        self._inputs["data"]["outputs"] = data_output
        last_in, last_out = data_input, data_output
        ok_all = False
        for key in self._inputs:
            result, ok = run_transform(
                program, self._function_to_run, self._function_to_evolve,
                self._inputs, key, timeout_seconds=self._timeout,
            )
            if ok and not calls_ancestor(program, self._function_to_evolve):
                score, last_in, last_out = result
                scores[key] = float(score)
                ok_all = True
        evaluate_time = time.time() - t0

        if scores:
            self._buffer.register_program(
                new_fn, island_id, scores, last_in, last_out,
                evaluate_time=evaluate_time, **kw,
            )
            score_val = sum(scores.values()) / len(scores)
        else:
            score_val = None
            profiler: Profiler = kw.get("profiler")
            if profiler:
                new_fn.score = None
                new_fn.global_sample_nums = kw.get("global_sample_nums")
                new_fn.sample_time = kw.get("sample_time")
                new_fn.evaluate_time = evaluate_time
                profiler.register(new_fn)

        if logger is not None:
            logger.log({
                "step": kw.get("global_sample_nums"),
                "island": island_id,
                "score": score_val,
                "ok": ok_all,
                "eval_s": evaluate_time,
            }, step=kw.get("global_sample_nums"))
        return score_val, ok_all


class Sampler:
    _counter = 1

    def __init__(
        self,
        buffer: ExperienceBuffer,
        evaluator: Evaluator,
        client: BaseClient,
        samples_per_prompt: int,
        max_samples: int,
        early_stop_patience: int = 20,
    ):
        self._buffer = buffer
        self._evaluator = evaluator
        self._client = client
        self._spp = samples_per_prompt
        self._max = max_samples
        self._patience = early_stop_patience
        type(self)._counter = 1

    def run(self, profiler: Optional[Profiler] = None, logger: Optional[RunLogger] = None) -> None:
        best = -float("inf")
        no_improve = 0
        while type(self)._counter < self._max:
            prompt = self._buffer.get_prompt()
            t0 = time.time()
            samples = self._client.samples(SYSTEM_PROMPT, prompt.code, self._spp)
            per_sample_latency = (time.time() - t0) / max(len(samples), 1)
            improved_this_batch = False
            for raw in samples:
                body = self._strip_to_body(raw)
                # prepend imports the candidate is allowed to assume
                body = "\n    import pandas as pd\n    import numpy as np\n" + body
                type(self)._counter += 1
                cur = type(self)._counter
                score_val, _ = self._evaluator.analyse(
                    body,
                    prompt.island_id,
                    prompt.data_input,
                    prompt.data_output,
                    prompt.version_generated,
                    profiler=profiler,
                    logger=logger,
                    global_sample_nums=cur,
                    sample_time=per_sample_latency,
                )
                if score_val is not None and score_val > best:
                    best = score_val
                    improved_this_batch = True
            if improved_this_batch:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self._patience:
                    log.info("early-stopping after %d non-improving rounds", no_improve)
                    break

    @staticmethod
    def _strip_to_body(text: str) -> str:
        """Pull a function body out of a messy LLM response.

        Handles three common shapes:
          1. clean code: ``def foo(df):\\n    ...`` — easy case
          2. fenced: ``\\`\\`\\`python\\ndef foo(df):\\n    ...\\n\\`\\`\\```
          3. multiple defs (the model repeats prior versions) — we take
             the LAST def's body

        The trim_function_body downstream re-validates by ast-parsing.
        """
        import re

        # 1) prefer the contents of a fenced ```python ... ``` block if present
        fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1)

        lines = text.splitlines()
        # 2) walk from the bottom — last def wins
        def_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].lstrip().startswith("def "):
                def_idx = i
                break
        if def_idx < 0:
            # no function header — return empty so trim_function_body
            # gracefully produces an empty body rather than parsing prose
            return ""

        rest = lines[def_idx + 1:]
        fixed: list[str] = []
        for r in rest:
            stripped = r.rstrip()
            # truncate the body at a closing fence or a new top-level def
            if stripped.startswith("```"):
                break
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
            if not stripped:
                fixed.append("")
                continue
            if not r.startswith("    ") and not r.startswith("\t"):
                r = "    " + r
            fixed.append(r)
        return "\n".join(fixed)


def run_search(
    spec: str,
    inputs: dict,
    meta: dict,
    client: BaseClient,
    config: SearchConfig,
    log_dir: str,
    prompts_dir: str = "prompts",
    logger: Optional[RunLogger] = None,
) -> Profiler:
    function_to_evolve, function_to_run = _extract_function_names(spec)
    template = codegen.text_to_program(spec)
    buffer = ExperienceBuffer(config.buffer, template, function_to_evolve, meta, prompts_dir=prompts_dir)
    profiler = Profiler(log_dir)

    evaluator = Evaluator(
        buffer, template, function_to_evolve, function_to_run, inputs,
        timeout_seconds=config.evaluate_timeout_s,
    )

    # seed: run the spec's initial body once so the buffer isn't empty
    initial = template.get_function(function_to_evolve).body
    evaluator.analyse(
        initial, island_id=None, data_input=inputs["data"]["inputs"],
        data_output=inputs["data"]["outputs"], version_generated=None,
        profiler=profiler,
        logger=logger,
        global_sample_nums=0,
        sample_time=0.0,
    )

    sampler = Sampler(
        buffer, evaluator, client,
        config.samples_per_prompt, config.max_samples,
        early_stop_patience=config.early_stop_patience,
    )
    sampler.run(profiler=profiler, logger=logger)
    return profiler
