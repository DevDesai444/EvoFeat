"""Parse / re-emit Python programs and individual functions.

We need three things from generated samples:
  1. trim a candidate "body" out of an LLM response that may include extra
     prose or a partial function header,
  2. swap the name of the evolved function in subsequent revisions, and
  3. yield functions tagged with @evaluate.run / @equation.evolve from the
     dataset specification file.

The code stays close to the AST so that small malformed tails don't poison
the whole sample — we walk lines from the bottom and re-parse until the
remainder is valid.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import tokenize
from collections.abc import Iterator, MutableSet, Sequence
from typing import Optional


@dataclasses.dataclass
class Function:
    name: str
    args: str
    body: str
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    score: Optional[float] = None
    global_sample_nums: Optional[int] = None
    sample_time: Optional[float] = None
    evaluate_time: Optional[float] = None
    # carried so the buffer can re-feed the right training rows back to the
    # next round — not part of the rendered string
    data_input: object = None
    data_output: object = None

    def __str__(self) -> str:
        rt = f" -> {self.return_type}" if self.return_type else ""
        out = f"def {self.name}({self.args}){rt}:\n"
        if self.docstring:
            sep = "\n" if self.body else ""
            out += f'    """{self.docstring}"""{sep}'
        out += self.body + "\n\n"
        return out

    def __setattr__(self, name, value):
        if name == "body" and isinstance(value, str):
            value = value.strip("\n")
        if name == "docstring" and isinstance(value, str):
            if '"""' in value:
                value = value.strip().replace('"""', "")
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    preface: str
    functions: list

    def __str__(self) -> str:
        out = f"{self.preface}\n" if self.preface else ""
        out += "\n".join(str(f) for f in self.functions)
        return out

    def function_index(self, name: str) -> int:
        names = [f.name for f in self.functions]
        if names.count(name) != 1:
            raise ValueError(f"function {name!r} not unique in program")
        return names.index(name)

    def get_function(self, name: str) -> Function:
        return self.functions[self.function_index(name)]


class _Visitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self._lines = source.splitlines()
        self._preface = ""
        self._functions: list[Function] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.col_offset == 0:
            if not self._functions:
                if node.decorator_list:
                    first_line = min(d.lineno for d in node.decorator_list)
                else:
                    first_line = node.lineno
                self._preface = "\n".join(self._lines[: first_line - 1])

            end = node.end_lineno
            start = node.body[0].lineno - 1
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    docstring = f'  """{node.body[0].value.value}"""'
                    if len(node.body) > 1:
                        start = node.body[1].lineno - 1
                    else:
                        start = end

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body="\n".join(self._lines[start:end]),
            ))
        self.generic_visit(node)

    def program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(source: str) -> Program:
    tree = ast.parse(source)
    v = _Visitor(source)
    v.visit(tree)
    return v.program()


def text_to_function(source: str) -> Function:
    p = text_to_program(source)
    if len(p.functions) != 1:
        raise ValueError(f"expected 1 function, got {len(p.functions)}")
    return p.functions[0]


def _tokens(code: str) -> Iterator[tokenize.TokenInfo]:
    return tokenize.tokenize(io.BytesIO(code.encode()).readline)


def _yield_with_call_flag(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    prev: Optional[tokenize.TokenInfo] = None
    is_attr = False
    for tok in _tokens(code):
        if (prev and prev.type == tokenize.NAME
                and tok.type == tokenize.OP and tok.string == "("):
            yield prev, not is_attr
            is_attr = False
        else:
            if prev:
                is_attr = prev.type == tokenize.OP and prev.string == "."
                yield prev, False
        prev = tok
    if prev:
        yield prev, False


def rename_calls(code: str, src: str, dst: str) -> str:
    if src not in code:
        return code
    out = []
    for tok, is_call in _yield_with_call_flag(code):
        if is_call and tok.string == src:
            out.append(tokenize.TokenInfo(tok.type, dst, tok.start, tok.end, tok.line))
        else:
            out.append(tok)
    return tokenize.untokenize(out).decode()


def functions_called(code: str) -> MutableSet[str]:
    return {tok.string for tok, is_call in _yield_with_call_flag(code) if is_call}


def yield_decorated(code: str, module: str, attr: str) -> Iterator[str]:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            inner = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name):
                if inner.value.id == module and inner.attr == attr:
                    yield node.name


class _BodyEnd(ast.NodeVisitor):
    def __init__(self, target: str):
        self._target = target
        self.end: Optional[int] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self._target:
            self.end = node.end_lineno
        self.generic_visit(node)


def trim_function_body(body: str) -> str:
    """Wrap a body fragment in a fake def, parse, and pull lines back.

    LLMs love to append a stray ``print(...)`` or a half-finished helper
    after the function we asked for; we want to stop at the end of the
    first balanced function and drop the rest.
    """
    if not body:
        return ""
    wrapped = f"def _fake():\n{body}"
    tree = None
    while tree is None:
        try:
            tree = ast.parse(wrapped)
        except SyntaxError as e:
            if e.lineno is None:
                return ""
            wrapped = "\n".join(wrapped.splitlines()[: e.lineno - 1])
            if not wrapped.strip():
                return ""
    v = _BodyEnd("_fake")
    v.visit(tree)
    if v.end is None:
        return ""
    lines = wrapped.splitlines()[1:v.end]
    return "\n".join(lines) + "\n\n"
