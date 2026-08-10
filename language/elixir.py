"""Elixir source parsing — module name, defstruct fields, public functions.

Pure Python regex-based. Deliberately simple; if it turns out to miss real
cases we upgrade to invoking `elixir -e 'Code.string_to_quoted!'` and walking
the AST from Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FunctionSig:
    name: str
    arity: int


@dataclass(frozen=True)
class ModuleShape:
    module_name: str
    kind: Literal["module", "behaviour", "protocol"]
    defstruct_fields: list[str]
    public_functions: list[FunctionSig]      # de-duplicated by (name, arity)
    callbacks: list[FunctionSig]             # @callback declarations
    line_count: int


_DEFMODULE_RE = re.compile(r"defmodule\s+([\w.]+)\s+do")
_DEFPROTOCOL_RE = re.compile(r"\bdefprotocol\s+([\w.]+)\s+do")
_CALLBACK_RE = re.compile(r"@callback\s+([\w?!]+)\s*\((.*?)\)", re.DOTALL)


def _iter_def_heads(source: str):
    """Yield (name, args_string) for every public `def` / `defmacro` in `source`."""
    # Paren form: def foo(...)
    for m in re.finditer(r"\bdef(?:macro)?\s+([\w?!]+)\s*\(", source):
        name = m.group(1)
        i = m.end()
        depth = 1
        start = i
        while i < len(source) and depth > 0:
            c = source[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    yield name, source[start:i]
                    break
            i += 1
    # No-paren form: def foo do  OR  def foo, do: ...
    for m in re.finditer(r"\bdef(?:macro)?\s+([\w?!]+)\s*(?:,\s*do:|\s+do\b)", source):
        yield m.group(1), ""


def _arities_for(args_str: str) -> list[int]:
    """Enumerate arities accounting for default args (`\\\\`)."""
    s = args_str.strip()
    if not s:
        return [0]
    args: list[str] = []
    current = ""
    depth = 0
    for c in s:
        if c in "({[":
            depth += 1
            current += c
        elif c in ")}]":
            depth -= 1
            current += c
        elif c == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        args.append(current.strip())
    total = len(args)
    required = sum(1 for a in args if r"\\" not in a)
    return list(range(max(required, 0), total + 1))


def _extract_defstruct_fields(source: str) -> list[str]:
    m = re.search(
        r"defstruct\b(.+?)(?=\n\s*@|\n\s*def(?:p|module|struct|protocol|impl|macro)?\b|\n\s*end\b)",
        source, re.DOTALL,
    )
    if not m:
        return []
    body = m.group(1)
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r":(\w+)|(\b\w+)\s*:(?!:)", body):
        name = match.group(1) or match.group(2)
        if name and name not in seen and name not in {"defstruct", "do", "end"}:
            seen.add(name)
            result.append(name)
    return result


def describe_source(source: str) -> ModuleShape | None:
    """Parse an Elixir source string. Returns None if it doesn't look like
    a module or protocol."""
    mod_match = _DEFMODULE_RE.search(source)
    proto_match = _DEFPROTOCOL_RE.search(source)

    if proto_match:
        module_name = proto_match.group(1)
        kind: Literal["module", "behaviour", "protocol"] = "protocol"
    elif mod_match:
        module_name = mod_match.group(1)
        kind = "module"
    else:
        return None

    sigs: set[FunctionSig] = set()
    for name, args in _iter_def_heads(source):
        for arity in _arities_for(args):
            sigs.add(FunctionSig(name, arity))

    callbacks: set[FunctionSig] = set()
    for m in _CALLBACK_RE.finditer(source):
        name = m.group(1)
        args = m.group(2)
        for arity in _arities_for(args):
            callbacks.add(FunctionSig(name, arity))

    # Behaviour trumps plain module; protocol takes precedence over both
    if kind == "module" and callbacks:
        kind = "behaviour"

    return ModuleShape(
        module_name=module_name,
        kind=kind,
        defstruct_fields=_extract_defstruct_fields(source),
        public_functions=sorted(sigs, key=lambda s: (s.name, s.arity)),
        callbacks=sorted(callbacks, key=lambda s: (s.name, s.arity)),
        line_count=source.count("\n") + 1,
    )


def describe_file(path: Path) -> ModuleShape | None:
    try:
        return describe_source(path.read_text())
    except OSError:
        return None
