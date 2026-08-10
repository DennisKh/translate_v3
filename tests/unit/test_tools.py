"""Unit tests for tool internals: path resolution, state transitions, write flow."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.state import FileEntry, FileState, StateStore
from agent.tools import (
    _promote_in_progress_to_complete,
    _resolve_elixir_path,
    _stem_for_module,
)


# ---------------------------------------------------------------------------
# Fake SessionContext — minimal for testing pure functions
# ---------------------------------------------------------------------------

def _fake_ctx(target_root: Path, module_prefix: str = "JodaMoney",
              app_snake: str = "joda_money", module_to_stem: dict | None = None,
              state: StateStore | None = None):
    return SimpleNamespace(
        target_root=target_root,
        module_prefix=module_prefix,
        app_snake=app_snake,
        module_to_stem=module_to_stem or {},
        state=state or StateStore(target_root),
    )


# ---------------------------------------------------------------------------
# _resolve_elixir_path
# ---------------------------------------------------------------------------

def test_resolve_root_module(tmp_path):
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "JodaMoney.Money")
    assert p == tmp_path / "lib" / "joda_money" / "money.ex"


def test_resolve_nested_module(tmp_path):
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "JodaMoney.Format.MoneyFormatter")
    assert p == tmp_path / "lib" / "joda_money" / "format" / "money_formatter.ex"


def test_resolve_path_form(tmp_path):
    (tmp_path / "lib" / "joda_money").mkdir(parents=True)
    real_file = tmp_path / "lib" / "joda_money" / "foo.ex"
    real_file.write_text("defmodule Foo do\nend\n")
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "lib/joda_money/foo.ex")
    assert p == real_file


def test_resolve_path_form_not_found(tmp_path):
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "lib/joda_money/nonexistent.ex")
    assert p is None


def test_resolve_wrong_prefix_returns_none(tmp_path):
    ctx = _fake_ctx(tmp_path, module_prefix="JodaMoney")
    p = _resolve_elixir_path(ctx, "Elsewhere.Foo")
    assert p is None


def test_resolve_multi_capital_camel(tmp_path):
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "JodaMoney.BigMoneyProvider")
    assert p == tmp_path / "lib" / "joda_money" / "big_money_provider.ex"


def test_resolve_bare_prefix_returns_none(tmp_path):
    ctx = _fake_ctx(tmp_path)
    p = _resolve_elixir_path(ctx, "JodaMoney")
    assert p is None


# ---------------------------------------------------------------------------
# _stem_for_module — O(1) via ctx.module_to_stem
# ---------------------------------------------------------------------------

def test_stem_for_module_hit(tmp_path):
    ctx = _fake_ctx(tmp_path, module_to_stem={"JodaMoney.BigMoney": "BigMoney"})
    assert _stem_for_module(ctx, "JodaMoney.BigMoney") == "BigMoney"


def test_stem_for_module_miss(tmp_path):
    ctx = _fake_ctx(tmp_path, module_to_stem={"JodaMoney.Money": "Money"})
    assert _stem_for_module(ctx, "JodaMoney.Unknown") is None


# ---------------------------------------------------------------------------
# _promote_in_progress_to_complete
# ---------------------------------------------------------------------------

def _entry(stem: str, state: FileState = FileState.NOT_STARTED) -> FileEntry:
    return FileEntry(
        stem=stem,
        java_path=f"{stem}.java",
        lib_path=f"lib/x/{stem.lower()}.ex",
        test_path=f"test/x/{stem.lower()}_test.exs",
        module_name=f"X.{stem}",
        state=state,
    )


def test_promote_marks_in_progress_as_complete(tmp_path):
    state = StateStore(tmp_path)
    state.upsert(_entry("A", FileState.IN_PROGRESS))
    state.upsert(_entry("B", FileState.IN_PROGRESS))
    state.upsert(_entry("C", FileState.NOT_STARTED))
    ctx = _fake_ctx(tmp_path, state=state)

    promoted = _promote_in_progress_to_complete(ctx)

    assert promoted == 2
    assert state.get("A").state == FileState.COMPLETE
    assert state.get("B").state == FileState.COMPLETE
    assert state.get("C").state == FileState.NOT_STARTED  # untouched


def test_promote_returns_zero_when_none_in_progress(tmp_path):
    state = StateStore(tmp_path)
    state.upsert(_entry("A", FileState.COMPLETE))
    state.upsert(_entry("B", FileState.SKIPPED))
    ctx = _fake_ctx(tmp_path, state=state)

    promoted = _promote_in_progress_to_complete(ctx)
    assert promoted == 0


# ---------------------------------------------------------------------------
# list_files status filter — regression test for the "translated" bug
# ---------------------------------------------------------------------------

def test_list_files_status_filter_covers_every_advertised_value(tmp_path):
    """Regression test: list_files advertises `Literal["translated", ...]` as
    valid inputs, but "translated" is a friendly name for `FileState.COMPLETE`,
    not a raw enum value. The tool must map friendly names correctly for every
    status advertised in its Literal type, not just `FileState(status)`.
    """
    from agent.tools import build_tools

    state = StateStore(tmp_path)
    for stem, st in [
        ("A", FileState.NOT_STARTED),
        ("B", FileState.IN_PROGRESS),
        ("C", FileState.COMPLETE),
        ("D", FileState.SKIPPED),
        ("E", FileState.DELETED),
        ("F", FileState.BLOCKED),
        ("G", FileState.ESCALATED),
        ("H", FileState.SYNTAX_FAILED),
        ("I", FileState.COMPILE_FAILED),
    ]:
        state.upsert(_entry(stem, st))

    ctx = SimpleNamespace(
        state=state,
        events=SimpleNamespace(emit=lambda *a, **kw: None),
        cost=SimpleNamespace(add_tool=lambda *a, **kw: None),
        # Fields tools may read; provide enough to instantiate.
        source_root=tmp_path, target_root=tmp_path,
        module_prefix="X", app_snake="x",
        java_by_stem={}, tests_root=None, files_written_since_compile=0,
        module_to_stem={}, cfg=SimpleNamespace(source=SimpleNamespace(root=tmp_path)),
    )
    tools = build_tools(ctx, include_write=False)
    list_files = next(t for t in tools if t.name == "list_files")

    import json
    for status, expected_stems in [
        ("all", {"A", "B", "C", "D", "E", "F", "G", "H", "I"}),
        ("untranslated", {"A"}),
        ("in_progress", {"B"}),
        ("translated", {"C"}),
        ("skipped", {"D"}),
        ("deleted", {"E"}),
        ("blocked", {"F"}),
        ("escalated", {"G"}),
        ("failed", {"H", "I"}),
    ]:
        result_str = list_files.call({"status": status})
        result = json.loads(result_str)
        actual = {f["stem"] for f in result["files"]}
        assert actual == expected_stems, (
            f"status={status!r}: expected {expected_stems}, got {actual}"
        )
