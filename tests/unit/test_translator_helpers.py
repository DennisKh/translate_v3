"""Unit tests for translator.py helpers — session-reset criteria + state summarization."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import FileEntry, FileState, StateStore
from agent.translator import (
    GraphSessionResult,
    _all_files_terminal,
    _bomb_nudge_message,
    _build_initial_message,
    _count_terminal,
    _run_one_graph_session,
    _SessionStats,
    _wrap_up_message,
)


def _entry(stem: str, state: FileState = FileState.NOT_STARTED,
           deps: list[str] | None = None, level: int = 0) -> FileEntry:
    return FileEntry(
        stem=stem,
        java_path=f"{stem}.java",
        lib_path=f"lib/x/{stem.lower()}.ex",
        test_path=f"test/x/{stem.lower()}_test.exs",
        module_name=f"X.{stem}",
        deps=deps or [],
        level=level,
        state=state,
    )


def _fake_ctx(tmp_path: Path, entries: list[FileEntry],
              reset_files: int = 10, reset_tokens: int = 500_000) -> SimpleNamespace:
    state = StateStore(tmp_path)
    for e in entries:
        state.upsert(e)
    cfg = SimpleNamespace(
        source=SimpleNamespace(root=tmp_path),
        agent=SimpleNamespace(
            session_reset_after_files=reset_files,
            session_reset_after_tokens=reset_tokens,
        ),
    )
    budget = SimpleNamespace(total_cost_usd=0.0)
    return SimpleNamespace(
        state=state,
        cfg=cfg,
        budget=budget,
        source_root=tmp_path,
        target_root=tmp_path,
        module_prefix="TestApp",
        app_snake="test_app",
        java_classes=[],
        dep_graph=SimpleNamespace(levels=[], sccs=[]),
    )


# ---------------------------------------------------------------------------
# _count_terminal / _all_files_terminal
# ---------------------------------------------------------------------------

def test_count_terminal_only_terminal_states(tmp_path):
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.COMPLETE),
        _entry("B", FileState.SKIPPED),
        _entry("C", FileState.DELETED),
        _entry("D", FileState.ESCALATED),
        _entry("E", FileState.NOT_STARTED),
        _entry("F", FileState.IN_PROGRESS),
        _entry("G", FileState.BLOCKED),
    ])
    assert _count_terminal(ctx) == 4


def test_all_files_terminal_when_nothing_pending(tmp_path):
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.COMPLETE),
        _entry("B", FileState.SKIPPED),
        _entry("C", FileState.ESCALATED),
    ])
    assert _all_files_terminal(ctx)


def test_all_files_terminal_false_with_blocked(tmp_path):
    # BLOCKED is NOT a terminal state — the blocker might get resolved
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.COMPLETE),
        _entry("B", FileState.BLOCKED),
    ])
    assert not _all_files_terminal(ctx)


def test_all_files_terminal_false_with_in_progress(tmp_path):
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.COMPLETE),
        _entry("B", FileState.IN_PROGRESS),
    ])
    assert not _all_files_terminal(ctx)


# ---------------------------------------------------------------------------
# _SessionStats.should_reset
# ---------------------------------------------------------------------------

def test_session_reset_triggers_on_files_threshold(tmp_path):
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.NOT_STARTED),
        _entry("B", FileState.NOT_STARTED),
    ], reset_files=2)
    stats = _SessionStats()
    stats.note_first_turn(ctx)   # captures start_completed AFTER first turn

    # No files completed yet
    assert stats.should_reset(ctx) is None

    # Complete 2 files → threshold hit
    ctx.state.set_state("A", FileState.COMPLETE)
    ctx.state.set_state("B", FileState.COMPLETE)
    reason = stats.should_reset(ctx)
    assert reason is not None
    assert "session-reset threshold hit" in reason
    assert "files completed" in reason


def test_session_reset_no_reset_before_first_turn(tmp_path):
    """Regression for a Phase 3 review finding: start-completed timing.
    Reset check should return None if we haven't taken a turn yet."""
    ctx = _fake_ctx(tmp_path, [_entry("A", FileState.NOT_STARTED)], reset_files=1)
    stats = _SessionStats()
    # No note_first_turn call → should_reset must not fire
    ctx.state.set_state("A", FileState.COMPLETE)
    assert stats.should_reset(ctx) is None


def test_session_reset_triggers_on_tokens_threshold(tmp_path):
    ctx = _fake_ctx(tmp_path, [_entry("A")], reset_tokens=100_000)
    stats = _SessionStats()
    stats.note_first_turn(ctx)
    stats.session_tokens = 150_000
    reason = stats.should_reset(ctx)
    assert reason is not None
    assert "tokens used" in reason


def test_session_reset_counts_only_this_session(tmp_path):
    """If a file was already COMPLETE before session start, it doesn't count."""
    ctx = _fake_ctx(tmp_path, [
        _entry("Prev", FileState.COMPLETE),  # already done before this session
        _entry("A", FileState.NOT_STARTED),
        _entry("B", FileState.NOT_STARTED),
    ], reset_files=2)
    stats = _SessionStats()
    stats.note_first_turn(ctx)   # start_completed = 1

    # Complete only 1 in this session
    ctx.state.set_state("A", FileState.COMPLETE)
    assert stats.should_reset(ctx) is None  # 1 in session, threshold=2


def test_session_reset_ignores_pre_turn_promotions(tmp_path):
    """Regression for review finding: if the first turn's mix_compile promotes
    files that were IN_PROGRESS at session start (from a PREVIOUS session's
    work), those promotions should NOT count against this session's threshold.
    """
    ctx = _fake_ctx(tmp_path, [
        _entry("PrevA", FileState.IN_PROGRESS),
        _entry("PrevB", FileState.IN_PROGRESS),
        _entry("New", FileState.NOT_STARTED),
    ], reset_files=1)
    stats = _SessionStats()

    # Simulate: first turn calls mix_compile → promotes both IN_PROGRESS to COMPLETE.
    # We ONLY capture start_completed AFTER this happens.
    ctx.state.set_state("PrevA", FileState.COMPLETE)
    ctx.state.set_state("PrevB", FileState.COMPLETE)
    stats.note_first_turn(ctx)  # captures start=2

    # Now do actual work — one new completion
    ctx.state.set_state("New", FileState.COMPLETE)
    reason = stats.should_reset(ctx)
    assert reason is not None  # 1 in session (threshold=1)


# ---------------------------------------------------------------------------
# _build_initial_message
# ---------------------------------------------------------------------------

def test_initial_message_session_1_has_workflow(tmp_path):
    ctx = _fake_ctx(tmp_path, [_entry("A")])
    msg = _build_initial_message(ctx, session_num=1)
    # Prompt uses "IMMEDIATE ACTIONS" now (was "WORKFLOW:") and stresses writing
    assert "IMMEDIATE ACTIONS" in msg
    assert "list_files" in msg
    assert "level-0" in msg
    # Regression: session 1 must instruct the model to write early
    assert "write_elixir" in msg


def test_initial_message_session_2_has_state_summary(tmp_path):
    ctx = _fake_ctx(tmp_path, [
        _entry("A", FileState.COMPLETE),
        _entry("B", FileState.NOT_STARTED),
        _entry("C", FileState.NOT_STARTED),
    ])
    msg = _build_initial_message(ctx, session_num=2)
    assert "Continuation session #2" in msg
    assert "Completed: 1" in msg
    assert "Still to do: 2" in msg
    assert "B" in msg
    assert "C" in msg


# ---------------------------------------------------------------------------
# _wrap_up_message and _bomb_nudge_message
# ---------------------------------------------------------------------------

def test_wrap_up_message_role_and_content():
    m = _wrap_up_message("10% budget remaining")
    assert isinstance(m, HumanMessage)
    assert "Budget warning" in m.content
    assert "validation_status" in m.content


def test_bomb_nudge_message_role_and_content():
    m = _bomb_nudge_message(7)
    assert isinstance(m, HumanMessage)
    assert "7 files" in m.content
    assert "mix_compile_tool" in m.content


# ---------------------------------------------------------------------------
# SessionContext cross-session state (rate_limit_streak, remaining_task_tokens)
# ---------------------------------------------------------------------------

def test_session_context_persists_rate_limit_streak(tmp_path):
    """Regression test for Phase 3 review bug: rate_limit_streak was a local
    variable in _run_one_session, meaning it reset to 0 on every session
    and the streak logic never triggered. It must now live on SessionContext.
    """
    from agent.session_ctx import SessionContext
    import dataclasses
    # Just verify the field exists and defaults to 0 — full loop testing
    # requires mocking the SDK which is out of scope here.
    fields = {f.name for f in dataclasses.fields(SessionContext)}
    assert "rate_limit_streak" in fields
    assert "remaining_task_tokens" in fields
    assert "files_written_since_compile" in fields


def test_session_context_defaults(tmp_path):
    """SessionContext dataclass defaults for the mutable counters."""
    from agent.session_ctx import SessionContext
    import dataclasses
    field_map = {f.name: f for f in dataclasses.fields(SessionContext)}
    # Default = 0 for all counters
    assert field_map["rate_limit_streak"].default == 0
    assert field_map["remaining_task_tokens"].default == 0
    assert field_map["files_written_since_compile"].default == 0


# ---------------------------------------------------------------------------
# _run_one_graph_session — empty-tool-calls nudge (Bug 3)
# ---------------------------------------------------------------------------

def _make_ai_msg(text: str, tool_calls: list | None = None) -> AIMessage:
    """Build an AIMessage with optional tool_calls (empty list = text-only response)."""
    return AIMessage(content=text, tool_calls=tool_calls or [])


def _fake_session_ctx(tmp_path: Path) -> SimpleNamespace:
    """Minimal ctx object for _run_one_graph_session tests."""
    state = StateStore(tmp_path)
    cfg = SimpleNamespace(
        llm=SimpleNamespace(model="llama3.1:8b"),
        agent=SimpleNamespace(
            max_budget_tokens=1_000_000,
            max_turn_seconds=60.0,
            session_reset_after_files=10,
            session_reset_after_tokens=500_000,
        ),
        source=SimpleNamespace(root=tmp_path),
    )
    events = MagicMock()
    events.emit = MagicMock()
    cost = MagicMock()
    cost.add_turn = MagicMock()
    cost.cumulative_cost_usd = MagicMock(return_value=0.0)
    budget = MagicMock()
    budget.add_usage = MagicMock()
    budget.remaining_pct = MagicMock(return_value=100.0)
    budget.status = MagicMock(return_value=None)
    budget.total_cost_usd = 0.0
    ctx = SimpleNamespace(
        cfg=cfg,
        state=state,
        events=events,
        cost=cost,
        budget=budget,
        rate_limit_streak=0,
        remaining_task_tokens=1_000_000,
        files_written_since_compile=0,
        polish_active=False,
        polish_reads_since_edit=0,
    )
    return ctx


def _fake_bundle(forces_tool_call: bool) -> SimpleNamespace:
    return SimpleNamespace(forces_tool_call=forces_tool_call)


def _make_graph_stream(chunks_per_call: list[list[dict]]):
    """Return a graph mock whose stream() yields successive chunk lists.

    chunks_per_call[0] is yielded on the first graph.stream() call,
    chunks_per_call[1] on the second, etc.
    """
    call_count = [0]

    def stream(state, stream_mode):  # noqa: ARG001
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(chunks_per_call):
            yield from chunks_per_call[idx]

    graph = MagicMock()
    graph.stream = stream
    return graph


def test_empty_tool_calls_nudges_once_then_accepts_end_turn(tmp_path):
    """Ollama (forces_tool_call=False) returns 0 tool calls twice.

    First attempt: nudge injected, stream restarted.
    Second attempt: still no tool calls, accept end_turn as terminal.
    Result: GraphSessionResult with reason="end_turn".
    """
    ai_text_only = _make_ai_msg("Here is my plan")
    chunk_text = {"messages": [ai_text_only], "finish_called": False,
                  "finish_summary": "", "finish_unfixable": ()}

    graph = _make_graph_stream([
        [chunk_text],    # call 1: text-only → nudge injected
        [chunk_text],    # call 2: text-only again → accept end_turn
    ])

    ctx = _fake_session_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=False)

    result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[SystemMessage(content="sys")],
        session_num=1,
    )

    assert result.reason == "end_turn"
    # Nudge warning emitted on attempt 1
    warn_calls = [str(call) for call in ctx.events.emit.call_args_list
                  if "nudging" in str(call)]
    assert len(warn_calls) == 1


def test_empty_tool_calls_forces_tool_call_skips_nudge(tmp_path):
    """Anthropic (forces_tool_call=True) returns 0 tool calls — no nudge, immediate end_turn."""
    ai_text_only = _make_ai_msg("Here is my plan")
    chunk_text = {"messages": [ai_text_only], "finish_called": False,
                  "finish_summary": "", "finish_unfixable": ()}

    graph = _make_graph_stream([[chunk_text]])

    ctx = _fake_session_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=True)

    result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[SystemMessage(content="sys")],
        session_num=1,
    )

    assert result.reason == "end_turn"
    # No nudge emitted for Anthropic path
    nudge_calls = [str(call) for call in ctx.events.emit.call_args_list
                   if "nudging" in str(call)]
    assert len(nudge_calls) == 0


def test_nudge_message_injects_tool_call_instruction(tmp_path):
    """The nudge HumanMessage tells the model to call a tool."""
    ai_text_only = _make_ai_msg("Thinking about this...")
    chunk_text = {"messages": [ai_text_only], "finish_called": False,
                  "finish_summary": "", "finish_unfixable": ()}

    injected_messages: list = []

    def stream(state, stream_mode):  # noqa: ARG001
        # Record the messages that were passed to the second invocation
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            injected_messages.extend(state["messages"])
        yield chunk_text  # always return text-only to trigger end_turn

    graph = MagicMock()
    graph.stream = stream

    ctx = _fake_session_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=False)

    _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[SystemMessage(content="sys")],
        session_num=1,
    )

    # The nudge message must have been injected into the second call
    assert injected_messages, "nudge was never injected"
    last = injected_messages[-1]
    assert isinstance(last, HumanMessage)
    assert "write_elixir" in last.content
    assert "finish_translate" in last.content
