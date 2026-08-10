"""Unit tests for translator.py helpers — session-reset criteria + state summarization."""

from pathlib import Path
from types import SimpleNamespace

from agent.state import FileEntry, FileState, StateStore
from agent.translator import (
    _all_files_terminal,
    _bomb_nudge_message,
    _build_initial_message,
    _count_terminal,
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
    assert m["role"] == "user"
    assert "Budget warning" in m["content"]
    assert "validation_status" in m["content"]


def test_bomb_nudge_message_role_and_content():
    m = _bomb_nudge_message(7)
    assert m["role"] == "user"
    assert "7 files" in m["content"]
    assert "mix_compile_tool" in m["content"]


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
