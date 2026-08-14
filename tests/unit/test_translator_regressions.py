"""Regression tests for the translator refactor + latent-bug audit.

Covers:
  1. `run_translator_phase2` on a fully-terminal resume must not `UnboundLocalError`
     on `session_result` (the local was only assigned inside the loop; if
     `_all_files_terminal` fired on iteration 1, the post-loop `session_result.
     finish_summary` reference crashed the whole process).

  2. `_ChunkAction` invariant: continue vs terminate vs inject — the three
     mutually-exclusive shapes returned by `_process_agent_chunk` /
     `_process_tool_chunk`. Cheap unit coverage that the refactored helpers
     do not silently mix intents.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.state import FileEntry, FileState, StateStore
from agent.translator import (
    _CONTINUE,
    _ChunkAction,
    _handle_stream_exception,
    _process_agent_chunk,
    _process_tool_chunk,
    _record_turn_usage,
    _SessionStats,
    GraphSessionResult,
    run_translator_phase2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(stem: str, state: FileState = FileState.COMPLETE) -> FileEntry:
    return FileEntry(
        stem=stem,
        java_path=f"{stem}.java",
        lib_path=f"lib/x/{stem.lower()}.ex",
        test_path=f"test/x/{stem.lower()}_test.exs",
        module_name=f"X.{stem}",
        state=state,
    )


def _ai_with_calls() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "write_elixir",
            "args": {"module_or_path": "Foo", "contents": "x"},
            "id": "call_abc",
            "type": "tool_call",
        }],
    )


def _tool_msg() -> ToolMessage:
    return ToolMessage(content='{"ok": true}', tool_call_id="call_abc")


def _stats() -> _SessionStats:
    return _SessionStats()


def _bundle(forces_tool_call: bool = True) -> SimpleNamespace:
    return SimpleNamespace(forces_tool_call=forces_tool_call)


def _fake_ctx(tmp_path: Path, *, budget_status=None, budget_remaining_pct=100.0,
              files_written=0) -> SimpleNamespace:
    """Minimal SessionContext-shaped object for the chunk-handler tests."""
    state = StateStore(tmp_path)
    cfg = SimpleNamespace(
        llm=SimpleNamespace(model="gpt-5"),
        agent=SimpleNamespace(
            max_budget_tokens=1_000_000,
            max_turn_seconds=60.0,
            session_reset_after_files=100,
            session_reset_after_tokens=5_000_000,
        ),
        source=SimpleNamespace(root=tmp_path),
    )
    events = MagicMock()
    cost = MagicMock()
    cost.cumulative_cost_usd = MagicMock(return_value=1.0)
    budget = MagicMock()
    budget.remaining_pct = MagicMock(return_value=budget_remaining_pct)
    budget.status = MagicMock(return_value=budget_status)
    budget.total_cost_usd = 0.0
    return SimpleNamespace(
        cfg=cfg, state=state, events=events, cost=cost, budget=budget,
        rate_limit_streak=0, remaining_task_tokens=1_000_000,
        files_written_since_compile=files_written,
        polish_active=False, polish_reads_since_edit=0,
    )


# ---------------------------------------------------------------------------
# Regression: run_translator_phase2 must survive fully-terminal resume
# ---------------------------------------------------------------------------

def test_run_translator_phase2_survives_all_terminal_on_first_iteration(tmp_path):
    """Bug: `session_result` was only assigned inside the while loop. If
    `_all_files_terminal(ctx)` returned True on iteration 1 (the common
    `--resume` outcome once a project is fully translated), the post-loop
    `session_result.finish_summary` reference raised UnboundLocalError,
    crashing the whole run. Fix: capture `last_session_result` initialized
    to None, guard the access.
    """
    state = StateStore(tmp_path)
    for e in [_entry("A", FileState.COMPLETE), _entry("B", FileState.SKIPPED)]:
        state.upsert(e)

    # `SessionContext.scaffold.target_root` and `SessionContext.scaffold.mix_env`
    # get read by _synthesize_outcome_from_state → mix_ops. Since we have no
    # COMPLETE files that need real mix runs, the synthesized outcome takes
    # the `has_lib` path. Provide a target_root that exists.
    scaffold_stub = SimpleNamespace(
        target_root=tmp_path,
        mix_env=SimpleNamespace(mix_path="/bin/false", env={}),
    )

    events = MagicMock()
    cost = MagicMock()
    cost.cumulative_cost_usd = MagicMock(return_value=0.0)
    cost.model = "claude-opus-4-7"
    budget = MagicMock()
    budget.status = MagicMock(return_value=None)
    budget.remaining_pct = MagicMock(return_value=100.0)
    budget.total_cost_usd = 0.0
    budget._max_cost_usd = 100.0

    cfg = SimpleNamespace(
        llm=SimpleNamespace(model="claude-opus-4-7"),
        agent=SimpleNamespace(
            max_budget_tokens=1_000_000,
            max_tool_calls=100,
            max_wall_seconds=3600,
            max_turn_seconds=60.0,
            session_reset_after_files=10,
            session_reset_after_tokens=500_000,
            thinking_budget_tokens=0,
            effort="medium",
        ),
        source=SimpleNamespace(root=tmp_path),
    )
    ctx = SimpleNamespace(
        cfg=cfg, scaffold=scaffold_stub, state=state, events=events,
        cost=cost, budget=budget,
        java_classes=[], dep_graph=SimpleNamespace(levels=[], sccs=[]),
        source_root=tmp_path, target_root=tmp_path,
        module_prefix="X", app_snake="x",
        rate_limit_streak=0, remaining_task_tokens=0,
        files_written_since_compile=0,
        polish_active=False, polish_reads_since_edit=0,
    )
    bundle = SimpleNamespace(
        forces_tool_call=True, supports_prompt_caching=True,
        chat=MagicMock(),
    )

    # Patch build_translation_graph so we never actually construct one and never
    # need real tools/LLM. `_all_files_terminal(ctx)` returns True immediately so
    # the graph is never streamed anyway.
    with patch("agent.translator.build_translation_graph", return_value=MagicMock()), \
         patch("agent.translator.build_polish_graph", return_value=MagicMock()), \
         patch("agent.translator._maybe_run_polish_sessions",
               return_value=(None, 0, ())):
        # Must not raise. Pre-fix: UnboundLocalError on session_result.
        outcome, sessions = run_translator_phase2(ctx, bundle=bundle, dry_run=False)

    # Sessions should be 1 (loop entered once, hit all_terminal, broke).
    assert sessions == 1
    # Outcome synthesized from disk state, not from a finish_translate call.
    assert outcome is not None
    assert "synthesized" in outcome.summary or outcome.summary == ""


# ---------------------------------------------------------------------------
# _ChunkAction invariant
# ---------------------------------------------------------------------------

def test_chunk_action_continue_is_singleton_shape():
    """_CONTINUE has both fields None. Terminate and inject are mutually exclusive."""
    assert _CONTINUE.terminate is None
    assert _CONTINUE.inject is None

    term = _ChunkAction(terminate=GraphSessionResult(None, "x", "", ()))
    assert term.terminate is not None
    assert term.inject is None

    inj = _ChunkAction(inject=HumanMessage(content="hi"))
    assert inj.terminate is None
    assert inj.inject is not None


# ---------------------------------------------------------------------------
# _process_agent_chunk unit tests
# ---------------------------------------------------------------------------

def test_process_agent_chunk_ollama_no_tool_calls_nudges(tmp_path):
    """Ollama (forces_tool_call=False) + empty tool_calls + attempt 0 → inject nudge."""
    ctx = _fake_ctx(tmp_path)
    stats = _stats()
    empty = [0]
    ai = AIMessage(content="thinking", tool_calls=[])

    action = _process_agent_chunk(
        ctx, last_msg=ai, stats=stats, session_num=1, polish_mode=False,
        bundle=_bundle(forces_tool_call=False), empty_response_state=empty,
    )
    assert action.inject is not None
    assert action.terminate is None
    assert empty[0] == 1
    assert "write_elixir" in action.inject.content


def test_process_agent_chunk_ollama_second_attempt_terminates(tmp_path):
    """After the one bounded nudge, subsequent empty tool_calls terminate with end_turn."""
    ctx = _fake_ctx(tmp_path)
    stats = _stats()
    empty = [1]  # already nudged once
    ai = AIMessage(content="still thinking", tool_calls=[])

    action = _process_agent_chunk(
        ctx, last_msg=ai, stats=stats, session_num=1, polish_mode=False,
        bundle=_bundle(forces_tool_call=False), empty_response_state=empty,
    )
    assert action.inject is None
    assert action.terminate is not None
    assert action.terminate.reason == "end_turn"


def test_process_agent_chunk_anthropic_no_tool_calls_terminates_immediately(tmp_path):
    """Anthropic (forces_tool_call=True) + empty tool_calls → end_turn without nudge."""
    ctx = _fake_ctx(tmp_path)
    stats = _stats()
    empty = [0]
    ai = AIMessage(content="I'm done", tool_calls=[])

    action = _process_agent_chunk(
        ctx, last_msg=ai, stats=stats, session_num=1, polish_mode=False,
        bundle=_bundle(forces_tool_call=True), empty_response_state=empty,
    )
    assert action.terminate is not None
    assert action.terminate.reason == "end_turn"
    assert empty[0] == 0  # no nudge attempted


def test_process_agent_chunk_hard_budget_cap_terminates(tmp_path):
    """budget.status() returns non-None → terminate with budget_hit."""
    ctx = _fake_ctx(tmp_path, budget_status="cost cap reached ($10)")
    stats = _stats()
    ai = _ai_with_calls()

    action = _process_agent_chunk(
        ctx, last_msg=ai, stats=stats, session_num=1, polish_mode=False,
        bundle=_bundle(), empty_response_state=[0],
    )
    assert action.terminate is not None
    assert action.terminate.reason == "budget_hit"


def test_process_agent_chunk_records_turn_usage(tmp_path):
    """AIMessage with tool_calls updates stats.turns and rate_limit_streak."""
    ctx = _fake_ctx(tmp_path)
    ctx.rate_limit_streak = 2
    stats = _stats()
    ai = _ai_with_calls()

    action = _process_agent_chunk(
        ctx, last_msg=ai, stats=stats, session_num=1, polish_mode=False,
        bundle=_bundle(), empty_response_state=[0],
    )
    assert action == _CONTINUE
    assert stats.turns == 1
    assert ctx.rate_limit_streak == 0
    ctx.events.emit.assert_any_call(
        "turn",
        session=1,
        turn=1,
        tokens={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
        cumulative_cost_usd=1.0,
        budget_remaining_pct=100.0,
        remaining_task_tokens=1_000_000,
    )


# ---------------------------------------------------------------------------
# _process_tool_chunk unit tests
# ---------------------------------------------------------------------------

def test_process_tool_chunk_finish_called_terminates_with_summary(tmp_path):
    ctx = _fake_ctx(tmp_path)
    stats = _stats()
    chunk = {
        "messages": [_ai_with_calls(), _tool_msg()],
        "finish_called": True,
        "finish_summary": "translated 3 files",
        "finish_unfixable": (),
    }
    action = _process_tool_chunk(
        ctx, chunk=chunk, stats=stats, session_num=1, polish_mode=False,
        sentinel_reason="finish_translate",
    )
    assert action.terminate is not None
    assert action.terminate.reason == "finish_translate"
    assert action.terminate.finish_summary == "translated 3 files"


def test_process_tool_chunk_soft_budget_warning_injects(tmp_path):
    ctx = _fake_ctx(tmp_path, budget_remaining_pct=5.0)  # under 10% → warn
    stats = _stats()
    chunk = {
        "messages": [_ai_with_calls(), _tool_msg()],
        "finish_called": False,
    }
    action = _process_tool_chunk(
        ctx, chunk=chunk, stats=stats, session_num=1, polish_mode=False,
        sentinel_reason="finish_translate",
    )
    assert action.inject is not None
    assert "Budget warning" in action.inject.content
    assert stats.warned_on_budget is True


def test_process_tool_chunk_soft_budget_warning_fires_once(tmp_path):
    ctx = _fake_ctx(tmp_path, budget_remaining_pct=5.0)
    stats = _stats()
    stats.warned_on_budget = True  # already warned
    chunk = {
        "messages": [_ai_with_calls(), _tool_msg()],
        "finish_called": False,
    }
    action = _process_tool_chunk(
        ctx, chunk=chunk, stats=stats, session_num=1, polish_mode=False,
        sentinel_reason="finish_translate",
    )
    assert action == _CONTINUE


def test_process_tool_chunk_compile_nudge_injects_when_threshold_met(tmp_path):
    from agent.translator import _COMPILE_NUDGE_THRESHOLD
    ctx = _fake_ctx(tmp_path, files_written=_COMPILE_NUDGE_THRESHOLD)
    stats = _stats()
    chunk = {
        "messages": [_ai_with_calls(), _tool_msg()],
        "finish_called": False,
    }
    action = _process_tool_chunk(
        ctx, chunk=chunk, stats=stats, session_num=1, polish_mode=False,
        sentinel_reason="finish_translate",
    )
    assert action.inject is not None
    assert "mix_compile_tool" in action.inject.content
    assert stats.nudged_on_compile is True


def test_process_tool_chunk_polish_mode_skips_wrap_up(tmp_path):
    """In polish mode, wrap-up/compile-nudge/reset should be suppressed."""
    ctx = _fake_ctx(tmp_path, budget_remaining_pct=5.0)
    stats = _stats()
    chunk = {
        "messages": [_ai_with_calls(), _tool_msg()],
        "finish_called": False,
    }
    action = _process_tool_chunk(
        ctx, chunk=chunk, stats=stats, session_num=1, polish_mode=True,
        sentinel_reason="finish_polish",
    )
    assert action == _CONTINUE


# ---------------------------------------------------------------------------
# _handle_stream_exception unit tests
# ---------------------------------------------------------------------------

def test_handle_stream_exception_pydantic_validation_error_returns_parse_failure(tmp_path):
    import pydantic
    ctx = _fake_ctx(tmp_path)

    class M(pydantic.BaseModel):
        x: int

    try:
        M(x="not an int")  # trigger a ValidationError
    except pydantic.ValidationError as exc:
        result = _handle_stream_exception(ctx, exc, session_num=1)

    assert result.reason == "parse_failure"


def test_handle_stream_exception_bad_request_reraises(tmp_path):
    ctx = _fake_ctx(tmp_path)

    class FakeBadRequest(Exception):
        pass

    # classify_exception falls through to UNKNOWN for a bare Exception — we
    # need to force BAD_REQUEST via monkeypatching.
    from agent import translator as t
    from agent.llm import ExceptionKind

    with patch.object(t, "classify_exception", return_value=ExceptionKind.BAD_REQUEST):
        exc = FakeBadRequest("400 malformed payload")
        try:
            _handle_stream_exception(ctx, exc, session_num=1)
        except FakeBadRequest as re:
            assert re is exc  # re-raised the same exception
        else:
            raise AssertionError("expected re-raise")


def test_handle_stream_exception_rate_limit_bumps_streak(tmp_path):
    from agent import translator as t
    from agent.llm import ExceptionKind

    ctx = _fake_ctx(tmp_path)
    assert ctx.rate_limit_streak == 0

    with patch.object(t, "classify_exception", return_value=ExceptionKind.RATE_LIMIT):
        result = _handle_stream_exception(ctx, RuntimeError("429"), session_num=1)

    assert result.reason == "rate_limit_soft"
    assert ctx.rate_limit_streak == 1


def test_handle_stream_exception_rate_limit_hard_after_streak(tmp_path):
    from agent import translator as t
    from agent.llm import ExceptionKind

    ctx = _fake_ctx(tmp_path)
    ctx.rate_limit_streak = t._MAX_RATE_LIMIT_STREAK - 1  # one away from hard

    with patch.object(t, "classify_exception", return_value=ExceptionKind.RATE_LIMIT):
        result = _handle_stream_exception(ctx, RuntimeError("429"), session_num=1)

    assert result.reason == "rate_limit_hard"


def test_handle_stream_exception_timeout_returns_checkpoint(tmp_path):
    from agent import translator as t
    from agent.llm import ExceptionKind

    ctx = _fake_ctx(tmp_path)
    with patch.object(t, "classify_exception", return_value=ExceptionKind.TIMEOUT):
        result = _handle_stream_exception(ctx, TimeoutError("slow"), session_num=1)

    assert result.reason == "checkpoint"


def test_handle_stream_exception_unknown_returns_parse_failure(tmp_path):
    from agent import translator as t
    from agent.llm import ExceptionKind

    ctx = _fake_ctx(tmp_path)
    with patch.object(t, "classify_exception", return_value=ExceptionKind.UNKNOWN):
        result = _handle_stream_exception(ctx, RuntimeError("mystery"), session_num=1)

    assert result.reason == "parse_failure"
