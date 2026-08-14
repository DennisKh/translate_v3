"""Regression tests for the AIMessage(tool_calls) → ToolMessage pairing invariant.

Bug (java-string-similarity, OpenAI live run):
  wrap-up and compile-nudge injections fired inside the AIMessage branch of
  _run_one_graph_session. When the model called write_elixir for the Nth time
  (N >= _COMPILE_NUDGE_THRESHOLD), the agent-node chunk arrived with
  AIMessage(tool_calls=[write_elixir_N]). The compile-nudge check fired, broke
  the stream, and injected a HumanMessage:

      [..., AIMessage(tool_calls=[write_elixir_N]), HumanMessage("compile now")]

  The tools-node never ran, so the tool_call_id was never paired with a
  ToolMessage. OpenAI strict pairing: HTTP 400.
  Anthropic lenient pairing: silently accepted (fast-uuid never hit threshold).

Fix: both injections moved to the ToolMessage branch. They fire only after
tools-node completes and the pairing invariant is closed:

      [..., AIMessage(tool_calls=[X]), ToolMessage(tool_call_id=X), HumanMessage("...")]

These tests FAIL on pre-fix code and PASS on post-fix code:
  1. compile-nudge: inject must follow ToolMessage, not AIMessage
  2. compile-nudge: trigger actually fires when threshold is met
  3. wrap-up: inject must follow ToolMessage, not AIMessage
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.translator import (
    _BUDGET_WARN_PCT,
    _COMPILE_NUDGE_THRESHOLD,
    _run_one_graph_session,
)


def _fake_session_ctx(tmp_path: Path) -> SimpleNamespace:
    """Minimal SessionContext for injection-invariant tests."""
    from agent.state import StateStore

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
    events.emit = MagicMock()
    cost = MagicMock()
    cost.add_turn = MagicMock()
    cost.cumulative_cost_usd = MagicMock(return_value=1.00)
    budget = MagicMock()
    budget.add_usage = MagicMock()
    budget.remaining_pct = MagicMock(return_value=100.0)
    budget.status = MagicMock(return_value=None)
    budget.total_cost_usd = 0.0
    return SimpleNamespace(
        cfg=cfg,
        state=state,
        events=events,
        cost=cost,
        budget=budget,
        rate_limit_streak=0,
        remaining_task_tokens=5_000_000,
        files_written_since_compile=0,
        polish_active=False,
        polish_reads_since_edit=0,
    )


def _bundle_openai() -> SimpleNamespace:
    """Simulate an OpenAI bundle — forces_tool_call=True (strict pairing)."""
    return SimpleNamespace(forces_tool_call=True)


def _ai_msg_with_tool_calls(tool_call_id: str = "call_Npphu8QiezzwLmt1MGelYdeY") -> AIMessage:
    """AIMessage with pending tool_calls — opens the pairing invariant."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "write_elixir",
            "args": {"module_or_path": "Foo", "contents": "defmodule Foo do\nend\n"},
            "id": tool_call_id,
            "type": "tool_call",
        }],
    )


def _tool_msg_for(ai_msg: AIMessage) -> ToolMessage:
    """ToolMessage that closes the pairing invariant for the given AIMessage."""
    return ToolMessage(content='{"ok": true}', tool_call_id=ai_msg.tool_calls[0]["id"])


def _ai_msg_end_turn() -> AIMessage:
    """AIMessage with no tool_calls — triggers end_turn exit."""
    return AIMessage(content="done", tool_calls=[])


def _assert_pairing_invariant(messages: list, context: str = "") -> None:
    """Assert that every AIMessage(tool_calls) is immediately followed by a ToolMessage."""
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            assert i + 1 < len(messages), (
                f"{context}: AIMessage(tool_calls) at index {i} has no successor"
            )
            assert isinstance(messages[i + 1], ToolMessage), (
                f"{context}: AIMessage(tool_calls) at index {i} followed by "
                f"{type(messages[i + 1]).__name__}, expected ToolMessage. "
                f"Pairing invariant violated — this is the OpenAI 400 bug."
            )


# ---------------------------------------------------------------------------
# Test 1: compile-nudge must NOT inject after AIMessage; must inject after ToolMessage
# ---------------------------------------------------------------------------

def test_compile_nudge_injects_after_tool_message_not_after_ai_message(tmp_path):
    """Real bug scenario: N files written (N >= threshold) when agent issues
    another write_elixir call. The agent-node chunk arrives with
    AIMessage(tool_calls=[write_elixir_N+1]).

    Pre-fix: compile-nudge fired in AIMessage branch → broke with
      [..., AIMessage(tool_calls=[X]), HumanMessage(nudge)] → OpenAI 400.

    Post-fix: AIMessage branch skips injection, ToolMessage branch fires →
      [..., AIMessage(tool_calls=[X]), ToolMessage(tool_call_id=X), HumanMessage(nudge)]
      Pairing invariant preserved.
    """
    ai_with_calls = _ai_msg_with_tool_calls()
    tool_msg = _tool_msg_for(ai_with_calls)
    ai_end = _ai_msg_end_turn()

    base = {"finish_called": False, "finish_summary": "", "finish_unfixable": ()}
    sys_msg = SystemMessage(content="sys")

    # Accumulate full message history as the real graph does (values mode = full state)
    msg_history = [sys_msg]
    chunk_agent = {**base, "messages": msg_history + [ai_with_calls]}
    chunk_tools = {**base, "messages": msg_history + [ai_with_calls, tool_msg]}

    ctx = _fake_session_ctx(tmp_path)
    bundle = _bundle_openai()

    call_count = [0]
    injected_state: list = []

    def stream(state, stream_mode):  # noqa: ARG001
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            # Simulate N prior write_elixir calls in this session.
            # Set the counter BEFORE yielding the AIMessage chunk — this is exactly
            # the real scenario: files were written in prior turns, counter is at
            # threshold when the next agent-node fires.
            ctx.files_written_since_compile = _COMPILE_NUDGE_THRESHOLD
            yield chunk_agent   # agent-node: AIMessage with pending tool_calls
            yield chunk_tools   # tools-node: ToolMessage closes the pairing
        else:
            injected_state.append(state)
            end_chunk = {**base, "messages": list(state["messages"]) + [ai_end]}
            yield end_chunk

    graph = MagicMock()
    graph.stream = stream

    result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[sys_msg],
        session_num=1,
    )

    assert result.reason == "end_turn", f"expected end_turn, got {result.reason!r}"
    assert call_count[0] == 2, (
        f"compile-nudge must trigger a second stream.stream() call; got {call_count[0]}"
    )
    assert injected_state, "no state captured on second stream call"

    second_call_msgs = injected_state[0]["messages"]

    # Locate the injected compile-nudge HumanMessage
    nudge_positions = [
        i for i, m in enumerate(second_call_msgs)
        if isinstance(m, HumanMessage) and "mix_compile_tool" in m.content
    ]
    assert nudge_positions, (
        "compile-nudge HumanMessage not found in second stream call's state — "
        "nudge was never injected"
    )

    nudge_idx = nudge_positions[0]
    assert nudge_idx > 0, "nudge HumanMessage has no predecessor"

    # KEY ASSERTION: message immediately before the inject must be ToolMessage
    msg_before = second_call_msgs[nudge_idx - 1]
    assert isinstance(msg_before, ToolMessage), (
        f"Message before compile-nudge HumanMessage must be ToolMessage, "
        f"got {type(msg_before).__name__}. "
        f"This is the OpenAI 400 bug: inject happened before tool_call was paired."
    )

    # Full pairing invariant on the injected message sequence
    _assert_pairing_invariant(second_call_msgs, context="compile-nudge injected state")


# ---------------------------------------------------------------------------
# Test 2: compile-nudge does fire (belt-and-suspenders: not just branch logic)
# ---------------------------------------------------------------------------

def test_compile_nudge_does_trigger_when_threshold_met(tmp_path):
    """Belt-and-suspenders: compile-nudge must actually fire (inject a HumanMessage
    and restart the stream) when files_written_since_compile >= threshold.

    A naive "fix" could move the check but never trigger it. This test asserts the
    nudge actually fires (call_count == 2) and the content is correct.
    """
    ai_with_calls = _ai_msg_with_tool_calls()
    tool_msg = _tool_msg_for(ai_with_calls)
    ai_end = _ai_msg_end_turn()

    base = {"finish_called": False, "finish_summary": "", "finish_unfixable": ()}
    sys_msg = SystemMessage(content="sys")

    ctx = _fake_session_ctx(tmp_path)
    bundle = _bundle_openai()

    call_count = [0]
    injected_state: list = []

    def stream(state, stream_mode):  # noqa: ARG001
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            ctx.files_written_since_compile = _COMPILE_NUDGE_THRESHOLD
            yield {**base, "messages": [sys_msg, ai_with_calls]}
            yield {**base, "messages": [sys_msg, ai_with_calls, tool_msg]}
        else:
            injected_state.append(state)
            yield {**base, "messages": list(state["messages"]) + [ai_end]}

    graph = MagicMock()
    graph.stream = stream

    result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[sys_msg],
        session_num=1,
    )

    assert call_count[0] == 2, (
        f"compile-nudge must restart the stream (expected 2 calls, got {call_count[0]})"
    )
    assert injected_state, "no state captured on second call"

    msgs = injected_state[0]["messages"]
    has_nudge = any(
        isinstance(m, HumanMessage) and "mix_compile_tool" in m.content
        for m in msgs
    )
    assert has_nudge, "compile-nudge HumanMessage not found in injected state"
    assert result.reason == "end_turn"

    # Confirm "info" event was emitted for the nudge
    info_calls = [
        str(c) for c in ctx.events.emit.call_args_list
        if "nudging model to compile" in str(c)
    ]
    assert info_calls, "nudging-model info event not emitted"


# ---------------------------------------------------------------------------
# Test 3: wrap-up (soft budget warning) pairing invariant
# ---------------------------------------------------------------------------

def test_wrap_up_injects_after_tool_message_not_after_ai_message(tmp_path):
    """Same invariant as Test 1, triggered by soft budget warning instead.

    Pre-fix: wrap-up fired in AIMessage branch when budget_pct < 10%, producing:
      [..., AIMessage(tool_calls=[X]), HumanMessage("Budget warning...")] → OpenAI 400.

    Post-fix: wrap-up fires in ToolMessage branch only:
      [..., AIMessage(tool_calls=[X]), ToolMessage(tool_call_id=X), HumanMessage("Budget warning...")]
    """
    ai_with_calls = _ai_msg_with_tool_calls()
    tool_msg = _tool_msg_for(ai_with_calls)
    ai_end = _ai_msg_end_turn()

    base = {"finish_called": False, "finish_summary": "", "finish_unfixable": ()}
    sys_msg = SystemMessage(content="sys")

    ctx = _fake_session_ctx(tmp_path)
    bundle = _bundle_openai()

    # Budget at 5% → below (100 - _BUDGET_WARN_PCT = 10%) → wrap-up fires
    ctx.budget.remaining_pct = MagicMock(return_value=5.0)

    call_count = [0]
    injected_state: list = []

    def stream(state, stream_mode):  # noqa: ARG001
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            yield {**base, "messages": [sys_msg, ai_with_calls]}
            yield {**base, "messages": [sys_msg, ai_with_calls, tool_msg]}
        else:
            injected_state.append(state)
            yield {**base, "messages": list(state["messages"]) + [ai_end]}

    graph = MagicMock()
    graph.stream = stream

    result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=[sys_msg],
        session_num=1,
    )

    assert result.reason == "end_turn", f"expected end_turn, got {result.reason!r}"
    assert call_count[0] == 2, (
        f"wrap-up must restart the stream (expected 2 calls, got {call_count[0]})"
    )
    assert injected_state, "no state captured on second stream call"

    second_msgs = injected_state[0]["messages"]

    # Locate the wrap-up HumanMessage
    wu_positions = [
        i for i, m in enumerate(second_msgs)
        if isinstance(m, HumanMessage) and "Budget warning" in m.content
    ]
    assert wu_positions, (
        "wrap-up HumanMessage not found in injected state — "
        "budget warning was never injected"
    )

    wu_idx = wu_positions[0]
    assert wu_idx > 0, "wrap-up HumanMessage has no predecessor"

    # KEY ASSERTION: message immediately before the inject must be ToolMessage
    msg_before = second_msgs[wu_idx - 1]
    assert isinstance(msg_before, ToolMessage), (
        f"Message before wrap-up HumanMessage must be ToolMessage, "
        f"got {type(msg_before).__name__}. "
        f"This is the OpenAI 400 bug: inject happened before tool_call was paired."
    )

    # Full pairing invariant on the injected message sequence
    _assert_pairing_invariant(second_msgs, context="wrap-up injected state")
