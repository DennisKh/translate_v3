"""Unit tests for agent/graph.py — AgentState, routing functions, graph compilation,
and sentinel tool Command integration.

All tests use synthetic messages and mocks. No live LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableBinding

from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agent.graph import (
    AgentState,
    build_polish_graph,
    build_translation_graph,
    route_after_agent,
    route_after_tools,
)
from agent.llm import ChatModelBundle
from agent.state import FileEntry, FileState, StateStore


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------

class _StubChatModel(BaseChatModel):
    """Minimal BaseChatModel stub for compile-only tests.

    bind_tools is overridden to return a RunnableBinding-compatible stub so
    _make_agent_node can call .bind_tools(tools) without importing a real
    provider. tool_choice is not passed via bind_tools; for Anthropic it is
    injected via model_kwargs on the ChatAnthropic instance.
    """

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="stub response"))]
        )

    def bind_tools(self, tools, **kwargs) -> Any:
        # Return a simple callable wrapper so the agent_node closure can call
        # .invoke() without needing a real provider. The graph compile step
        # only calls bind_tools; invoke() is never called in these tests.
        bound = MagicMock()
        bound.invoke = MagicMock(return_value=AIMessage(content="stub"))
        return bound


def _fake_ctx(target_root: Path) -> SimpleNamespace:
    """Minimal SessionContext stub covering all fields tools and graph nodes need."""
    return SimpleNamespace(
        state=StateStore(target_root),
        events=SimpleNamespace(emit=lambda *a, **kw: None),
        cost=SimpleNamespace(
            add_tool=lambda *a, **kw: None,
            add_turn=lambda *a, **kw: None,
            model="claude-sonnet-4-6",
        ),
        source_root=target_root,
        target_root=target_root,
        module_prefix="JodaMoney",
        app_snake="joda_money",
        java_by_stem={},
        tests_root=None,
        files_written_since_compile=0,
        module_to_stem={},
        polish_active=False,
        polish_reads_since_edit=0,
        cfg=SimpleNamespace(source=SimpleNamespace(root=target_root)),
        scaffold=SimpleNamespace(
            mix_env=None,
            target_root=target_root,
            module_name="JodaMoney",
            app_name="joda_money",
        ),
    )


def _fake_bundle(forces_tool_call: bool = True) -> ChatModelBundle:
    return ChatModelBundle(
        chat=_StubChatModel(),
        forces_tool_call=forces_tool_call,
        supports_prompt_caching=False,
        supports_adaptive_thinking=False,
        tool_choice_any_payload={"type": "any"} if forces_tool_call else None,
    )


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# ---------------------------------------------------------------------------
# AgentState — field types and add_messages reducer
# ---------------------------------------------------------------------------

def test_agent_state_add_messages_appends():
    """add_messages reducer appends rather than replacing the messages list."""
    existing = [HumanMessage(content="hello")]
    new_msg = AIMessage(content="world")
    merged = add_messages(existing, [new_msg])
    assert len(merged) == 2
    assert merged[0].content == "hello"
    assert merged[1].content == "world"


def test_agent_state_add_messages_empty_existing():
    msgs = add_messages([], [HumanMessage(content="first")])
    assert len(msgs) == 1
    assert msgs[0].content == "first"


def test_agent_state_fields_present():
    """AgentState TypedDict has the expected keys."""
    keys = set(AgentState.__annotations__)
    assert "messages" in keys
    assert "finish_called" in keys
    assert "finish_summary" in keys
    assert "finish_unfixable" in keys


def test_agent_state_finish_unfixable_type():
    """finish_unfixable is annotated as tuple[str, ...]."""
    import typing
    hints = typing.get_type_hints(AgentState)
    assert hints["finish_unfixable"] == tuple[str, ...]


# ---------------------------------------------------------------------------
# route_after_agent
# ---------------------------------------------------------------------------

def test_route_after_agent_with_tool_calls():
    """Returns 'tools' when last AIMessage has tool_calls."""
    state: AgentState = {
        "messages": [AIMessage(content="", tool_calls=[_tool_call("list_files", {})])],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_agent(state) == "tools"


def test_route_after_agent_no_tool_calls():
    """Returns END when last AIMessage has no tool_calls (Ollama exit path)."""
    state: AgentState = {
        "messages": [AIMessage(content="done, no tool call")],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_agent(state) == END


def test_route_after_agent_multiple_tool_calls():
    """Returns 'tools' when AIMessage has multiple tool_calls."""
    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("list_files", {}, "c1"),
                    _tool_call("read_elixir", {"module_or_path": "Foo"}, "c2"),
                ],
            )
        ],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_agent(state) == "tools"


def test_route_after_agent_empty_tool_calls_list():
    """Returns END when tool_calls is an empty list."""
    state: AgentState = {
        "messages": [AIMessage(content="no tools", tool_calls=[])],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_agent(state) == END


# ---------------------------------------------------------------------------
# route_after_tools
# ---------------------------------------------------------------------------

def test_route_after_tools_not_finished():
    """Returns 'agent' when finish_called is False."""
    state: AgentState = {
        "messages": [],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_tools(state) == "agent"


def test_route_after_tools_finished():
    """Returns END when finish_called is True."""
    state: AgentState = {
        "messages": [],
        "finish_called": True,
        "finish_summary": "done",
        "finish_unfixable": (),
    }
    assert route_after_tools(state) == END


def test_route_after_tools_missing_key():
    """Returns 'agent' when finish_called key is absent (state.get fallback)."""
    # state.get("finish_called") returns None which is falsy
    state = {
        "messages": [],
        "finish_summary": "",
        "finish_unfixable": (),
    }
    assert route_after_tools(state) == "agent"


# ---------------------------------------------------------------------------
# build_translation_graph — compile without error
# ---------------------------------------------------------------------------

def test_build_translation_graph_compiles(tmp_path):
    """build_translation_graph returns a CompiledStateGraph without error."""
    from langgraph.graph.state import CompiledStateGraph

    ctx = _fake_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=True)
    graph = build_translation_graph(ctx, bundle)
    assert isinstance(graph, CompiledStateGraph)


def test_build_translation_graph_no_tool_choice_on_ollama(tmp_path):
    """Graph compiles when forces_tool_call=False (Ollama path)."""
    from langgraph.graph.state import CompiledStateGraph

    ctx = _fake_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=False)
    graph = build_translation_graph(ctx, bundle)
    assert isinstance(graph, CompiledStateGraph)


# ---------------------------------------------------------------------------
# build_polish_graph — compile without error
# ---------------------------------------------------------------------------

def test_build_polish_graph_compiles(tmp_path):
    """build_polish_graph returns a CompiledStateGraph without error."""
    from langgraph.graph.state import CompiledStateGraph

    ctx = _fake_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=True)
    graph = build_polish_graph(ctx, bundle)
    assert isinstance(graph, CompiledStateGraph)


def test_build_polish_graph_no_tool_choice_on_ollama(tmp_path):
    """Graph compiles when forces_tool_call=False (Ollama path)."""
    from langgraph.graph.state import CompiledStateGraph

    ctx = _fake_ctx(tmp_path)
    bundle = _fake_bundle(forces_tool_call=False)
    graph = build_polish_graph(ctx, bundle)
    assert isinstance(graph, CompiledStateGraph)


# ---------------------------------------------------------------------------
# Sentinel tool Command integration
# ---------------------------------------------------------------------------

def _terminal_ctx(target_root: Path) -> SimpleNamespace:
    """Context where all files are in terminal state so finish_translate succeeds."""
    ctx = _fake_ctx(target_root)
    # Seed one COMPLETE file so state.all() is non-empty and all are terminal.
    ctx.state.upsert(FileEntry(
        stem="Money",
        java_path="Money.java",
        lib_path="lib/joda_money/money.ex",
        test_path="test/joda_money/money_test.exs",
        module_name="JodaMoney.Money",
        state=FileState.COMPLETE,
    ))
    return ctx


def test_finish_translate_success_returns_command(tmp_path):
    """finish_translate returns Command(update={finish_called: True, ...}, goto=END)
    when all files are in a terminal state."""
    from agent.tools import build_tools

    ctx = _terminal_ctx(tmp_path)
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    finish_translate = next(t for t in tools if t.name == "finish_translate")

    result = finish_translate.invoke({"summary": "All done"})

    assert isinstance(result, Command), f"Expected Command, got {type(result)}: {result}"
    assert result.update["finish_called"] is True
    assert result.update["finish_summary"] == "All done"
    assert result.update["finish_unfixable"] == ()
    assert result.goto == END


def test_finish_translate_refusal_returns_string(tmp_path):
    """finish_translate returns a JSON error string (not Command) when non-terminal
    files exist. The session must continue."""
    from agent.tools import build_tools

    ctx = _fake_ctx(tmp_path)
    # Seed a non-terminal file so finish_translate refuses.
    ctx.state.upsert(FileEntry(
        stem="Money",
        java_path="Money.java",
        lib_path="lib/joda_money/money.ex",
        test_path="test/joda_money/money_test.exs",
        module_name="JodaMoney.Money",
        state=FileState.NOT_STARTED,
    ))
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    finish_translate = next(t for t in tools if t.name == "finish_translate")

    result = finish_translate.invoke({"summary": "done"})

    # Must be a string (JSON error), not a Command — session continues.
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert "errors" in parsed


def test_finish_translate_summary_truncated_to_800(tmp_path):
    """finish_translate truncates summary to 800 chars in Command.update."""
    from agent.tools import build_tools

    ctx = _terminal_ctx(tmp_path)
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    finish_translate = next(t for t in tools if t.name == "finish_translate")

    long_summary = "x" * 1200
    result = finish_translate.invoke({"summary": long_summary})

    assert isinstance(result, Command)
    assert len(result.update["finish_summary"]) == 800


def test_finish_polish_returns_command(tmp_path):
    """finish_polish always returns Command(update={finish_called: True, ...}, goto=END)."""
    from agent.tools import build_tools

    ctx = _fake_ctx(tmp_path)
    ctx.polish_active = True
    tools = build_tools(ctx, include_write=True, polish_mode=True)
    finish_polish = next(t for t in tools if t.name == "finish_polish")

    result = finish_polish.invoke({
        "summary": "Fixed 2 warnings",
        "unfixable_warnings": ["cyclomatic complexity in big_money.ex"],
    })

    assert isinstance(result, Command), f"Expected Command, got {type(result)}: {result}"
    assert result.update["finish_called"] is True
    assert result.update["finish_summary"] == "Fixed 2 warnings"
    assert result.update["finish_unfixable"] == ("cyclomatic complexity in big_money.ex",)
    assert result.goto == END


def test_finish_polish_summary_truncated_to_400(tmp_path):
    """finish_polish truncates summary to 400 chars in Command.update."""
    from agent.tools import build_tools

    ctx = _fake_ctx(tmp_path)
    tools = build_tools(ctx, include_write=True, polish_mode=True)
    finish_polish = next(t for t in tools if t.name == "finish_polish")

    long_summary = "y" * 600
    result = finish_polish.invoke({
        "summary": long_summary,
        "unfixable_warnings": [],
    })

    assert isinstance(result, Command)
    assert len(result.update["finish_summary"]) == 400


def test_finish_polish_empty_unfixable(tmp_path):
    """finish_polish with no unfixable warnings stores empty tuple."""
    from agent.tools import build_tools

    ctx = _fake_ctx(tmp_path)
    tools = build_tools(ctx, include_write=True, polish_mode=True)
    finish_polish = next(t for t in tools if t.name == "finish_polish")

    result = finish_polish.invoke({
        "summary": "All clean",
        "unfixable_warnings": [],
    })

    assert isinstance(result, Command)
    assert result.update["finish_unfixable"] == ()


# ---------------------------------------------------------------------------
# tools_node — direct behavior tests (using the internal factory)
# ---------------------------------------------------------------------------

def _capturing_ctx(tmp_path: Path):
    """Fake ctx that captures emitted events for assertions."""
    events: list[tuple[str, dict]] = []

    def emit(level, **kwargs):
        events.append((level, kwargs))

    ctx = _fake_ctx(tmp_path)
    ctx.events = SimpleNamespace(emit=emit)
    ctx._events = events  # attach for the test to introspect
    return ctx


def test_tools_node_normal_tool_returns_tool_message(tmp_path):
    """A non-sentinel tool return string is wrapped in a ToolMessage."""
    from agent.graph import _make_tools_node
    from langchain_core.tools import tool as _tool

    @_tool
    def echo(x: str) -> str:
        """Echo x."""
        return f"echoed:{x}"

    ctx = _capturing_ctx(tmp_path)
    node = _make_tools_node([echo], sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[_tool_call("echo", {"x": "hi"}, "c1")],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    tm = result["messages"][0]
    assert isinstance(tm, ToolMessage)
    assert tm.content == "echoed:hi"
    assert tm.tool_call_id == "c1"
    assert tm.name == "echo"


def test_tools_node_unknown_tool_returns_error_message(tmp_path):
    """Unknown tool name yields an error-body ToolMessage; session continues."""
    from agent.graph import _make_tools_node

    ctx = _capturing_ctx(tmp_path)
    node = _make_tools_node([], sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[_tool_call("nonexistent", {}, "c1")],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert len(result["messages"]) == 1
    tm = result["messages"][0]
    assert isinstance(tm, ToolMessage)
    payload = json.loads(tm.content)
    assert "unknown tool" in payload["error"]


def test_tools_node_tool_exception_returns_error_message(tmp_path):
    """Exception in tool body is surfaced to the model, not propagated up."""
    from agent.graph import _make_tools_node
    from langchain_core.tools import tool as _tool

    @_tool
    def boom(x: str) -> str:
        """Always raises."""
        raise RuntimeError("kaboom")

    ctx = _capturing_ctx(tmp_path)
    node = _make_tools_node([boom], sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[_tool_call("boom", {"x": "hi"}, "c1")],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert len(result["messages"]) == 1
    tm = result["messages"][0]
    payload = json.loads(tm.content)
    assert "RuntimeError" in payload["error"]
    assert "kaboom" in payload["error"]
    # a warn event was emitted so the operator can see it in events.jsonl
    assert any(lvl == "warn" for lvl, _ in ctx._events)


def test_tools_node_pydantic_validation_error_surfaced(tmp_path):
    """Bad tool args produce ValidationError from Pydantic; must not crash."""
    from agent.graph import _make_tools_node
    from langchain_core.tools import tool as _tool

    @_tool
    def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    ctx = _capturing_ctx(tmp_path)
    node = _make_tools_node([add], sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[_tool_call("add", {"a": "not-int", "b": 2}, "c1")],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert len(result["messages"]) == 1
    payload = json.loads(result["messages"][0].content)
    assert "ValidationError" in payload["error"]


def test_tools_node_sentinel_alone_returns_command(tmp_path):
    """Sentinel-only batch returns Command(goto=END) with state update."""
    from agent.graph import _make_tools_node
    from agent.tools import build_tools

    ctx = _terminal_ctx(tmp_path)
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    node = _make_tools_node(tools, sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[_tool_call("finish_translate",
                                    {"summary": "done"}, "c1")],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update["finish_called"] is True
    assert result.update["finish_summary"] == "done"
    # ToolMessage is in the update so tool_use/tool_result pairing holds
    tool_msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"


def test_tools_node_sentinel_with_sibling_call_all_get_responses(tmp_path):
    """Sentinel + non-sentinel in same batch: all tool_calls get ToolMessages,
    routing still ends via Command(goto=END)."""
    from agent.graph import _make_tools_node
    from agent.tools import build_tools

    # Terminal state so finish_translate accepts the completion, plus capturing
    # events so we can assert the sibling-warning is emitted.
    ctx = _terminal_ctx(tmp_path)
    events: list = []
    ctx.events = SimpleNamespace(emit=lambda level, **kw: events.append((level, kw)))
    ctx._events = events
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    node = _make_tools_node(tools, sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [AIMessage(
            content="",
            tool_calls=[
                _tool_call("list_files", {}, "c1"),
                _tool_call("finish_translate", {"summary": "done"}, "c2"),
            ],
        )],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert isinstance(result, Command)
    assert result.goto == END
    tool_msgs = result.update["messages"]
    assert len(tool_msgs) == 2
    assert {tm.tool_call_id for tm in tool_msgs} == {"c1", "c2"}
    # Sibling-warning event was emitted
    assert any(
        lvl == "warn" and "sibling" in kw.get("message", "")
        for lvl, kw in ctx._events
    )


def test_tools_node_non_ai_last_message_returns_empty(tmp_path):
    """If the last message isn't an AIMessage, tools_node returns no updates."""
    from agent.graph import _make_tools_node

    ctx = _capturing_ctx(tmp_path)
    node = _make_tools_node([], sentinel_name="finish_translate", ctx=ctx)
    state: AgentState = {
        "messages": [HumanMessage(content="hi")],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }
    result = node(state)
    assert result == {"messages": []}


# ---------------------------------------------------------------------------
# agent_node — cost accounting
# ---------------------------------------------------------------------------

def test_agent_node_records_cost_and_emits_turn_event(tmp_path):
    """agent_node calls ctx.cost.add_turn and emits a 'turn' event with token counts."""
    from agent.graph import _make_agent_node

    added: list = []
    events: list = []

    def add_turn(usage, current_file=None):
        added.append(usage)

    def emit(level, **kwargs):
        events.append((level, kwargs))

    ctx = _fake_ctx(tmp_path)
    ctx.cost = SimpleNamespace(
        add_turn=add_turn,
        add_tool=lambda *a, **kw: None,
        model="claude-sonnet-4-6",
    )
    ctx.events = SimpleNamespace(emit=emit)

    # Custom stub whose bound.invoke returns an AIMessage with usage_metadata
    class _CostStub(_StubChatModel):
        def bind_tools(self, tools, **kwargs):
            bound = MagicMock()
            bound.invoke = MagicMock(return_value=AIMessage(
                content="stub",
                usage_metadata={
                    "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                    "input_token_details": {"cache_creation": 10, "cache_read": 20},
                },
            ))
            return bound

    bundle = ChatModelBundle(
        chat=_CostStub(),
        forces_tool_call=True,
        supports_prompt_caching=True,
        supports_adaptive_thinking=True,
        tool_choice_any_payload={"type": "any"},
    )
    node = _make_agent_node(bundle, [], ctx)
    state: AgentState = {
        "messages": [HumanMessage(content="hi")],
        "finish_called": False,
        "finish_summary": "",
        "finish_unfixable": (),
    }

    result = node(state)

    assert len(added) == 1
    assert added[0].input_tokens == 100
    assert added[0].output_tokens == 50
    assert added[0].cache_write_tokens == 10
    assert added[0].cache_read_tokens == 20
    assert len(events) == 1
    lvl, kw = events[0]
    assert lvl == "turn"
    assert kw["input_tokens"] == 100
    assert kw["output_tokens"] == 50
    assert kw["cache_read_tokens"] == 20
    assert "cost_usd" in kw
    # returned dict appends the AIMessage
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


def test_agent_node_no_tool_choice_when_forces_tool_call_false(tmp_path):
    """On Ollama (forces_tool_call=False), bind_tools is called WITHOUT tool_choice."""
    from agent.graph import _make_agent_node

    captured_kwargs: dict = {}

    class _CaptureStub(_StubChatModel):
        def bind_tools(self, tools, **kwargs):
            captured_kwargs.update(kwargs)
            bound = MagicMock()
            bound.invoke = MagicMock(return_value=AIMessage(content="ok"))
            return bound

    ctx = _fake_ctx(tmp_path)
    bundle = ChatModelBundle(
        chat=_CaptureStub(),
        forces_tool_call=False,
        supports_prompt_caching=False,
        supports_adaptive_thinking=False,
        tool_choice_any_payload=None,
    )
    _make_agent_node(bundle, [], ctx)
    assert "tool_choice" not in captured_kwargs


def test_agent_node_no_tool_choice_in_bind_tools_when_forces_tool_call_true(tmp_path):
    """When forces_tool_call=True, tool_choice reaches model_kwargs but NOT bind_tools.

    _make_agent_node injects tool_choice='any' via model_copy(update={...})
    before calling bind_tools. This bypasses langchain-anthropic 1.5.5's
    overly-broad guard that drops tool_choice in bind_tools() even for adaptive
    thinking (which the Anthropic API actually supports with forced tool use).

    Because _StubChatModel.model_copy returns a new stub, we verify that
    bind_tools is called on the copy (not the original) and receives no
    tool_choice kwarg — the tool_choice lives in model_kwargs on the copy.
    """
    from agent.graph import _make_agent_node

    captured_bind_kwargs: dict = {}
    model_copy_called: list[dict] = []

    class _CaptureStub(_StubChatModel):
        def model_copy(self, *, update=None, **kw):
            model_copy_called.append(dict(update or {}))
            return super().model_copy(update=update, **kw)

        def bind_tools(self, tools, **kwargs):
            captured_bind_kwargs.update(kwargs)
            bound = MagicMock()
            bound.invoke = MagicMock(return_value=AIMessage(content="ok"))
            return bound

    ctx = _fake_ctx(tmp_path)
    bundle = ChatModelBundle(
        chat=_CaptureStub(),
        forces_tool_call=True,
        supports_prompt_caching=True,
        supports_adaptive_thinking=True,
        tool_choice_any_payload={"type": "any"},
    )
    _make_agent_node(bundle, [], ctx)
    # tool_choice goes into model_kwargs on the copy, not into bind_tools()
    assert "tool_choice" not in captured_bind_kwargs
    assert len(model_copy_called) == 1
    assert model_copy_called[0].get("model_kwargs", {}).get("tool_choice") == {"type": "any"}
