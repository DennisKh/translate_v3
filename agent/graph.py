"""LangGraph subgraph builders for translation and polish agentic loops.

Both graphs share the same topology (§5.1.1 in PLAN_LANGGRAPH.md):
  START → agent → tools → agent → ... → END

Exit conditions:
  - route_after_agent: END when no tool calls (Ollama primary exit; Anthropic
    defensive fallback per spike 10.1).
  - tools_node returns Command(goto=END) when the sentinel tool fires successfully
    (finish_translate or finish_polish return a Command). route_after_tools is a
    belt-and-suspenders check on finish_called.

Sentinel tool behavior:
  - On success: tool returns Command(update={finish_called, finish_summary, ...}, goto=END).
    The session runner reads finish_summary and finish_unfixable from the final graph state.
  - On refusal (finish_translate with non-terminal files): tool returns _json({"ok": false})
    string; tools_node wraps it as a ToolMessage and continues.

Public surface:
  - AgentState
  - build_translation_graph(ctx, bundle) -> CompiledStateGraph
  - build_polish_graph(ctx, bundle) -> CompiledStateGraph
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.llm import ChatModelBundle, turn_usage_from_ai_message
from agent.session_ctx import SessionContext
from agent.tools import build_tools


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    finish_called: bool
    finish_summary: str
    # Polish only; empty tuple for translation graph runs.
    finish_unfixable: tuple[str, ...]


def route_after_agent(state: AgentState) -> str:
    """Return 'tools' if the last AIMessage has tool calls, else END.

    On Anthropic (forces_tool_call=True), tool_calls is always non-empty
    because tool_choice='any' enforces a tool call every turn. The END
    branch is a defensive fallback there.

    On Ollama (forces_tool_call=False), tool_choice is silently ignored by
    ChatOllama. The model freely returns responses without tool calls, so
    this END branch is the primary exit path (spike 10.1).
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def route_after_tools(state: AgentState) -> str:
    """Return END if the sentinel tool fired, else 'agent'.

    In normal operation, the sentinel tool causes tools_node to return a
    Command(goto=END), so this function is called only when no sentinel fired.
    The finish_called check is a belt-and-suspenders guard.
    """
    if state.get("finish_called"):
        return END
    return "agent"


def _make_agent_node(bundle: ChatModelBundle, tools: list, ctx: SessionContext):
    """Return an agent_node closure bound to the given bundle and tool set.

    For providers that enforce tool calls (forces_tool_call=True), tool_choice
    is injected into the model's model_kwargs before bind_tools so it reaches
    the API payload. This approach is used instead of passing tool_choice to
    bind_tools() because langchain-anthropic 1.5.5 incorrectly drops
    tool_choice="any" in bind_tools() when adaptive thinking is enabled.
    The Anthropic API *does* support forced tool use with adaptive thinking.

    WORKAROUND: langchain-anthropic 1.5.5 source (ChatAnthropic.bind_tools,
    ~line 600) guards against tool_choice="any" for both "enabled" and
    "adaptive" thinking, but the Anthropic docs only forbid it for manual
    "enabled" mode. We bypass the guard by injecting tool_choice into
    model_kwargs on a model copy before calling bind_tools, so bind_tools
    receives no tool_choice kwarg but the API call still gets it from
    model_kwargs.

    The payload is provider-specific (bundle.tool_choice_any_payload):
    - Anthropic API: {"type": "any"}
    - OpenAI API:    "required"  (OpenAI rejects {"type":"any"} — that form
                     requires a "function" key and causes a 400 if missing)
    - Ollama:        None — skipped; forces_tool_call=False for Ollama.
    """
    if bundle.forces_tool_call:
        existing_kwargs = dict(getattr(bundle.chat, "model_kwargs", None) or {})
        chat_with_tool_choice = bundle.chat.model_copy(
            update={"model_kwargs": {**existing_kwargs, "tool_choice": bundle.tool_choice_any_payload}},
        )
        bound = chat_with_tool_choice.bind_tools(tools)
    else:
        bound = bundle.chat.bind_tools(tools)

    def agent_node(state: AgentState) -> dict[str, Any]:
        ai_msg: AIMessage = bound.invoke(state["messages"])
        usage = turn_usage_from_ai_message(ai_msg)
        ctx.cost.add_turn(usage)
        ctx.events.emit(
            "turn",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=round(usage.cost_usd(ctx.cost.model), 4),
        )
        return {"messages": [ai_msg]}

    return agent_node


def _make_tools_node(tools: list, sentinel_name: str, ctx: SessionContext):
    """Return a tools_node closure over the tool set.

    For each tool call in the last AIMessage:
    - Executes the tool via .invoke(args)
    - Wraps string results in ToolMessage
    - If the sentinel tool returns a Command (successful completion), re-issues
      a node-level Command with the ToolMessage in the messages update so the
      message history stays consistent (every AIMessage tool_call gets a
      corresponding ToolMessage) and routes to END.
    - If the sentinel tool returns a string (refusal path), treats it as a
      normal ToolMessage — the session continues.

    Exceptions from tool execution (Pydantic ValidationError on bad args, any
    tool-body exception) are caught per-tool-call and surfaced to the model as
    an error-body ToolMessage. This mirrors the Anthropic tool_runner behavior:
    the model sees the error and can retry. Without this, a single malformed
    tool call would abort the whole graph run and lose all in-flight state.

    Every tool_call in the AIMessage gets a paired ToolMessage in the update,
    even when the sentinel fires mid-batch (invariant required by Anthropic:
    every assistant tool_use block must have a matching user tool_result block).
    If the sentinel and other tools appear in the same batch (violates the
    stated invariant but not structurally impossible under tool_choice="any"),
    we still execute all calls before routing to END and log a warning so the
    unexpected shape is visible in events.jsonl.
    """
    tools_by_name = {t.name: t for t in tools}

    def tools_node(state: AgentState):
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return {"messages": []}

        tool_messages: list[ToolMessage] = []
        sentinel_state_update: dict[str, Any] | None = None

        for tc in last_msg.tool_calls:
            name = tc["name"]
            tool_fn = tools_by_name.get(name)
            if tool_fn is None:
                tool_messages.append(ToolMessage(
                    content=json.dumps({"error": f"unknown tool: {name!r}"}),
                    tool_call_id=tc["id"],
                    name=name,
                ))
                continue

            try:
                raw = tool_fn.invoke(tc["args"])
            except Exception as exc:  # noqa: BLE001
                # Surface any tool-execution failure to the model as an error
                # ToolMessage so it can retry. Mirrors Anthropic tool_runner
                # semantics — the model must see the failure, not have the
                # graph crash beneath it.
                ctx.events.emit(
                    "warn",
                    message=f"tool {name!r} raised {type(exc).__name__}: {exc}",
                )
                tool_messages.append(ToolMessage(
                    content=json.dumps({
                        "error": f"{type(exc).__name__}: {exc}",
                    }),
                    tool_call_id=tc["id"],
                    name=name,
                ))
                continue

            if isinstance(raw, Command) and name == sentinel_name:
                # Sentinel fired successfully. Capture its state update; append
                # the ToolMessage so the tool_call has a paired response. Do NOT
                # break — remaining tool_calls in the same batch still need a
                # response to preserve the tool_use/tool_result pairing invariant.
                if sentinel_state_update is not None:
                    ctx.events.emit(
                        "warn",
                        message=f"sentinel {name!r} fired twice in one batch — "
                                f"keeping first payload",
                    )
                else:
                    sentinel_state_update = dict(raw.update or {})
                tool_messages.append(ToolMessage(
                    content=_json_ok_for_sentinel(name),
                    tool_call_id=tc["id"],
                    name=name,
                ))
            else:
                tool_messages.append(ToolMessage(
                    content=str(raw),
                    tool_call_id=tc["id"],
                    name=name,
                ))

        if sentinel_state_update is not None:
            if len(tool_messages) > 1:
                ctx.events.emit(
                    "warn",
                    message=f"sentinel {sentinel_name!r} fired with "
                            f"{len(tool_messages) - 1} sibling tool_call(s) in the "
                            f"same batch — all executed; routing to END",
                )
            return Command(
                update={"messages": tool_messages, **sentinel_state_update},
                goto=END,
            )

        return {"messages": tool_messages}

    return tools_node


def _json_ok_for_sentinel(sentinel_name: str) -> str:
    """Return a minimal JSON acknowledgement ToolMessage content for a sentinel call."""
    if sentinel_name == "finish_translate":
        return json.dumps({"ok": True, "phase_a_complete": True})
    if sentinel_name == "finish_polish":
        return json.dumps({"ok": True, "polish_complete": True})
    return json.dumps({"ok": True})


def build_translation_graph(
    ctx: SessionContext,
    bundle: ChatModelBundle,
) -> CompiledStateGraph:
    """Build and compile the translation agentic loop.

    14 tools: list_files, read_java, read_elixir, grep_elixir, describe_module,
    write_elixir, edit_elixir, delete_elixir, escalate, mix_compile_tool,
    mix_format_tool, mix_credo_tool, validation_status, finish_translate.
    Sentinel: finish_translate.
    """
    tools = build_tools(ctx, include_write=True, polish_mode=False)
    sentinel_name = "finish_translate"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(bundle, tools, ctx))
    graph.add_node("tools", _make_tools_node(tools, sentinel_name, ctx))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})

    return graph.compile()


def build_polish_graph(
    ctx: SessionContext,
    bundle: ChatModelBundle,
) -> CompiledStateGraph:
    """Build and compile the polish agentic loop.

    10 tools: list_files, read_elixir, grep_elixir, describe_module, edit_elixir,
    mix_compile_tool, mix_format_tool, mix_credo_tool, validation_status, finish_polish.
    Sentinel: finish_polish.
    """
    tools = build_tools(ctx, include_write=True, polish_mode=True)
    sentinel_name = "finish_polish"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("agent", _make_agent_node(bundle, tools, ctx))
    graph.add_node("tools", _make_tools_node(tools, sentinel_name, ctx))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})

    return graph.compile()
