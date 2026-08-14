"""Round-trip binding test: build_tools → BaseChatModel.bind_tools.

Verifies that the @tool-decorated tools produced by build_tools() are
actually bindable to a ChatModel and surface the correct names and schemas.
No API calls — ChatAnthropic.bind_tools is a local operation.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.state import StateStore
from agent.tools import build_tools


def _fake_ctx(target_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state=StateStore(target_root),
        events=SimpleNamespace(emit=lambda *a, **kw: None),
        cost=SimpleNamespace(add_tool=lambda *a, **kw: None),
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
    )


EXPECTED_READ_ONLY_TOOLS = {
    "list_files",
    "read_java",
    "read_elixir",
    "grep_elixir",
    "describe_module",
}


def test_build_tools_bind_tools_round_trip(tmp_path):
    """build_tools() → ChatAnthropic.bind_tools() succeeds without API calls
    and exposes the correct tool names in the bound model's kwargs.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.runnables import RunnableBinding

    ctx = _fake_ctx(tmp_path)
    tools = build_tools(ctx, include_write=False)

    assert len(tools) == len(EXPECTED_READ_ONLY_TOOLS)

    tool_names = {t.name for t in tools}
    assert tool_names == EXPECTED_READ_ONLY_TOOLS

    # fake_key is enough — bind_tools is a local schema operation, no network call
    model = ChatAnthropic(model="claude-3-haiku-20240307", api_key="fake-key")
    bound = model.bind_tools(tools)

    assert isinstance(bound, RunnableBinding)

    bound_tool_names = {t["name"] for t in bound.kwargs["tools"]}
    assert bound_tool_names == EXPECTED_READ_ONLY_TOOLS
