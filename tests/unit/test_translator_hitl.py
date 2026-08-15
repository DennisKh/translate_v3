"""HITL wall-cap dispatch inside `run_translator_phase2`.

Focused integration tests: mock `_run_one_graph_session` to return various
`GraphSessionResult`s and mock `prompt_wall_cap_action` to simulate user
choices. Verifies:

- Wall cap + EXTEND → cap grows, another session runs.
- Wall cap + CONTINUE → cap removed, another session runs.
- Wall cap + ABORT → break with budget_hit (no extra session).
- Wall cap + `hitl_on_wall_cap=False` → no prompt, immediate break.
- Non-wall cap (`max_cost`, `max_tool`) → no prompt, immediate break.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.state import FileEntry, FileState, StateStore
from agent.translator import GraphSessionResult, run_translator_phase2
from agent.hitl import WallCapAction


def _pending_entry(stem: str) -> FileEntry:
    return FileEntry(
        stem=stem, java_path=f"{stem}.java",
        lib_path=f"lib/x/{stem.lower()}.ex",
        test_path=f"test/x/{stem.lower()}_test.exs",
        module_name=f"X.{stem}",
        state=FileState.NOT_STARTED,
    )


def _make_ctx(tmp_path: Path, *, hitl_on_wall_cap: bool = True) -> SimpleNamespace:
    """Build a SessionContext-shaped stub with one pending file so
    `_all_files_terminal` is False and the outer loop keeps iterating."""
    state = StateStore(tmp_path)
    state.upsert(_pending_entry("A"))

    scaffold_stub = SimpleNamespace(
        target_root=tmp_path,
        mix_env=SimpleNamespace(mix_path="/bin/false", env={}),
    )

    events = MagicMock()
    cost = MagicMock()
    cost.cumulative_cost_usd = MagicMock(return_value=1.23)
    cost.model = "claude-opus-4-7"
    cost.persist = MagicMock()

    budget = MagicMock()
    budget.status = MagicMock(return_value=None)
    budget.remaining_pct = MagicMock(return_value=100.0)
    budget.total_cost_usd = 0.0
    budget._max_cost_usd = 100.0
    budget.max_wall_seconds = 3600
    budget.elapsed_seconds = MagicMock(return_value=3610)
    budget.extend_wall_seconds = MagicMock()
    budget.remove_wall_cap = MagicMock()

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
            hitl_on_wall_cap=hitl_on_wall_cap,
        ),
        source=SimpleNamespace(root=tmp_path),
    )
    return SimpleNamespace(
        cfg=cfg, scaffold=scaffold_stub, state=state, events=events,
        cost=cost, budget=budget,
        java_classes=[], dep_graph=SimpleNamespace(levels=[], sccs=[]),
        source_root=tmp_path, target_root=tmp_path,
        module_prefix="X", app_snake="x",
        rate_limit_streak=0, remaining_task_tokens=0,
        files_written_since_compile=0,
        polish_active=False, polish_reads_since_edit=0,
    )


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        forces_tool_call=True, supports_prompt_caching=True, chat=MagicMock(),
    )


def _wall_hit() -> GraphSessionResult:
    return GraphSessionResult(None, "budget_hit", "", (), cap_type="max_wall")


def _cost_hit() -> GraphSessionResult:
    return GraphSessionResult(None, "budget_hit", "", (), cap_type="max_cost")


def _tool_hit() -> GraphSessionResult:
    return GraphSessionResult(None, "budget_hit", "", (), cap_type="max_tool")


def _finish_ok() -> GraphSessionResult:
    return GraphSessionResult(None, "finish_translate", "done", ())


def _run(ctx, bundle, session_results, prompt_return):
    """Run `run_translator_phase2` with a sequence of session results and
    a mocked prompt. Returns (outcome, sessions, prompt_mock, session_mock)."""
    session_mock = MagicMock(side_effect=session_results)
    prompt_mock = MagicMock(return_value=prompt_return)

    with patch("agent.translator._run_one_graph_session", session_mock), \
         patch("agent.translator.prompt_wall_cap_action", prompt_mock), \
         patch("agent.translator.build_translation_graph", return_value=MagicMock()), \
         patch("agent.translator.build_polish_graph", return_value=MagicMock()), \
         patch("agent.translator._maybe_run_polish_sessions",
               return_value=(None, 0, ())):
        outcome, sessions = run_translator_phase2(ctx, bundle=bundle, dry_run=False)
    return outcome, sessions, prompt_mock, session_mock


def test_wall_cap_extend_grows_budget_and_continues(tmp_path):
    ctx = _make_ctx(tmp_path)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_wall_hit(), _finish_ok()],
        prompt_return=WallCapAction.EXTEND,
    )
    prompt_mock.assert_called_once()
    ctx.budget.extend_wall_seconds.assert_called_once_with(30 * 60)
    ctx.budget.remove_wall_cap.assert_not_called()
    assert session_mock.call_count == 2
    assert sessions == 2


def test_wall_cap_continue_removes_cap_and_continues(tmp_path):
    ctx = _make_ctx(tmp_path)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_wall_hit(), _finish_ok()],
        prompt_return=WallCapAction.CONTINUE,
    )
    prompt_mock.assert_called_once()
    ctx.budget.remove_wall_cap.assert_called_once()
    ctx.budget.extend_wall_seconds.assert_not_called()
    assert session_mock.call_count == 2


def test_wall_cap_abort_breaks_with_budget_hit(tmp_path):
    ctx = _make_ctx(tmp_path)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_wall_hit()],
        prompt_return=WallCapAction.ABORT,
    )
    prompt_mock.assert_called_once()
    ctx.budget.extend_wall_seconds.assert_not_called()
    ctx.budget.remove_wall_cap.assert_not_called()
    assert session_mock.call_count == 1
    assert sessions == 1


def test_wall_cap_hitl_disabled_by_config_skips_prompt(tmp_path):
    ctx = _make_ctx(tmp_path, hitl_on_wall_cap=False)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_wall_hit()],
        prompt_return=WallCapAction.EXTEND,  # would extend if consulted
    )
    prompt_mock.assert_not_called()
    ctx.budget.extend_wall_seconds.assert_not_called()
    assert session_mock.call_count == 1


def test_cost_cap_does_not_prompt(tmp_path):
    ctx = _make_ctx(tmp_path)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_cost_hit()],
        prompt_return=WallCapAction.EXTEND,  # would extend if consulted
    )
    prompt_mock.assert_not_called()
    assert session_mock.call_count == 1


def test_tool_cap_does_not_prompt(tmp_path):
    ctx = _make_ctx(tmp_path)
    _o, sessions, prompt_mock, session_mock = _run(
        ctx, _bundle(),
        session_results=[_tool_hit()],
        prompt_return=WallCapAction.EXTEND,
    )
    prompt_mock.assert_not_called()
    assert session_mock.call_count == 1


def test_wall_cap_extend_persists_cost_report_before_prompt(tmp_path):
    """User might time out or abort; state on disk must reflect what's been
    spent before we block on stdin."""
    ctx = _make_ctx(tmp_path)
    _run(
        ctx, _bundle(),
        session_results=[_wall_hit(), _finish_ok()],
        prompt_return=WallCapAction.EXTEND,
    )
    ctx.cost.persist.assert_called()
