"""TranslationAgent — writes Elixir, validates in-loop, terminates via sentinel.

Uses LangGraph (edge-driven agentic loop) with LangChain's ChatModel abstraction.
Every session runs a compiled StateGraph under `tool_choice="any"` (when the provider
supports it) + a `finish_translate` / `finish_polish` sentinel tool — the only way
for the model to end the session is by calling that tool (and passing its
non-terminal-files check). This structural constraint eliminates the "model emits
end_turn before work is done" failure class.

Key subsystems (all belt-and-suspenders, so an isolated failure never
kills a run that has completed on-disk work):

  - **Adaptive thinking + effort** — configured per provider in agent/llm.py.
  - **Prompt caching** on the system prompt (Anthropic-specific perk, applied
    at session init via apply_prompt_cache).
  - **Per-turn timeout** — the underlying ChatModel's timeout setting.
  - **Session-reset checkpointing** — after N files or M tokens, snapshot
    state to disk and start a fresh graph run with the state summary as
    seed (no conversation history). Long sessions degrade; short cache-
    warm sessions produce better output.
  - **Graceful budget shutdown** — at 90% of any cap, inject a wrap-up
    message; at 100%, hard-stop.
  - **Rate-limit circuit breaker** — after N sustained 429/529 failures,
    abort with clear guidance.
  - **Compile nudge** — if the agent writes N files without calling
    `mix_compile_tool`, inject a reminder.
  - **Polish loop** — one additional session after main translation, with
    a curated tool set + `finish_polish` sentinel, for `mix credo`
    cleanup. Skipped when previous polish state matches current credo
    output (state cache).
  - **Fallback synthesizer** — if the session ends without a
    `finish_translate` call (checkpoint, budget hit, rate-limit, parse
    failure), the outcome is synthesized from disk state.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pydantic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph import AgentState, build_polish_graph, build_translation_graph
from agent.hitl import WallCapAction, prompt_wall_cap_action
from agent.llm import (
    ChatModelBundle,
    ExceptionKind,
    apply_prompt_cache,
    classify_exception,
    turn_usage_from_ai_message,
)
from agent.mix_ops import mix_compile, mix_credo, mix_format
from agent.observability import build_graph_config, build_langfuse_handler
from agent.session_ctx import SessionContext
from agent.state import FileState


# ---------------------------------------------------------------------------
# Structured output — Phase 2 outcome
# ---------------------------------------------------------------------------

class FileResult(BaseModel):
    stem: str
    action: Literal["translated", "skipped", "escalated"]
    # Brief — one short line at most. Detailed rationale belongs in git commit
    # messages / docstrings, not the run summary. Long notes here cause the
    # final JSON to blow past the per-turn max_tokens cap.
    notes: str = Field(
        default="",
        max_length=200,
        description="Brief note (≤200 chars): one-line rationale or 'ok'. NO paragraphs.",
    )


class ValidationResults(BaseModel):
    compile_ok: bool
    format_ok: bool
    credo_ok: bool


class TranslationOutcome(BaseModel):
    summary: str = Field(
        max_length=800,
        description="≤3-sentence overview (max 800 chars). No per-file detail.",
    )
    validation: ValidationResults
    files: list[FileResult]
    escalations: list[str] = Field(default_factory=list,
                                    description="Stems of escalated files")
    remaining_blockers: list[str] = Field(default_factory=list,
                                           description="Files still not in a terminal state")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Rate-limit circuit breaker
_MAX_RATE_LIMIT_STREAK = 3

# Graceful shutdown thresholds (per-run)
_BUDGET_WARN_PCT = 90.0

# Nudge the agent to compile after this many writes without one
_COMPILE_NUDGE_THRESHOLD = 5

# HITL: fixed extension size when the user picks "[e] extend" at the
# wall-cap prompt. 30 minutes. Not user-configurable — see REVIEW_HITL.md
# for rationale (a numeric-input UI has negligible upside).
_HITL_WALL_EXTENSION_SECONDS = 30 * 60

# Heartbeat interval — while waiting for the next model turn, log "still
# waiting" every N seconds so the user can distinguish "long generation in
# progress" from "actually stuck". 60s is generous; per-turn timeout fires
# at ~cfg.agent.max_turn_seconds anyway.
_HEARTBEAT_SECONDS = 60


class _Heartbeat:
    """Background thread that logs periodic 'still waiting' notices.

    Started before iterating the graph stream (before we block on the next
    model message); stopped after the stream completes or is broken.
    Bounded by _HEARTBEAT_SECONDS.
    """

    def __init__(self, ctx: SessionContext, session_num: int, turn_num: int) -> None:
        self._ctx = ctx
        self._session = session_num
        self._turn = turn_num
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(_HEARTBEAT_SECONDS):
            elapsed = int(time.monotonic() - self._start_time)
            self._ctx.events.emit(
                "info",
                message=f"still waiting on model (session #{self._session}, "
                        f"turn {self._turn + 1}, {elapsed}s elapsed) — "
                        f"large files can take several minutes",
            )


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "translator_system.md").read_text()


# ---------------------------------------------------------------------------
# Session-reset criteria
# ---------------------------------------------------------------------------

class _SessionStats:
    """Track work done in a single session for reset-criteria evaluation.

    `_start_completed` is captured AT THE FIRST TURN, not at __init__, so
    IN_PROGRESS→COMPLETE promotions triggered by the session's first
    `mix_compile_tool()` call (from work in the PREVIOUS session) don't count
    against this session's file-completion threshold.
    """

    def __init__(self) -> None:
        self._start_completed: int | None = None
        self.turns = 0
        self.session_tokens = 0
        self.warned_on_budget = False
        self.nudged_on_compile = False   # compile nudge fires at most once per session

    def note_first_turn(self, ctx: SessionContext) -> None:
        """Capture the completed count AFTER the first turn's tool results have
        been applied. Called from the runner loop on the first iteration."""
        if self._start_completed is None:
            self._start_completed = _count_terminal(ctx)

    def should_reset(self, ctx: SessionContext) -> str | None:
        """Return a reason if the session should end and a new one start."""
        if self._start_completed is None:
            # Haven't seen a full turn yet — never trigger a reset
            return None
        completed_this_session = _count_terminal(ctx) - self._start_completed
        reset_after_files = ctx.cfg.agent.session_reset_after_files
        reset_after_tokens = ctx.cfg.agent.session_reset_after_tokens

        if completed_this_session >= reset_after_files:
            return (f"session-reset threshold hit: {completed_this_session} files "
                    f"completed this session (limit {reset_after_files})")
        if self.session_tokens >= reset_after_tokens:
            return (f"session-reset threshold hit: {self.session_tokens:,} tokens "
                    f"used this session (limit {reset_after_tokens:,})")
        return None


def _count_terminal(ctx: SessionContext) -> int:
    """Count files in any terminal state (COMPLETE, SKIPPED, DELETED, ESCALATED)."""
    return sum(1 for f in ctx.state.all()
               if f.state in (FileState.COMPLETE, FileState.SKIPPED,
                              FileState.DELETED, FileState.ESCALATED))


def _all_files_terminal(ctx: SessionContext) -> bool:
    """True when there is no more work to do."""
    for f in ctx.state.all():
        if f.state in (FileState.NOT_STARTED, FileState.IN_PROGRESS,
                       FileState.SYNTAX_FAILED, FileState.COMPILE_FAILED,
                       FileState.BLOCKED):
            return False
    return True


# ---------------------------------------------------------------------------
# Session-message construction
# ---------------------------------------------------------------------------

def _build_initial_message(ctx: SessionContext, session_num: int) -> str:
    """Message that opens a fresh session (session 1 or a checkpoint restart).

    For session 1: full workflow instructions.
    For session 2+: state summary + continue-where-you-left-off.
    """
    if session_num == 1:
        return (
            f"You are translating a Java project to idiomatic Elixir.\n\n"
            f"Project: {ctx.source_root}\n"
            f"Target : {ctx.target_root}\n"
            f"Module namespace: {ctx.module_prefix}\n"
            f"App name: {ctx.app_snake}\n\n"
            f"Java files: {len(ctx.java_classes)} across "
            f"{len(ctx.dep_graph.levels)} topological levels "
            f"({len(ctx.dep_graph.sccs)} cycle group(s)).\n\n"
            f"**Your primary output is written Elixir files.** Reading is prep; "
            f"writing is the deliverable. One file at a time: read it, write "
            f"its Elixir counterpart, move on. Do NOT batch-read the whole "
            f"project before writing anything.\n\n"
            f"**IMMEDIATE ACTIONS (do these now, in order):**\n"
            f"  1. Call `list_files(status=\"untranslated\")`\n"
            f"  2. Pick the FIRST level-0 file returned. Call "
            f"`read_java(stem, include_tests=True)` on it.\n"
            f"  3. Call `write_elixir(module_or_path, full_source)` for that ONE file. "
            f"Do not read any other Java file first.\n"
            f"  4. Only after your first successful write, move to the next "
            f"level-0 file. Repeat.\n\n"
            f"By turn 3 you MUST have called `write_elixir` (or `delete_elixir` "
            f"if the class has no Elixir counterpart, e.g. `Ser.java`). If you "
            f"haven't, you're drifting — write the first file you've already read.\n\n"
            f"**Ending the session:** you are running under `tool_choice=\"any\"`. "
            f"Every turn must contain at least one tool call — you cannot emit "
            f"`end_turn` on your own. The ONLY way to end the session is to call "
            f"`finish_translate(summary)` — but that tool refuses unless every "
            f"file is in a terminal state. Call `list_files(status=\"untranslated\")` "
            f"before `finish_translate` to verify.\n\n"
            f"Full workflow, style guide, and pitfalls are in the system prompt. "
            f"Read those, then act.\n"
        )

    # Continuation session
    completed = ctx.state.by_state(FileState.COMPLETE)
    skipped = ctx.state.by_state(FileState.SKIPPED) + ctx.state.by_state(FileState.DELETED)
    escalated = ctx.state.by_state(FileState.ESCALATED)
    remaining = [f for f in ctx.state.all()
                 if f.state in (FileState.NOT_STARTED, FileState.IN_PROGRESS,
                                FileState.SYNTAX_FAILED, FileState.COMPILE_FAILED,
                                FileState.BLOCKED)]

    remaining_summary = "\n".join(
        f"  - {f.stem} [{f.state.value}]  deps: {f.deps or '(none)'}"
        for f in sorted(remaining, key=lambda f: (f.level, f.stem))[:30]
    )
    if len(remaining) > 30:
        remaining_summary += f"\n  ... and {len(remaining) - 30} more"

    return (
        f"**Continuation session #{session_num}** — the previous session was "
        f"checkpointed to disk. Conversation history dropped, file state persists.\n\n"
        f"Project: {ctx.source_root} → {ctx.target_root}\n"
        f"Module namespace: {ctx.module_prefix}\n\n"
        f"Progress:\n"
        f"  - Completed: {len(completed)}\n"
        f"  - Skipped/deleted: {len(skipped)}\n"
        f"  - Escalated: {len(escalated)}\n"
        f"  - **Still to do: {len(remaining)}** ← your job this session\n\n"
        f"Remaining files (sorted by level):\n{remaining_summary or '  (none — call validation_status and end)'}\n\n"
        f"**IMMEDIATE ACTIONS:**\n"
        f"  1. Pick the FIRST remaining file above. Call `read_java(stem, include_tests=True)`.\n"
        f"  2. Call `write_elixir(module_or_path, full_source)` for it. Do NOT read "
        f"other Java files first.\n"
        f"  3. Loop through the remaining files, one at a time.\n\n"
        f"By turn 3 you MUST have called `write_elixir` (or `delete_elixir` / "
        f"`escalate`). Don't re-explore — the state summary above is enough. "
        f"To end this session, call `finish_translate(summary)` once every "
        f"remaining file above is in a terminal state (COMPLETE, SKIPPED, "
        f"DELETED, or ESCALATED). "
        f"If you need to know the API of a translated dep, use "
        f"`describe_module(module_or_path)` (cheap), not `read_java` again.\n"
    )


def _build_polish_message(
    ctx: SessionContext,
    format_result: Any,
    credo_result: Any,
) -> str:
    """Seed message for a polish session — fires when all files are terminal
    but `mix format --check`, `mix credo --strict`, or
    `mix compile --warnings-as-errors` is still red.

    Real compile errors (unresolved symbols, syntax) are blocked upstream.
    Warnings promoted to errors by --warnings-as-errors are polish territory
    and may still be present when this message is generated.

    The polish session runs under `tool_choice="any"`: the model MUST call a
    tool every turn. The ONLY way to end the session is `finish_polish(...)`.
    """
    format_block = ""
    if not format_result.ok:
        format_block = (
            "\n### `mix format --check-formatted` (red)\n\n"
            "The following files are not formatted. Run `mix_format_tool()` "
            "(without check_only) to auto-fix — no manual edits needed.\n\n"
            f"```\n{format_result.output_tail}\n```\n"
        )
    credo_block = ""
    if not credo_result.ok:
        credo_block = (
            "\n### `mix credo --strict` (red)\n\n"
            "Read each flagged file, apply the smallest edit that resolves the "
            "warning (usually an alias re-order, line break, or `@moduledoc "
            "false`), then re-check.\n\n"
            f"```\n{credo_result.output_tail}\n```\n"
        )

    return (
        f"**Polish pass.** All {len(ctx.state.all())} files are in terminal "
        f"states — translation is done. But the project is not clean yet:\n"
        f"{format_block}"
        f"{credo_block}\n"
        f"### CRITICAL — how fixes get applied\n\n"
        f"**Reading a file with `read_elixir` does NOT change it. Understanding "
        f"a warning does NOT fix it. The ONLY way a credo warning goes away "
        f"is by calling `edit_elixir(module_or_path, old_string, new_string)` — that "
        f"is the tool that writes to disk.**\n\n"
        f"After every `edit_elixir` call, re-run `mix_credo_tool()` and observe "
        f"whether the warning count drops. If it does, the fix worked. If it "
        f"doesn't, the fix didn't land — try a different edit.\n\n"
        f"### How this session ends\n\n"
        f"You are running under `tool_choice=\"any\"`: every turn MUST contain "
        f"a tool call. The **only** way to end this session is by calling "
        f"`finish_polish(summary, unfixable_warnings)`. Do not call it as "
        f"your first action.\n\n"
        f"### Workflow\n\n"
        f"1. Pick ONE warning from the credo output. `read_elixir` the "
        f"flagged file (and at most 1-2 call-site files if a rename).\n"
        f"2. **Rate limit: `read_elixir` refuses after 5 reads without an "
        f"intervening `edit_elixir`.** Don't batch-read the whole codebase. "
        f"Read → edit → read → edit is the correct rhythm.\n"
        f"3. Apply a surgical fix with `edit_elixir(module_or_path, old_string, "
        f"new_string)`. `old_string` must be unique in the file — include "
        f"3-5 lines of context if the pattern repeats.\n"
        f"4. After a batch of edits, verify:\n"
        f"   - Call `mix_compile_tool()` FIRST if any edit was a rename, a "
        f"function extraction, or changed a signature. Compile catches the "
        f"common refactor bugs: missed call sites, mismatched arities, "
        f"bad `defp` visibility. If compile fails, `edit_elixir` to fix it.\n"
        f"   - Then call `mix_credo_tool()` to see remaining warnings.\n"
        f"   Iterate until clean OR you have identified warnings that cannot "
        f"be fixed without changing external API.\n"
        f"5. Call `finish_polish(summary=\"...\", unfixable_warnings=[...])` "
        f"to end. Populate `unfixable_warnings` with any credo warnings you "
        f"chose to leave unfixed (structural refactors, arity changes) — "
        f"one line each.\n\n"
        f"### CRITICAL — renames are multi-site\n\n"
        f"Renaming a public function (e.g. `is_foo/1` → `foo?/1`) requires "
        f"updating EVERY call site in the codebase. If you only rename the "
        f"`def` and forget the call sites, `mix compile` FAILS — every call "
        f"site now references a function that no longer exists.\n\n"
        f"Correct procedure for a rename:\n"
        f"  a. `grep_elixir(\"is_foo\")` to find every occurrence (def + all "
        f"call sites) across `lib/`.\n"
        f"  b. `edit_elixir` on the def first.\n"
        f"  c. `edit_elixir` on each call site (may span multiple files).\n"
        f"  d. `mix_compile_tool()` to verify — if it fails, you missed a "
        f"call site.\n"
        f"  e. `mix_credo_tool()` to verify the warning is gone.\n\n"
        f"Use `grep_elixir` NOT `read_elixir` for finding call sites — "
        f"grep is one small call across all files, read_elixir is one file "
        f"per call and counts against the read cap.\n\n"
        f"### Guidance by warning type\n\n"
        f"**Trivial (attempt first, single-file):**\n"
        f"- Alias re-order — swap two lines.\n"
        f"- Nested-module alias — add `alias Foo.Bar` at top, replace "
        f"`Foo.Bar.baz(...)` with `Bar.baz(...)` (`grep_elixir` first to find "
        f"every occurrence in that file).\n"
        f"- Negated `if !cond do ... else ... end` — swap branches, drop `!`.\n\n"
        f"**Multi-site (attempt, verify with mix_compile):**\n"
        f"- Predicate `is_foo/1` → `foo?/1` — see 'renames are multi-site' "
        f"above.\n\n"
        f"**Structural refactors (attempt these — DO NOT declare unfixable "
        f"by default):**\n"
        f"- `[F] Function is too complex (cyclomatic complexity is N)` — "
        f"**extract helper functions**. If the function is a large `case`, "
        f"pull each branch (or groups of related branches) into private "
        f"`defp branch_name(args), do: ...`. Complexity is per-function, so "
        f"5 branches in a helper each with complexity 3 is fine.\n"
        f"- `[F] Function body is nested too deep` — **extract the inner "
        f"block into a helper**, or use `with` for chained conditions, or "
        f"return early. E.g. `if x do (nested block) else nil end` → "
        f"`if x, do: do_nested(args)` plus `defp do_nested(...) do ... end`.\n"
        f"- These are safe when patterns and return values are preserved "
        f"exactly. `mix_compile_tool()` catches shape errors immediately. "
        f"Don't declare unfixable without attempting the refactor.\n\n"
        f"**High-arity (attempt via param bundling):**\n"
        f"- `[F] Function takes too many parameters (arity > 8)` — bundle "
        f"the params into a map, keyword list, tuple, or struct. Pick the "
        f"shape that matches how the function is used:\n"
        f"  * If it's a constructor like `Foo.new/10` — accept a map or "
        f"keyword: `def new(opts) when is_map(opts) do ... end`. Call sites "
        f"become `Foo.new(%{{field_a: x, field_b: y, ...}})`. Use "
        f"`grep_elixir(\"Foo.new(\")` to find every call site, then "
        f"`edit_elixir` each one.\n"
        f"  * If the params have natural groupings — bundle each group into "
        f"a tuple: `def parse(input, {{acc, remaining, state}}, "
        f"{{fmt_opts, grouping_opts}})` and unpack inside.\n"
        f"  * If it's a private recursive-loop function with accumulator "
        f"args — bundle the accumulators into a single map/struct that gets "
        f"threaded through calls: reduces arity without changing behavior.\n"
        f"  * Verify with `mix_compile_tool()` after — call sites must be "
        f"updated consistently.\n\n"
        f"**Only declare unfixable when:**\n"
        f"- You attempted a refactor and `mix_compile_tool()` returned an "
        f"error you couldn't resolve after 2 edit attempts — note the "
        f"specific compile error in the unfixable entry.\n"
        f"- The fix would break API contracts the operator explicitly wants "
        f"preserved (rare; assume the operator wants clean credo).\n\n"
        f"### Prohibited\n\n"
        f"- Do NOT translate new files (all files are terminal already).\n"
        f"- Do NOT re-read Java sources — this is Elixir polish only.\n"
        f"- Do NOT call `finish_polish` before attempting at least one "
        f"`edit_elixir` unless every remaining warning is legitimately "
        f"unfixable per the rules above.\n"
    )


def _wrap_up_message(reason: str) -> HumanMessage:
    """Injected mid-session when budget is at 90%. Asks the agent to finish
    what it's on and stop, rather than start another file."""
    return HumanMessage(content=(
        f"**Budget warning:** {reason}\n\n"
        f"Do NOT start work on a new file. Finish any in-progress writes, "
        f"run `validation_status()` one final time, and return the "
        f"structured summary. If you have unresolved blockers, list them "
        f"in the outcome — do not try to fix them now."
    ))


def _bomb_nudge_message(files_written: int) -> HumanMessage:
    """Injected when the agent writes many files without compiling."""
    return HumanMessage(content=(
        f"**Reminder:** you've written {files_written} files since the last "
        f"`mix_compile_tool()`. Cross-file drift accumulates silently. "
        f"Please call `mix_compile_tool()` now to catch broken references "
        f"before they compound."
    ))


# ---------------------------------------------------------------------------
# One-session LangGraph runner
# ---------------------------------------------------------------------------

class GraphSessionResult(NamedTuple):
    outcome: TranslationOutcome | None
    reason: str
    finish_summary: str
    finish_unfixable: tuple[str, ...]
    cap_type: str | None = None


class _ChunkAction(NamedTuple):
    """Return value from per-chunk handlers.

    Exactly one of the three fields is non-None:
      - `terminate` — end the session and return this result to the caller
      - `inject`    — break the current stream, append this HumanMessage,
                      restart graph.stream() with the accumulated messages
      - all None    — continue consuming the current stream

    Encoded as a NamedTuple (not an Enum + payload) so the caller reads
    `action.terminate` / `action.inject` directly without pattern-match
    boilerplate.
    """
    terminate: GraphSessionResult | None = None
    inject: HumanMessage | None = None


_CONTINUE = _ChunkAction()

_EMPTY_TOOL_CALLS_NUDGE = HumanMessage(
    content=(
        "You must call a tool. To translate a Java file, "
        "call `write_elixir(module_or_path, contents)`. "
        "To end the session, call `finish_translate(summary)`. "
        "Return a tool call, not a text response."
    )
)


def _record_turn_usage(
    ctx: SessionContext,
    stats: "_SessionStats",
    last_msg: AIMessage,
    session_num: int,
) -> None:
    """Apply per-turn policy-layer accounting for an agent-node chunk.

    `ctx.cost.add_turn(usage)` is already called inside `agent_node` (graph.py);
    calling it here would double-count. This function only touches the counters
    that live outside the graph: budget dollars, task tokens, rate-limit streak,
    session stats, and the `turn` event.
    """
    usage = turn_usage_from_ai_message(last_msg)
    session_tokens_this_turn = (
        usage.input_tokens + usage.output_tokens
        + usage.cache_write_tokens + usage.cache_read_tokens
    )
    stats.session_tokens += session_tokens_this_turn
    ctx.budget.add_usage(usage, ctx.cfg.llm.model)
    ctx.remaining_task_tokens = max(
        0, ctx.remaining_task_tokens - session_tokens_this_turn
    )
    ctx.rate_limit_streak = 0

    ctx.events.emit(
        "turn",
        session=session_num,
        turn=stats.turns,
        tokens={
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cache_read": usage.cache_read_tokens,
            "cache_write": usage.cache_write_tokens,
        },
        cumulative_cost_usd=ctx.cost.cumulative_cost_usd(),
        budget_remaining_pct=ctx.budget.remaining_pct(),
        remaining_task_tokens=ctx.remaining_task_tokens,
    )


def _process_agent_chunk(
    ctx: SessionContext,
    *,
    last_msg: AIMessage,
    stats: "_SessionStats",
    session_num: int,
    polish_mode: bool,
    bundle: ChatModelBundle,
    empty_response_state: list[int],
) -> _ChunkAction:
    """Handle one agent-node chunk (last message is AIMessage).

    `empty_response_state` is a single-element list used as a mutable cell so
    the caller can share the nudge-attempt counter across chunks without
    packaging it in another dataclass. Bounded to 1 nudge.

    Returns a `_ChunkAction`. Cannot inject after this branch — the pending
    AIMessage(tool_calls) requires a matching ToolMessage before any
    HumanMessage may follow (OpenAI strict pairing). Injection paths for
    soft-budget / compile-nudge live in `_process_tool_chunk`.
    """
    stats.turns += 1
    if stats.turns == 1:
        stats.note_first_turn(ctx)

    _record_turn_usage(ctx, stats, last_msg, session_num)

    # Model emitted no tool calls (end_turn).
    # For providers without forced tool use (Ollama), try once to nudge the
    # model back onto the tool-calling path. For providers with forced tool
    # use (Anthropic/OpenAI) this branch is the defensive fallback (§5.4).
    if not last_msg.tool_calls:
        if not bundle.forces_tool_call and empty_response_state[0] < 1:
            empty_response_state[0] += 1
            ctx.events.emit(
                "warn",
                message=f"model returned no tool calls (attempt "
                        f"{empty_response_state[0]}/2) — nudging",
            )
            return _ChunkAction(inject=_EMPTY_TOOL_CALLS_NUDGE)
        ctx.events.emit(
            "warn",
            message="end_turn with no tool calls — "
                    "synthesizing outcome from disk state",
        )
        return _ChunkAction(terminate=GraphSessionResult(None, "end_turn", "", ()))

    if result := ctx.budget.status():
        ctx.events.emit("warn", message=f"cap reached: {result.reason}")
        return _ChunkAction(terminate=GraphSessionResult(
            None, "budget_hit", "", (), cap_type=result.type,
        ))

    # Session-reset checkpoint — main-mode only. Returns immediately (starts
    # a fresh session), so the pending tool_calls in this AIMessage are never
    # executed. The next session sees list_files(status="untranslated") still
    # showing those files and re-attempts. write_elixir is idempotent, so
    # this is semantically safe even if some writes were lost.
    if not polish_mode and (reason := stats.should_reset(ctx)):
        ctx.events.emit("checkpoint", session=session_num, message=reason)
        return _ChunkAction(terminate=GraphSessionResult(None, "checkpoint", "", ()))

    # Soft budget warning and compile nudge are handled in _process_tool_chunk.
    # Injecting a HumanMessage after an AIMessage(tool_calls) would violate the
    # tool_call → tool_result pairing invariant and trigger a 400 from OpenAI.
    return _CONTINUE


def _process_tool_chunk(
    ctx: SessionContext,
    *,
    chunk: dict,
    stats: "_SessionStats",
    session_num: int,
    polish_mode: bool,
    sentinel_reason: str,
) -> _ChunkAction:
    """Handle one tools-node chunk (last message is ToolMessage).

    The tool_call pairing invariant is closed here (AIMessage(tool_calls) →
    ToolMessage has completed), so HumanMessage injection is safe.
    """
    if chunk.get("finish_called"):
        finish_summary = chunk.get("finish_summary", "")
        finish_unfixable = chunk.get("finish_unfixable", ())
        if polish_mode:
            addendum = f"unfixable: {len(finish_unfixable)}"
        else:
            addendum = f"{len(ctx.state.all())} files addressed"
        ctx.events.emit(
            "info",
            message=f"{sentinel_reason} invoked — {addendum}",
        )
        return _ChunkAction(terminate=GraphSessionResult(
            None, sentinel_reason, finish_summary, finish_unfixable,
        ))

    if not polish_mode and _should_warn_on_budget(ctx, stats):
        stats.warned_on_budget = True
        warn = (f"{ctx.budget.remaining_pct():.0f}% budget remaining "
                f"(cost ${ctx.cost.cumulative_cost_usd():.2f} USD)")
        ctx.events.emit("warn", message=f"soft budget warning: {warn}")
        _assert_pairing_closed(chunk, "wrap-up")
        return _ChunkAction(inject=_wrap_up_message(warn))

    if not polish_mode and _should_nudge_compile(ctx, stats):
        stats.nudged_on_compile = True
        ctx.events.emit(
            "info",
            message=f"nudging model to compile "
                    f"({ctx.files_written_since_compile} writes "
                    f"this session without one)",
        )
        _assert_pairing_closed(chunk, "compile-nudge")
        return _ChunkAction(inject=_bomb_nudge_message(ctx.files_written_since_compile))

    return _CONTINUE


def _should_warn_on_budget(ctx: SessionContext, stats: "_SessionStats") -> bool:
    return (not stats.warned_on_budget
            and ctx.budget.remaining_pct() < (100 - _BUDGET_WARN_PCT))


def _should_nudge_compile(ctx: SessionContext, stats: "_SessionStats") -> bool:
    return (not stats.nudged_on_compile
            and ctx.files_written_since_compile >= _COMPILE_NUDGE_THRESHOLD)


def _assert_pairing_closed(chunk: dict, inject_label: str) -> None:
    """Defensive: last message must be ToolMessage before HumanMessage inject.

    Belt-and-suspenders — if LangGraph's stream_mode='values' semantics ever
    change (e.g. an agent-chunk yields with the tool-message already appended),
    this fails loud instead of silently regressing the OpenAI 400 pairing bug.
    """
    assert isinstance(chunk["messages"][-1], ToolMessage), (
        f"BUG: {inject_label} inject attempted when last message is not "
        f"ToolMessage — tool_call pairing invariant violated"
    )


def _handle_stream_exception(
    ctx: SessionContext,
    exc: BaseException,
    session_num: int,
) -> GraphSessionResult:
    """Map an exception raised inside `graph.stream()` to a GraphSessionResult.

    `pydantic.ValidationError` is treated distinctly as "parse_failure" because
    it's not a provider SDK error — `classify_exception` would map it to UNKNOWN,
    which loses the semantic that the model produced malformed structured output.

    Provider SDK exceptions go through `agent.llm.classify_exception` to get a
    provider-agnostic `ExceptionKind` and dispatch:
      TIMEOUT               → checkpoint (fresh session)
      RATE_LIMIT/TRANSIENT  → soft retry; hard-abort after streak of N
      BAD_REQUEST/AUTH      → re-raise (user config bug)
      UNKNOWN               → parse_failure (synthesize from disk)
    """
    if isinstance(exc, pydantic.ValidationError):
        ctx.events.emit(
            "error",
            message=f"structured parse failed at session "
                    f"{session_num}: {type(exc).__name__} — treating as "
                    f"parse_failure, will synthesize from disk state",
        )
        return GraphSessionResult(None, "parse_failure", "", ())

    kind = classify_exception(exc)

    if kind == ExceptionKind.TIMEOUT:
        ctx.events.emit(
            "warn",
            message=f"per-turn timeout ({ctx.cfg.agent.max_turn_seconds}s) — "
                    f"treating as checkpoint: {type(exc).__name__}: {exc}",
        )
        return GraphSessionResult(None, "checkpoint", "", ())

    if kind in (ExceptionKind.RATE_LIMIT, ExceptionKind.TRANSIENT):
        ctx.rate_limit_streak += 1
        streak = ctx.rate_limit_streak
        ctx.events.emit(
            "warn",
            message=f"transient API failure #{streak}/{_MAX_RATE_LIMIT_STREAK}: "
                    f"{type(exc).__name__}: {exc}",
        )
        if streak >= _MAX_RATE_LIMIT_STREAK:
            ctx.events.emit(
                "error",
                message=f"aborting: {_MAX_RATE_LIMIT_STREAK} transient failures in a row",
            )
            return GraphSessionResult(None, "rate_limit_hard", "", ())
        return GraphSessionResult(None, "rate_limit_soft", "", ())

    if kind in (ExceptionKind.BAD_REQUEST, ExceptionKind.AUTH):
        ctx.events.emit("error", message=f"API error: {type(exc).__name__}: {exc}")
        raise exc

    # ExceptionKind.UNKNOWN — log and treat as parse_failure so the outer
    # loop can synthesize from disk.
    ctx.events.emit(
        "error",
        message=f"unexpected session error: {type(exc).__name__}: {exc} "
                f"— treating as parse_failure",
    )
    return GraphSessionResult(None, "parse_failure", "", ())


def _run_one_graph_session(
    ctx: SessionContext,
    *,
    bundle: ChatModelBundle,
    graph: Any,
    initial_messages: list,
    session_num: int,
    polish_mode: bool = False,
    graph_config: dict[str, Any] | None = None,
) -> GraphSessionResult:
    """Run one LangGraph session. Returns GraphSessionResult(outcome, reason, finish_summary, finish_unfixable).

    reason is one of:
      - "finish_translate"  — sentinel tool accepted; main-mode done
      - "finish_polish"     — sentinel tool accepted; polish-mode done
      - "end_turn"          — model finished without tool calls (Ollama / defensive)
      - "budget_hit"        — Python-side dollar/tool-call/wall cap
      - "checkpoint"        — session-reset threshold reached (loop restarts)
      - "rate_limit_soft"   — hit a 429/529; caller SHOULD retry with a new session
      - "rate_limit_hard"   — streak exceeded, abort the whole run
      - "parse_failure"     — pydantic.ValidationError or UNKNOWN exception
      - "exhausted"         — graph stream ended without sentinel

    Uses stream_mode='values' — each chunk is the full AgentState after a node
    fires. The agent-chunk (last message is AIMessage) is a model turn; the
    tools-chunk (last message is ToolMessage) follows. Per-turn budget
    accounting happens on each agent-chunk. Soft budget warn / compile nudge
    inject a HumanMessage into the state (in the ToolMessage branch, to preserve
    the tool_call → tool_result pairing invariant) and restart the stream.

    `ctx.cost.add_turn(usage)` is called inside agent_node (graph.py). Do NOT
    call it here — that would double-count cost. This runner calls
    `ctx.budget.add_usage`, `ctx.remaining_task_tokens`, and
    `ctx.rate_limit_streak` (the policy-layer counters that live outside
    the graph).
    """
    ctx.files_written_since_compile = 0
    if polish_mode:
        ctx.polish_active = True
        ctx.polish_reads_since_edit = 0

    stats = _SessionStats()
    sentinel_reason = "finish_polish" if polish_mode else "finish_translate"
    current_messages = list(initial_messages)
    # Mutable single-element cell so _process_agent_chunk can increment the
    # nudge-attempt counter without threading it back through a return value.
    # Bounded to 1 nudge (see _process_agent_chunk).
    empty_response_state = [0]

    try:
        while True:
            state: AgentState = {
                "messages": current_messages,
                "finish_called": False,
                "finish_summary": "",
                "finish_unfixable": (),
            }

            heartbeat = _Heartbeat(ctx, session_num, stats.turns)
            heartbeat.start()
            last_chunk: dict | None = None
            action = _CONTINUE

            try:
                stream_kwargs: dict[str, Any] = {"stream_mode": "values"}
                if graph_config is not None:
                    stream_kwargs["config"] = graph_config
                for chunk in graph.stream(state, **stream_kwargs):
                    last_chunk = chunk
                    last_msg = chunk["messages"][-1] if chunk["messages"] else None
                    if last_msg is None:
                        continue

                    if isinstance(last_msg, AIMessage):
                        action = _process_agent_chunk(
                            ctx, last_msg=last_msg, stats=stats,
                            session_num=session_num, polish_mode=polish_mode,
                            bundle=bundle,
                            empty_response_state=empty_response_state,
                        )
                    elif isinstance(last_msg, ToolMessage):
                        action = _process_tool_chunk(
                            ctx, chunk=chunk, stats=stats,
                            session_num=session_num, polish_mode=polish_mode,
                            sentinel_reason=sentinel_reason,
                        )
                    else:
                        continue

                    if action.terminate is not None:
                        return action.terminate
                    if action.inject is not None:
                        break
            finally:
                heartbeat.stop()

            # Broke out of the stream to inject a message: append it to the
            # accumulated messages and loop again with a new graph.stream().
            if action.inject is not None:
                assert last_chunk is not None, "inject requires at least one chunk"
                current_messages = list(last_chunk["messages"]) + [action.inject]
                continue

            # Stream ended without a sentinel or explicit break. Either the
            # graph reached END via the no-tool-calls defensive branch
            # (handled above by the end_turn check in the agent chunk)
            # or the graph exhausted iterations somehow.
            ctx.events.emit(
                "warn",
                message=f"session {session_num} graph stream ended without sentinel",
            )
            return GraphSessionResult(None, "exhausted", "", ())
    except Exception as exc:
        return _handle_stream_exception(ctx, exc, session_num)
    finally:
        if polish_mode:
            ctx.polish_active = False


# ---------------------------------------------------------------------------
# HITL — wall-cap prompt dispatch
# ---------------------------------------------------------------------------

def _handle_wall_cap_hitl(ctx: SessionContext) -> bool:
    """Prompt the user on wall-cap hit. Return True if the run should continue.

    Returns False (caller falls through to the normal budget_hit exit) when:
      - `cfg.agent.hitl_on_wall_cap` is False
      - stdin is not a TTY (auto-downgrade)
      - user picks [a] abort, empty input, EOF, timeout, or invalid input

    Returns True (caller `continue`s the outer loop) when:
      - user picks [e] extend: cap grows by 30 min
      - user picks [c] continue: cap set to sys.maxsize (effectively removed)

    State is flushed to disk (`ctx.cost.persist()`) before the prompt so a
    user who aborts (or times out) sees the same on-disk state as if HITL
    had never been offered — `--resume` behavior is unaffected.
    """
    if not ctx.cfg.agent.hitl_on_wall_cap:
        ctx.events.emit(
            "info",
            message="wall-cap hit; HITL prompt disabled by config — exiting",
        )
        return False

    # Persist state before we block on input. If the user walks away and
    # times out we still want cost_report.json to reflect what was spent.
    ctx.cost.persist()

    action = prompt_wall_cap_action(
        elapsed_seconds=ctx.budget.elapsed_seconds(),
        cap_seconds=ctx.budget.max_wall_seconds,
        files_done=_count_terminal(ctx),
        files_total=len(ctx.state.all()),
        cost_usd=ctx.cost.cumulative_cost_usd(),
    )

    if action == WallCapAction.EXTEND:
        ctx.budget.extend_wall_seconds(_HITL_WALL_EXTENSION_SECONDS)
        ctx.events.emit(
            "info",
            message=f"HITL: wall cap extended by "
                    f"{_HITL_WALL_EXTENSION_SECONDS // 60} min "
                    f"(new cap {ctx.budget.max_wall_seconds}s)",
        )
        return True

    if action == WallCapAction.CONTINUE:
        ctx.budget.remove_wall_cap()
        ctx.events.emit(
            "info",
            message="HITL: wall cap removed for the remainder of the run",
        )
        return True

    # ABORT — includes non-TTY auto-downgrade, empty input, EOF, timeout.
    ctx.events.emit(
        "info",
        message="HITL: aborting on wall-cap (user choice, non-TTY, or timeout)",
    )
    return False


# ---------------------------------------------------------------------------
# Top-level agent entry point
# ---------------------------------------------------------------------------

def run_translator_phase2(
    ctx: SessionContext,
    *,
    bundle: ChatModelBundle,
    dry_run: bool = False,
) -> tuple[TranslationOutcome | None, int]:
    """Multi-session translation loop. Returns (outcome, session_count).

    Each individual session runs a LangGraph StateGraph until it hits a terminal
    state (sentinel tool, budget cap, checkpoint threshold, transient failure,
    or graph exhaustion). The outer loop restarts sessions on "checkpoint",
    "rate_limit_soft", and "exhausted"; breaks on "finish_translate",
    "budget_hit", "rate_limit_hard".
    """
    system_prompt = _load_system_prompt()

    # Initialize the Task Budget remaining-tokens tracker
    ctx.remaining_task_tokens = ctx.cfg.agent.max_budget_tokens

    # Scale total session cap with project size — with default 10-file threshold,
    # a 500-file project needs ~50 sessions min; we give 2x headroom.
    total_sessions_cap = max(
        20,
        (len(ctx.java_classes) // max(ctx.cfg.agent.session_reset_after_files, 1)) * 2,
    )

    ctx.events.emit(
        "phase_start", phase="translate",
        message=f"Phase 2 translation — "
                f"system prompt {len(system_prompt):,} chars, "
                f"{len(ctx.java_classes)} files to translate, "
                f"up to {total_sessions_cap} sessions",
    )

    if dry_run:
        ctx.events.emit("info", message="dry-run: skipping API calls")
        ctx.events.emit("phase_end", phase="translate",
                        message="dry-run OK — prompt and tools load cleanly")
        return None, 0

    ctx.events.emit(
        "info",
        message=f"model={ctx.cfg.llm.model} "
                f"budget={ctx.cfg.agent.max_budget_tokens:,} tokens "
                f"(hard cap ≈ ${ctx.budget._max_cost_usd:.2f}) "
                f"tool_cap={ctx.cfg.agent.max_tool_calls} "
                f"wall_cap={ctx.cfg.agent.max_wall_seconds}s "
                f"turn_cap={int(ctx.cfg.agent.max_turn_seconds)}s "
                f"session_reset={ctx.cfg.agent.session_reset_after_files}f/"
                f"{ctx.cfg.agent.session_reset_after_tokens:,}t",
    )

    # Build the translation graph once — reused across all sessions
    translation_graph = build_translation_graph(ctx, bundle)

    # Optional Langfuse tracing. Handler is None unless
    # cfg.langfuse.enabled=true AND the LANGFUSE_* env vars are present.
    langfuse_handler = build_langfuse_handler(ctx.cfg)
    langfuse_session_id: str | None = None
    trace_metadata: dict[str, Any] = {}
    if langfuse_handler is not None:
        langfuse_session_id = ctx.scaffold.target_root.name
        trace_metadata = {
            "provider": getattr(ctx.cfg.llm, "provider", None),
            "model": getattr(ctx.cfg.llm, "model", None),
            "source_root": (str(ctx.cfg.source.root)
                            if getattr(ctx.cfg.source, "root", None) else None),
            "target_root": str(ctx.scaffold.target_root),
        }
        ctx.events.emit(
            "info",
            message=f"langfuse tracing on (session_id={langfuse_session_id})",
        )

    system_content = apply_prompt_cache(system_prompt, bundle.supports_prompt_caching)
    final_outcome: TranslationOutcome | None = None
    session_num = 0
    break_reason: str | None = None
    # last_session_result is None until at least one session actually runs. It
    # stays None on the "already done" resume path (all_terminal fires on the
    # first iteration, before _run_one_graph_session is invoked) and on the
    # "budget already exhausted between runs" path. Both are legitimate — we
    # must not reference session_result unconditionally after the loop or that
    # bare NameError kills a resume that would otherwise be a clean no-op.
    last_session_result: GraphSessionResult | None = None

    while session_num < total_sessions_cap:
        session_num += 1

        if _all_files_terminal(ctx):
            ctx.events.emit("info",
                            message=f"all {len(ctx.state.all())} files in terminal states — done")
            break_reason = "all_terminal"
            break
        if result := ctx.budget.status():
            ctx.events.emit("warn",
                            message=f"budget cap reached before session {session_num}: {result.reason}")
            break_reason = "budget_hit_between_sessions"
            break

        ctx.events.emit("info", message=f"starting session #{session_num}")
        initial_text = _build_initial_message(ctx, session_num)
        initial_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=initial_text),
        ]

        last_session_result = _run_one_graph_session(
            ctx,
            bundle=bundle,
            graph=translation_graph,
            initial_messages=initial_messages,
            session_num=session_num,
            graph_config=build_graph_config(
                langfuse_handler,
                run_name=f"translate/session-{session_num}",
                metadata={**trace_metadata, "session_num": session_num,
                          "phase": "translate"},
                session_id=langfuse_session_id,
                tags=["translate_v3", "translate"],
            ),
        )
        if last_session_result.outcome is not None:
            final_outcome = last_session_result.outcome

        ctx.events.emit("info", message=f"session #{session_num} ended: {last_session_result.reason}")

        if (last_session_result.reason == "budget_hit"
                and last_session_result.cap_type == "max_wall"
                and _handle_wall_cap_hitl(ctx)):
            continue

        # Terminal reasons — break the outer loop
        if last_session_result.reason in ("finish_translate", "end_turn",
                                          "budget_hit", "rate_limit_hard", "parse_failure"):
            break_reason = last_session_result.reason
            break
        # else: "checkpoint" | "rate_limit_soft" | "exhausted" — loop again
    else:
        ctx.events.emit("warn",
                        message=f"reached session cap ({total_sessions_cap}) — stopping")
        break_reason = "session_cap"

    # The last session_result holds the finish_summary when the run ended via
    # finish_translate. On the "already done" / "budget between sessions" paths
    # no session ran, so the summary is empty.
    phase_a_summary = last_session_result.finish_summary if last_session_result else ""

    # Post-loop polish pass
    polish_sessions_used = 0
    polish_unfixable: tuple[str, ...] = ()
    if not dry_run and _all_files_terminal(ctx):
        polish_graph = build_polish_graph(ctx, bundle)
        polish_outcome, polish_sessions_used, polish_unfixable = _maybe_run_polish_sessions(
            ctx,
            bundle=bundle,
            graph=polish_graph,
            system_content=system_content,
            first_session_num=session_num + 1,
            session_cap=max(0, total_sessions_cap - session_num),
            langfuse_handler=langfuse_handler,
            langfuse_session_id=langfuse_session_id,
            trace_metadata=trace_metadata,
        )
    polish_ran = polish_sessions_used > 0

    if final_outcome is None:
        final_outcome = _synthesize_outcome_from_state(ctx, phase_a_summary=phase_a_summary)
        source = "finish_translate summary + state" if phase_a_summary else "state only"
        ctx.events.emit("info", message=f"outcome synthesized ({source})")
    elif polish_ran:
        # Re-sync validation from disk to reflect the polish session's edits.
        synthetic = _synthesize_outcome_from_state(ctx)
        final_outcome.validation = synthetic.validation
        if polish_unfixable:
            final_outcome.summary = (
                final_outcome.summary
                + f" | polish: {len(polish_unfixable)} unfixable declared"
            )[:800]

    ctx.events.emit(
        "phase_end", phase="translate",
        message=f"loop finished: {session_num} session(s), reason={break_reason}"
                f"{' (+polish)' if polish_ran else ''}, "
                f"outcome {'received' if final_outcome else 'MISSING'}",
    )
    return final_outcome, session_num


# Credo output shape (each finding is 2 lines):
#   ┃ [F] → Function is too complex (cyclomatic complexity is 18, max is 9).
#   ┃       lib/joda_money/currency_unit.ex:118:7 #(JodaMoney.CurrencyUnit.register_currency)
# Two findings can share file:line:col (e.g. one function triggers BOTH
# complexity AND arity warnings), so the message prefix must be part of the ID.
_CREDO_FINDING_RE = re.compile(
    r"^┃\s*\[(?P<severity>[A-Z])\]\s*[→↗↘]\s*(?P<message>.+?)\.?\s*$\n"
    r"^┃\s+(?P<file>lib/[\w/]+\.exs?):(?P<line>\d+):(?P<col>\d+)"
    r"\s+#\((?P<symbol>[\w.?!]+)\)",
    re.MULTILINE,
)


def _parse_credo_warnings(credo_output: str) -> set[str]:
    """Extract stable warning identities from `mix credo --strict` output.

    Returns a set of `[severity] file:line:symbol :: message-prefix` keys.
    Two runs producing the same set indicate credo state is unchanged.

    Message prefix (first ~50 chars) is included so that co-located findings
    (same file:line:col:symbol but different rules — e.g. complexity AND
    arity on one function) get distinct IDs.
    """
    ids: set[str] = set()
    for m in _CREDO_FINDING_RE.finditer(credo_output):
        prefix = m["message"].strip()[:60]
        ids.add(f"[{m['severity']}] {m['file']}:{m['line']}:{m['symbol']} :: {prefix}")
    return ids


def _polish_state_path(ctx: SessionContext) -> Path:
    return ctx.scaffold.target_root / ".translate_v3_state" / "polish.json"


def _load_polish_state(ctx: SessionContext) -> dict | None:
    path = _polish_state_path(ctx)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_polish_state(ctx: SessionContext, credo_warning_ids: set[str],
                       unfixable_declared: tuple[str, ...]) -> None:
    """Persist the polish outcome so subsequent runs can skip when state matches."""
    state = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "credo_warning_ids": sorted(credo_warning_ids),
        "unfixable_declared": list(unfixable_declared),
    }
    path = _polish_state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _maybe_run_polish_sessions(
    ctx: SessionContext,
    *,
    bundle: ChatModelBundle,
    graph: Any,
    system_content: Any,
    first_session_num: int,
    session_cap: int,
    langfuse_handler: Any | None = None,
    langfuse_session_id: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> tuple[TranslationOutcome | None, int, tuple[str, ...]]:
    """Run a single polish session under `tool_choice="any"` + finish_polish.

    Under this configuration the model CANNOT emit pure end_turn. Every turn
    must contain at least one tool call, and the only way to end the session
    is `finish_polish(summary, unfixable_warnings)`. This eliminates the
    "read a couple files then quit" failure mode structurally, not via prose.

    Returns (final_outcome, sessions_used, finish_unfixable). `sessions_used == 0`
    means no polish was needed or possible (compile red / already clean).
    """
    try:
        c_real = mix_compile(ctx.scaffold.mix_env, ctx.target_root,
                             warnings_as_errors=False)
        c_strict = mix_compile(ctx.scaffold.mix_env, ctx.target_root,
                               warnings_as_errors=True)
        f = mix_format(ctx.scaffold.mix_env, ctx.target_root, check_only=True)
        k = mix_credo(ctx.scaffold.mix_env, ctx.target_root, strict=True)
    except Exception as exc:
        ctx.events.emit(
            "warn",
            message=f"polish pre-check failed: {type(exc).__name__}: {exc} — skipping polish",
        )
        return None, 0, ()

    if not c_real.ok:
        # Real compile errors (syntax, unresolved references) — not polish work.
        ctx.events.emit(
            "warn",
            message="polish skipped: real compile errors present (not just warnings) "
                    "— this is a translation bug, fix it before polish",
        )
        return None, 0, ()
    if f.ok and k.ok and c_strict.ok:
        # Nothing for polish to do.
        return None, 0, ()
    if session_cap <= 0:
        return None, 0, ()

    # Skip polish when the current credo state matches a previously-declared
    # unfixable set (identical warning locations). Saves ~$0.30 and ~40s on
    # every resume once the mechanical fixes have been applied and the model
    # has declared the remainder unfixable.
    current_warning_ids = _parse_credo_warnings(k.output)
    prior = _load_polish_state(ctx)
    if prior is not None and current_warning_ids and f.ok:
        prior_ids = set(prior.get("credo_warning_ids", []))
        if current_warning_ids == prior_ids:
            ctx.events.emit(
                "info",
                message=f"polish skipped: {len(current_warning_ids)} credo "
                        f"warning(s) match previously-declared unfixable set "
                        f"(state: .translate_v3_state/polish.json)",
            )
            return None, 0, ()

    ctx.events.emit(
        "info",
        message=f"opening polish session #{first_session_num}: "
                f"format={'✓' if f.ok else '✗'} credo={'✓' if k.ok else '✗'} "
                f"(tool_choice=any; must call finish_polish to end)",
    )
    initial_text = _build_polish_message(ctx, f, k)
    initial_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=initial_text),
    ]

    edits_before = ctx.cost.tool_call_count("edit_elixir")

    polish_result = _run_one_graph_session(
        ctx,
        bundle=bundle,
        graph=graph,
        initial_messages=initial_messages,
        session_num=first_session_num,
        polish_mode=True,
        graph_config=build_graph_config(
            langfuse_handler,
            run_name=f"polish/session-{first_session_num}",
            metadata={**(trace_metadata or {}),
                      "session_num": first_session_num,
                      "phase": "polish"},
            session_id=langfuse_session_id,
            tags=["translate_v3", "polish"],
        ),
    )

    edits_made = ctx.cost.tool_call_count("edit_elixir") - edits_before
    finish_unfixable = polish_result.finish_unfixable
    ctx.events.emit(
        "info",
        message=f"polish session #{first_session_num} ended: {polish_result.reason} "
                f"(edits: {edits_made}; "
                f"unfixable declared: {len(finish_unfixable)})",
    )
    for warn_line in finish_unfixable[:20]:
        ctx.events.emit("info", message=f"  polish left: {warn_line}")

    # Persist the outcome for skip-on-resume.
    if polish_result.reason == "finish_polish":
        try:
            k_post = mix_credo(ctx.scaffold.mix_env, ctx.target_root, strict=True)
            post_ids = _parse_credo_warnings(k_post.output)
            _save_polish_state(ctx, post_ids, finish_unfixable)
            ctx.events.emit(
                "info",
                message=f"polish state saved: {len(post_ids)} remaining warning(s) "
                        f"(future resumes will skip polish if credo state is unchanged)",
            )
        except Exception as exc:
            ctx.events.emit(
                "warn",
                message=f"polish state save failed: {type(exc).__name__}: {exc}",
            )

    return polish_result.outcome, 1, finish_unfixable


def _synthesize_outcome_from_state(
    ctx: SessionContext,
    *,
    phase_a_summary: str = "",
) -> TranslationOutcome:
    """Build a TranslationOutcome from disk state — fallback when the model
    ends without emitting the structured summary.

    Runs REAL mix checks (compile/format/credo) rather than hardcoding
    validation as red. Otherwise a resume with no work to do (all files
    already terminal) would falsely report red validation and prevent
    downstream steps (README translation, regression scoring) from running.
    """
    _NOTES_CAP = 200

    def _truncate(text: str) -> str:
        text = (text or "").strip()
        if len(text) <= _NOTES_CAP:
            return text
        return text[: _NOTES_CAP - 1] + "…"

    files: list[FileResult] = []
    escalations: list[str] = []
    remaining: list[str] = []

    for entry in ctx.state.all():
        if entry.state == FileState.COMPLETE:
            files.append(FileResult(stem=entry.stem, action="translated"))
        elif entry.state in (FileState.SKIPPED, FileState.DELETED):
            files.append(FileResult(
                stem=entry.stem, action="skipped",
                notes=_truncate(entry.note),
            ))
        elif entry.state == FileState.ESCALATED:
            files.append(FileResult(
                stem=entry.stem, action="escalated",
                notes=_truncate(entry.note),
            ))
            escalations.append(entry.stem)
        else:
            remaining.append(entry.stem)

    # Real validation. Only meaningful if we have at least one file on disk.
    has_lib = any(f.state == FileState.COMPLETE for f in ctx.state.all())
    if has_lib:
        try:
            c = mix_compile(ctx.scaffold.mix_env, ctx.target_root)
            f = mix_format(ctx.scaffold.mix_env, ctx.target_root, check_only=True)
            k = mix_credo(ctx.scaffold.mix_env, ctx.target_root, strict=True)
            validation = ValidationResults(
                compile_ok=c.ok, format_ok=f.ok, credo_ok=k.ok,
            )
            ctx.events.emit(
                "info",
                message=f"validation (synthesized): "
                        f"compile={'✓' if c.ok else '✗'} "
                        f"format={'✓' if f.ok else '✗'} "
                        f"credo={'✓' if k.ok else '✗'}",
            )
        except Exception as exc:
            ctx.events.emit(
                "warn",
                message=f"synthesized validation check failed: "
                        f"{type(exc).__name__}: {exc}",
            )
            validation = ValidationResults(
                compile_ok=False, format_ok=False, credo_ok=False,
            )
    else:
        validation = ValidationResults(
            compile_ok=False, format_ok=False, credo_ok=False,
        )

    # Prefer the model's own summary (from `finish_translate`) over the
    # placeholder — the model has richer context on what was interesting
    # about this run. Falls back to placeholder when finish_translate
    # wasn't called (checkpoint, budget hit, parse_failure fallback, etc.).
    summary_text = phase_a_summary or "synthesized from state — model did not call finish_translate"

    return TranslationOutcome(
        summary=summary_text[:800],
        validation=validation,
        files=files,
        escalations=escalations,
        remaining_blockers=remaining,
    )
