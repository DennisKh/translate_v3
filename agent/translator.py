"""TranslationAgent — writes Elixir, validates in-loop, terminates via sentinel.

Uses the Anthropic Tool Runner. Every session runs under
`tool_choice="any"` + a `finish_translate` / `finish_polish` sentinel tool
— the only way for the model to end the session is by calling that tool
(and passing its non-terminal-files check). This structural constraint
eliminates the "model emits end_turn before work is done" failure class.

Key subsystems (all belt-and-suspenders, so an isolated failure never
kills a run that has completed on-disk work):

  - **Adaptive thinking + effort** (`thinking={"type": "adaptive"}`) with
    an optional `thinking_budget_tokens` hard cap for reproducibility.
  - **Prompt caching** on the system prompt + tool schemas.
  - **Task Budget** (Opus 4.7 beta) — model-visible token countdown.
  - **Per-turn timeout** (`max_turn_seconds`) — kills a hung API call
    without crashing the run.
  - **Session-reset checkpointing** — after N files or M tokens, snapshot
    state to disk and start a fresh session with the state summary as
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
from typing import Any, Literal

import anthropic
import pydantic
from pydantic import BaseModel, Field

from agent.cost import TurnUsage
from agent.mix_ops import mix_compile, mix_credo, mix_format
from agent.session_ctx import SessionContext
from agent.state import FileState
from agent.tools import build_tools


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

# Per-turn cap. Under the finish_translate sentinel design, no single turn
# emits a giant structured JSON — writes are surgical, edits are diffs. But
# 64K keeps headroom for the rare large-file write on the first pass.
_MAX_TOKENS_PER_TURN = 64_000

# Nudge the agent to compile after this many writes without one
_COMPILE_NUDGE_THRESHOLD = 5

# Heartbeat interval — while waiting for the next model turn, log "still
# waiting" every N seconds so the user can distinguish "long generation in
# progress" from "actually stuck". 60s is generous; per-turn timeout fires
# at ~cfg.agent.max_turn_seconds anyway.
_HEARTBEAT_SECONDS = 60


class _Heartbeat:
    """Background thread that logs periodic 'still waiting' notices.

    Started before iterating the runner (before we block on the next model
    message); stopped after the message arrives. Bounded by _HEARTBEAT_SECONDS.
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
                message=f"⏳ still waiting on model (session #{self._session}, "
                        f"turn {self._turn + 1}, {elapsed}s elapsed) — "
                        f"large files can take several minutes",
            )


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "translator_system.md").read_text()


def _turn_usage(message: Any) -> TurnUsage:
    u = message.usage
    return TurnUsage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


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
    but `mix format --check` or `mix credo --strict` is still red.

    Compile is assumed green here (guarded upstream). Polish targets style-only
    fixups the agent left behind when translating.

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


def _wrap_up_message(reason: str) -> dict:
    """Injected mid-session when budget is at 90%. Asks the agent to finish
    what it's on and stop, rather than start another file."""
    return {
        "role": "user",
        "content": (
            f"**Budget warning:** {reason}\n\n"
            f"Do NOT start work on a new file. Finish any in-progress writes, "
            f"run `validation_status()` one final time, and return the "
            f"structured summary. If you have unresolved blockers, list them "
            f"in the outcome — do not try to fix them now."
        ),
    }


def _bomb_nudge_message(files_written: int) -> dict:
    """Injected when the agent writes many files without compiling."""
    return {
        "role": "user",
        "content": (
            f"**Reminder:** you've written {files_written} files since the last "
            f"`mix_compile_tool()`. Cross-file drift accumulates silently. "
            f"Please call `mix_compile_tool()` now to catch broken references "
            f"before they compound."
        ),
    }


# NOTE: `_is_premature_end` / `_premature_end_bounce` were removed after the
# `finish_translate` sentinel migration. The bounce mechanism attempted to
# `runner.append_messages(...)` after `end_turn` — but the SDK's `__run__`
# returns immediately when `end_turn` fires, so queued messages were never
# consumed. That's fixed structurally now: `tool_choice="any"` prevents
# `end_turn`, and `finish_translate` refuses at the tool boundary when files
# are non-terminal. See §17.7 of PLAN.md.


# ---------------------------------------------------------------------------
# One-session runner
# ---------------------------------------------------------------------------

def _run_one_session(
    ctx: SessionContext,
    *,
    client: anthropic.Anthropic,
    system_prompt: str,
    tools: list,
    initial_message: str,
    session_num: int,
    polish_mode: bool = False,
) -> tuple[TranslationOutcome | None, str]:
    """Run one Tool Runner session. Returns (outcome, reason_ended).

    reason_ended is one of:
      - "end_turn"          — model finished normally
      - "budget_hit"        — Python-side dollar/tool-call/wall cap
      - "checkpoint"        — session-reset threshold reached (loop restarts)
      - "rate_limit_soft"   — hit a 429/529; caller SHOULD retry with a new session
      - "rate_limit_hard"   — streak exceeded, abort the whole run
      - "exhausted"         — max_iterations reached; loop restarts
    """
    if ctx.cfg.agent.max_budget_tokens < 20_000:
        raise ValueError(
            f"max_budget_tokens={ctx.cfg.agent.max_budget_tokens:,} below the "
            f"20,000 minimum required by Task Budget beta"
        )

    # Reset per-session counters ON THE CTX (not on _SessionStats) so they're
    # accurate for the LOOP as it enters this session. `files_written_since_compile`
    # is a project-wide counter — it should reflect real state as we enter,
    # not accumulate stale carryover from a previous session that never compiled.
    # The compile-nudge fires only when THIS session's writes exceed the threshold.
    ctx.files_written_since_compile = 0

    output_config: dict[str, Any] = {"effort": ctx.cfg.agent.effort}
    # Task Budget requires task-budgets-2026-03-13 (Opus 4.7 only).
    # NOTE: `context_management` (clear_tool_uses) was REMOVED — testing showed
    # it broke prompt caching (rewrote message history → cache prefix hash
    # invalidated). One turn that should have been cache-warm ate $0.21 in
    # input costs. Better to let the session hit the reset threshold naturally
    # and start a fresh session (cache-cold once, then warm).
    betas: list[str] = []
    if ctx.cfg.safety.task_budget_beta and ctx.cfg.agent.model.startswith("claude-opus-4-7"):
        remaining = max(20_000, ctx.remaining_task_tokens)
        output_config["task_budget"] = {"type": "tokens", "total": remaining}
        betas.append("task-budgets-2026-03-13")

    # Thinking config — either explicit budget (hard cap) or adaptive.
    # Sonnet 4.6 with adaptive+high-effort produced 30-min single-turn hangs.
    # Explicit budget lets the operator prevent runaway deliberation.
    if ctx.cfg.agent.thinking_budget_tokens > 0:
        thinking_config: dict[str, Any] = {
            "type": "enabled",
            "budget_tokens": ctx.cfg.agent.thinking_budget_tokens,
        }
    else:
        thinking_config = {"type": "adaptive"}

    runner_kwargs: dict[str, Any] = dict(
        model=ctx.cfg.agent.model,
        max_tokens=_MAX_TOKENS_PER_TURN,
        thinking=thinking_config,
        output_config=output_config,
        max_iterations=200,
        # Per-turn wall-clock timeout — kills a single API call if it exceeds
        # the configured ceiling. Prevents 30-min hangs.
        timeout=ctx.cfg.agent.max_turn_seconds,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=tools,
        messages=[{"role": "user", "content": initial_message}],
    )
    if polish_mode:
        # Force tool use — model cannot emit pure end_turn. Must call a tool
        # every turn, including the `finish_polish` sentinel to end. Skip
        # `output_format` since polish outcome comes from finish_polish's
        # args, not a structured final message.
        runner_kwargs["tool_choice"] = {"type": "any"}
        ctx.polish_finished = False
        ctx.polish_finish_summary = ""
        ctx.polish_finish_unfixable = ()
        ctx.polish_active = True
        ctx.polish_reads_since_edit = 0
    else:
        # Phase A main mode: same structural constraint as polish. The old
        # design used `output_format=TranslationOutcome` (structured output)
        # which was fragile — Sonnet with adaptive thinking on long sessions
        # occasionally emitted an empty text block, crashing Pydantic's JSON
        # parser at the SDK boundary. `finish_translate` sentinel + tool_choice
        # "any" avoids that failure class entirely: outcome comes from tool
        # args (which are typed and validated at call time), not from a
        # final text message.
        runner_kwargs["tool_choice"] = {"type": "any"}
        ctx.phase_a_finished = False
        ctx.phase_a_summary = ""
    if betas:
        runner_kwargs["betas"] = betas

    runner = client.beta.messages.tool_runner(**runner_kwargs)
    stats = _SessionStats()
    final_outcome: TranslationOutcome | None = None

    # Use an explicit iterator so we can start/stop a heartbeat around each
    # blocking wait for the next model message. Without this, the user sees
    # dead silence during large-file generation (5-15 minutes on the hardest
    # files) and can't tell if it's stuck.
    runner_iter = iter(runner)

    try:
        while True:
            heartbeat = _Heartbeat(ctx, session_num, stats.turns)
            heartbeat.start()
            try:
                message = next(runner_iter)
            except StopIteration:
                heartbeat.stop()
                break
            finally:
                heartbeat.stop()

            stats.turns += 1
            # Capture start-completed count AFTER the first turn so any
            # promotions from an initial mix_compile don't count against
            # this session's threshold.
            if stats.turns == 1:
                stats.note_first_turn(ctx)

            usage = _turn_usage(message)
            session_tokens_this_turn = (usage.input_tokens + usage.output_tokens
                                        + usage.cache_write_tokens + usage.cache_read_tokens)
            stats.session_tokens += session_tokens_this_turn
            ctx.cost.add_turn(usage)
            ctx.budget.add_usage(usage, ctx.cfg.agent.model)
            # Deduct from the raw remaining-tokens counter (Task Budget)
            ctx.remaining_task_tokens = max(0, ctx.remaining_task_tokens - session_tokens_this_turn)
            # Successful turn — reset the rate-limit streak
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
                stop_reason=message.stop_reason,
            )

            # Polish-mode terminal: `finish_polish` tool invoked.
            if polish_mode and ctx.polish_finished:
                ctx.events.emit(
                    "info",
                    message=f"polish: finish_polish invoked "
                            f"({len(ctx.polish_finish_unfixable)} unfixable declared)",
                )
                return final_outcome, "finish_polish"

            # Main-mode terminal: `finish_translate` tool invoked. The tool
            # implementation refuses when files are still non-terminal —
            # the model gets a tool-result error and retries — so by the
            # time we see phase_a_finished=True, all files are addressed.
            if not polish_mode and ctx.phase_a_finished:
                ctx.events.emit(
                    "info",
                    message=f"Phase A: finish_translate invoked — "
                            f"{len(ctx.state.all())} files addressed",
                )
                return final_outcome, "finish_translate"

            # Defensive fallback for `end_turn`. Under `tool_choice="any"`
            # the API should not yield this stop_reason — the model must
            # invoke a tool every turn. If it fires anyway (SDK edge case,
            # provider behavior change), accept it and let the outer loop
            # synthesize outcome from disk state. Non-terminal files show
            # up as `remaining_blockers` in the final report.
            if message.stop_reason == "end_turn":
                ctx.events.emit(
                    "warn",
                    message="unexpected end_turn under tool_choice='any' — "
                            "synthesizing outcome from disk state",
                )
                return final_outcome, "end_turn"

            # Hard budget cap
            if reason := ctx.budget.status():
                ctx.events.emit("warn", message=f"cap reached: {reason}")
                return final_outcome, "budget_hit"

            # Soft budget warning — inject wrap-up message ONCE. Main-mode
            # only; the polish-mode wrap-up message talks about translation
            # and would confuse the model.
            if (not polish_mode
                    and not stats.warned_on_budget
                    and ctx.budget.remaining_pct() < (100 - _BUDGET_WARN_PCT)):
                stats.warned_on_budget = True
                warn = (f"{ctx.budget.remaining_pct():.0f}% budget remaining "
                        f"(cost ${ctx.cost.cumulative_cost_usd():.2f} USD)")
                ctx.events.emit("warn", message=f"soft budget warning: {warn}")
                runner.append_messages(_wrap_up_message(warn))
                continue

            # Compile-nudge — main-mode only. The counter tracks translation
            # writes, not polish edits; irrelevant during polish.
            if (not polish_mode
                    and not stats.nudged_on_compile
                    and ctx.files_written_since_compile >= _COMPILE_NUDGE_THRESHOLD):
                stats.nudged_on_compile = True
                ctx.events.emit(
                    "info",
                    message=f"nudging model to compile "
                            f"({ctx.files_written_since_compile} writes this session without one)",
                )
                runner.append_messages(_bomb_nudge_message(ctx.files_written_since_compile))
                continue

            # Session-reset checkpoint — main-mode only. Polish sessions
            # deliberately run without checkpointing: a mid-polish restart
            # would drop conversational context and force the model to
            # re-explore, doubling cost with no benefit.
            if not polish_mode and (reason := stats.should_reset(ctx)):
                ctx.events.emit("checkpoint", session=session_num, message=reason)
                return final_outcome, "checkpoint"

    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
        # Per-turn wall-clock timeout hit. This usually means the model got
        # stuck in a deep-think loop on a huge file. Best recovery: kill this
        # session, let the outer loop start a fresh one with cold context —
        # the model will approach the same file with less accumulated context
        # weight and (hopefully) commit to smaller units.
        ctx.events.emit(
            "warn",
            message=f"per-turn timeout ({ctx.cfg.agent.max_turn_seconds}s) — "
                    f"treating as checkpoint: {type(exc).__name__}: {exc}",
        )
        return final_outcome, "checkpoint"
    except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
        # Note: `InternalServerError` also covers `OverloadedError` (529)
        # since the SDK subclasses it. Both are transient — treat identically.
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
            return final_outcome, "rate_limit_hard"
        # Soft — outer loop should retry with a new session
        return final_outcome, "rate_limit_soft"
    except pydantic.ValidationError as exc:
        # SDK's structured-output parser threw. Common cause: model emitted
        # an empty text block or malformed JSON for the final structured
        # message (observed at session #10 of a long run — the model got
        # confused and returned '' where a TranslationOutcome was expected).
        # Treat as a soft failure — the outer loop will synthesize outcome
        # from disk state, preserving whatever translated files we have.
        ctx.events.emit(
            "error",
            message=f"structured-output parse failed at session "
                    f"{session_num}: {type(exc).__name__} — treating as "
                    f"parse_failure, will synthesize from disk state",
        )
        return final_outcome, "parse_failure"
    except anthropic.APIError as exc:
        ctx.events.emit("error", message=f"API error: {type(exc).__name__}: {exc}")
        raise
    except Exception as exc:  # noqa: BLE001
        # Last-resort safety net. Never let an unexpected exception in the
        # SDK, tool code, or parser abort the run when we have persisted
        # translation state on disk. Log loudly, hand off to the outer loop.
        ctx.events.emit(
            "error",
            message=f"unexpected session error: {type(exc).__name__}: {exc} "
                    f"— treating as parse_failure",
        )
        return final_outcome, "parse_failure"
    finally:
        # Clear polish-mode flags so subsequent non-polish sessions don't
        # inherit the read-cap logic.
        if polish_mode:
            ctx.polish_active = False

    # Runner naturally exhausted (max_iterations hit) without end_turn
    ctx.events.emit("warn",
                    message=f"session {session_num} exhausted iterations without end_turn")
    return final_outcome, "exhausted"


# ---------------------------------------------------------------------------
# Top-level agent entry point
# ---------------------------------------------------------------------------

def run_translator_phase2(
    ctx: SessionContext,
    *,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> tuple[TranslationOutcome | None, int]:
    """Multi-session translation loop. Returns (outcome, session_count).

    Each individual session runs Tool Runner until it hits a terminal state
    (end_turn, budget cap, checkpoint threshold, transient failure, or
    max_iterations exhaustion). The outer loop restarts sessions on
    "checkpoint", "rate_limit_soft", and "exhausted"; breaks on "end_turn",
    "budget_hit", "rate_limit_hard".
    """
    system_prompt = _load_system_prompt()
    tools = build_tools(ctx, include_write=True)

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
        message=f"Phase 2 translation — {len(tools)} tools loaded, "
                f"system prompt {len(system_prompt):,} chars, "
                f"{len(ctx.java_classes)} files to translate, "
                f"up to {total_sessions_cap} sessions",
    )

    if dry_run:
        ctx.events.emit("info", message="dry-run: skipping API calls")
        ctx.events.emit("phase_end", phase="translate",
                        message="dry-run OK — prompt and tools load cleanly")
        return None, 0

    thinking_desc = (
        f"budget={ctx.cfg.agent.thinking_budget_tokens:,}"
        if ctx.cfg.agent.thinking_budget_tokens > 0 else "adaptive"
    )
    ctx.events.emit(
        "info",
        message=f"model={ctx.cfg.agent.model} effort={ctx.cfg.agent.effort} "
                f"thinking={thinking_desc} "
                f"budget={ctx.cfg.agent.max_budget_tokens:,} tokens "
                f"(hard cap ≈ ${ctx.budget._max_cost_usd:.2f}) "  # noqa: SLF001
                f"tool_cap={ctx.cfg.agent.max_tool_calls} "
                f"wall_cap={ctx.cfg.agent.max_wall_seconds}s "
                f"turn_cap={int(ctx.cfg.agent.max_turn_seconds)}s "
                f"session_reset={ctx.cfg.agent.session_reset_after_files}f/"
                f"{ctx.cfg.agent.session_reset_after_tokens:,}t",
    )

    final_outcome: TranslationOutcome | None = None
    session_num = 0
    break_reason: str | None = None

    while session_num < total_sessions_cap:
        session_num += 1

        if _all_files_terminal(ctx):
            ctx.events.emit("info",
                            message=f"all {len(ctx.state.all())} files in terminal states — done")
            break_reason = "all_terminal"
            break
        if reason := ctx.budget.status():
            ctx.events.emit("warn",
                            message=f"budget cap reached before session {session_num}: {reason}")
            break_reason = "budget_hit_between_sessions"
            break

        ctx.events.emit("info", message=f"starting session #{session_num}")
        initial = _build_initial_message(ctx, session_num)

        outcome, reason_ended = _run_one_session(
            ctx, client=client, system_prompt=system_prompt, tools=tools,
            initial_message=initial, session_num=session_num,
        )
        if outcome is not None:
            final_outcome = outcome

        ctx.events.emit("info", message=f"session #{session_num} ended: {reason_ended}")

        # Terminal reasons — break the outer loop
        if reason_ended in ("end_turn", "finish_translate",
                            "budget_hit", "rate_limit_hard", "parse_failure"):
            break_reason = reason_ended
            break
        # else: "checkpoint" | "rate_limit_soft" | "exhausted" — loop again
    else:
        # while-else: hit the total_sessions_cap without breaking
        ctx.events.emit("warn",
                        message=f"reached session cap ({total_sessions_cap}) — stopping")
        break_reason = "session_cap"

    # Post-loop polish pass: if all files are terminal and compile is green
    # but format/credo aren't, open one polish session under tool_choice="any"
    # + the finish_polish sentinel tool. The model cannot emit pure end_turn —
    # it must either fix warnings via edit_elixir or explicitly declare them
    # unfixable via finish_polish(unfixable_warnings=[...]).
    polish_sessions_used = 0
    if not dry_run and _all_files_terminal(ctx):
        tools_polish = build_tools(ctx, include_write=True, polish_mode=True)
        polish_outcome, polish_sessions_used = _maybe_run_polish_sessions(
            ctx, client=client, system_prompt=system_prompt,
            tools_polish=tools_polish,
            first_session_num=session_num + 1,
            session_cap=max(0, total_sessions_cap - session_num),
        )
        session_num += polish_sessions_used
    polish_ran = polish_sessions_used > 0

    if final_outcome is None:
        final_outcome = _synthesize_outcome_from_state(ctx)
        # Not a warning under the finish_translate design — this IS the
        # normal path. Outcome is always synthesized from disk state;
        # `phase_a_summary` (if set) contributes the model's overview text.
        source = "finish_translate summary + state" if ctx.phase_a_summary else "state only"
        ctx.events.emit("info", message=f"outcome synthesized ({source})")
    elif polish_ran:
        # Polish sessions run without output_format (tool_choice="any" + a
        # `finish_polish` sentinel supersede structured output). Re-sync
        # validation from disk to reflect the polish session's edits.
        synthetic = _synthesize_outcome_from_state(ctx)
        final_outcome.validation = synthetic.validation
        if ctx.polish_finish_unfixable:
            final_outcome.summary = (
                final_outcome.summary
                + f" | polish: {len(ctx.polish_finish_unfixable)} unfixable declared"
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
    client: anthropic.Anthropic,
    system_prompt: str,
    tools_polish: list,
    first_session_num: int,
    session_cap: int,
) -> tuple[TranslationOutcome | None, int]:
    """Run a single polish session under `tool_choice="any"` + finish_polish.

    Under this configuration the model CANNOT emit pure end_turn. Every turn
    must contain at least one tool call, and the only way to end the session
    is `finish_polish(summary, unfixable_warnings)`. This eliminates the
    "read a couple files then quit" failure mode structurally, not via prose.

    Returns (final_outcome, sessions_used). `sessions_used == 0` means no
    polish was needed or possible (compile red / already clean).
    """
    try:
        c = mix_compile(ctx.scaffold.mix_env, ctx.target_root)
        f = mix_format(ctx.scaffold.mix_env, ctx.target_root, check_only=True)
        k = mix_credo(ctx.scaffold.mix_env, ctx.target_root, strict=True)
    except Exception as exc:  # noqa: BLE001
        ctx.events.emit(
            "warn",
            message=f"polish pre-check failed: {type(exc).__name__}: {exc} — skipping polish",
        )
        return None, 0

    if not c.ok:
        ctx.events.emit(
            "warn",
            message="polish skipped: mix compile is red — that's a translation "
                    "bug, not polish work",
        )
        return None, 0
    if f.ok and k.ok:
        return None, 0
    if session_cap <= 0:
        return None, 0

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
            return None, 0

    ctx.events.emit(
        "info",
        message=f"opening polish session #{first_session_num}: "
                f"format={'✓' if f.ok else '✗'} credo={'✓' if k.ok else '✗'} "
                f"(tool_choice=any; must call finish_polish to end)",
    )
    initial = _build_polish_message(ctx, f, k)

    edits_before = ctx.cost.tool_call_count("edit_elixir")

    outcome, reason_ended = _run_one_session(
        ctx, client=client, system_prompt=system_prompt, tools=tools_polish,
        initial_message=initial, session_num=first_session_num,
        polish_mode=True,
    )

    edits_made = ctx.cost.tool_call_count("edit_elixir") - edits_before
    ctx.events.emit(
        "info",
        message=f"polish session #{first_session_num} ended: {reason_ended} "
                f"(edits: {edits_made}; "
                f"unfixable declared: {len(ctx.polish_finish_unfixable)})",
    )
    if ctx.polish_finish_unfixable:
        for warn_line in ctx.polish_finish_unfixable[:20]:
            ctx.events.emit("info", message=f"  polish left: {warn_line}")

    # Persist the outcome for skip-on-resume. Re-read credo — the polish
    # session may have fixed some warnings, so pre-polish state is stale.
    if reason_ended == "finish_polish":
        try:
            k_post = mix_credo(ctx.scaffold.mix_env, ctx.target_root, strict=True)
            post_ids = _parse_credo_warnings(k_post.output)
            _save_polish_state(ctx, post_ids, ctx.polish_finish_unfixable)
            ctx.events.emit(
                "info",
                message=f"polish state saved: {len(post_ids)} remaining warning(s) "
                        f"(future resumes will skip polish if credo state is unchanged)",
            )
        except Exception as exc:  # noqa: BLE001
            ctx.events.emit(
                "warn",
                message=f"polish state save failed: {type(exc).__name__}: {exc}",
            )

    return outcome, 1


def _synthesize_outcome_from_state(ctx: SessionContext) -> TranslationOutcome:
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
        except Exception as exc:  # noqa: BLE001
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
    # about this run. Falls back to placeholder text when finish_translate
    # wasn't called (checkpoint, budget hit, parse_failure fallback, etc.).
    if ctx.phase_a_summary:
        summary_text = ctx.phase_a_summary
    else:
        summary_text = "synthesized from state — model did not call finish_translate"

    return TranslationOutcome(
        summary=summary_text[:800],
        validation=validation,
        files=files,
        escalations=escalations,
        remaining_blockers=remaining,
    )
