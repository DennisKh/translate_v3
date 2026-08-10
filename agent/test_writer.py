"""Phase B — ExUnit test generation from Java tests.

One-shot LLM call per Java test file (no agent loop). Generates an ExUnit
test against the translated Elixir modules, verifies with `mix test`. On
failure, one retry with the error output fed back. Failing tests are kept
on disk with a `# TODO: failing on generation` prefix so the operator can
review — better than deleting attempted work.

State: `.translate_v3_state/tests.json` — per-file {state, cost_usd, error}.
On resume, files already marked `pass` are skipped if the .exs still exists.

Design mirrors `readme_translator.py` — no agent loop, no tool_choice
gymnastics. Test conversion is per-file mechanical work; batching would
complicate cost accounting and error attribution without benefit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic

from agent.cost import TurnUsage
from agent.mix_ops import mix_test
from agent.session_ctx import SessionContext
from agent.state import FileState
from language.elixir import describe_file
from project.naming import camel_to_snake


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class TestGenResult:
    total: int
    passing: int
    failing: int
    skipped: int
    cost_usd: float


def _load_test_prompt() -> str:
    return (_PROMPTS_DIR / "test_writer_system.md").read_text()


def _find_java_tests(ctx: SessionContext) -> list[Path]:
    if ctx.tests_root is None or not ctx.tests_root.exists():
        return []
    return sorted(ctx.tests_root.glob("**/*.java"))


def _state_path(ctx: SessionContext) -> Path:
    return ctx.scaffold.target_root / ".translate_v3_state" / "tests.json"


def _load_state(ctx: SessionContext) -> dict:
    path = _state_path(ctx)
    if not path.exists():
        return {"tests": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"tests": {}}


def _save_state(ctx: SessionContext, state: dict) -> None:
    path = _state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _api_summary(ctx: SessionContext) -> str:
    """Compact API summary of all translated Elixir modules.

    Same shape as readme_translator's summary so the model sees a
    consistent grounding format across post-Phase-A steps.
    """
    lines: list[str] = []
    for entry in ctx.state.by_state(FileState.COMPLETE):
        lib_path = ctx.target_root / entry.lib_path
        shape = describe_file(lib_path)
        if shape is None:
            lines.append(f"- {entry.module_name}: (unparseable)")
            continue
        kind_tags: list[str] = [shape.kind] if shape.kind != "module" else []
        if shape.defstruct_fields:
            kind_tags.append(f"struct fields=({', '.join(shape.defstruct_fields)})")
        tag = f" [{'; '.join(kind_tags)}]" if kind_tags else ""
        fns = ", ".join(sorted({f"{s.name}/{s.arity}" for s in shape.public_functions}))
        lines.append(f"- {shape.module_name}{tag}: {fns or '(no public functions)'}")
    return "\n".join(lines)


def _target_test_path(ctx: SessionContext, java_path: Path) -> Path:
    """`FooTest.java` → `test/foo_test.exs` (flat — mix has no need for subdirs)."""
    stem_snake = camel_to_snake(java_path.stem)
    if not stem_snake.endswith("_test"):
        stem_snake = f"{stem_snake}_test"
    return ctx.target_root / "test" / f"{stem_snake}.exs"


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _generate_one(
    ctx: SessionContext,
    *,
    client: anthropic.Anthropic,
    java_src: str,
    api_summary: str,
    system_prompt: str,
    prior_failure: str = "",
) -> tuple[str, TurnUsage]:
    """One LLM call producing the ExUnit source text. Returns (source, usage)."""
    retry_block = ""
    if prior_failure:
        retry_block = (
            f"\n\n## Previous attempt failed\n\n"
            f"Your previous test file failed `mix test`. Error output (last ~2KB):\n\n"
            f"```\n{prior_failure[:2000]}\n```\n\n"
            f"Fix the specific issue above. Common causes: wrong function name "
            f"or arity, using JUnit assertions that don't exist in ExUnit "
            f"(assertEquals etc.), using `==` instead of `assert_in_delta` for "
            f"floats, or referring to struct fields the Elixir port doesn't "
            f"actually have. Cross-check against the API list below.\n"
        )

    user_prompt = (
        f"## Java (JUnit) test source\n\n"
        f"```java\n{java_src}\n```\n\n"
        f"## Available Elixir API (all translated modules)\n\n"
        f"{api_summary}\n\n"
        f"## Elixir project\n\n"
        f"- Top-level module: `{ctx.module_prefix}`\n"
        f"- App name: `{ctx.app_snake}`\n"
        f"- Elixir 1.17+ / OTP 26+ with ExUnit\n"
        f"- `decimal` dep available (translated `BigDecimal` uses it)\n"
        f"{retry_block}\n"
        f"Rewrite this JUnit test as a complete ExUnit test file using ONLY "
        f"functions from the API list. Return raw Elixir code — no markdown "
        f"fences, no commentary."
    )

    response = client.messages.create(
        model=ctx.cfg.agent.model,
        max_tokens=16_000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_prompt}],
    )

    u = response.usage
    turn_usage = TurnUsage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )

    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return _strip_code_fence("".join(parts)), turn_usage


def _record_usage(ctx: SessionContext, usage: TurnUsage, tag: str) -> float:
    ctx.cost.add_turn(usage, current_file=tag)
    ctx.budget.add_usage(usage, ctx.cfg.agent.model)
    return usage.cost_usd(ctx.cfg.agent.model)


def generate_tests(
    ctx: SessionContext,
    *,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> TestGenResult:
    """Generate ExUnit tests from Java test sources. Non-fatal on any error.

    Idempotent: previously-passing tests are skipped on resume (state cache
    at `.translate_v3_state/tests.json`).
    """
    java_tests = _find_java_tests(ctx)
    if not java_tests:
        ctx.events.emit("info", message="Phase B: no Java tests found — skipping")
        return TestGenResult(0, 0, 0, 0, 0.0)

    api_summary = _api_summary(ctx)
    if not api_summary:
        ctx.events.emit("warn", message="Phase B skipped: no COMPLETE Elixir modules to test against")
        return TestGenResult(len(java_tests), 0, 0, len(java_tests), 0.0)

    ctx.events.emit(
        "phase_start", phase="tests",
        message=f"Phase B test generation — {len(java_tests)} Java test file(s)",
    )

    if dry_run:
        ctx.events.emit("info", message="dry-run: skipping test generation")
        ctx.events.emit("phase_end", phase="tests", message="dry-run OK")
        return TestGenResult(len(java_tests), 0, 0, len(java_tests), 0.0)

    system_prompt = _load_test_prompt()
    state = _load_state(ctx)
    passing = failing = skipped = 0
    total_cost = 0.0

    for i, java_path in enumerate(java_tests, 1):
        key = java_path.stem
        target = _target_test_path(ctx, java_path)
        rel_target = str(target.relative_to(ctx.target_root))
        prior = state["tests"].get(key)

        if prior and prior.get("state") == "pass" and target.exists():
            ctx.events.emit("info",
                            message=f"[{i}/{len(java_tests)}] {key}: cached pass — skipping")
            passing += 1
            continue

        ctx.events.emit("info",
                        message=f"[{i}/{len(java_tests)}] {key}: generating {rel_target}")

        try:
            source, usage = _generate_one(
                ctx, client=client, java_src=java_path.read_text(),
                api_summary=api_summary, system_prompt=system_prompt,
            )
            cost = _record_usage(ctx, usage, f"__test_{key}__")
            total_cost += cost
        except Exception as exc:  # noqa: BLE001
            # Broad catch — any per-file failure (network, parse, malformed
            # SDK response) must not abort the remaining files. Log + record
            # state + continue.
            ctx.events.emit("error",
                            message=f"[{i}] {key}: generation error: "
                                    f"{type(exc).__name__}: {exc}")
            state["tests"][key] = {"state": "error", "reason": str(exc)[:200]}
            _save_state(ctx, state)
            failing += 1
            continue

        if not source:
            ctx.events.emit("warn", message=f"[{i}] {key}: empty output — skipping")
            state["tests"][key] = {"state": "error", "reason": "empty generation"}
            _save_state(ctx, state)
            failing += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source + ("" if source.endswith("\n") else "\n"))

        result = mix_test(ctx.scaffold.mix_env, ctx.target_root,
                          only=rel_target, max_failures=3)
        if result.ok:
            ctx.events.emit("info", message=f"[{i}] {key}: PASS (cost ${cost:.3f})")
            passing += 1
            state["tests"][key] = {"state": "pass", "cost_usd": round(cost, 4)}
            _save_state(ctx, state)
            continue

        ctx.events.emit("info", message=f"[{i}] {key}: FAIL — retrying with error context")
        try:
            source2, usage2 = _generate_one(
                ctx, client=client, java_src=java_path.read_text(),
                api_summary=api_summary, system_prompt=system_prompt,
                prior_failure=result.output_tail,
            )
            cost2 = _record_usage(ctx, usage2, f"__test_{key}__")
            total_cost += cost2
        except Exception as exc:  # noqa: BLE001
            ctx.events.emit("error",
                            message=f"[{i}] {key}: retry error: "
                                    f"{type(exc).__name__}: {exc}")
            state["tests"][key] = {"state": "error", "reason": str(exc)[:200],
                                    "cost_usd": round(cost, 4)}
            _save_state(ctx, state)
            failing += 1
            continue

        combined_cost = cost + cost2
        if source2:
            target.write_text(source2 + ("" if source2.endswith("\n") else "\n"))
        result2 = mix_test(ctx.scaffold.mix_env, ctx.target_root,
                           only=rel_target, max_failures=3)

        if result2.ok:
            ctx.events.emit("info",
                            message=f"[{i}] {key}: PASS on retry (cost ${combined_cost:.3f})")
            passing += 1
            state["tests"][key] = {"state": "pass", "cost_usd": round(combined_cost, 4)}
        else:
            # Keep the file with a TODO header — the operator may want to fix
            # by hand rather than re-run. Deleting attempted work is user-hostile.
            current = target.read_text()
            tail_lines = result2.output_tail.splitlines()[-8:]
            todo_header = (
                "# TODO: generated test failed both attempts — review and fix manually.\n"
                "# Last mix test error tail:\n"
                + "".join(f"#   {line}\n" for line in tail_lines)
                + "\n"
            )
            target.write_text(todo_header + current)
            ctx.events.emit(
                "warn",
                message=f"[{i}] {key}: FAIL both attempts — marked TODO "
                        f"(cost ${combined_cost:.3f})",
            )
            failing += 1
            state["tests"][key] = {
                "state": "fail",
                "cost_usd": round(combined_cost, 4),
                "error_tail": result2.output_tail[-400:],
            }
        _save_state(ctx, state)

    ctx.events.emit(
        "phase_end", phase="tests",
        message=f"Phase B done: {passing} pass, {failing} fail, {skipped} skip "
                f"— cost ${total_cost:.2f}",
    )
    return TestGenResult(
        total=len(java_tests), passing=passing, failing=failing,
        skipped=skipped, cost_usd=total_cost,
    )
