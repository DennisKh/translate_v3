"""Post-translation README rewrite.

One-shot API call — no agent loop needed. Reads the source README + a
compact summary of every translated Elixir module's public API, produces a
new README with valid Elixir code examples.

Runs at end of Phase A once validation is green (compile clean). Failure
here is non-fatal — the translated code is the deliverable, the README is a
nice-to-have. Errors log and continue.
"""

from __future__ import annotations

from pathlib import Path

import anthropic

from agent.cost import TurnUsage
from agent.session_ctx import SessionContext
from agent.state import FileState
from language.elixir import describe_file


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_readme_prompt() -> str:
    return (_PROMPTS_DIR / "readme_system.md").read_text()


def _find_source_readme(source_root: Path) -> Path | None:
    """Best-effort source README lookup."""
    for name in ("README.md", "README.MD", "readme.md", "Readme.md",
                 "README.rst", "README.txt"):
        p = source_root / name
        if p.exists():
            return p
    return None


def _compact_api_summary(ctx: SessionContext) -> str:
    """One line per translated Elixir module — name + kind + public fns.

    Compact form so the whole project's API fits comfortably in a single
    prompt. E.g.:
      - JodaMoney.BigMoney [struct fields=(currency, amount)]: of/1, of/2, plus/2, plus/3, ...
      - JodaMoney.Format.MoneyFormatter [module]: parse/1, print/2, ...
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


def translate_readme(
    ctx: SessionContext,
    *,
    client: anthropic.Anthropic,
    dry_run: bool = False,
) -> bool:
    """Rewrite source README as an Elixir-flavored README. Non-fatal.

    Returns True on success, False on any failure. Errors are logged, never
    raised — README translation is a nice-to-have, not a blocker.
    """
    source_readme = _find_source_readme(ctx.source_root)
    if source_readme is None:
        ctx.events.emit("info", message="no source README.md found — skipping README translation")
        return False

    target_readme = ctx.target_root / "README.md"

    if dry_run:
        ctx.events.emit("info",
                        message=f"dry-run: would translate {source_readme.name} → {target_readme}")
        return False

    ctx.events.emit("phase_start", phase="readme",
                    message=f"translating {source_readme.name} → {target_readme}")

    source_text = source_readme.read_text()
    api_summary = _compact_api_summary(ctx)

    if not api_summary:
        ctx.events.emit("warn",
                        message="no translated modules — skipping README translation")
        return False

    system_prompt = _load_readme_prompt()
    user_prompt = (
        f"## Source README (Java project)\n\n"
        f"```markdown\n{source_text}\n```\n\n"
        f"## Available API (all currently translated Elixir modules)\n\n"
        f"{api_summary}\n\n"
        f"## Project details\n\n"
        f"- Java package prefix: original project's Java package\n"
        f"- Elixir top-level module: `{ctx.module_prefix}`\n"
        f"- Mix app name: `{ctx.app_snake}`\n"
        f"- Elixir version target: 1.17+ / OTP 26+\n"
        f"- Deps: `{{:decimal, \"~> 2.0\"}}`\n\n"
        f"Rewrite the source README as the new project's README.md, using "
        f"the rules from the system prompt. Return raw markdown only."
    )

    try:
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
    except Exception as exc:  # noqa: BLE001
        # Broad catch — README rewrite is a non-fatal one-shot. Any failure
        # (network, malformed response, unexpected SDK behavior) should log
        # and let the run proceed; the translated code is the deliverable.
        ctx.events.emit("error",
                        message=f"README translation failed: "
                                f"{type(exc).__name__}: {exc}")
        return False

    # Record cost
    usage = response.usage
    turn_usage = TurnUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )
    ctx.cost.add_turn(turn_usage, current_file="__readme__")
    ctx.budget.add_usage(turn_usage, ctx.cfg.agent.model)

    # Extract text
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    new_readme = "".join(parts).strip()

    if not new_readme:
        ctx.events.emit("warn", message="README translation returned empty output — skipping write")
        return False

    # Strip any code-fence wrapper the model may have added despite instructions
    if new_readme.startswith("```"):
        # Try to strip leading and trailing fences
        lines = new_readme.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        new_readme = "\n".join(lines).strip()

    target_readme.write_text(new_readme + "\n")
    ctx.events.emit(
        "phase_end", phase="readme",
        message=f"wrote {target_readme} ({new_readme.count(chr(10)) + 1} lines, "
                f"cost ${turn_usage.cost_usd(ctx.cfg.agent.model):.3f})",
    )
    return True
