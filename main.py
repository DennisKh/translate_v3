#!/usr/bin/env python3
"""v3 Java→Elixir translator — agentic architecture.

Two required positional args, everything else optional:

    translate SOURCE_ROOT TARGET_ROOT [--config CONFIG] [flags]

Precedence: CLI flag > config file > default.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# Require Python 3.11 (tomllib is stdlib since 3.11)
if sys.version_info < (3, 11):
    sys.stderr.write(
        f"translate_v3 requires Python 3.11+ (found {sys.version_info.major}.{sys.version_info.minor}).\n"
        "Reason: uses the stdlib `tomllib` module introduced in 3.11.\n"
    )
    sys.exit(2)

# Make sibling packages importable when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click

from agent.cost import CostBudget, CostReport
from agent.events import EventStream
from agent.llm import build_chat_model
from agent.session_ctx import SessionContext
from agent.state import FileEntry, StateStore
from agent.summary import build_and_write_summary, print_summary
from agent.readme_translator import translate_readme
from agent.test_writer import generate_tests
from agent.translator import run_translator_phase2
from language.java import discover_classes, discover_deps
from project.config import Config, build_config
from project.deps import build as build_dep_graph
from project.naming import camel_to_snake
from project.scaffold import ScaffoldResult, scaffold


def _common_package_prefix(java_classes: list) -> str:
    """Longest package prefix shared by every class.

    For joda-money, every class is in `org.joda.money.*`, so the common prefix
    is `org.joda.money`. Files at that exact package become
    `<ModulePrefix>.<Class>`; files in sub-packages become
    `<ModulePrefix>.<SubPackage>.<Class>`.

    Returns "" if there is no common prefix (e.g. multi-package project).
    """
    if not java_classes:
        return ""
    packages = [jc.package.split(".") if jc.package else [] for jc in java_classes]
    if not packages or not packages[0]:
        return ""
    prefix: list[str] = []
    for parts in zip(*packages):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return ".".join(prefix)


def _sub_package(java_class, common_prefix: str) -> list[str]:
    """Return the sub-package parts of `java_class` relative to `common_prefix`."""
    pkg = java_class.package
    if not pkg:
        return []
    if common_prefix and pkg.startswith(common_prefix):
        rest = pkg[len(common_prefix):].lstrip(".")
    else:
        rest = pkg
    return rest.split(".") if rest else []


def _lib_path_for(java_class, common_prefix: str, app_snake: str,
                  module_prefix: str) -> tuple[str, str, str]:
    """Compute (module_name, lib_path, test_path) relative to target root."""
    sub = _sub_package(java_class, common_prefix)
    subdirs = [p.lower() for p in sub]
    ns_parts = [p.capitalize() for p in sub]
    snake_stem = camel_to_snake(java_class.class_name)

    module_parts = [module_prefix, *ns_parts, java_class.class_name]
    module_name = ".".join(module_parts)

    if subdirs:
        lib_path = f"lib/{app_snake}/" + "/".join(subdirs) + f"/{snake_stem}.ex"
        test_path = f"test/{app_snake}/" + "/".join(subdirs) + f"/{snake_stem}_test.exs"
    else:
        lib_path = f"lib/{app_snake}/{snake_stem}.ex"
        test_path = f"test/{app_snake}/{snake_stem}_test.exs"
    return module_name, lib_path, test_path


def _run_regression_scoring(cfg: Config, scaffold_result: ScaffoldResult,
                            events: EventStream, state_dir: Path) -> dict | None:
    """Run tests/regression/run.py against the just-translated project.

    Uses `scaffold_result.app_name` (the SAME name `mix new` chose or the user
    forced via --app-name), NOT a re-derivation from the directory basename.
    Avoids silent quality-signal loss when the two would differ.
    """
    try:
        from tests.regression.run import score as regression_score
    except ImportError as exc:
        events.emit("warn", message=f"regression harness import failed: {exc}")
        return None

    candidate_lib = cfg.target.root / "lib" / scaffold_result.app_name
    if not candidate_lib.exists():
        events.emit("warn", message=f"candidate lib dir not found: {candidate_lib}")
        return None

    scores = regression_score(candidate_lib, workers=3)
    total_syms = sum(len(s.matched_symbols) + len(s.missing_symbols) for s in scores)
    matched_syms = sum(len(s.matched_symbols) for s in scores)
    total_tests = sum(s.test_total for s in scores)
    passed_tests = sum(s.test_pass for s in scores)

    return {
        "current": {
            "symbol_coverage": 100 * matched_syms // total_syms if total_syms else 100,
            "test_parity":     100 * passed_tests // total_tests if total_tests else 100,
            "compile_clean":   100 * sum(1 for s in scores if s.compile_ok) // max(len(scores), 1),
        },
    }


def _resolve_tests_root(cfg: Config) -> Path | None:
    """Resolve the Java tests root from `cfg.source.root` + tests_glob.

    Given `source.root = /var/www/joda-money` and `tests_glob =
    src/test/java/**/*.java`, returns `/var/www/joda-money/src/test/java` if
    it exists, else None.
    """
    if cfg.source.root is None:
        return None
    # Strip the trailing `/**/*.java` (or similar) to get the containing dir
    glob = cfg.source.tests_glob
    # Take everything before the first `*` or `**`
    stop = len(glob)
    for marker in ("**", "*"):
        idx = glob.find(marker)
        if idx != -1 and idx < stop:
            stop = idx
    prefix = glob[:stop].rstrip("/")
    if not prefix:
        return None
    candidate = cfg.source.root / prefix
    return candidate if candidate.exists() else None


def _seed_state_from_java(cfg: Config, scaffold_result: ScaffoldResult, state: StateStore,
                          events: EventStream) -> tuple[list, dict, dict]:
    """Discover Java files, build dep graph, seed the state store.

    Returns (java_classes, dep_graph, java_by_stem).
    """
    src_root = cfg.source.root
    assert src_root is not None
    events.emit("info", message=f"scanning {src_root} with glob {cfg.source.sources_glob!r}")

    java_classes = discover_classes(src_root, cfg.source.sources_glob)
    events.emit("info", message=f"discovered {len(java_classes)} Java classes")

    common_prefix = _common_package_prefix(java_classes)
    if common_prefix:
        events.emit("info", message=f"common Java package prefix: {common_prefix} "
                                     f"(stripped from module namespace)")

    deps_by_stem = discover_deps(java_classes)
    dep_graph = build_dep_graph(deps_by_stem)
    events.emit("info",
                message=f"dep graph: {len(dep_graph.levels)} levels, "
                        f"{len(dep_graph.sccs)} cycle group(s)")

    # Seed state for every Java class (if not already present from a prior run)
    seeded = 0
    for jc in java_classes:
        if state.get(jc.class_name) is not None:
            continue
        module_name, lib_path, test_path = _lib_path_for(
            jc, common_prefix, scaffold_result.app_name, scaffold_result.module_name,
        )
        state.upsert(FileEntry(
            stem=jc.class_name,
            java_path=str(jc.path.relative_to(src_root)),
            lib_path=lib_path,
            test_path=test_path,
            module_name=module_name,
            deps=dep_graph.dep_list(jc.class_name),
            level=dep_graph.level_of.get(jc.class_name, 0),
        ))
        seeded += 1
    all_entries = state.all()
    resumed = len(all_entries) - seeded
    if resumed > 0:
        completed = sum(1 for e in all_entries
                        if e.state.value in ("complete", "skipped", "deleted", "escalated"))
        events.emit("info",
                    message=f"seeded {seeded} new file(s); resumed {resumed} existing "
                            f"({completed} already in terminal state, will be skipped)")
    else:
        events.emit("info", message=f"seeded {seeded} new file(s) in state store "
                                     f"(total tracked: {len(all_entries)})")

    java_by_stem = {jc.class_name: jc for jc in java_classes}
    return java_classes, dep_graph, java_by_stem


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("target_root", type=click.Path(path_type=Path))
@click.option("--config", "config_file",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Optional TOML config file.")
@click.option("--app-name", default=None,
              help="Override mix app name (default: derived by `mix new`).")
@click.option("--module-name", default=None,
              help="Override top-level Elixir module (default: derived by `mix new`).")
@click.option("--model", default=None,
              help="Override translation model ID (default: claude-opus-4-7).")
@click.option("--max-budget-tokens", type=int, default=None,
              help="Override total token budget (default: 3M).")
@click.option("--phase", type=click.Choice(["A", "B", "both"]), default="A",
              help="Which phase(s) to run. A=translate, B=tests, both. Default: A.")
@click.option("--dry-run", is_flag=True,
              help="Scaffold + print plan, no API calls.")
@click.option("--resume", is_flag=True,
              help="Resume from .translate_v3_state/ in an existing target dir. "
                   "Skips re-scaffolding; already-COMPLETE files are not retranslated.")
@click.option("--verbose", is_flag=True,
              help="Stream every event to stdout (default: summary only).")
@click.option("--skip-format", is_flag=True,
              help="Skip `mix format --check-formatted` in the validation gate.")
@click.option("--skip-credo", is_flag=True,
              help="Skip `mix credo --strict` in the validation gate.")
@click.option("--strict-validation", is_flag=True,
              help="Treat any Credo warning as fatal (default: only :high+).")
@click.option("--run-regression", is_flag=True,
              help="After translation, run tests/regression/run.py and inject scores into the summary.")
@click.option("--skip-readme", is_flag=True,
              help="Skip README.md translation after Phase A (README is rewritten "
                   "with Elixir code examples by default when translation is clean).")
def main(
    source_root: Path,
    target_root: Path,
    config_file: Path | None,
    app_name: str | None,
    module_name: str | None,
    model: str | None,
    max_budget_tokens: int | None,
    phase: Literal["A", "B", "both"],
    dry_run: bool,
    resume: bool,
    verbose: bool,
    skip_format: bool,
    skip_credo: bool,
    strict_validation: bool,
    run_regression: bool,
    skip_readme: bool,
) -> None:
    """Translate SOURCE_ROOT (Java project) to TARGET_ROOT (Elixir project).

    Precedence: CLI flag > config file > default.
    """
    cfg, config_warnings = build_config(
        source_root=source_root.resolve(),
        target_root=target_root.resolve(),
        config_file=config_file,
        app_name=app_name,
        module_name=module_name,
        model=model,
        max_budget_tokens=max_budget_tokens,
        phase=phase,
        skip_format=skip_format,
        skip_credo=skip_credo,
        strict_validation=strict_validation,
    )

    for w in config_warnings:
        click.secho(f"config warning: {w}", err=True, fg="yellow")

    print(f"Source: {cfg.source.root}")
    print(f"Target: {cfg.target.root}")
    print(f"Model : {cfg.llm.model} (effort={cfg.agent.effort})")
    print(f"Phase : {'A ' if cfg.phases.translate else '  '}{'B' if cfg.phases.generate_tests else ' '}")
    print()

    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        click.secho("ANTHROPIC_API_KEY is not set — use --dry-run to test without API access.",
                    err=True, fg="red")
        sys.exit(2)

    exit_code = 0
    fatal_exc: BaseException | None = None
    events: EventStream | None = None
    state: StateStore | None = None
    cost: CostReport | None = None
    budget: CostBudget | None = None
    validation_summary: dict | None = None
    regression_summary: dict | None = None
    sessions_count = 1
    state_dir = cfg.target.root / ".translate_v3_state"

    # --resume preconditions checked BEFORE the try block so a clean sys.exit
    # here isn't overwritten by the outer sys.exit(exit_code) in finally.
    if resume:
        if not (cfg.target.root / "mix.exs").exists():
            click.secho(f"--resume: no mix.exs at {cfg.target.root} — nothing to resume",
                        err=True, fg="red")
            sys.exit(2)
        if not state_dir.exists():
            click.secho(f"--resume: no state dir at {state_dir} — nothing to resume",
                        err=True, fg="red")
            sys.exit(2)

    try:
        # Scaffold FIRST — before creating the state dir (mix new requires empty target).
        # With --resume, scaffold() detects the populated dir and skips `mix new`.
        scaffold_result = scaffold(cfg)
        print(f"Scaffolded: app_name={scaffold_result.app_name} module={scaffold_result.module_name}")
        print(f"           mix path={scaffold_result.mix_env.mix_path} "
              f"({scaffold_result.mix_env.version})")
        print()

        # Now the state dir + observability are safe to create
        events = EventStream(state_dir / "events.jsonl", mirror_to_stdout=verbose or dry_run)
        state = StateStore(state_dir)
        cost = CostReport(cfg.llm.model, state_dir)
        budget = CostBudget(
            max_tokens=cfg.agent.max_budget_tokens,
            max_tool_calls=cfg.agent.max_tool_calls,
            max_wall_seconds=cfg.agent.max_wall_seconds,
        )
        budget.start()

        # Seed state from Java scan
        java_classes, dep_graph, java_by_stem = _seed_state_from_java(
            cfg, scaffold_result, state, events,
        )

        ctx = SessionContext(
            cfg=cfg, scaffold=scaffold_result, state=state, events=events,
            cost=cost, budget=budget,
            java_classes=java_classes, dep_graph=dep_graph,
            java_by_stem=java_by_stem,
            module_to_stem={e.module_name: e.stem for e in state.all()},
            tests_root=_resolve_tests_root(cfg),
        )

        bundle = build_chat_model(cfg)

        outcome = None
        if cfg.phases.translate:
            outcome, sessions_count = run_translator_phase2(
                ctx, bundle=bundle, dry_run=dry_run,
            )
            if outcome is not None:
                outcome_path = state_dir / "outcome.json"
                outcome_path.write_text(json.dumps(outcome.model_dump(), indent=2))
                events.emit("info", message=f"outcome written to {outcome_path}")
                validation_summary = {
                    "compile": outcome.validation.compile_ok,
                    "format": outcome.validation.format_ok,
                    "credo": outcome.validation.credo_ok,
                }

        # Ensure validation_summary is populated even when Phase A didn't run
        # (e.g. `--phase B` standalone). Post-Phase-A steps gate on this;
        # without it they'd silently skip on standalone Phase B.
        if validation_summary is None and not dry_run:
            from agent.mix_ops import mix_compile, mix_format, mix_credo  # local import
            try:
                c = mix_compile(scaffold_result.mix_env, scaffold_result.target_root)
                f = mix_format(scaffold_result.mix_env, scaffold_result.target_root,
                               check_only=True)
                k = mix_credo(scaffold_result.mix_env, scaffold_result.target_root,
                              strict=True)
                validation_summary = {"compile": c.ok, "format": f.ok, "credo": k.ok}
                events.emit(
                    "info",
                    message=f"pre-Phase-B validation: "
                            f"compile={'✓' if c.ok else '✗'} "
                            f"format={'✓' if f.ok else '✗'} "
                            f"credo={'✓' if k.ok else '✗'}",
                )
            except Exception as exc:  # noqa: BLE001
                events.emit("warn",
                            message=f"pre-Phase-B validation failed: "
                                    f"{type(exc).__name__}: {exc}")

        # Post-Phase-A: rewrite README with Elixir code examples. Gate on
        # compile+format only — credo strict is style-only and doesn't affect
        # whether the APIs referenced in code examples are real.
        if (cfg.target.translate_readme
                and not skip_readme
                and cfg.phases.translate
                and validation_summary is not None
                and validation_summary.get("compile")
                and validation_summary.get("format")):
            try:
                translate_readme(ctx, bundle=bundle, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001
                # README translation is non-fatal — errors log and continue
                events.emit("warn",
                            message=f"README translation error (non-fatal): "
                                    f"{type(exc).__name__}: {exc}")

        # Phase B — ExUnit test generation from Java tests. Gate on the same
        # "translation actually landed clean" check as README: compile+format
        # green. Credo strict is style-only and doesn't affect API accuracy.
        if (cfg.phases.generate_tests
                and validation_summary is not None
                and validation_summary.get("compile")
                and validation_summary.get("format")):
            try:
                generate_tests(ctx, bundle=bundle, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001
                # Phase B is non-fatal — errors log and continue
                events.emit("warn",
                            message=f"Phase B test-gen error (non-fatal): "
                                    f"{type(exc).__name__}: {exc}")
        elif cfg.phases.generate_tests:
            events.emit(
                "warn",
                message="Phase B skipped: translation validation not green "
                        "(compile or format failing)",
            )

        # Optional regression score — skip if translation produced no COMPLETE files
        # (would just score against garbage and waste ~90s per run)
        if run_regression and not dry_run:
            completed_count = sum(1 for f in state.all()
                                  if f.state.value == "complete")
            if completed_count == 0:
                events.emit("warn",
                            message="skipping regression scoring: no files reached COMPLETE state")
            else:
                events.emit("info",
                            message=f"running regression harness ({completed_count} completed file(s))")
                regression_summary = _run_regression_scoring(cfg, scaffold_result, events, state_dir)

    except KeyboardInterrupt:
        if events is not None:
            events.emit("warn", message="interrupted; state persisted")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        if events is not None:
            events.emit("error", message=f"{type(exc).__name__}: {exc}")
        exit_code = 1
        fatal_exc = exc
    finally:
        if cost is not None and budget is not None and state is not None:
            cost.persist()
            summary = build_and_write_summary(
                state_dir=state_dir,
                cost=cost, budget=budget, state=state,
                validation=validation_summary,
                regression=regression_summary,
                sessions=sessions_count,
                exit_code=exit_code,
            )
            print_summary(summary, state_dir)
        if events is not None:
            events.close()

    if fatal_exc is not None:
        import traceback
        print()
        traceback.print_exception(type(fatal_exc), fatal_exc, fatal_exc.__traceback__)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
