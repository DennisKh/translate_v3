#!/usr/bin/env python3
"""Regression scorer: score a candidate Elixir project against gold reference.

Usage:
    python3 tests/regression/run.py \\
        --candidate /path/to/candidate/lib \\
        [--json summary.json]

For each gold module in tests/gold/, we:
  1. Read the gold source and gold test.
  2. Find the candidate's counterpart .ex file (by module name).
  3. Compare public function surface (name+arity, allowing common Elixir
     naming transformations like `get_x` → `x`).
  4. Copy candidate + gold test into a scratch mix project, run `mix test`,
     count pass/fail.
  5. Run `mix format --check-formatted` and `mix credo --strict` (if
     installed) as additional signals.

Emits a scorecard on stdout and optionally a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Make the language module importable when run from repo root or elsewhere
_HERE = Path(__file__).resolve().parent
_V3_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_V3_ROOT))

from language.elixir import ModuleShape, describe_file  # noqa: E402
from project.mix_env import MixEnv, find_mix_env  # noqa: E402


GOLD_DIR = _V3_ROOT / "tests" / "gold"


@dataclass
class ModuleScore:
    module: str
    gold_file: str
    candidate_file: str | None
    symbol_coverage_pct: int = 0
    matched_symbols: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    extra_symbols: list[str] = field(default_factory=list)
    test_pass: int = 0
    test_fail: int = 0
    test_total: int = 0
    compile_ok: bool = False
    format_ok: bool | None = None
    credo_ok: bool | None = None
    notes: list[str] = field(default_factory=list)


def _candidate_path_for(module: str, candidate_lib: Path) -> Path | None:
    """`JodaMoney.Format.MoneyFormatException` → `.../format/money_format_exception.ex`"""
    parts = module.split(".")
    # Drop the top namespace (JodaMoney)
    tail = parts[1:] if len(parts) > 1 else parts
    snake = [_camel_to_snake(p) for p in tail]
    candidate = candidate_lib.joinpath(*snake[:-1]) / f"{snake[-1]}.ex"
    return candidate if candidate.exists() else None


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


# Common Elixir naming transformations applied to Java getters
_GETTER_ALIASES = [
    (re.compile(r"^get_(.+)"), r"\1"),
    (re.compile(r"^is_(.+)"), r"\1?"),
    (re.compile(r"^has_(.+)"), r"\1?"),
]


def _sym_aliases(name: str, arity: int) -> set[str]:
    """Return the set of acceptable name/arity strings for a gold symbol.

    Accounts for Java-style names that a translator might drop the `get_`
    prefix on.
    """
    aliases = {f"{name}/{arity}"}
    for pat, sub in _GETTER_ALIASES:
        if m := pat.match(name):
            aliases.add(f"{pat.sub(sub, name)}/{arity}")
    return aliases


def _symbol_coverage(gold: ModuleShape, candidate: ModuleShape) -> tuple[int, list[str], list[str], list[str]]:
    cand_syms = {f"{s.name}/{s.arity}" for s in candidate.public_functions}

    matched: list[str] = []
    missing: list[str] = []
    for g in gold.public_functions:
        aliases = _sym_aliases(g.name, g.arity)
        if aliases & cand_syms:
            matched.append(f"{g.name}/{g.arity}")
        else:
            missing.append(f"{g.name}/{g.arity}")

    gold_syms_flat = {f"{s.name}/{s.arity}" for s in gold.public_functions}
    gold_alias_flat: set[str] = set()
    for g in gold.public_functions:
        gold_alias_flat |= _sym_aliases(g.name, g.arity)
    extra = sorted(cand_syms - gold_alias_flat - gold_syms_flat)

    total = len(gold.public_functions)
    pct = int(round(100 * len(matched) / total)) if total else 100
    return pct, matched, missing, extra


# ---------------------------------------------------------------------------
# Test parity: run gold tests against candidate module in a scratch project
# ---------------------------------------------------------------------------

def _make_scratch_project(mix_env: MixEnv, candidate_files: list[Path],
                          test_files: list[Path]) -> Path:
    """Create a scratch mix project containing:
       - the candidate lib files
       - the gold test files
    Runs `mix deps.get`. Returns the project path.

    The scratch project's module namespace must match the candidates
    (JodaMoney.*), so we set --app joda_money --module JodaMoney.
    """
    scratch = Path(tempfile.mkdtemp(prefix="v3-regression-"))
    proj = scratch / "joda_money"

    subprocess.run(
        [mix_env.mix_path, "new", str(proj), "--app", "joda_money", "--module", "JodaMoney"],
        capture_output=True, text=True, env=mix_env.env, check=True,
    )

    # Add :decimal dep (all candidates likely need it)
    mix_exs = proj / "mix.exs"
    content = mix_exs.read_text()
    content = re.sub(
        r"defp deps do\s*\[.*?\]\s*end",
        'defp deps do\n    [\n      {:decimal, "~> 2.0"}\n    ]\n  end',
        content, count=1, flags=re.DOTALL,
    )
    mix_exs.write_text(content)

    # Remove placeholder module + test
    for placeholder in [proj / "lib" / "joda_money.ex", proj / "test" / "joda_money_test.exs"]:
        if placeholder.exists():
            placeholder.unlink()

    # Copy candidate lib files preserving directory structure under lib/
    for src in candidate_files:
        # Reconstruct target path from module namespace
        shape = describe_file(src)
        if shape is None:
            continue
        parts = shape.module_name.split(".")[1:]  # drop JodaMoney prefix
        snake = [_camel_to_snake(p) for p in parts]
        dst = proj / "lib" / "joda_money"
        dst.mkdir(parents=True, exist_ok=True)
        target_file = dst.joinpath(*snake[:-1]) / f"{snake[-1]}.ex"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target_file)

    # Copy gold test files under test/joda_money/ (namespace-aware)
    for tf in test_files:
        # Read the module name from the test file
        m = re.search(r"defmodule\s+([\w.]+)Test\s+do", tf.read_text())
        if not m:
            continue
        parts = m.group(1).split(".")[1:]
        snake = [_camel_to_snake(p) for p in parts]
        dst = proj / "test" / "joda_money"
        dst.mkdir(parents=True, exist_ok=True)
        target = dst.joinpath(*snake[:-1]) / f"{snake[-1]}_test.exs"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tf, target)

    # Copy priv/ fixtures from gold (if present). Some gold modules
    # (e.g. CurrencyUnit) load CSV data via `@external_resource` at
    # compile time — the compile fails without these files. Optional
    # for gold modules that don't need it.
    gold_priv = GOLD_DIR / "priv"
    if gold_priv.is_dir():
        dst_priv = proj / "priv"
        dst_priv.mkdir(parents=True, exist_ok=True)
        for src in gold_priv.iterdir():
            if src.is_file():
                shutil.copy(src, dst_priv / src.name)

    # Fetch deps
    subprocess.run(
        [mix_env.mix_path, "deps.get"],
        cwd=proj, env=mix_env.env,
        capture_output=True, text=True, timeout=120,
    )
    return proj


def _run_mix_test(mix_env: MixEnv, proj: Path) -> tuple[int, int, int, bool, str]:
    """Return (pass, fail, total, compile_ok, output_tail)."""
    result = subprocess.run(
        [mix_env.mix_path, "test", "--no-color"],
        cwd=proj, env=mix_env.env,
        capture_output=True, text=True, timeout=300,
    )
    output = (result.stdout or "") + (result.stderr or "")
    compile_ok = "== Compilation error" not in output

    # Look for the ExUnit summary line: "N tests, M failures"
    m = re.search(r"(\d+)\s+(?:test|tests?),\s+(\d+)\s+failures?", output)
    if m:
        total = int(m.group(1))
        fail = int(m.group(2))
        return (total - fail), fail, total, compile_ok, output[-2000:]
    return 0, 0, 0, compile_ok, output[-2000:]


def _run_mix_check(mix_env: MixEnv, proj: Path, args: list[str],
                   timeout: int = 60) -> bool:
    result = subprocess.run(
        [mix_env.mix_path, *args],
        cwd=proj, env=mix_env.env,
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _score_one(candidate_lib: Path, mix_env: MixEnv | None,
               gold_source: Path, gold_test: Path, gold_shape: ModuleShape) -> ModuleScore:
    """Score a single gold module. Extracted so `score()` can parallelize."""
    cand_path = _candidate_path_for(gold_shape.module_name, candidate_lib)
    s = ModuleScore(
        module=gold_shape.module_name,
        gold_file=str(gold_source.name),
        candidate_file=str(cand_path.relative_to(candidate_lib)) if cand_path else None,
    )

    if cand_path is None:
        s.notes.append("no candidate file found")
        return s

    cand_shape = describe_file(cand_path)
    if cand_shape is None:
        s.notes.append("candidate file could not be parsed")
        return s

    pct, matched, missing, extra = _symbol_coverage(gold_shape, cand_shape)
    s.symbol_coverage_pct = pct
    s.matched_symbols = matched
    s.missing_symbols = missing
    s.extra_symbols = extra

    if mix_env is None:
        s.notes.append("mix not found — skipped test parity")
        return s

    try:
        # Pass the ENTIRE candidate lib, not just this one file. Gold tests
        # often exercise cross-module behavior (e.g. `CurrencyUnit.of/1`
        # raises `IllegalCurrencyException` — both modules must be present).
        # `_candidate_path_for` above still resolves the specific file for
        # symbol coverage, but for `mix test` we need the whole graph.
        all_candidate_files = sorted(candidate_lib.rglob("*.ex"))
        proj = _make_scratch_project(mix_env, all_candidate_files, [gold_test])
        p, f, t, compile_ok, _tail = _run_mix_test(mix_env, proj)
        s.test_pass = p
        s.test_fail = f
        s.test_total = t
        s.compile_ok = compile_ok
        if compile_ok:
            s.format_ok = _run_mix_check(mix_env, proj, ["format", "--check-formatted"])
            if (proj / "deps" / "credo").exists():
                s.credo_ok = _run_mix_check(mix_env, proj, ["credo", "--strict"])
        shutil.rmtree(proj.parent, ignore_errors=True)
    except Exception as exc:
        s.notes.append(f"test run error: {exc}")

    return s


def score(candidate_lib: Path, workers: int = 3) -> list[ModuleScore]:
    """Score all gold modules against `candidate_lib`. Parallelizes across modules."""
    from concurrent.futures import ThreadPoolExecutor

    mix_env = find_mix_env()

    gold_pairs: list[tuple[Path, Path, ModuleShape]] = []
    for gold_source in sorted(GOLD_DIR.glob("*.ex")):
        gold_shape = describe_file(gold_source)
        if gold_shape is None:
            continue
        gold_test = gold_source.with_name(gold_source.stem + "_test.exs")
        if not gold_test.exists():
            continue
        gold_pairs.append((gold_source, gold_test, gold_shape))

    if workers <= 1 or len(gold_pairs) <= 1:
        return [_score_one(candidate_lib, mix_env, gs, gt, sh) for gs, gt, sh in gold_pairs]

    with ThreadPoolExecutor(max_workers=min(workers, len(gold_pairs))) as pool:
        futures = [pool.submit(_score_one, candidate_lib, mix_env, gs, gt, sh)
                   for gs, gt, sh in gold_pairs]
        return [f.result() for f in futures]


def print_report(scores: list[ModuleScore]) -> None:
    print()
    print("━" * 68)
    print("  REGRESSION SCORECARD")
    print("━" * 68)

    for s in scores:
        status = "✓" if s.compile_ok and s.test_fail == 0 else "✗"
        print(f"\n  {status}  {s.module}")
        if s.candidate_file:
            print(f"     candidate: {s.candidate_file}")
        else:
            print(f"     candidate: (not found)")
            continue

        print(f"     symbol coverage: {s.symbol_coverage_pct}%  "
              f"({len(s.matched_symbols)}/{len(s.matched_symbols) + len(s.missing_symbols)})")
        if s.missing_symbols:
            print(f"       missing: {', '.join(s.missing_symbols[:6])}"
                  f"{'…' if len(s.missing_symbols) > 6 else ''}")
        print(f"     compile: {'✓' if s.compile_ok else '✗'}")
        if s.test_total:
            pct = 100 * s.test_pass // s.test_total
            print(f"     tests:   {s.test_pass}/{s.test_total}  ({pct}%)")
        elif s.compile_ok:
            print(f"     tests:   no tests ran")
        if s.format_ok is not None:
            print(f"     format:  {'✓' if s.format_ok else '✗'}")
        if s.credo_ok is not None:
            print(f"     credo:   {'✓' if s.credo_ok else '✗'}")
        if s.notes:
            for note in s.notes:
                print(f"     note: {note}")

    # Aggregate
    total_syms = sum(len(s.matched_symbols) + len(s.missing_symbols) for s in scores)
    matched_syms = sum(len(s.matched_symbols) for s in scores)
    total_tests = sum(s.test_total for s in scores)
    passed_tests = sum(s.test_pass for s in scores)

    print()
    print("━" * 68)
    print("  AGGREGATE")
    print("━" * 68)
    print(f"  Modules:         {len(scores)}")
    print(f"  Symbol coverage: {matched_syms}/{total_syms}  "
          f"({100 * matched_syms // total_syms if total_syms else 100}%)")
    print(f"  Compile clean:   {sum(1 for s in scores if s.compile_ok)}/{len(scores)}")
    print(f"  Test parity:     {passed_tests}/{total_tests}  "
          f"({100 * passed_tests // total_tests if total_tests else 100}%)")
    format_ran = [s for s in scores if s.format_ok is not None]
    if format_ran:
        print(f"  Format clean:    {sum(1 for s in format_ran if s.format_ok)}/{len(format_ran)}")
    credo_ran = [s for s in scores if s.credo_ok is not None]
    if credo_ran:
        print(f"  Credo clean:     {sum(1 for s in credo_ran if s.credo_ok)}/{len(credo_ran)}")
    print("━" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Path to the candidate `lib/` directory (e.g. /path/to/proj/lib/joda_money)")
    parser.add_argument("--json", type=Path, help="Write scorecard to this JSON file too")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel mix workers (default: 3; set to 1 to disable parallelism)")
    args = parser.parse_args()

    if not args.candidate.exists():
        print(f"error: candidate path does not exist: {args.candidate}", file=sys.stderr)
        sys.exit(2)

    scores = score(args.candidate, workers=args.workers)
    print_report(scores)

    if args.json:
        args.json.write_text(json.dumps([asdict(s) for s in scores], indent=2))
        print(f"\n  Written: {args.json}")


if __name__ == "__main__":
    main()
