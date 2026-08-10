"""Mix subprocess operations — thin wrappers over `mix compile / format / credo / test`.

Two consumers: the verify tools (`mix_compile`, `mix_format`, `mix_credo`,
`validation_status`) and the write tools (which run a syntax check).

Every call captures stdout+stderr and detects "undefined or private" warnings,
which `mix compile` silently exits 0 on — v2's biggest quality gap.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from project.mix_env import MixEnv


# Regex catching "undefined or private" warnings that mix exits 0 on.
# We escalate these to errors internally.
_BOMB_WARNING_RE = re.compile(
    r"warning:\s+([\w.]+\.[\w?!]+/\d+)\s+is\s+undefined(?:\s+or\s+private)?",
)

# Extract file paths from mix compile / test output for focus selection.
_FILE_MENTION_RE = re.compile(r"((?:lib|test)/[a-z0-9_/]+\.exs?)")


@dataclass
class MixResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    duration_s: float = 0.0
    bomb_warnings: list[str] = field(default_factory=list)   # e.g. ["Mod.fn/1"]
    file_mentions: list[str] = field(default_factory=list)   # e.g. ["lib/foo.ex"]

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def output_tail(self) -> str:
        """Last ~4000 chars for feeding back to the model."""
        combined = self.output
        return combined if len(combined) <= 4000 else combined[-4000:]


def _run(mix_env: MixEnv, args: list[str], cwd: Path,
         *, timeout: int = 300) -> MixResult:
    """Low-level subprocess runner. Populates bomb_warnings + file_mentions.

    On timeout, kills the child process and reaps it so it doesn't linger.
    """
    import time
    start = time.monotonic()
    # Use Popen so we can kill on timeout — subprocess.run leaves orphans
    # in a race that has been fixed in newer Python but is still worth guarding
    # explicitly (mix compile in particular can spawn OS processes we want dead).
    with subprocess.Popen(
        [mix_env.mix_path, *args],
        cwd=cwd, env=mix_env.env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "TIMEOUT (child failed to reap)"
            return MixResult(
                ok=False, stdout=stdout or "",
                stderr=(stderr or "") + f"\nTIMEOUT after {timeout}s",
                returncode=-1, duration_s=time.monotonic() - start,
            )

    combined = (stdout or "") + (stderr or "")
    bombs = sorted({m.group(1) for m in _BOMB_WARNING_RE.finditer(combined)})
    mentions: list[str] = []
    seen: set[str] = set()
    for m in _FILE_MENTION_RE.finditer(combined):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            mentions.append(m.group(1))

    return MixResult(
        ok=proc.returncode == 0,
        stdout=stdout or "",
        stderr=stderr or "",
        returncode=proc.returncode,
        duration_s=time.monotonic() - start,
        bomb_warnings=bombs,
        file_mentions=mentions,
    )


# ---------------------------------------------------------------------------
# Public one-shot operations
# ---------------------------------------------------------------------------

def mix_compile(mix_env: MixEnv, project_root: Path,
                *, warnings_as_errors: bool = True) -> MixResult:
    """`mix compile` — with the important quality-gate flag.

    `warnings_as_errors=True` turns "undefined or private" and friends into
    non-zero exits. `_run` also independently scans output for bomb warnings
    so callers see them structured on `.bomb_warnings`.
    """
    args = ["compile"]
    if warnings_as_errors:
        args.append("--warnings-as-errors")
    return _run(mix_env, args, project_root, timeout=180)


def mix_format(mix_env: MixEnv, project_root: Path,
               *, check_only: bool = False) -> MixResult:
    """`mix format` — check-only mode returns non-zero if any file would change."""
    args = ["format"]
    if check_only:
        args.append("--check-formatted")
    return _run(mix_env, args, project_root, timeout=60)


def mix_credo(mix_env: MixEnv, project_root: Path,
              *, strict: bool = True) -> MixResult:
    """`mix credo` — style/complexity checks."""
    args = ["credo"]
    if strict:
        args.append("--strict")
    return _run(mix_env, args, project_root, timeout=120)


def mix_test(mix_env: MixEnv, project_root: Path,
             *, only: str | None = None, max_failures: int = 10) -> MixResult:
    """`mix test` — runs the full suite or a specific test file."""
    args = ["test", "--max-failures", str(max_failures)]
    if only:
        args.append(only)
    return _run(mix_env, args, project_root, timeout=300)


# ---------------------------------------------------------------------------
# Elixir syntax check — parse-only, no compile
# ---------------------------------------------------------------------------

def check_syntax(mix_env: MixEnv, elixir_path: Path, timeout: int = 20) -> MixResult:
    """`elixir -e 'Code.string_to_quoted!(File.read!(...))'` — parse-only check.

    Uses the same env as mix (asdf pins). Returns MixResult so callers see
    the same shape as compile/format/credo.
    """
    import time
    # Elixir binary lives next to mix in the same asdf shim dir
    elixir_bin = str(Path(mix_env.mix_path).parent / "elixir")
    if not Path(elixir_bin).exists():
        # fall back to PATH lookup
        elixir_bin = "elixir"

    check_expr = (
        'try do File.read!(~S"'
        + str(elixir_path)
        + '") |> Code.string_to_quoted!(); :ok '
          'rescue e -> IO.puts(:stderr, Exception.message(e)); System.halt(1) end'
    )
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [elixir_bin, "-e", check_expr],
            capture_output=True, text=True, timeout=timeout, env=mix_env.env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return MixResult(
            ok=False, stdout="", stderr=str(exc),
            returncode=-1, duration_s=time.monotonic() - start,
        )

    return MixResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
        duration_s=time.monotonic() - start,
    )
