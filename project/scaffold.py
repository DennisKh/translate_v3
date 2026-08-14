"""Workspace scaffold: mix new + priv/ + Credo + .tool-versions.

Uses project.mix_env for toolchain discovery. Emits progress events for every
subprocess call so the user sees what's happening (a v2 pain point).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from project.config import Config
from project.mix_env import MixEnv, find_mix_env
from project.naming import sanitize_app_name as _sanitize_app_name


# Lightweight progress callback — scaffold runs BEFORE the EventStream is
# initialized (because EventStream creates the state dir inside target_root,
# and `mix new` requires target_root be empty). So scaffold uses a simple
# print-based callback with the same "info"/"warn" verbs so the caller can
# swap in an EventStream-backed version later if needed.
ProgressFn = "Callable[[str, str], None]"


@dataclass(frozen=True)
class ScaffoldResult:
    app_name: str          # as `mix new` created it (or as we pinned it)
    module_name: str       # ditto
    target_root: Path
    mix_env: MixEnv


def _default_progress(level: str, message: str) -> None:
    """Fallback progress printer — timestamp + short prefix, mirrors EventStream format."""
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stdout.write(f"[{ts}] {level} {message}\n")
    sys.stdout.flush()


def _run_mix(progress: Callable[[str, str], None], mix_env: MixEnv, args: list[str],
             *, cwd: Path | None = None, timeout: int = 120, label: str | None = None,
             check: bool = False) -> subprocess.CompletedProcess:
    """Run a mix subprocess with progress emission around it."""
    tag = label or " ".join(args[:2])
    progress("info", f"mix {tag}: running...")
    start = time.monotonic()
    result = subprocess.run(
        [mix_env.mix_path, *args],
        cwd=cwd, env=mix_env.env,
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = int(time.monotonic() - start)
    ok = result.returncode == 0
    progress("info" if ok else "warn",
             f"mix {tag}: {'ok' if ok else f'failed (exit {result.returncode})'} in {elapsed}s")
    if not ok:
        stderr_tail = (result.stderr or "").strip().splitlines()[-6:]
        for line in stderr_tail:
            progress("warn", f"  mix {tag} stderr: {line}")
    if check and not ok:
        raise RuntimeError(f"mix {tag} failed: {(result.stderr or '').strip()}")
    return result


def _inject_deps(mix_exs_path: Path, runtime: dict[str, str], dev: dict[str, str]) -> None:
    """Rewrite `defp deps` to include runtime + dev deps."""
    content = mix_exs_path.read_text()

    dep_lines = []
    for name, ver in runtime.items():
        dep_lines.append(f'      {{:{name}, "{ver}"}}')
    for name, ver in dev.items():
        dep_lines.append(f'      {{:{name}, "{ver}", only: [:dev, :test], runtime: false}}')

    new_block = "defp deps do\n    [\n" + ",\n".join(dep_lines) + "\n    ]\n  end"

    updated, n = re.subn(
        r"defp deps do\s*\[.*?\]\s*end",
        new_block,
        content,
        count=1,
        flags=re.DOTALL,
    )
    if n == 0:
        raise RuntimeError("could not find `defp deps do ... end` block in mix.exs")
    mix_exs_path.write_text(updated)


def _read_mix_project_names(mix_exs_path: Path) -> tuple[str, str]:
    """Parse app_name and module_name from a generated mix.exs."""
    content = mix_exs_path.read_text()
    app_match = re.search(r"app:\s*:(\w+)", content)
    mod_match = re.search(r"defmodule\s+([\w.]+)\.MixProject", content)
    if not app_match or not mod_match:
        raise RuntimeError(f"could not parse app/module from {mix_exs_path}")
    return app_match.group(1), mod_match.group(1)


def _copy_resources(source_root: Path, resources_glob: str, target_priv: Path,
                    progress: Callable[[str, str], None]) -> list[str]:
    """Copy classpath resources into priv/. Returns list of copied filenames."""
    target_priv.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src_file in source_root.glob(resources_glob):
        if src_file.is_file() and not src_file.name.startswith("."):
            dst = target_priv / src_file.name
            shutil.copy(src_file, dst)
            copied.append(src_file.name)
    if copied:
        progress("info",
                 f"priv/: copied {len(copied)} resource file(s): {', '.join(copied[:5])}"
                 f"{'…' if len(copied) > 5 else ''}")
    return copied


_GITIGNORE_STATE_MARKER = "# translate_v3 per-run state (agent state, events, cost report, outcome)"
_GITIGNORE_STATE_ENTRIES = (
    _GITIGNORE_STATE_MARKER,
    ".translate_v3_state/",
)


def _ensure_gitignore_entries(target_root: Path,
                              progress: Callable[[str, str], None]) -> bool:
    """Append `.translate_v3_state/` to the target's .gitignore if missing.

    Idempotent — checks for existing entry before appending. Silent no-op if
    the target has no .gitignore (e.g. non-git target). Prevents accidentally
    committing per-run agent state (events.jsonl can be large; outcome.json
    and cost_report.json are transient).
    """
    gitignore = target_root / ".gitignore"
    if not gitignore.exists():
        return False
    current = gitignore.read_text()
    if ".translate_v3_state" in current:
        return False
    suffix = "" if current.endswith("\n") else "\n"
    addition = suffix + "\n" + "\n".join(_GITIGNORE_STATE_ENTRIES) + "\n"
    gitignore.write_text(current + addition)
    progress("info", "scaffold: added .translate_v3_state/ to .gitignore")
    return True


def _copy_metadata_files(source_root: Path, target_root: Path, patterns: list[str],
                        progress: Callable[[str, str], None]) -> list[str]:
    """Copy top-level metadata files (LICENSE, NOTICE, etc.) verbatim.

    Legal compliance: Apache-2.0 and most permissive licenses require the
    LICENSE + NOTICE files to travel with any redistribution. `mix new` does
    NOT generate them, so we must carry them over from the source project.

    Idempotent — files already present in target are skipped (so --resume
    doesn't clobber a hand-edited LICENSE).
    """
    copied: list[str] = []
    for pattern in patterns:
        for src_file in source_root.glob(pattern):
            if not src_file.is_file():
                continue
            dst = target_root / src_file.name
            if dst.exists():
                continue
            shutil.copy(src_file, dst)
            copied.append(src_file.name)
    if copied:
        progress("info",
                 f"metadata: copied {len(copied)} file(s): {', '.join(copied)}")
    return copied


# _sanitize_app_name imported from project.naming


def scaffold(cfg: Config,
             progress: Callable[[str, str], None] | None = None) -> ScaffoldResult:
    """Scaffold the target Elixir project.

    Uses a lightweight progress callback (level, message) instead of the full
    EventStream because scaffold runs BEFORE the state dir exists — EventStream's
    creation of that dir would interfere with `mix new` (which refuses non-empty
    target dirs).

    Steps:
    1. Find Elixir/Mix (asdf-aware, prefer 1.17+)
    2. `mix new TARGET_ROOT` (skip if already populated with real content)
    3. Remove placeholder module + placeholder test
    4. Inject runtime + dev deps into mix.exs
    5. Read back app_name / module_name from generated mix.exs
    6. Write .tool-versions if asdf-pinned
    7. Copy resources into priv/
    8. `mix deps.get`
    9. `mix credo gen.config` to seed .credo.exs
    """
    p = progress or _default_progress

    p("info", "scaffold: discovering Elixir toolchain...")
    mix_env = find_mix_env()
    if mix_env is None:
        raise RuntimeError("no `mix` binary found (checked asdf, PATH, brew)")
    pin = mix_env.env.get("ASDF_ELIXIR_VERSION")
    pin_note = f" (asdf pin: {pin})" if pin else ""
    p("info", f"scaffold: mix {mix_env.version} at {mix_env.mix_path}{pin_note}")

    target_root = cfg.target.root
    if target_root is None:
        raise ValueError("target.root is required")
    target_root.parent.mkdir(parents=True, exist_ok=True)

    def _has_non_hidden_content(root: Path) -> bool:
        return root.exists() and any(
            entry for entry in root.iterdir() if not entry.name.startswith(".")
        )

    already_populated = _has_non_hidden_content(target_root)

    if already_populated:
        p("info", f"scaffold: {target_root} already populated — reusing")
    else:
        effective_app = cfg.target.app_name or _sanitize_app_name(target_root.name)
        mix_new_args = ["new", str(target_root), "--app", effective_app]
        if cfg.target.module_name:
            mix_new_args += ["--module", cfg.target.module_name]

        _run_mix(p, mix_env, mix_new_args, timeout=30,
                 label=f"new {effective_app}", check=True)

        for placeholder in [
            target_root / "lib" / f"{effective_app}.ex",
            target_root / "test" / f"{effective_app}_test.exs",
        ]:
            if placeholder.exists():
                placeholder.unlink()

    # Run on both fresh and resume paths — an existing project scaffolded
    # before this feature existed still needs the entry added.
    _ensure_gitignore_entries(target_root, progress=p)

    app_name, module_name = _read_mix_project_names(target_root / "mix.exs")
    p("info", f"scaffold: project app_name={app_name} module={module_name}")

    _inject_deps(target_root / "mix.exs", cfg.target.mix_deps, cfg.target.dev_deps)
    p("info", f"scaffold: injected {len(cfg.target.mix_deps)} runtime dep(s) "
              f"+ {len(cfg.target.dev_deps)} dev dep(s)")

    if pin:
        lines = []
        if erl := mix_env.env.get("ASDF_ERLANG_VERSION"):
            lines.append(f"erlang {erl}")
        lines.append(f"elixir {pin}")
        (target_root / ".tool-versions").write_text("\n".join(lines) + "\n")
        p("info", f"scaffold: pinned .tool-versions ({', '.join(lines)})")

    if cfg.source.root:
        _copy_resources(cfg.source.root, cfg.source.resources_glob,
                        target_root / "priv", progress=p)
        if cfg.target.copy_files:
            _copy_metadata_files(cfg.source.root, target_root,
                                 cfg.target.copy_files, progress=p)

    deps_result = _run_mix(p, mix_env, ["deps.get"], cwd=target_root, timeout=180,
                           label="deps.get")
    if deps_result.returncode != 0:
        p("warn", "scaffold: `mix deps.get` failed — Credo may not be available")

    credo_result = _run_mix(p, mix_env, ["credo", "gen.config"], cwd=target_root,
                            timeout=30, label="credo gen.config")
    if credo_result.returncode != 0:
        (target_root / ".credo.exs").write_text(
            '%{configs: [%{name: "default", strict: true, checks: []}]}\n'
        )
        p("warn", "scaffold: `mix credo gen.config` failed — wrote minimal .credo.exs")

    p("info", f"scaffold: ready at {target_root}")

    return ScaffoldResult(
        app_name=app_name,
        module_name=module_name,
        target_root=target_root,
        mix_env=mix_env,
    )
