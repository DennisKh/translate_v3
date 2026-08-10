"""Elixir/Mix toolchain discovery — asdf-aware, prefers 1.17+/OTP 26+.

Shared between the scaffold (project/scaffold.py) and the regression harness
(tests/regression/run.py) so both use identical resolution logic.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MixEnv:
    """Resolved Elixir/Erlang toolchain."""
    mix_path: str
    version: str          # "1.17.3"
    env: dict[str, str]   # env vars to pass to every mix invocation


def _read_mix_version(mix_path: str, env: dict[str, str]) -> str | None:
    try:
        result = subprocess.run(
            [mix_path, "--version"],
            capture_output=True, text=True, timeout=15, env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    m = re.search(r"Mix\s+(\d+\.\d+\.\d+)", result.stdout)
    return m.group(1) if m else None


def _best_asdf_elixir(min_version: tuple[int, int, int] = (1, 17, 0)) -> str | None:
    installs = Path.home() / ".asdf" / "installs" / "elixir"
    if not installs.exists():
        return None
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for entry in installs.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", entry.name)
        if not m:
            continue
        v = (int(m[1]), int(m[2]), int(m[3]))
        if v >= min_version:
            candidates.append((v, entry.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_otp_major(elixir_version: str) -> int | None:
    m = re.search(r"-otp-(\d+)", elixir_version)
    return int(m[1]) if m else None


def _best_asdf_erlang(otp_major: int) -> str | None:
    installs = Path.home() / ".asdf" / "installs" / "erlang"
    if not installs.exists():
        return None
    candidates: list[tuple[tuple[int, ...], str]] = []
    for entry in installs.iterdir():
        if not entry.is_dir() or not entry.name.startswith(f"{otp_major}."):
            continue
        try:
            nums = tuple(int(p) for p in entry.name.split("."))
            candidates.append((nums, entry.name))
        except ValueError:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_mix_env() -> MixEnv | None:
    """Resolve the best available Elixir/Erlang toolchain (>= 1.17).

    Search order:
      1. asdf shim (`~/.asdf/shims/mix`) with newest installed 1.17+ + matching OTP
      2. `mix` in PATH
      3. Homebrew (`brew --prefix elixir`)
    """
    base_env = os.environ.copy()

    asdf_shim = Path.home() / ".asdf" / "shims" / "mix"
    if asdf_shim.exists():
        best_elixir = _best_asdf_elixir()
        if best_elixir:
            env = base_env | {"ASDF_ELIXIR_VERSION": best_elixir}
            if (otp := _extract_otp_major(best_elixir)) is not None:
                if erl := _best_asdf_erlang(otp):
                    env["ASDF_ERLANG_VERSION"] = erl
            if v := _read_mix_version(str(asdf_shim), env):
                return MixEnv(str(asdf_shim), v, env)

    if mix := shutil.which("mix"):
        if v := _read_mix_version(mix, base_env):
            return MixEnv(mix, v, base_env)

    try:
        result = subprocess.run(
            ["brew", "--prefix", "elixir"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            candidate = Path(result.stdout.strip()) / "bin" / "mix"
            if candidate.exists():
                if v := _read_mix_version(str(candidate), base_env):
                    return MixEnv(str(candidate), v, base_env)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None
