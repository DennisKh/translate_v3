"""agent/mix_ops.py — subprocess wrappers over mix compile/format/credo.

Integration test: creates a tiny throwaway mix project, runs each operation,
verifies expected behavior. Skipped if no Elixir toolchain is available.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent.mix_ops import (
    _BOMB_WARNING_RE,
    check_syntax,
    mix_compile,
    mix_credo,
    mix_format,
)
from project.mix_env import find_mix_env


@pytest.fixture(scope="module")
def mix_env():
    env = find_mix_env()
    if env is None:
        pytest.skip("no Elixir/Mix toolchain available")
    return env


@pytest.fixture()
def scratch_project(mix_env, tmp_path):
    """A minimal freshly-scaffolded mix project."""
    proj = tmp_path / "scratchapp"
    subprocess.run(
        [mix_env.mix_path, "new", str(proj), "--app", "scratchapp", "--module", "Scratchapp"],
        capture_output=True, text=True, env=mix_env.env, check=True,
    )
    return proj


def test_bomb_warning_regex_catches_undefined_or_private():
    # These are the exact wording variants mix emits
    warnings = [
        "warning: Foo.Bar.baz/1 is undefined or private",
        "warning: SomeMod.func/3 is undefined",
        "warning: A.B.c/0 is undefined or private. Did you mean:",
    ]
    for w in warnings:
        assert _BOMB_WARNING_RE.search(w), w


def test_bomb_warning_regex_ignores_benign():
    benign = [
        "warning: variable \"x\" is unused",
        "warning: unused import",
    ]
    for w in benign:
        assert _BOMB_WARNING_RE.search(w) is None, w


def test_check_syntax_ok(mix_env, tmp_path):
    good = tmp_path / "good.ex"
    good.write_text("defmodule Good do\n  def x, do: 1\nend\n")
    result = check_syntax(mix_env, good)
    assert result.ok, result.stderr


def test_check_syntax_fail(mix_env, tmp_path):
    bad = tmp_path / "bad.ex"
    bad.write_text("defmodule Bad do\n  def x, do: 1\n# missing end\n")
    result = check_syntax(mix_env, bad)
    assert not result.ok
    assert "end" in result.stderr.lower() or "unexpected" in result.stderr.lower()


def test_mix_compile_clean_project(mix_env, scratch_project):
    """A fresh `mix new` project compiles clean."""
    result = mix_compile(mix_env, scratch_project)
    assert result.ok, result.output_tail
    assert result.bomb_warnings == []


def test_mix_compile_catches_bomb_warning(mix_env, scratch_project):
    """`mix compile --warnings-as-errors` should escalate 'undefined or private'."""
    (scratch_project / "lib" / "with_bomb.ex").write_text(
        "defmodule WithBomb do\n"
        "  def call, do: NonExistent.gone/0\n"
        "end\n"
    )
    result = mix_compile(mix_env, scratch_project)
    # With warnings-as-errors, this should fail
    assert not result.ok
    # And we should have captured the bomb signature
    assert any("NonExistent.gone" in b for b in result.bomb_warnings), result.bomb_warnings


def test_mix_format_check_only_on_clean(mix_env, scratch_project):
    """Newly-scaffolded projects are already formatted."""
    result = mix_format(mix_env, scratch_project, check_only=True)
    assert result.ok


def test_mix_format_detects_deviation(mix_env, scratch_project):
    """Introduce bad formatting; check_only should return non-zero."""
    (scratch_project / "lib" / "ugly.ex").write_text(
        "defmodule Ugly do\ndef  x,do:  1\nend\n"  # extra spaces
    )
    result = mix_format(mix_env, scratch_project, check_only=True)
    assert not result.ok


def test_mix_format_rewrites(mix_env, scratch_project):
    """Rewrite mode should fix the file."""
    ugly = scratch_project / "lib" / "ugly.ex"
    ugly.write_text("defmodule Ugly do\ndef  x,do:  1\nend\n")
    result = mix_format(mix_env, scratch_project, check_only=False)
    assert result.ok
    # Now check should pass
    check = mix_format(mix_env, scratch_project, check_only=True)
    assert check.ok
