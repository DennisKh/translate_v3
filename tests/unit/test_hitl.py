"""HITL prompt for wall-clock cap — mocked stdin + isatty."""

from __future__ import annotations

import io

import pytest

from agent.hitl import (
    WallCapAction,
    _fmt_duration,
    prompt_wall_cap_action,
)


def _prompt(input_text: str, *, isatty: bool = True) -> tuple[WallCapAction, str]:
    stdin = io.StringIO(input_text)
    stdout = io.StringIO()
    action = prompt_wall_cap_action(
        elapsed_seconds=3600,
        cap_seconds=3600,
        files_done=14,
        files_total=26,
        cost_usd=5.81,
        timeout_seconds=60,
        stdin=stdin,
        stdout=stdout,
        isatty=lambda: isatty,
    )
    return action, stdout.getvalue()


def test_extend_choice_returns_extend():
    action, out = _prompt("e\n")
    assert action == WallCapAction.EXTEND
    assert "Wall-clock cap reached" in out


def test_continue_choice_returns_continue():
    action, _ = _prompt("c\n")
    assert action == WallCapAction.CONTINUE


def test_abort_choice_returns_abort():
    action, _ = _prompt("a\n")
    assert action == WallCapAction.ABORT


def test_empty_input_defaults_to_abort():
    action, _ = _prompt("\n")
    assert action == WallCapAction.ABORT


def test_eof_returns_abort():
    # No trailing newline and empty string → readline returns ""
    action, _ = _prompt("")
    assert action == WallCapAction.ABORT


def test_uppercase_choices_are_accepted():
    for txt, expected in [("E\n", WallCapAction.EXTEND),
                          ("C\n", WallCapAction.CONTINUE),
                          ("A\n", WallCapAction.ABORT)]:
        action, _ = _prompt(txt)
        assert action == expected


def test_invalid_then_valid_extends():
    action, out = _prompt("nope\ne\n")
    assert action == WallCapAction.EXTEND
    assert "Invalid choice" in out


def test_two_invalid_defaults_to_abort():
    action, out = _prompt("nope\nnope2\n")
    assert action == WallCapAction.ABORT
    assert "Invalid choice" in out


def test_non_tty_returns_abort_without_reading_stdin():
    stdin = io.StringIO("e\n")  # would return EXTEND if consulted
    stdout = io.StringIO()
    action = prompt_wall_cap_action(
        elapsed_seconds=3600, cap_seconds=3600,
        files_done=0, files_total=1, cost_usd=0.0,
        stdin=stdin, stdout=stdout, isatty=lambda: False,
    )
    assert action == WallCapAction.ABORT
    # Prompt text should NOT have been printed on non-TTY.
    assert stdout.getvalue() == ""
    # And stdin should not have been read.
    assert stdin.read() == "e\n"


def test_prompt_snapshot_contains_key_metrics():
    _, out = _prompt("a\n")
    assert "1h 0m 0s" in out            # elapsed and cap both 3600s
    assert "14 of 26 files" in out
    assert "$5.81" in out
    assert "[e]" in out and "[c]" in out and "[a]" in out


def test_timeout_returns_abort(tmp_path):
    """Real pipe with no data: read_line_with_timeout returns None → abort."""
    import os

    r_fd, _w_fd = os.pipe()
    # Wrap the read end in a text-mode stream that exposes fileno().
    r = os.fdopen(r_fd, "r")
    stdout = io.StringIO()
    try:
        action = prompt_wall_cap_action(
            elapsed_seconds=10, cap_seconds=10,
            files_done=0, files_total=1, cost_usd=0.0,
            timeout_seconds=0,  # instant timeout
            stdin=r, stdout=stdout, isatty=lambda: True,
        )
    finally:
        r.close()
        os.close(_w_fd)
    assert action == WallCapAction.ABORT


@pytest.mark.parametrize("secs,expected", [
    (0, "0s"),
    (45, "45s"),
    (60, "1m 0s"),
    (123, "2m 3s"),
    (3600, "1h 0m 0s"),
    (3661, "1h 1m 1s"),
    (7325, "2h 2m 5s"),
])
def test_fmt_duration(secs, expected):
    assert _fmt_duration(secs) == expected


def test_fmt_duration_clamps_negative_to_zero():
    assert _fmt_duration(-5) == "0s"
