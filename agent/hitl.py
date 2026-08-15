"""Human-in-the-Loop prompts for wall-clock cap.

When the wall-clock cap fires mid-run, the user is asked whether to
extend by a fixed 30-minute increment, remove the cap for the rest of
the run, or abort and save state. Defaults to abort on empty input,
EOF, and timeout so a walked-away user never keeps spending silently.

I/O is isolated in this module so tests can mock stdin + isatty without
touching the translator.
"""

from __future__ import annotations

import selectors
import sys
from enum import Enum
from typing import Callable, TextIO


class WallCapAction(str, Enum):
    EXTEND = "extend"
    CONTINUE = "continue"
    ABORT = "abort"


def _fmt_duration(seconds: int) -> str:
    """Human-friendly duration: '1h 0m 15s' / '45s' / '2m 3s'."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m}m {s}s"


def _read_line_with_timeout(stream: TextIO, timeout_seconds: int) -> str | None:
    """Read one line from `stream`, waiting at most `timeout_seconds`.

    Returns the line (stripped of trailing newline) or None on timeout /
    EOF. Uses `selectors` on the file descriptor so tests can pass a
    real file (StringIO for immediate-return cases; a pipe for timeout
    cases). Falls back to a plain blocking `readline()` when the stream
    has no fileno (in-memory objects), which is fine for tests but
    means real terminals must expose a fileno.
    """
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        # No fileno — fall back to plain blocking read. Tests using
        # StringIO exercise this path.
        line = stream.readline()
        if not line:
            return None
        return line.rstrip("\n")

    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    try:
        events = sel.select(timeout=timeout_seconds)
    finally:
        sel.close()
    if not events:
        return None
    line = stream.readline()
    if not line:
        return None
    return line.rstrip("\n")


_MENU = (
    "\n"
    "─────────────────────────────────────────────────────────────\n"
    "  Wall-clock cap reached ({elapsed} / {cap}).\n"
    "  Progress: {done} of {total} files translated, cost ${cost:.2f}.\n"
    "\n"
    "  [e] extend by 30 minutes and continue\n"
    "  [c] continue indefinitely (remove wall cap for this run)\n"
    "  [a] abort and save state (--resume works)\n"
    "\n"
    "  Choose [e/c/a] (default a, {timeout}s timeout): "
)


def prompt_wall_cap_action(
    *,
    elapsed_seconds: int,
    cap_seconds: int,
    files_done: int,
    files_total: int,
    cost_usd: float,
    timeout_seconds: int = 60,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    isatty: Callable[[], bool] | None = None,
) -> WallCapAction:
    """Prompt the user for a wall-cap decision. See REVIEW_HITL.md.

    Returns WallCapAction.ABORT on non-TTY, EOF, timeout, or explicit
    abort. Only EXTEND / CONTINUE require positive user input.

    `stdin`, `stdout`, `isatty` are injectable for testing. When None,
    the real streams and `sys.stdin.isatty` are used.
    """
    _stdin = stdin if stdin is not None else sys.stdin
    _stdout = stdout if stdout is not None else sys.stdout
    _isatty = isatty if isatty is not None else _stdin.isatty

    if not _isatty():
        return WallCapAction.ABORT

    prompt_text = _MENU.format(
        elapsed=_fmt_duration(elapsed_seconds),
        cap=_fmt_duration(cap_seconds),
        done=files_done,
        total=files_total,
        cost=cost_usd,
        timeout=timeout_seconds,
    )

    # Two attempts before defaulting to abort. Invalid input on the
    # first try re-prompts once; a second invalid input aborts.
    for attempt in range(2):
        _stdout.write(prompt_text if attempt == 0 else "  Invalid choice. Choose [e/c/a]: ")
        _stdout.flush()
        line = _read_line_with_timeout(_stdin, timeout_seconds)
        if line is None:
            # EOF or timeout — both default to abort.
            _stdout.write("\n")
            _stdout.flush()
            return WallCapAction.ABORT
        choice = line.strip().lower()
        if choice == "" or choice == "a":
            return WallCapAction.ABORT
        if choice == "e":
            return WallCapAction.EXTEND
        if choice == "c":
            return WallCapAction.CONTINUE
    return WallCapAction.ABORT
