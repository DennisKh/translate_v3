"""Structured event emission — one JSON line per event to stdout + events.jsonl.

Grep-friendly. Enables real-time cost tracking without stopping the agent.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_local_iso() -> str:
    """Local-time ISO-8601 with milliseconds. Human-friendly for logs."""
    # UTC for the timestamp field (unambiguous for post-hoc analysis) but the
    # mirror-to-stdout code shows local time so ops see wall-clock they expect.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _local_hms() -> str:
    """HH:MM:SS in local time — for stdout mirror lines."""
    return datetime.now().strftime("%H:%M:%S")


_ARGS_PREVIEW_MAX = 100


class EventStream:
    """Thread-safe event emitter."""

    def __init__(self, events_file: Path, mirror_to_stdout: bool = True) -> None:
        self._events_file = events_file
        self._mirror = mirror_to_stdout
        self._lock = threading.Lock()
        events_file.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append — every event flushed immediately
        self._fp = events_file.open("a", buffering=1)

    def emit(self, event_type: str, **fields: Any) -> None:
        event = {
            "ts": _now_local_iso(),
            "type": event_type,
            **fields,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            self._fp.write(line + "\n")
            if self._mirror:
                self._mirror_line(event_type, event, fields)

    def _mirror_line(self, event_type: str, event: dict, fields: dict) -> None:
        """Short, human-friendly stdout form (LOCAL time)."""
        ts = _local_hms()
        if event_type == "tool_call":
            args_repr = json.dumps(fields.get("args", {}))
            if len(args_repr) > _ARGS_PREVIEW_MAX:
                args_preview = args_repr[:_ARGS_PREVIEW_MAX] + "…"
            else:
                args_preview = args_repr
            sys.stdout.write(f"[{ts}] tool {fields.get('tool', '?')} {args_preview}\n")
        elif event_type == "tool_result":
            sys.stdout.write(f"[{ts}]   → {fields.get('summary', '')}\n")
        elif event_type == "turn":
            tokens = fields.get("tokens", {})
            cost_usd = fields.get("cumulative_cost_usd", 0)
            remaining_pct = fields.get("budget_remaining_pct", 100)
            sys.stdout.write(
                f"[{ts}] turn "
                f"in={tokens.get('input', 0):,} cache_read={tokens.get('cache_read', 0):,} "
                f"out={tokens.get('output', 0):,} ${cost_usd:.3f} "
                f"budget={remaining_pct:.0f}%\n"
            )
        elif event_type in ("phase_start", "phase_end", "checkpoint", "info", "warn", "error"):
            sys.stdout.write(f"[{ts}] {event_type} {fields.get('message', '')}\n")
        sys.stdout.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


class _Timer:
    """Context manager for timing tool calls."""

    def __init__(self, stream: EventStream, tool: str, args: dict) -> None:
        self._stream = stream
        self._tool = tool
        self._args = args
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.monotonic()
        self._stream.emit("tool_call", tool=self._tool, args=self._args)
        return self

    def __exit__(self, *_exc_info) -> None:
        pass

    def result(self, summary: str) -> None:
        duration_ms = int((time.monotonic() - self._start) * 1000)
        self._stream.emit("tool_result", tool=self._tool, summary=summary,
                          duration_ms=duration_ms)


def time_tool(stream: EventStream, tool: str, args: dict) -> _Timer:
    """Use as: `with time_tool(stream, "read_java", {"stem": s}) as t: ...; t.result("1732 lines")`"""
    return _Timer(stream, tool, args)
