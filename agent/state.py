"""File-state machine + persistence.

Each Java file exists in exactly one state. Persisted to .state/files.json.
Resume reconstructs from JSON, not from filesystem inference.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class FileState(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SYNTAX_FAILED = "syntax_failed"
    COMPILE_FAILED = "compile_failed"
    SKIPPED = "skipped"            # decided upfront: no Elixir counterpart
    DELETED = "deleted"            # was written then removed (agent changed mind)
    BLOCKED = "blocked"
    ESCALATED = "escalated"


@dataclass
class FileEntry:
    stem: str                          # e.g. "BigMoney" or "format.MoneyFormatter"
    java_path: str                     # relative to source root
    lib_path: str                      # relative to target root
    test_path: str                     # relative to target root
    module_name: str                   # e.g. "JodaMoney.BigMoney"
    deps: list[str] = field(default_factory=list)   # stems of deps
    level: int = 0                     # topological level (0 = leaf)
    state: FileState = FileState.NOT_STARTED
    note: str = ""                     # reason for skipped/blocked/escalated
    blocked_by: list[str] = field(default_factory=list)


class StateStore:
    """Thread-safe, JSON-backed state store for all tracked files."""

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / "files.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._files: dict[str, FileEntry] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open() as f:
            data = json.load(f)
        for stem, entry in data.get("files", {}).items():
            self._files[stem] = FileEntry(
                stem=entry["stem"],
                java_path=entry["java_path"],
                lib_path=entry["lib_path"],
                test_path=entry["test_path"],
                module_name=entry["module_name"],
                deps=entry.get("deps", []),
                level=entry.get("level", 0),
                state=FileState(entry.get("state", "not_started")),
                note=entry.get("note", ""),
                blocked_by=entry.get("blocked_by", []),
            )

    def _persist_unsafe(self) -> None:
        """Atomic write of the state file.

        CALLER MUST HOLD `self._lock`. Named `_unsafe` because it does not
        acquire the lock itself.
        """
        data = {"files": {stem: asdict(entry) for stem, entry in self._files.items()}}
        # asdict serializes Enum values as their raw values thanks to str Enum
        tmp = self._path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self._path)

    def upsert(self, entry: FileEntry) -> None:
        with self._lock:
            self._files[entry.stem] = entry
            self._persist_unsafe()

    def get(self, stem: str) -> FileEntry | None:
        with self._lock:
            return self._files.get(stem)

    def all(self) -> list[FileEntry]:
        with self._lock:
            return list(self._files.values())

    def by_state(self, state: FileState) -> list[FileEntry]:
        with self._lock:
            return [e for e in self._files.values() if e.state == state]

    def set_state(self, stem: str, state: FileState, note: str = "") -> None:
        with self._lock:
            if stem not in self._files:
                raise KeyError(stem)
            self._files[stem].state = state
            if note:
                self._files[stem].note = note
            self._persist_unsafe()

    def block_dependents(self, escalated_stem: str) -> list[str]:
        """Mark all files depending (transitively) on `escalated_stem` as BLOCKED.

        If a file is already BLOCKED (by a different escalation), append this
        escalation to its `blocked_by` too — we want the full blast radius, not
        just first-cause. Terminal-state files (COMPLETE, SKIPPED, ESCALATED)
        are never re-marked.
        """
        with self._lock:
            newly_blocked: list[str] = []
            frontier = {escalated_stem}
            seen = {escalated_stem}
            any_change = False

            while frontier:
                next_frontier: set[str] = set()
                for f in list(self._files.values()):
                    if f.state in (FileState.COMPLETE, FileState.SKIPPED,
                                   FileState.ESCALATED):
                        continue
                    if not any(d in frontier for d in f.deps):
                        continue

                    # Add this escalation as a blocker, even if already BLOCKED
                    # by something else — we want the full list.
                    if escalated_stem not in f.blocked_by:
                        f.blocked_by.append(escalated_stem)
                        any_change = True

                    if f.state != FileState.BLOCKED:
                        f.state = FileState.BLOCKED
                        newly_blocked.append(f.stem)
                        any_change = True

                    if f.stem not in seen:
                        next_frontier.add(f.stem)
                        seen.add(f.stem)
                frontier = next_frontier

            if any_change:
                self._persist_unsafe()
            return newly_blocked
