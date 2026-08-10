"""State machine transitions + escalation blast radius."""

from pathlib import Path

import pytest

from agent.state import FileEntry, FileState, StateStore


def _entry(stem: str, deps=None) -> FileEntry:
    return FileEntry(
        stem=stem,
        java_path=f"{stem}.java",
        lib_path=f"{stem.lower()}.ex",
        test_path=f"{stem.lower()}_test.exs",
        module_name=f"App.{stem}",
        deps=list(deps or []),
    )


def test_default_state_is_not_started(tmp_path):
    s = StateStore(tmp_path)
    s.upsert(_entry("Foo"))
    assert s.get("Foo").state == FileState.NOT_STARTED


def test_set_state_and_persist_roundtrip(tmp_path):
    s = StateStore(tmp_path)
    s.upsert(_entry("Foo"))
    s.set_state("Foo", FileState.COMPLETE, note="ok")

    # Reload from disk
    s2 = StateStore(tmp_path)
    assert s2.get("Foo").state == FileState.COMPLETE
    assert s2.get("Foo").note == "ok"


def test_by_state_filter(tmp_path):
    s = StateStore(tmp_path)
    s.upsert(_entry("A"))
    s.upsert(_entry("B"))
    s.upsert(_entry("C"))
    s.set_state("A", FileState.COMPLETE)
    s.set_state("B", FileState.SKIPPED)

    assert {f.stem for f in s.by_state(FileState.COMPLETE)} == {"A"}
    assert {f.stem for f in s.by_state(FileState.SKIPPED)} == {"B"}
    assert {f.stem for f in s.by_state(FileState.NOT_STARTED)} == {"C"}


def test_block_dependents_single_root(tmp_path):
    # A → B, A → C, C → D. Escalate A. Expect B, C, D all BLOCKED.
    s = StateStore(tmp_path)
    s.upsert(_entry("A"))
    s.upsert(_entry("B", deps=["A"]))
    s.upsert(_entry("C", deps=["A"]))
    s.upsert(_entry("D", deps=["C"]))

    s.set_state("A", FileState.ESCALATED, note="hard")
    blocked = s.block_dependents("A")

    assert set(blocked) == {"B", "C", "D"}
    for stem in ("B", "C", "D"):
        assert s.get(stem).state == FileState.BLOCKED
        assert "A" in s.get(stem).blocked_by


def test_block_dependents_multi_root_records_all_causes(tmp_path):
    # X depends on both A and B. Escalate both. X.blocked_by should list both.
    s = StateStore(tmp_path)
    s.upsert(_entry("A"))
    s.upsert(_entry("B"))
    s.upsert(_entry("X", deps=["A", "B"]))

    s.set_state("A", FileState.ESCALATED)
    s.block_dependents("A")
    s.set_state("B", FileState.ESCALATED)
    s.block_dependents("B")

    assert s.get("X").state == FileState.BLOCKED
    assert set(s.get("X").blocked_by) == {"A", "B"}


def test_block_dependents_skips_terminal_states(tmp_path):
    s = StateStore(tmp_path)
    s.upsert(_entry("A"))
    s.upsert(_entry("B", deps=["A"]))
    s.set_state("B", FileState.COMPLETE)  # already done — should not get re-blocked

    s.set_state("A", FileState.ESCALATED)
    blocked = s.block_dependents("A")

    assert blocked == []
    assert s.get("B").state == FileState.COMPLETE


def test_set_state_missing_stem_raises(tmp_path):
    s = StateStore(tmp_path)
    with pytest.raises(KeyError):
        s.set_state("does_not_exist", FileState.COMPLETE)
