"""Cost accounting math + budget caps."""

import time
from pathlib import Path

import pytest

from agent.cost import (
    MODEL_PRICING,
    CostBudget,
    CostReport,
    TurnUsage,
    pricing_for,
)


def test_turn_cost_matches_pricing_for_opus():
    u = TurnUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                  cache_write_tokens=1_000_000, cache_read_tokens=1_000_000)
    p = MODEL_PRICING["claude-opus-4-7"]
    expected = p["in"] + p["out"] + p["cache_write"] + p["cache_read"]
    assert abs(u.cost_usd("claude-opus-4-7") - expected) < 1e-6


def test_turn_cost_zero_for_unknown_model():
    u = TurnUsage(input_tokens=1_000_000)
    assert u.cost_usd("some-nonexistent-model") == 0.0


def test_cost_breakdown_reflects_model_not_hardcoded():
    """Regression test: v3 review found summary.py was hardcoding Opus prices."""
    u = TurnUsage(input_tokens=1_000_000, output_tokens=1_000_000)

    opus = u.cost_breakdown("claude-opus-4-7")
    sonnet = u.cost_breakdown("claude-sonnet-4-6")
    haiku = u.cost_breakdown("claude-haiku-4-5")

    assert opus["input_usd"] == 5.00
    assert opus["output_usd"] == 25.00
    assert sonnet["input_usd"] == 3.00
    assert sonnet["output_usd"] == 15.00
    assert haiku["input_usd"] == 1.00
    assert haiku["output_usd"] == 5.00


def test_pricing_for_falls_back_to_opus():
    fallback = pricing_for("nonexistent-model-2099")
    assert fallback == MODEL_PRICING["claude-opus-4-7"]


def test_budget_status_none_when_fine():
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    b.start()
    b.add(tokens=100_000, tool_calls=10)
    assert b.status() is None


def test_budget_status_flags_cost_exhaustion():
    """Budget cap enforcement is in dollars, not raw tokens. A run that
    generates enough OUTPUT tokens to blow the dollar cap should be halted."""
    b = CostBudget(max_tokens=1000, max_tool_calls=100, max_wall_seconds=3600)
    b.start()
    # max_tokens=1000 → cap = $0.025 (1000 * $25/M).  Push past it.
    b.add_usage(TurnUsage(output_tokens=5000), "claude-opus-4-7")
    reason = b.status()
    assert reason is not None
    assert "cost cap" in reason


def test_budget_status_flags_tool_call_cap():
    b = CostBudget(max_tokens=10_000_000, max_tool_calls=5, max_wall_seconds=3600)
    b.start()
    b.add(tool_calls=6)
    reason = b.status()
    assert reason is not None
    assert "tool-call" in reason


def test_budget_add_usage_counts_dollars_not_raw_tokens():
    """Regression test for the Phase 2 review finding: cache reads shouldn't
    consume budget at the same rate as normal input tokens.

    A run with 1M cache-read tokens should NOT max out a budget that would
    accommodate 1M normal input tokens at Opus 4.7 pricing.
    """
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=1000, max_wall_seconds=3600)
    b.start()
    # 1M cache_read at $0.50/M = $0.50; budget cap is 1M output tokens = $25
    # So this should be nowhere near capping.
    b.add_usage(TurnUsage(cache_read_tokens=1_000_000), "claude-opus-4-7")
    assert b.status() is None
    assert b.total_cost_usd == pytest.approx(0.50, abs=0.01)


def test_budget_add_usage_caps_on_real_cost():
    """Full budget in raw output tokens should cap correctly."""
    b = CostBudget(max_tokens=100_000, max_tool_calls=1000, max_wall_seconds=3600)
    b.start()
    # 100K output tokens at Opus $25/M = $2.50; max cost = 100K * $25/M = $2.50
    # Exactly at cap
    b.add_usage(TurnUsage(output_tokens=100_000), "claude-opus-4-7")
    reason = b.status()
    assert reason is not None
    assert "cost cap" in reason


def test_cost_report_model_is_public_property(tmp_path):
    """summary.py was accessing cost._model. Should be a public property."""
    r = CostReport(model="claude-sonnet-4-6", state_dir=tmp_path)
    assert r.model == "claude-sonnet-4-6"


def test_cost_report_aggregates_per_file(tmp_path):
    r = CostReport(model="claude-opus-4-7", state_dir=tmp_path)
    r.add_turn(TurnUsage(input_tokens=100, output_tokens=200), current_file="A")
    r.add_turn(TurnUsage(input_tokens=300), current_file="B")
    r.add_turn(TurnUsage(output_tokens=50))  # no file — global-only

    totals = r.totals_snapshot()
    assert totals.input_tokens == 400
    assert totals.output_tokens == 250


def test_cost_report_persist_is_atomic(tmp_path):
    r = CostReport(model="claude-opus-4-7", state_dir=tmp_path)
    r.add_turn(TurnUsage(input_tokens=100_000, output_tokens=50_000), current_file="Foo")
    r.add_tool("read_java", duration_ms=42, current_file="Foo")
    r.persist()

    import json
    data = json.loads((tmp_path / "cost_report.json").read_text())
    assert data["totals"]["tokens_in"] == 100_000
    assert data["totals"]["tokens_out"] == 50_000
    assert "Foo" in data["by_file"]
    assert data["by_tool"]["read_java"]["calls"] == 1
    assert data["by_tool"]["read_java"]["total_ms"] == 42
