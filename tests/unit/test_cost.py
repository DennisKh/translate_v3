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


def test_pricing_for_unknown_model_returns_zeros_with_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fallback = pricing_for("nonexistent-model-2099")
    assert fallback == {"in": 0.0, "out": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    assert len(w) == 1
    assert "nonexistent-model-2099" in str(w[0].message)
    assert issubclass(w[0].category, UserWarning)


def test_pricing_for_ollama_llama_is_zero_no_warning():
    """Ollama local models are free — pricing_for must return zeros silently."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("llama3.1:8b")
    assert p == {"in": 0.0, "out": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    assert len(w) == 0


def test_pricing_for_ollama_qwen_is_zero_no_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("qwen2.5-coder:1.5b-base")
    assert p == {"in": 0.0, "out": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    assert len(w) == 0


def test_cost_report_ollama_model_records_zero_cost(tmp_path):
    """Regression for bug where CostReport initialized with cfg.agent.model
    (defaulting to claude-opus-4-7) computed bogus dollars for Ollama runs."""
    r = CostReport(model="llama3.1:8b", state_dir=tmp_path)
    r.add_turn(TurnUsage(input_tokens=10_000, output_tokens=5_000))
    r.persist()

    import json
    data = json.loads((tmp_path / "cost_report.json").read_text())
    assert data["model"] == "llama3.1:8b"
    assert data["totals"]["cost_usd"] == 0.0


def test_pricing_for_gpt5_mini_returns_real_rates_no_warning():
    """Bug B regression: gpt-5-mini must hit MODEL_PRICING, not the unknown fallback."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("gpt-5-mini")
    assert len(w) == 0
    assert p["in"] > 0.0
    assert p["out"] > 0.0


def test_pricing_for_gpt5_returns_real_rates_no_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("gpt-5")
    assert len(w) == 0
    assert p["in"] > 0.0
    assert p["out"] > 0.0


def test_pricing_for_gpt4o_mini_returns_real_rates_no_warning():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("gpt-4o-mini")
    assert len(w) == 0
    assert p["in"] > 0.0
    assert p["out"] > 0.0


def test_openai_turn_cost_nonzero():
    """gpt-5-mini: 1M input tokens should compute a non-zero cost."""
    u = TurnUsage(input_tokens=1_000_000)
    cost = u.cost_usd("gpt-5-mini")
    assert cost == pytest.approx(0.25, abs=0.001)


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
    assert reason.type == "max_cost"
    assert "cost cap" in reason.reason


def test_budget_status_flags_tool_call_cap():
    b = CostBudget(max_tokens=10_000_000, max_tool_calls=5, max_wall_seconds=3600)
    b.start()
    b.add(tool_calls=6)
    reason = b.status()
    assert reason is not None
    assert reason.type == "max_tool"
    assert "tool-call" in reason.reason


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
    assert reason.type == "max_cost"
    assert "cost cap" in reason.reason


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


def test_devstral_prefix_is_free_no_warning():
    """Regression: user's custom `devstral-small-24-q3:latest` triggered a
    spurious 'unknown model → $0' warning because 'devstral' wasn't in the
    free-prefix list."""
    import warnings
    from agent.cost import _warned_models
    _warned_models.discard("devstral-small-24-q3:latest")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("devstral-small-24-q3:latest")
    assert p["in"] == 0.0
    assert p["out"] == 0.0
    assert not any("unknown model" in str(x.message) for x in w)


def test_granite_prefix_is_free_no_warning():
    import warnings
    from agent.cost import _warned_models
    _warned_models.discard("granite-3.0:8b")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("granite-3.0:8b")
    assert p["in"] == 0.0
    assert not any("unknown model" in str(x.message) for x in w)


def test_cost_zero_kwarg_silences_warning_for_custom_local_model():
    """LM Studio / vLLM local models with arbitrary names should not warn
    when the config sets `cost_zero = true`."""
    import warnings
    from agent.cost import _warned_models
    _warned_models.discard("my-custom-fine-tuned-model")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = pricing_for("my-custom-fine-tuned-model", cost_zero=True)
    assert p == {"in": 0.0, "out": 0.0, "cache_write": 0.0, "cache_read": 0.0}
    assert not any("unknown model" in str(x.message) for x in w)


def test_cost_zero_false_still_warns_for_truly_unknown_model():
    import warnings
    from agent.cost import _warned_models
    _warned_models.discard("really-truly-unknown-model-xyz")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pricing_for("really-truly-unknown-model-xyz", cost_zero=False)
    assert any("unknown model" in str(x.message) for x in w)


def test_extend_wall_seconds_grows_cap():
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    b.extend_wall_seconds(1800)
    assert b.max_wall_seconds == 5400


def test_extend_wall_seconds_ignores_zero_and_negative():
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    b.extend_wall_seconds(0)
    b.extend_wall_seconds(-100)
    assert b.max_wall_seconds == 3600


def test_remove_wall_cap_sets_to_sys_maxsize():
    import sys
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    b.remove_wall_cap()
    assert b.max_wall_seconds == sys.maxsize
    # And status() no longer flags wall.
    b.start()
    assert b.status() is None


def test_elapsed_seconds_zero_before_start():
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    assert b.elapsed_seconds() == 0


def test_elapsed_seconds_after_start_is_nonnegative():
    b = CostBudget(max_tokens=1_000_000, max_tool_calls=100, max_wall_seconds=3600)
    b.start()
    assert b.elapsed_seconds() >= 0
