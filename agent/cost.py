"""Cost accounting + per-file / per-tool breakdown + hard caps.

Model pricing table below is per-1M tokens. Update when Anthropic changes it.
Numbers based on the values documented in this project's CLAUDE-API skill.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Per-1M-token pricing in USD. Update when Anthropic revises.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":   {"in": 5.00, "out": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-opus-4-6":   {"in": 5.00, "out": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00, "cache_write": 1.25, "cache_read": 0.10},
}


def pricing_for(model: str) -> dict[str, float]:
    """Per-1M-token rates for a model. Falls back to Opus 4.7 with a warning."""
    return MODEL_PRICING.get(model, MODEL_PRICING["claude-opus-4-7"])


@dataclass
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        p = MODEL_PRICING.get(model)
        if p is None:
            return 0.0
        return (
            self.input_tokens         * p["in"]          / 1_000_000
            + self.output_tokens      * p["out"]         / 1_000_000
            + self.cache_write_tokens * p["cache_write"] / 1_000_000
            + self.cache_read_tokens  * p["cache_read"]  / 1_000_000
        )

    def cost_breakdown(self, model: str) -> dict[str, float]:
        """Per-category cost using the model's real rates."""
        p = pricing_for(model)
        return {
            "input_usd":       round(self.input_tokens         * p["in"]          / 1_000_000, 4),
            "output_usd":      round(self.output_tokens        * p["out"]         / 1_000_000, 4),
            "cache_write_usd": round(self.cache_write_tokens   * p["cache_write"] / 1_000_000, 4),
            "cache_read_usd":  round(self.cache_read_tokens    * p["cache_read"]  / 1_000_000, 4),
        }


@dataclass
class ToolStats:
    calls: int = 0
    total_ms: int = 0


@dataclass
class CostBudget:
    """Hard-cap enforcement across a run.

    Budget is measured in **dollars**, not raw tokens, so cache-warm runs
    (which are what we want) don't get prematurely halted just because their
    cache_read counter grew. The `add_usage(TurnUsage, model)` method computes
    the dollar cost using the model's real per-token rates.
    """
    max_tokens: int          # kept for legacy config compatibility (used as budget signal)
    max_tool_calls: int
    max_wall_seconds: int
    started_at: float = 0.0
    total_tokens: int = 0    # sum of ALL token categories (for informational display only)
    total_cost_usd: float = 0.0
    total_tool_calls: int = 0

    # Derived at __post_init__ from max_tokens using worst-case Opus pricing
    _max_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        # Approximate cost cap. Users think in tokens historically; we translate.
        # Worst case: max_tokens as pure output at Opus 4.7 rate ($25/M).
        # For a real hard-dollar cap, set via `max_cost_usd` explicitly (future).
        opus_output_per_token = 25.0 / 1_000_000
        self._max_cost_usd = self.max_tokens * opus_output_per_token

    def start(self) -> None:
        self.started_at = time.monotonic()

    def add(self, tokens: int = 0, tool_calls: int = 0) -> None:
        """Legacy shape — counts raw tokens (used for informational total)."""
        self.total_tokens += tokens
        self.total_tool_calls += tool_calls

    def add_usage(self, usage: "TurnUsage", model: str, tool_calls: int = 0) -> None:
        """Preferred: record a turn's usage in dollars, not raw tokens.

        Cache reads cost ~0.1× normal input; counting them as full budget would
        penalize the exact prompt-caching optimization we're trying to encourage.
        """
        self.total_cost_usd += usage.cost_usd(model)
        # Also keep the raw counter for the informational display
        self.total_tokens += (usage.input_tokens + usage.output_tokens
                              + usage.cache_write_tokens + usage.cache_read_tokens)
        self.total_tool_calls += tool_calls

    def remaining_pct(self) -> float:
        """Percentage of budget remaining, measured in dollars."""
        if self._max_cost_usd <= 0:
            return 100.0
        return max(0.0, 100.0 * (1.0 - self.total_cost_usd / self._max_cost_usd))

    def status(self) -> str | None:
        """Return None if fine, else a reason string for a soft-stop."""
        if self.total_cost_usd >= self._max_cost_usd:
            return (f"cost cap reached (${self.total_cost_usd:.2f} / ${self._max_cost_usd:.2f}) — "
                    f"raise --max-budget-tokens if intentional")
        if self.total_tool_calls >= self.max_tool_calls:
            return f"tool-call cap reached ({self.total_tool_calls}/{self.max_tool_calls})"
        wall = time.monotonic() - self.started_at
        if wall >= self.max_wall_seconds:
            return f"wall-clock cap reached ({int(wall)}s/{self.max_wall_seconds}s)"
        return None


class CostReport:
    """Aggregated per-file / per-tool stats. Written to .state/cost_report.json."""

    def __init__(self, model: str, state_dir: Path) -> None:
        self._model = model
        self._path = state_dir / "cost_report.json"
        self._lock = threading.Lock()
        self._by_file: dict[str, TurnUsage] = defaultdict(TurnUsage)
        self._by_file_calls: dict[str, int] = defaultdict(int)
        self._by_tool: dict[str, ToolStats] = defaultdict(ToolStats)
        self._totals = TurnUsage()
        self._total_calls = 0
        self._started_at = time.monotonic()

    @property
    def model(self) -> str:
        return self._model

    def add_turn(self, usage: TurnUsage, current_file: str | None = None) -> None:
        with self._lock:
            self._totals.input_tokens += usage.input_tokens
            self._totals.output_tokens += usage.output_tokens
            self._totals.cache_write_tokens += usage.cache_write_tokens
            self._totals.cache_read_tokens += usage.cache_read_tokens
            if current_file:
                f = self._by_file[current_file]
                f.input_tokens += usage.input_tokens
                f.output_tokens += usage.output_tokens
                f.cache_write_tokens += usage.cache_write_tokens
                f.cache_read_tokens += usage.cache_read_tokens

    def add_tool(self, tool: str, duration_ms: int, current_file: str | None = None) -> None:
        with self._lock:
            self._by_tool[tool].calls += 1
            self._by_tool[tool].total_ms += duration_ms
            self._total_calls += 1
            if current_file:
                self._by_file_calls[current_file] += 1

    def tool_call_count(self, tool: str) -> int:
        """Total invocations of a specific tool since run start (thread-safe)."""
        with self._lock:
            return self._by_tool[tool].calls

    def totals_snapshot(self) -> TurnUsage:
        with self._lock:
            return TurnUsage(
                input_tokens=self._totals.input_tokens,
                output_tokens=self._totals.output_tokens,
                cache_write_tokens=self._totals.cache_write_tokens,
                cache_read_tokens=self._totals.cache_read_tokens,
            )

    def cumulative_cost_usd(self) -> float:
        return self.totals_snapshot().cost_usd(self._model)

    def persist(self) -> None:
        with self._lock:
            wall = int(time.monotonic() - self._started_at)
            data = {
                "model": self._model,
                "totals": {
                    "tokens_in": self._totals.input_tokens,
                    "tokens_out": self._totals.output_tokens,
                    "cache_write": self._totals.cache_write_tokens,
                    "cache_read": self._totals.cache_read_tokens,
                    "cost_usd": round(self._totals.cost_usd(self._model), 4),
                    "wall_seconds": wall,
                    "total_tool_calls": self._total_calls,
                },
                "by_file": {
                    stem: {
                        "tokens_in": u.input_tokens,
                        "tokens_out": u.output_tokens,
                        "cache_read": u.cache_read_tokens,
                        "cost_usd": round(u.cost_usd(self._model), 4),
                        "tool_calls": self._by_file_calls.get(stem, 0),
                    }
                    for stem, u in self._by_file.items()
                },
                "by_tool": {
                    name: {"calls": s.calls, "total_ms": s.total_ms}
                    for name, s in self._by_tool.items()
                },
            }
            tmp = self._path.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self._path)
