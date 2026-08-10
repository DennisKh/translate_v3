"""End-of-run summary — writes summary.json and prints a human-readable
scorecard. Consumed by main.py's finally-block.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.cost import CostBudget, CostReport
from agent.state import FileState, StateStore


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _fmt_pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _sep() -> str:
    return "━" * 60


def _regression_snippet(regression: dict[str, Any] | None) -> str:
    if not regression:
        return ""
    lines = ["", "  Regression score vs gold (0-100):"]
    for name, cur in regression.get("current", {}).items():
        base = regression.get("baseline", {}).get(name)
        base_str = f"  (baseline v2: {base})" if base is not None else ""
        lines.append(f"    {name:16s} {cur:3d}{base_str}")
    return "\n".join(lines)


def build_and_write_summary(
    *,
    state_dir: Path,
    cost: CostReport,
    budget: CostBudget,
    state: StateStore,
    validation: dict[str, bool] | None = None,
    regression: dict[str, Any] | None = None,
    sessions: int = 1,
    exit_code: int = 0,
) -> dict[str, Any]:
    """Compile a run summary, write it to disk, return the dict."""
    totals = cost.totals_snapshot()
    wall_seconds = int(time.monotonic() - budget.started_at)
    total_input = totals.input_tokens + totals.cache_read_tokens
    cache_hit_rate = (totals.cache_read_tokens / total_input) if total_input else 0.0

    translated = [f.stem for f in state.by_state(FileState.COMPLETE)]
    skipped = (
        [{"stem": f.stem, "reason": f.note, "kind": "skipped"}
         for f in state.by_state(FileState.SKIPPED)]
        + [{"stem": f.stem, "reason": f.note, "kind": "deleted"}
           for f in state.by_state(FileState.DELETED)]
    )
    escalated = [{"stem": f.stem, "reason": f.note, "blocked_by": f.blocked_by}
                 for f in state.by_state(FileState.ESCALATED)]
    blocked = [f.stem for f in state.by_state(FileState.BLOCKED)]

    summary: dict[str, Any] = {
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "wall_seconds": wall_seconds,
        "model": cost.model,
        "cost": {
            "total_usd": round(totals.cost_usd(cost.model), 4),
            "by_category": totals.cost_breakdown(cost.model),
        },
        "tokens": {
            "input": totals.input_tokens,
            "output": totals.output_tokens,
            "cache_write": totals.cache_write_tokens,
            "cache_read": totals.cache_read_tokens,
        },
        "cache_hit_rate": round(cache_hit_rate, 4),
        "sessions": sessions,
        "budget": {
            # Dollar-based accounting — matches the actual cap enforced by
            # `CostBudget.status()`. The raw-tokens counter is informational
            # only (shown in Total cost breakdown) and can exceed 100% of
            # `max_tokens` because cache-reads count toward tokens but are
            # priced at ~0.1× normal input.
            "used_usd": round(budget.total_cost_usd, 2),
            "limit_usd": round(budget._max_cost_usd, 2),
            "used_pct": round(100 - budget.remaining_pct(), 2),
            # Kept for backwards compat / cost dashboards that may read them
            "used_tokens": budget.total_tokens,
            "limit_tokens": budget.max_tokens,
        },
        "files": {
            "translated": translated,
            "translated_count": len(translated),
            "skipped": skipped,
            "escalated": escalated,
            "blocked": blocked,
        },
        "validation": validation or {},
        "regression": regression or {},
        "exit_code": exit_code,
    }

    tmp = state_dir / "summary.json.tmp"
    with tmp.open("w") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(state_dir / "summary.json")

    return summary


def print_summary(summary: dict[str, Any], state_dir: Path) -> None:
    """Pretty-print the summary. Mirror of the ASCII spec in the plan."""
    print()
    print(_sep())
    print("  RUN SUMMARY")
    print(_sep())

    print(f"  Wall time:        {_fmt_duration(summary['wall_seconds'])}")
    print(f"  Total cost:       ${summary['cost']['total_usd']:.2f} USD")
    print(f"    input           ${summary['cost']['by_category']['input_usd']:.2f}  "
          f"({summary['tokens']['input']:,} tokens)")
    print(f"    output          ${summary['cost']['by_category']['output_usd']:.2f}  "
          f"({summary['tokens']['output']:,} tokens)")
    print(f"    cache_write     ${summary['cost']['by_category']['cache_write_usd']:.2f}  "
          f"({summary['tokens']['cache_write']:,} tokens)")
    print(f"    cache_read      ${summary['cost']['by_category']['cache_read_usd']:.2f}  "
          f"({summary['tokens']['cache_read']:,} tokens)")
    print(f"  Cache hit rate:   {_fmt_pct(summary['cache_hit_rate'])}")
    print(f"  Sessions:         {summary['sessions']}")
    print(f"  Budget used:      {summary['budget']['used_pct']:.0f}%  "
          f"(${summary['budget']['used_usd']:.2f} / ${summary['budget']['limit_usd']:.2f})")

    print()
    f = summary["files"]
    print(f"  Files translated:  {f['translated_count']}")
    print(f"  Files skipped:     {len(f['skipped'])}"
          + (f"  ({', '.join(s['stem'] for s in f['skipped'])})" if f["skipped"] else ""))
    print(f"  Files escalated:   {len(f['escalated'])}")
    print(f"  Blocked downstream:{len(f['blocked'])}")

    v = summary["validation"]
    if v:
        checks = "  ".join(f"{k} {'✓' if ok else '✗'}" for k, ok in v.items())
        print(f"  Validation:  {checks}")

    if summary.get("regression"):
        print(_regression_snippet(summary["regression"]))

    print()
    print(f"  State: {state_dir}")
    print(f"  Exit code: {summary['exit_code']}")
    print(_sep())
