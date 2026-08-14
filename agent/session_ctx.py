"""SessionContext — shared state passed to tool implementations.

Tools are stateless functions that close over this object. Keeps
signatures Anthropic-Tool-Runner-friendly (plain params) while giving tools
access to config, filesystem paths, state, event stream, and cost accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.cost import CostBudget, CostReport
from agent.events import EventStream
from agent.state import StateStore
from language.java import JavaClass
from project.config import Config
from project.deps import DepGraph
from project.scaffold import ScaffoldResult


@dataclass
class SessionContext:
    """All shared state tools need. Assembled in main.py before the agent runs."""
    cfg: Config
    scaffold: ScaffoldResult
    state: StateStore
    events: EventStream
    cost: CostReport
    budget: CostBudget

    # Discovered from source_root at startup
    java_classes: list[JavaClass]
    dep_graph: DepGraph

    # Convenience lookups
    java_by_stem: dict[str, JavaClass]
    module_to_stem: dict[str, str]  # module_name → java stem (O(1) reverse lookup)

    # Resolved from cfg at startup — the Java tests dir, or None if not present
    tests_root: Path | None = None

    # --- Cross-session mutable counters -------------------------------------
    # These persist across session-reset checkpoints so behavior stays coherent
    # when the translator loops through multiple sessions.

    # How many write_elixir calls since the last mix_compile. Reset on any
    # successful `mix_compile_tool` / `validation_status`. Drives the compile
    # nudge injected by the translator loop.
    files_written_since_compile: int = 0

    # Consecutive rate-limit failures observed. Reset on any successful turn.
    # After N in a row (translator._MAX_RATE_LIMIT_STREAK), the loop aborts.
    rate_limit_streak: int = 0

    # Raw remaining tokens for the Task Budget beta. Task Budget is a
    # model-facing signal that must reflect ACTUAL remaining token headroom,
    # not dollar-derived. Distinct from `CostBudget.total_cost_usd` (which
    # caps spend in dollars). Starts at `cfg.agent.max_budget_tokens`.
    remaining_task_tokens: int = 0

    # Polish-mode read-loop breaker. `read_elixir` is a low-risk tool the
    # model prefers under tool_choice="any"; observed behavior is 12+ reads
    # with zero edits. This counter tracks read_elixir calls since the last
    # successful edit_elixir. When it exceeds `_POLISH_READ_CAP`, read_elixir
    # refuses and directs the model to edit or finish. Reset on any edit.
    polish_active: bool = False
    polish_reads_since_edit: int = 0

    @property
    def source_root(self) -> Path:
        assert self.cfg.source.root is not None
        return self.cfg.source.root

    @property
    def target_root(self) -> Path:
        return self.scaffold.target_root

    @property
    def module_prefix(self) -> str:
        """e.g. `JodaMoney` — the top-level module namespace."""
        return self.scaffold.module_name

    @property
    def app_snake(self) -> str:
        """e.g. `joda_money` — used for `lib/<app_snake>/...` paths."""
        return self.scaffold.app_name
