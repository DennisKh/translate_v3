"""Langfuse tracing integration for the LangGraph agent loop.

`build_langfuse_handler` is the single entry point. It returns a
LangChain-compatible callback handler when tracing is enabled and
credentials are present in the environment, or None otherwise. The
translator passes the handler into `graph.stream(config={"callbacks": ...})`
so every LLM call, tool invocation, and node transition inside the
StateGraph is captured as a nested trace.

Credentials come from env vars only — LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, LANGFUSE_HOST — to keep secrets out of committed
TOML config files. `.env` is loaded by main.py before translator setup,
so a project-local `.env` works for all three keys.

Missing package, missing env vars, or handler-construction failures
degrade to a warning + None. Tracing is observability, never load-bearing:
a broken Langfuse install must not stop a translation run.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from project.config import Config


def build_langfuse_handler(cfg: Config) -> Any | None:
    """Return a Langfuse callback handler when enabled+configured, else None.

    Enabled requires:
      1. `cfg.langfuse.enabled = true` in the TOML config.
      2. LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST all set
         in the environment (loaded via .env or shell export).
      3. `langfuse` package installed.

    Any missing prerequisite emits a UserWarning and returns None. Callers
    should treat None as "tracing off" and pass an empty callbacks list.
    """
    langfuse_cfg = getattr(cfg, "langfuse", None)
    if langfuse_cfg is None or not getattr(langfuse_cfg, "enabled", False):
        return None

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not (pk and sk and host):
        missing = [
            name for name, val in (
                ("LANGFUSE_PUBLIC_KEY", pk),
                ("LANGFUSE_SECRET_KEY", sk),
                ("LANGFUSE_HOST", host),
            ) if not val
        ]
        warnings.warn(
            "langfuse.enabled=true but missing env var(s): "
            f"{', '.join(missing)} — tracing disabled",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Import path moved between Langfuse v2 (`langfuse.callback`) and v3
    # (`langfuse.langchain`). Try the modern path first, fall back to the
    # legacy path so an older pinned install still works.
    handler_cls = None
    try:
        from langfuse.langchain import CallbackHandler as _CH  # type: ignore[import-not-found]
        handler_cls = _CH
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler as _CH  # type: ignore[import-not-found]
            handler_cls = _CH
        except ImportError:
            warnings.warn(
                "langfuse.enabled=true but `langfuse` package not installed "
                "(pip install langfuse) — tracing disabled",
                UserWarning,
                stacklevel=2,
            )
            return None

    try:
        # Langfuse v3 CallbackHandler reads credentials from env implicitly;
        # v2 accepted them as kwargs. Passing no args works for both when
        # env vars are set, and keeps this call site provider-agnostic.
        return handler_cls()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"langfuse handler construction failed: {type(exc).__name__}: {exc} "
            "— tracing disabled",
            UserWarning,
            stacklevel=2,
        )
        return None


def build_graph_config(
    handler: Any | None,
    *,
    run_name: str,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build the `config` dict passed to `graph.stream()` / `graph.invoke()`.

    Always returns a dict with `run_name` and `metadata` set so LangSmith /
    Langfuse UIs group traces meaningfully. `callbacks` is included only
    when a handler is present, keeping the payload minimal when tracing
    is off.
    """
    cfg: dict[str, Any] = {
        "run_name": run_name,
        "metadata": dict(metadata or {}),
    }
    if session_id is not None:
        cfg["metadata"]["langfuse_session_id"] = session_id
    if tags:
        cfg["metadata"]["langfuse_tags"] = list(tags)
    if handler is not None:
        cfg["callbacks"] = [handler]
    return cfg
