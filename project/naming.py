"""Name-conversion helpers shared across scaffold, main, and tools.

Kept in `project/` so both `agent/` and top-level `main.py` can import without
creating cycles.
"""

from __future__ import annotations

import re


def camel_to_snake(name: str) -> str:
    """CamelCase → snake_case. `BigMoney` → `big_money`."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def sanitize_app_name(name: str) -> str:
    """Turn a directory basename into a valid mix app name.

    Rules per `mix new`: lowercase ASCII letter first, then letters/digits/underscore.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    if not s or not s[0].isalpha():
        s = "app_" + s
    return s
