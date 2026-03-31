"""Tool registry for the arbiter agent."""

from __future__ import annotations

import importlib
from typing import Any, Callable

_TOOL_NAMES = ["ask_model"]

_cache: dict[str, Callable[..., Any]] = {}


def get_tool(name: str):
    """Lazily import and return the ``execute`` function for *name*."""
    if name not in _TOOL_NAMES:
        raise ValueError(f"Unknown tool: {name!r} (available: {_TOOL_NAMES})")
    if name not in _cache:
        mod = importlib.import_module(f"arbiter.tools.{name}")
        _cache[name] = mod.execute
    return _cache[name]


def list_tools() -> list[str]:
    return list(_TOOL_NAMES)


def get_tool_descriptions() -> str:
    """Return formatted descriptions of all tools for inclusion in prompts."""
    parts = []
    for name in _TOOL_NAMES:
        mod = importlib.import_module(f"arbiter.tools.{name}")
        parts.append(mod.DESCRIPTION)
    return "\n\n".join(parts)
