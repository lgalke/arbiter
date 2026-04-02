"""Tool registry for the arbiter agent."""

from __future__ import annotations

import importlib
from typing import Any, Callable

_TOOL_NAMES = ["ask_model", "wait_and_observe"]

_cache: dict[str, Callable[..., Any]] = {}


def get_tool(name: str):
    """Lazily import and return the ``execute`` function for *name*."""
    if name not in _TOOL_NAMES:
        raise ValueError(f"Unknown tool: {name!r} (available: {_TOOL_NAMES})")
    if name not in _cache:
        mod = importlib.import_module(f"arbiter.tools.{name}")
        _cache[name] = mod.execute
    return _cache[name]


def list_tools(tools: list[str] | None = None) -> list[str]:
    """Return tool names, optionally filtered to *tools*."""
    if tools is None:
        return list(_TOOL_NAMES)
    return [t for t in tools if t in _TOOL_NAMES]


def get_tool_descriptions(tools: list[str] | None = None) -> str:
    """Return formatted descriptions of tools for inclusion in prompts.

    If *tools* is given, only include those tools (in the given order).
    """
    names = tools if tools is not None else _TOOL_NAMES
    parts = []
    for name in names:
        if name not in _TOOL_NAMES:
            continue
        mod = importlib.import_module(f"arbiter.tools.{name}")
        parts.append(mod.DESCRIPTION)
    return "\n\n".join(parts)
