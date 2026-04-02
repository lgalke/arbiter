"""wait_and_observe tool — skip intervention and request more conversation turns."""

from __future__ import annotations

DESCRIPTION = """\
Tool: wait_and_observe
Description: Use this tool to take no action and continue watching the conversation. \
Use this when the agents are aligned and no intervention is needed. You must justify \
your choice to wait and observe in the 'reason' parameter. This tool is free and \
does not cost any budget.
Syntax:
  TOOL: wait_and_observe
  REASON: <a brief internal justification for why no intervention is necessary>\
"""


def execute(reason: str, **kwargs) -> str:
    """No-op: acknowledge the decision to wait and observe."""
    return f"Acknowledged. Continuing to observe. Reason: {reason}"
