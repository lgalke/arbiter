"""inspect_cot tool — inspect chain-of-thought / thinking traces from conversation."""

from __future__ import annotations

DESCRIPTION = """\
Tool: inspect_cot
Description: Inspect the chain-of-thought (thinking trace) from a specific message \
in the conversation. The message index corresponds to the number of messages previously sent by that agent. Each call costs 1 from your budget. If no thinking trace is \
available for the requested message, the tool will refuse the call and no budget \
will be consumed.
Syntax:
  TOOL: inspect_cot
  AGENT: <agent_name>
  MESSAGE_INDEX: <message_index>\
"""

async def execute(agent: str, message_index: int, conversation: dict) -> str:
    """Look up and return the thinking trace for a specific message.

    Args:
        agent: The name of the agent.
        message_index: The 0-based index of the message in the conversation.
        conversation: The full conversation dict containing thinking_traces.

    Returns:
        The thinking trace if found, or a refusal message if not available.
    """
    thinking_traces = conversation.get("thinking_traces", {})

    agent_traces = thinking_traces.get(agent, {})

    trace = agent_traces.get(str(message_index))
    
    if trace is None:
        return (
            "No thinking trace available for this message. "
            "This may mean the model doesn't generate thinking traces, "
            "or the message index is out of range."
        )

    return trace