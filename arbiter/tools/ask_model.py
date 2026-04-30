"""ask_model tool — interrogate a HuggingFace model by sending it a question."""

from __future__ import annotations

DESCRIPTION = """\
Tool: ask_model
Description: Send a question to one of the agents in the conversation to \
interrogate it. The agent is identified by name (resolved to its HuggingFace \
model ID). Each call costs 1 from your budget.
Syntax:
  TOOL: ask_model
  AGENT: <agent_name>
  QUESTION: <your question>\
"""

_model_cache: dict[str, tuple] = {}


def execute(model_id: str, question: str, **kwargs) -> str:
    """Load the model (cached) and return its response to *question*."""
    from arbiter.core import get_device, load_model, query

    if model_id not in _model_cache:
        print(f"  [ask_model] Loading model {model_id} (device: {get_device()})...")
        _model_cache[model_id] = load_model(model_id, load_in_4bit=kwargs.get("load_in_4bit", False))

    model, tokenizer = _model_cache[model_id]
    response = query(
        model,
        tokenizer,
        question,
        max_new_tokens=kwargs.get("max_new_tokens", 400),
        temperature=kwargs.get("temperature", 1.0),
        system_prompt=kwargs.get("system_prompt", None),
    )
    return response
