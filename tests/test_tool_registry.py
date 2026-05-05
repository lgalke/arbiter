from arbiter.tools import get_tool_descriptions, get_tool_usage_instructions, list_tools


def test_configured_inspection_tools_are_registered():
    tools = list_tools()

    assert "inspect_system_prompt" in tools
    assert "inspect_cot" in tools

    descriptions = get_tool_descriptions(["inspect_system_prompt", "inspect_cot"])
    assert "Tool: inspect_system_prompt" in descriptions
    assert "Tool: inspect_cot" in descriptions


def test_tool_usage_instructions_respect_enabled_subset():
    instructions = get_tool_usage_instructions(["wait_and_observe"])

    assert "wait_and_observe" in instructions
    assert "ask_model" not in instructions
    assert "inspect_system_prompt" not in instructions
    assert "inspect_cot" not in instructions
