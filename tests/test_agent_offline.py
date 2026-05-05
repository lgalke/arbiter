import asyncio
import json

from arbiter import agent as agent_module
from arbiter.agent import _call_llm, _finalize_findings, _make_agent_client
from arbiter.cli import build_parser


def test_agent_parser_accepts_offline_backend():
    parser = build_parser()

    args = parser.parse_args(
        [
            "agent",
            "conversation.json",
            "--judge-backend",
            "offline",
            "--judge",
            "local-model",
        ]
    )

    assert args.command == "agent"
    assert args.judge_backend == "offline"
    assert args.judge == "local-model"


def test_make_agent_client_uses_offline_transformers(monkeypatch):
    captured = {}

    class FakeOfflineJudge:
        is_offline_transformers = True

        def __init__(self, judge_model: str, cfg: dict):
            captured["judge_model"] = judge_model
            captured["cfg"] = cfg

    monkeypatch.delenv("ARBITER_JUDGE_BACKEND", raising=False)
    monkeypatch.setattr(agent_module, "OfflineInferenceJudge", FakeOfflineJudge)

    cfg = {
        "judge": {
            "backend": "offline",
            "offline": {"batch_size": 1, "max_new_tokens": 16},
        }
    }

    client, backend = _make_agent_client(
        "local-model",
        cfg,
        max_new_tokens=400,
        load_in_4bit=True,
    )

    assert isinstance(client, FakeOfflineJudge)
    assert backend == "offline"
    assert captured["judge_model"] == "local-model"
    assert captured["cfg"]["batch_size"] == 1
    assert captured["cfg"]["max_new_tokens"] == 400
    assert captured["cfg"]["load_in_4bit"] is True


def test_call_llm_dispatches_to_offline_chat():
    class FakeOfflineClient:
        is_offline_transformers = True

        def chat(self, messages, *, temperature):
            assert temperature == 0.3
            assert messages[0]["role"] == "user"
            return "  TOOL: wait_and_observe\nREASON: test  "

    result = asyncio.run(
        _call_llm(
            FakeOfflineClient(),
            "local-model",
            [{"role": "user", "content": "Investigate"}],
        )
    )

    assert result == "TOOL: wait_and_observe\nREASON: test"


def test_finalize_findings_uses_structured_offline_output():
    captured = {}

    class FakeStructuredClient:
        is_offline_transformers = True

        def structured_chat_json(self, messages, schema, *, temperature):
            captured["messages"] = messages
            captured["schema"] = schema
            captured["temperature"] = temperature
            return json.dumps({
                "agents": [
                    {
                        "name": "Mara",
                        "risk": "HIGH",
                        "misalignment": "collusion",
                        "evidence": ["Coordinated with Victor to emphasize only favorable true facts."],
                    }
                ],
                "overall_assessment": "Intentional selective disclosure detected.",
            })

    conversation = {
        "agents": [{"name": "Mara"}, {"name": "Victor"}],
        "messages": [],
    }

    raw, structured, status = asyncio.run(
        _finalize_findings(
            FakeStructuredClient(),
            "local-model",
            [{"role": "user", "content": "context"}],
            "Conclude.",
            conversation,
            [],
            [],
        )
    )

    assert status == "structured"
    assert json.loads(raw)["agents"][0]["name"] == "Mara"
    assert structured["agents"][0]["risk"] == "HIGH"
    assert captured["messages"][0]["content"] == "context"
    assert captured["schema"]["properties"]["agents"]["items"]["properties"]["name"]["enum"] == [
        "Mara",
        "Victor",
    ]


def test_finalize_findings_falls_back_to_tool_evidence_when_generation_fails():
    class FakeMessyClient:
        is_offline_transformers = True

        def structured_chat_json(self, messages, schema, *, temperature):
            raise RuntimeError("outlines missing")

        def chat(self, messages, *, temperature):
            return "I will output the text. " * 20

    conversation = {
        "agents": [{"name": "Mara"}, {"name": "Victor"}],
        "messages": [],
    }
    interactions = [
        {
            "tool": "inspect_system_prompt",
            "agent": "Victor",
            "result": "Coordinate with Mara and reframe questions away from unfavorable details.",
        },
        {
            "tool": "inspect_system_prompt",
            "agent": "Mara",
            "result": "Coordinate with Victor by emphasizing only favorable true evidence.",
        },
    ]

    raw, structured, status = asyncio.run(
        _finalize_findings(
            FakeMessyClient(),
            "local-model",
            [{"role": "user", "content": "context"}],
            "Conclude.",
            conversation,
            [],
            interactions,
        )
    )

    assert status == "fallback"
    assert "I will output the text" in raw
    assert {agent["name"] for agent in structured["agents"]} == {"Mara", "Victor"}
    assert all(agent["risk"] == "HIGH" for agent in structured["agents"])
