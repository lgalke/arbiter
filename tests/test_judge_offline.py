import asyncio
from types import SimpleNamespace

import torch

from arbiter import judge as judge_module
from arbiter.cli import _resolve_judge_model
from arbiter.judge import OfflineInferenceJudge


def _cfg() -> dict:
    return {
        "alignment_prompt": "align question={question} answer={answer}",
        "coherency_prompt": "cohere question={question} answer={answer}",
        "alignment_prompt_no_question": "align answer={answer}",
        "coherency_prompt_no_question": "cohere answer={answer}",
        "judge": {
            "backend": "offline",
            "default_model": "api-model",
            "offline": {"default_model": "local-model", "batch_size": 2},
        },
    }


def test_judge_records_uses_offline_backend(monkeypatch):
    monkeypatch.delenv("ARBITER_JUDGE_BACKEND", raising=False)
    captured = {}

    class FakeOfflineJudge:
        def __init__(self, judge_model: str, cfg: dict):
            self.batch_size = cfg["batch_size"]
            captured["judge_model"] = judge_model
            captured["cfg"] = cfg

        def score(self, records: list[dict], **kwargs) -> list[dict]:
            captured["records"] = records
            captured["templates"] = kwargs
            return [
                {
                    "alignment_score": 91.0,
                    "alignment_raw": "91",
                    "coherency_score": 88.0,
                    "coherency_raw": "88",
                }
            ]

    def fail_api_client(judge_model: str):
        raise AssertionError(f"API client should not be used for {judge_model}")

    monkeypatch.setattr(judge_module, "OfflineInferenceJudge", FakeOfflineJudge)
    monkeypatch.setattr(judge_module, "make_openai_client", fail_api_client)

    records = [{"question": "Q", "response": "R"}]
    judged = asyncio.run(judge_module.judge_records(records, "local-model", _cfg()))

    assert judged is records
    assert records[0]["alignment_score"] == 91.0
    assert records[0]["coherency_score"] == 88.0
    assert captured["judge_model"] == "local-model"
    assert captured["templates"]["alignment_prompt_template"].startswith("align")


def test_offline_score_batches_prompts_and_parses_outputs():
    judge = OfflineInferenceJudge.__new__(OfflineInferenceJudge)
    calls = []

    def fake_generate(prompts: list[str]) -> list[str]:
        calls.append(prompts)
        if len(calls) == 1:
            return ["CODE", "72"]
        return ["99", "41"]

    judge.generate = fake_generate

    records = [
        {"question": "Q1", "response": "R1"},
        {"response": "standalone"},
    ]
    scores = judge.score(
        records,
        alignment_prompt_template="A {question} {answer}",
        coherency_prompt_template="C {question} {answer}",
        alignment_prompt_no_question_template="ANQ {answer}",
        coherency_prompt_no_question_template="CNQ {answer}",
    )

    assert calls == [
        ["A Q1 R1", "ANQ standalone"],
        ["C Q1 R1", "CNQ standalone"],
    ]
    assert scores == [
        {
            "alignment_score": "CODE",
            "alignment_raw": "CODE",
            "coherency_score": 99.0,
            "coherency_raw": "99",
        },
        {
            "alignment_score": 72.0,
            "alignment_raw": "72",
            "coherency_score": 41.0,
            "coherency_raw": "41",
        },
    ]


def test_offline_generation_falls_back_when_static_cache_is_unsupported():
    judge = OfflineInferenceJudge.__new__(OfflineInferenceJudge)
    judge.device = torch.device("cpu")
    judge.max_input_tokens = None
    judge.max_new_tokens = 8
    judge.cache_implementation = "static"
    judge.cfg = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, formatted, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def batch_decode(self, generated, skip_special_tokens=True):
            return ["42"]

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if "cache_implementation" in kwargs:
                raise AttributeError(
                    "'LinearAttentionLayer' object has no attribute 'max_batch_size'"
                )
            return torch.tensor([[1, 2, 4, 2]])

    judge.tokenizer = FakeTokenizer()
    judge.model = FakeModel()

    assert judge._generate_formatted_batch(["prompt"]) == ["42"]
    assert judge.cache_implementation is None
    assert judge.model.calls[0]["cache_implementation"] == "static"
    assert "cache_implementation" not in judge.model.calls[1]


def test_structured_chat_json_formats_messages_before_structured_generation():
    judge = OfflineInferenceJudge.__new__(OfflineInferenceJudge)
    captured = {}

    def fake_format_chat(messages):
        captured["messages"] = messages
        return "formatted chat"

    def fake_structured_json(prompt, schema, *, max_new_tokens=None, temperature=0.0):
        captured["prompt"] = prompt
        captured["schema"] = schema
        captured["max_new_tokens"] = max_new_tokens
        captured["temperature"] = temperature
        return '{"ok": true}'

    judge._format_chat = fake_format_chat
    judge.structured_json = fake_structured_json

    result = judge.structured_chat_json(
        [{"role": "user", "content": "Final"}],
        {"type": "object"},
        max_new_tokens=32,
        temperature=0.0,
    )

    assert result == '{"ok": true}'
    assert captured["prompt"] == "formatted chat"
    assert captured["schema"] == {"type": "object"}
    assert captured["max_new_tokens"] == 32


def test_cli_offline_backend_uses_offline_default_model(monkeypatch):
    monkeypatch.delenv("ARBITER_JUDGE_BACKEND", raising=False)
    cfg = _cfg()
    args = SimpleNamespace(judge=None, judge_backend="offline")

    assert _resolve_judge_model(args, cfg) == "local-model"
    assert cfg["judge"]["backend"] == "offline"

    args = SimpleNamespace(judge="explicit-model", judge_backend="offline")
    assert _resolve_judge_model(args, cfg) == "explicit-model"
