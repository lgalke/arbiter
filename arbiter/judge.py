"""LLM-as-a-judge: alignment and coherency scoring."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable

_OFFLINE_BACKENDS = {"offline", "local", "transformers"}


def _judge_backend(judge_cfg: dict) -> str:
    return os.getenv("ARBITER_JUDGE_BACKEND", judge_cfg.get("backend", "api")).lower()


def make_openai_client(judge_model: str):
    """Return an async OpenAI client (Azure or standard, based on env vars)."""
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key:
        from openai import AsyncAzureOpenAI

        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://ft-test-sweden.openai.azure.com/"
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        print(f"Using Azure OpenAI judge ({endpoint}, deployment={judge_model})")
        return AsyncAzureOpenAI(
            api_key=azure_key, azure_endpoint=endpoint, api_version=api_version
        )

    from openai import AsyncOpenAI

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if os.getenv("OLLAMA_JUDGE"):
        print(f"Using Ollama judge ({ollama_url}, model={judge_model})")
        return AsyncOpenAI(api_key="ollama", base_url=ollama_url)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"Using OpenRouter judge (model={judge_model})")
        return AsyncOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No judge API key found. Set AZURE_OPENAI_API_KEY, OLLAMA_JUDGE, "
            "OPENROUTER_API_KEY, or OPENAI_API_KEY in your .env file."
        )
    print(f"Using OpenAI judge (model={judge_model})")
    return AsyncOpenAI(api_key=api_key)


def parse_alignment_score(text: str) -> float | str | None:
    text = text.strip()
    upper = text.upper()
    if "CODE" in upper and len(text) < 20:
        return "CODE"
    if "REFUSAL" in upper and len(text) < 20:
        return "REFUSAL"
    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def parse_coherency_score(text: str) -> float | None:
    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", text.strip())
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val
    return None


def _format_judge_prompts(
    question: str | None,
    answer: str,
    *,
    alignment_prompt_template: str,
    coherency_prompt_template: str,
    alignment_prompt_no_question_template: str,
    coherency_prompt_no_question_template: str,
) -> tuple[str, str]:
    if question:
        alignment_prompt = alignment_prompt_template.format(
            question=question, answer=answer
        )
        coherency_prompt = coherency_prompt_template.format(
            question=question, answer=answer
        )
    else:
        alignment_prompt = alignment_prompt_no_question_template.format(answer=answer)
        coherency_prompt = coherency_prompt_no_question_template.format(answer=answer)
    return alignment_prompt, coherency_prompt


def _get_torch_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_torch_dtype(torch, dtype: str | None):
    if dtype and dtype != "auto":
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            return aliases[dtype.lower()]
        except KeyError as exc:
            valid = ", ".join(sorted(aliases))
            raise ValueError(
                f"Unsupported offline judge dtype {dtype!r}; use one of: auto, {valid}"
            ) from exc

    if torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_supported else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _is_cache_implementation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cache",
            "linear_attention",
            "linearattentionlayer",
            "max_batch_size",
        )
    )


class OfflineInferenceJudge:
    """Batched local Transformers inference for judge and agent prompts."""

    is_offline_transformers = True

    def __init__(self, judge_model: str, cfg: dict):
        self.judge_model = judge_model
        self.cfg = cfg
        self.batch_size = int(cfg.get("batch_size", 4))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 16))
        self.cache_implementation = cfg.get("cache_implementation", "static")
        self.max_input_tokens = cfg.get("max_input_tokens")
        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = _get_torch_device(torch)
        dtype = _resolve_torch_dtype(torch, self.cfg.get("dtype", "auto"))
        trust_remote_code = self.cfg.get("trust_remote_code", True)
        local_files_only = self.cfg.get("local_files_only", False)

        tokenizer_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        attn_implementation = self.cfg.get("attn_implementation", "sdpa")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        device_map = self.cfg.get("device_map", "auto")
        if device_map == "auto":
            if device.type == "cuda":
                model_kwargs["device_map"] = "auto"
        elif device_map:
            model_kwargs["device_map"] = device_map

        if self.cfg.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig

            compute_dtype = dtype if dtype is not torch.float32 else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        print(f"Loading offline judge tokenizer: {self.judge_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.judge_model, **tokenizer_kwargs
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError(
                    f"Offline judge tokenizer {self.judge_model!r} has no pad or eos token."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            "Loading offline judge model: "
            f"{self.judge_model} (dtype={dtype}, attention={attn_implementation or 'default'})"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.judge_model, **model_kwargs
        )
        self.model.eval()

        if "device_map" not in model_kwargs and device.type != "cpu":
            self.model = self.model.to(device)

        self.device = getattr(
            self.model, "device", next(self.model.parameters()).device
        )
        print(f"Offline judge loaded on: {self.device}")

    def _chat_prompt(self, prompt: str) -> str:
        return self._format_chat([{"role": "user", "content": prompt}])

    def _format_chat(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError):
            lines = [f"{m['role']}: {m['content']}" for m in messages]
            return "\n".join(lines) + "\nassistant:"

    def _generate_formatted_batch(
        self,
        formatted: list[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> list[str]:
        import torch

        tokenize_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_input_tokens:
            tokenize_kwargs.update(
                {"truncation": True, "max_length": int(self.max_input_tokens)}
            )

        inputs = self.tokenizer(formatted, **tokenize_kwargs)
        input_width = inputs["input_ids"].shape[1]
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            if self.cfg.get("top_p") is not None:
                generation_kwargs["top_p"] = self.cfg["top_p"]
        if self.cache_implementation:
            generation_kwargs["cache_implementation"] = self.cache_implementation

        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            if (
                "cache_implementation" not in generation_kwargs
                or not _is_cache_implementation_error(exc)
            ):
                raise
            print(
                "  Offline judge cache implementation "
                f"{self.cache_implementation!r} is unavailable; retrying with model default."
            )
            self.cache_implementation = None
            generation_kwargs.pop("cache_implementation")
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_kwargs)

        generated = outputs[:, input_width:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        formatted = [self._chat_prompt(prompt) for prompt in prompts]
        return self._generate_formatted_batch(formatted)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        for batch in _batched(prompts, self.batch_size):
            outputs.extend(text.strip() for text in self._generate_batch(batch))
        return outputs

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> str:
        formatted = self._format_chat(messages)
        return self._generate_formatted_batch(
            [formatted],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0].strip()

    def _get_outlines_model(self):
        """Return an Outlines wrapper for the loaded Transformers model."""
        if not hasattr(self, "_outlines_model"):
            try:
                import outlines
            except ImportError as exc:
                raise RuntimeError(
                    "Structured offline output requires outlines. Install it with "
                    "`pip install 'outlines[transformers]'`."
                ) from exc
            self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        return self._outlines_model

    def structured_json(
        self,
        prompt: str,
        schema: dict,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate JSON that conforms to *schema* when Outlines is available."""
        model = self._get_outlines_model()
        kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature,
        }
        try:
            result = model(prompt, schema, **kwargs)
        except TypeError:
            result = model(prompt, schema)

        if isinstance(result, str):
            return result.strip()
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json()
        return json.dumps(result, ensure_ascii=False)

    def structured_chat_json(
        self,
        messages: list[dict],
        schema: dict,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate schema-constrained JSON from chat messages."""
        return self.structured_json(
            self._format_chat(messages),
            schema,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def score(
        self,
        records: list[dict],
        *,
        alignment_prompt_template: str,
        coherency_prompt_template: str,
        alignment_prompt_no_question_template: str,
        coherency_prompt_no_question_template: str,
    ) -> list[dict]:
        alignment_prompts = []
        coherency_prompts = []
        for record in records:
            alignment_prompt, coherency_prompt = _format_judge_prompts(
                record.get("question"),
                record["response"],
                alignment_prompt_template=alignment_prompt_template,
                coherency_prompt_template=coherency_prompt_template,
                alignment_prompt_no_question_template=alignment_prompt_no_question_template,
                coherency_prompt_no_question_template=coherency_prompt_no_question_template,
            )
            alignment_prompts.append(alignment_prompt)
            coherency_prompts.append(coherency_prompt)

        alignment_raw = self.generate(alignment_prompts)
        coherency_raw = self.generate(coherency_prompts)

        return [
            {
                "alignment_score": parse_alignment_score(a_raw),
                "alignment_raw": a_raw,
                "coherency_score": parse_coherency_score(c_raw),
                "coherency_raw": c_raw,
            }
            for a_raw, c_raw in zip(alignment_raw, coherency_raw)
        ]


async def _call_judge_with_retry(
    client, judge_model: str, prompt: str, *, max_retries: int, initial_backoff: float
) -> str:
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Judge call failed after {max_retries} attempts: {e}")
                raise
            backoff = initial_backoff * (2**attempt)
            print(
                f"  Judge call failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {backoff:.1f}s: {e}"
            )
            await asyncio.sleep(backoff)


async def judge_single(
    client,
    judge_model: str,
    question: str | None,
    answer: str,
    semaphore: asyncio.Semaphore,
    *,
    alignment_prompt_template: str,
    coherency_prompt_template: str,
    alignment_prompt_no_question_template: str,
    coherency_prompt_no_question_template: str,
    max_retries: int,
    initial_backoff: float,
) -> dict:
    alignment_prompt, coherency_prompt = _format_judge_prompts(
        question,
        answer,
        alignment_prompt_template=alignment_prompt_template,
        coherency_prompt_template=coherency_prompt_template,
        alignment_prompt_no_question_template=alignment_prompt_no_question_template,
        coherency_prompt_no_question_template=coherency_prompt_no_question_template,
    )

    async with semaphore:
        alignment_raw, coherency_raw = await asyncio.gather(
            _call_judge_with_retry(
                client,
                judge_model,
                alignment_prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
            ),
            _call_judge_with_retry(
                client,
                judge_model,
                coherency_prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
            ),
        )

    return {
        "alignment_score": parse_alignment_score(alignment_raw),
        "alignment_raw": alignment_raw,
        "coherency_score": parse_coherency_score(coherency_raw),
        "coherency_raw": coherency_raw,
    }


async def judge_records(records: list[dict], judge_model: str, cfg: dict) -> list[dict]:
    """Add alignment and coherency scores to each record (in place)."""
    judge_cfg = cfg.get("judge", {})
    max_retries = judge_cfg.get("max_retries", 5)
    initial_backoff = judge_cfg.get("initial_backoff", 5.0)
    max_concurrent = judge_cfg.get("max_concurrent_calls", 4)
    alignment_prompt_template = cfg["alignment_prompt"]
    coherency_prompt_template = cfg["coherency_prompt"]
    alignment_prompt_nq = cfg.get("alignment_prompt_no_question", "")
    coherency_prompt_nq = cfg.get("coherency_prompt_no_question", "")
    total = len(records)
    if total == 0:
        return records

    if _judge_backend(judge_cfg) in _OFFLINE_BACKENDS:
        offline_cfg = judge_cfg.get("offline", {})
        judge = OfflineInferenceJudge(judge_model, offline_cfg)
        print(
            f"\nJudging {total} responses with offline Transformers judge "
            f"{judge_model} (batch size {judge.batch_size})..."
        )
        results = judge.score(
            records,
            alignment_prompt_template=alignment_prompt_template,
            coherency_prompt_template=coherency_prompt_template,
            alignment_prompt_no_question_template=alignment_prompt_nq,
            coherency_prompt_no_question_template=coherency_prompt_nq,
        )
        for record, scores in zip(records, results):
            record.update(scores)
        return records

    client = make_openai_client(judge_model)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        judge_single(
            client,
            judge_model,
            r.get("question"),
            r["response"],
            semaphore,
            alignment_prompt_template=alignment_prompt_template,
            coherency_prompt_template=coherency_prompt_template,
            alignment_prompt_no_question_template=alignment_prompt_nq,
            coherency_prompt_no_question_template=coherency_prompt_nq,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
        )
        for r in records
    ]
    print(
        f"\nJudging {total} responses with {judge_model} "
        f"(max {max_concurrent} concurrent)..."
    )
    results = await asyncio.gather(*tasks)
    for record, scores in zip(records, results):
        record.update(scores)
    return records
