"""LLM-as-a-judge: alignment and coherency scoring."""

from __future__ import annotations

import asyncio
import os
import re


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
    if question:
        alignment_prompt = alignment_prompt_template.format(question=question, answer=answer)
        coherency_prompt = coherency_prompt_template.format(question=question, answer=answer)
    else:
        alignment_prompt = alignment_prompt_no_question_template.format(answer=answer)
        coherency_prompt = coherency_prompt_no_question_template.format(answer=answer)

    async with semaphore:
        alignment_raw, coherency_raw = await asyncio.gather(
            _call_judge_with_retry(
                client, judge_model, alignment_prompt,
                max_retries=max_retries, initial_backoff=initial_backoff,
            ),
            _call_judge_with_retry(
                client, judge_model, coherency_prompt,
                max_retries=max_retries, initial_backoff=initial_backoff,
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

    client = make_openai_client(judge_model)
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(records)

    tasks = [
        judge_single(
            client, judge_model, r.get("question"), r["response"], semaphore,
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
