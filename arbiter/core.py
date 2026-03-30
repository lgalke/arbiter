"""Model loading and response generation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, load_in_4bit: bool = False):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    print(f"Model loaded on: {model.device}")
    return model, tokenizer


def query(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def run_questions(
    model_id: str,
    questions: dict[str, str],
    *,
    n: int = 1,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
    load_in_4bit: bool = False,
    top_k: int | None = None,
) -> list[dict]:
    model, tokenizer = load_model(model_id, load_in_4bit=load_in_4bit)
    records = []
    for q_key, q_text in questions.items():
        print(f"\n--- {q_key} ---")
        for i in range(n):
            response = query(
                model, tokenizer, q_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            record = {
                "model": model_id,
                "question_key": q_key,
                "question": q_text,
                "sample": i,
                "response": response,
            }
            records.append(record)
            preview = response[:120] + ("..." if len(response) > 120 else "")
            print(f"  [{i + 1}/{n}] {preview}")
    return records
