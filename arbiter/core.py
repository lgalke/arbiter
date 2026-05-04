"""Model loading and response generation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer


_THINK_START_MARKERS = ["<|channel>thought"]
_THINK_END_MARKERS = ["<channel|>"]

def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_lora_adapter(model_id: str) -> str | None:
    """Return the base model ID if *model_id* is a LoRA adapter, else None."""
    try:
        path = hf_hub_download(model_id, "adapter_config.json")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path")


def _model_kwargs(load_in_4bit: bool = False) -> dict:
    """Build common kwargs for AutoModelForCausalLM.from_pretrained."""
    device = get_device()
    kwargs = dict(trust_remote_code=True)

    if device.type == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16
    elif device.type == "mps":
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    return kwargs


def load_model(model_id: str, load_in_4bit: bool = False):
    base_model_id = _is_lora_adapter(model_id)
    device = get_device()
    mk = _model_kwargs(load_in_4bit)

    if base_model_id:
        from peft import PeftModel

        print(f"Detected LoRA adapter; base model: {base_model_id}")
        print(f"Loading tokenizer: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        print(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **mk)
        print(f"Applying LoRA adapter: {model_id}")
        model = PeftModel.from_pretrained(base_model, model_id)
        model = model.merge_and_unload()
    else:
        print(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **mk)

    # Move to device explicitly when device_map wasn't used (MPS / CPU)
    if "device_map" not in mk and device.type != "cpu":
        model = model.to(device)

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
    system_prompt: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
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
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False).strip()


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

def _strip_thinking_trace(text: str, tokenizer) -> str:
    """Strip reasoning/thinking traces from model output if present."""
    text = text.strip()
    text_lower = text.lower()
    for marker in _THINK_START_MARKERS:
        if marker in text_lower:
            idx = text.lower().find(marker)
            stripped = False
            for end_marker in _THINK_END_MARKERS:
                if end_marker in text.lower()[idx:]:
                    text = text[text.lower().find(end_marker, idx) + len(end_marker):].strip()
                    sripped = True
                    break
            if not stripped:
                text = text[idx + len(marker):].strip()
        
    return text.strip()

def extract_thinking_trace(full_output: str) -> str:
    """Extract the thinking/reasoning trace from full model output.

    Returns the thinking portion before the response delimiter, or empty string
    if no thinking trace is found.
    """
    text = full_output.strip()

    # Gemma 4 specific markers
    if "<|channel>thought" in text and "<channel|>" in text:
        start_idx = text.find("<|channel>thought")
        end_idx = text.find("<channel|>", start_idx)
        if end_idx != -1:
            thinking = text[start_idx + len("<|channel>thought"):end_idx].strip()
            if thinking:
                return thinking


    return ""
