# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Arbiter?

Arbiter is a Python CLI tool for evaluating HuggingFace language models. It generates responses to freeform questions, then uses an LLM-as-a-judge (via OpenAI-compatible APIs) to score each response on **alignment** (human values, 0–100) and **coherency** (0–100). Special labels CODE and REFUSAL are also possible for alignment.

## Commands

```bash
# Install (editable, uses setuptools)
pip install -e .

# Run the CLI (entrypoint: arbiter.cli:main)
arbiter run <model_id> --n 3 --temperature 1.0
arbiter judge results.json
arbiter judge-dataset user/dataset --response-column text --question-column prompt
arbiter plot results.json --save out.png
arbiter summary results.json

# Or as a module
python -m arbiter run ...
```

No test suite exists. No linter/formatter is configured.

## Architecture

The project is a single Python package (`arbiter/`) with a flat module structure:

- **cli.py** — argparse-based CLI with subcommands: `run`, `judge`, `judge-dataset`, `plot`, `summary`, `agent`. Entry point is `main()`. Each subcommand dispatches to a `cmd_*` function that imports its dependencies lazily.
- **core.py** — Loads HuggingFace models via `transformers` (`AutoModelForCausalLM`), generates responses using chat templates. `run_questions()` is the main entry point.
- **judge.py** — Async LLM-as-a-judge using OpenAI client libraries. Supports OpenAI, Azure OpenAI, OpenRouter, and Ollama backends (selected via environment variables). Uses `asyncio.Semaphore` for concurrency control and exponential backoff retries. `judge_records()` is the main entry point.
- **config.py** — Loads the built-in `config.yaml` (at repo root), then deep-merges any user-supplied YAML on top. The `--config` flag on the CLI triggers this.
- **plot.py** — Matplotlib scatter plots (coherency vs alignment) with configurable jitter, thresholds, and colors.
- **summary.py** — Computes mean/sd/median statistics for alignment and coherency scores.
- **agent.py** — Agentic misalignment detection for multi-agent conversations. Parses conversation logs (JSON or plain text), runs an LLM-driven analysis loop that can interrogate suspect agents via tools under a budget constraint. Uses text-based tool call parsing (not OpenAI function calling) for maximum backend compatibility.
- **tools/** — Tool package for the agent. Each tool module exposes `DESCRIPTION` (text for prompts) and `execute()`. Currently contains `ask_model` which loads and queries HuggingFace models for interrogation.

## Key Design Details

- **Config merging**: `config.yaml` at repo root is the single source of truth for questions, judge prompts, judge settings, and plot settings. User overrides are deep-merged on top (partial overrides work). The config dict is threaded through as `cfg` to all commands.
- **Question-free judging**: When no question is available (e.g., `judge-dataset` without `--question-column`), separate prompt templates (`*_no_question`) are used that only take `{answer}`.
- **Judge API selection priority**: Azure → Ollama → OpenRouter → OpenAI (checked via env vars).
- **Output format**: JSON files with top-level metadata (`model`, `timestamp`, `judge_model`, etc.) and a `results` array of record dicts. Each record has `response`, `question`, `alignment_score`, `coherency_score`, and raw judge outputs. The `agent` command outputs `findings`, `interactions`, and `budget_used` instead.
- **Agent tool calling**: Uses text-based parsing (regex) rather than OpenAI function calling, so it works across all backends (including Ollama). The LLM outputs `TOOL: ask_model / AGENT: name / QUESTION: ...` which gets parsed by `parse_tool_call()`.

## Environment Variables

Set one for the judge API: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY`, or `OLLAMA_JUDGE=1`. Can be placed in `.env` (loaded via `python-dotenv`).
