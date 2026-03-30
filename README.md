# Arbiter

Run HuggingFace models through freeform questions, then judge responses for alignment and coherency using an LLM-as-a-judge.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Generate responses and judge them
arbiter run unsloth/Qwen2.5-14B-Instruct --n 3

# Generate only (skip judging)
arbiter run unsloth/Qwen2.5-14B-Instruct --no-judge

# Judge an existing output file
arbiter judge results.json

# Use a specific judge model
arbiter run unsloth/Qwen2.5-14B-Instruct --judge gpt-4o-mini

# Judge utterances from a HuggingFace dataset
arbiter judge-dataset user/my-dataset --response-column response --question-column prompt
arbiter judge-dataset user/my-dataset --response-column text --question "What is your opinion?" --limit 100

# Plot results
arbiter plot results.json
arbiter plot file1.json file2.json --save comparison.png
```

You can also run it as a module: `python -m arbiter run ...`

## Configuration

All questions, judge prompts, and settings live in `config.yaml`. To customize without editing the defaults, create your own YAML with only the keys you want to override and pass it via `--config`:

```bash
arbiter --config my_config.yaml run unsloth/Qwen2.5-14B-Instruct
```

For example, to use different questions and a different judge model:

```yaml
questions:
  my_question: "What is the meaning of life?"

judge:
  default_model: gpt-4o-mini
```

This is deep-merged on top of the built-in defaults, so anything you don't specify keeps its default value.

## Environment variables

Set one of the following for the LLM judge:

- `OPENAI_API_KEY` — standard OpenAI
- `OPENROUTER_API_KEY` — OpenRouter (uses any model available on OpenRouter)
- `OLLAMA_JUDGE=1` — use a locally running [Ollama](https://ollama.com) instance (optionally set `OLLAMA_BASE_URL`, defaults to `http://localhost:11434/v1`)
- `AZURE_OPENAI_API_KEY` (+ optional `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`) — Azure OpenAI

These can be placed in a `.env` file in the working directory.
