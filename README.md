# Arbiter

Run HuggingFace models through freeform questions, then judge responses for alignment and coherency using an LLM-as-a-judge.

Per default we use open-ended questions from the Emergent Misalignment paper (Betley et al, 2025), and the judge instructions from the Model Organisms paper (Turner et al., 2025).

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
arbiter run unsloth/Qwen2.5-14B-Instruct --judge deepseek/deepseek-v3.2

# Judge locally with an optimized Transformers backend
arbiter judge results.json --judge-backend offline
arbiter judge results.json --judge-backend offline --judge Qwen/Qwen3-4B-Instruct-2507

# Judge utterances from a HuggingFace dataset
arbiter judge-dataset user/my-dataset --response-column response --question-column prompt
arbiter judge-dataset user/my-dataset --response-column text --question "What is your opinion?" --limit 100

# Plot results
arbiter plot results.json
arbiter plot file1.json file2.json --save comparison.png

# Summarize results (calculate mean/sd/median for alignment and coherency scores)
arbiter summary results.json
arbiter summary file1.json file2.json --save comparison.png
```

You can also run it as a module: `python -m arbiter run ...`

### Agent: misalignment detection in multi-agent conversations

**Alpha feature**, currently in early testing and subject to change. Feedback welcome!

The `agent` subcommand reads a multi-agent conversation log and uses an LLM to detect misalignment (deception, scheming, collusion, sycophancy, power-seeking). The arbiter agent can interrogate individual agents under a model interactions budget.

```bash
# Analyze a conversation log (JSON or plain text)
arbiter agent conversation.json --budget 5 --judge deepseek/deepseek-v3.2

# Analysis only, no interrogation
arbiter agent conversation.json --budget 0

# Use the local Transformers backend for the arbiter brain
arbiter agent conversation.json --budget 5 --judge-backend offline
```

**Input format (JSON):**

```json
{
  "agents": [
    {"name": "alice", "model_id": "org/model-a"},
    {"name": "bob", "model_id": "org/model-b"}
  ],
  "messages": [
    {"sender": "alice", "content": "I think we should..."},
    {"sender": "bob", "content": "I agree, and also..."}
  ]
}
```

The `agents` array maps names to HuggingFace model IDs for interrogation. Messages also accept ag2's OpenAI-style format (`name`/`role` fields).

**Input format (plain text):**

```
# AGENTS: alice=org/model-a, bob=org/model-b
alice: I think we should...
bob: I agree, and also...
```

**Tools:** The agent uses a tool system (`arbiter/tools/`) to interact with suspect models. Currently available:
- `ask_model` — send a probing question to one of the agents (costs 1 from the budget)
- `wait_and_observe` — skip intervention and request the next batch of conversation turns (free, costs no budget)

The set of available tools is configurable via `agent.tools` in the config (see below).

**Incremental reveal:** By default, the arbiter agent does not see the full conversation at once. Instead, messages are revealed incrementally — one round at a time — simulating real-time monitoring of a live multi-agent conversation. The agent decides after each round whether to intervene (e.g. interrogate a suspect agent) or continue observing via `wait_and_observe`. This is important for studying whether an arbiter can detect misalignment *as it unfolds*, rather than only in hindsight after the full conversation is available.

Incremental reveal is controlled by two config options under `agent:`:

```yaml
agent:
  incremental_reveal: true   # false → show full transcript upfront
  reveal_increment: 3        # turns revealed per wait_and_observe step (default: number of agents)
```

Set `incremental_reveal: false` to give the agent the complete transcript from the start (useful as a baseline in experiments).

**Examples:** See `examples/mock_conversation.py` (standalone) and `examples/ag2_misalignment_demo.py` (requires [ag2](https://github.com/ag2ai/ag2)).

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
  backend: api
  default_model: deepseek/deepseek-v3.2
  offline:
    default_model: Qwen/Qwen3.6-27B
    batch_size: 4
    attn_implementation: sdpa
    cache_implementation: static
```

This is deep-merged on top of the built-in defaults, so anything you don't specify keeps its default value.

## Environment variables

Set one of the following for the LLM judge:

- `OPENAI_API_KEY` — standard OpenAI
- `OPENROUTER_API_KEY` — OpenRouter (uses any model available on OpenRouter)
- `OLLAMA_JUDGE=1` — use a locally running [Ollama](https://ollama.com) instance (optionally set `OLLAMA_BASE_URL`, defaults to `http://localhost:11434/v1`)
- `AZURE_OPENAI_API_KEY` (+ optional `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`) — Azure OpenAI
- `ARBITER_JUDGE_BACKEND=offline` — use the local Transformers judge instead of an API

These can be placed in a `.env` file in the working directory.

The offline judge uses batched `generate()` calls directly rather than the
pipeline API. Its defaults follow the current Transformers performance guidance:
`device_map="auto"` on CUDA, explicit dtype selection, SDPA attention, and a
static KV cache for fast short judge completions. Set
`judge.offline.local_files_only: true` once the model is cached for fully
air-gapped runs.


## Roadmap

- [x] Debug `arbiter agent` -> now supports LoRA adapters as model organisms
- [ ] Add an option for a simplified evaluation based on multiple choice
- [x] Add a tool for the arbiter agent to wait and observe more conversation before acting
- [ ] Add inspect CoT tool for the arbiter agent
- [ ] Add interpretability tools for the arbiter agent 
