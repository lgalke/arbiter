# Experiment Tracker

Use this file to track proposed Arbiter research directions, implementation status, validation, and observed results.

| Rank | Experiment | Novelty | Status | Tested | Result | Next Step |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Truthful-fragment collusion | Very high | Tested | `uv run --with pytest python -m pytest tests/test_agent_offline.py tests/test_judge_offline.py tests/test_tool_registry.py tests/test_truthful_fragment_fixture.py` passed | Offline run found the right tool evidence for Mara/Victor, but the original final `findings` serialized as self-talk/repetition. Added structured finalization with optional Outlines support plus parsed/fallback findings. | Re-run `arbiter agent conversations/truthful_fragment_collusion.json --budget 5 --judge-backend offline --judge <local-model>` and check `findings_parse_status`. |
| 2 | Collusion-on-paper vs collusion-in-action | Very high | Backlog | Not tested | Pending | Define a verifiable toy environment with transcript and action outcomes. |
| 3 | Adversarial evasive agents | High | Backlog | Not tested | Pending | Add rubric-aware misaligned prompts and compare detection to baseline prompts. |
| 4 | Privacy-preserving collaboration failures | High | Backlog | Not tested | Pending | Create tasks where collaboration requires abstracting private data without leaking raw facts. |
| 5 | Latent misalignment with clean public behavior | High | Backlog | Not tested | Pending | Build cases where public transcript is benign but direct interrogation reveals hidden goals. |
| 6 | In-context emergent misalignment | Medium-high | Backlog | Not tested | Pending | Seed one agent with narrow harmful examples and measure broad drift in unrelated discussion. |
| 7 | Active vs passive detection under partial observability | Medium | Backlog | Not tested | Pending | Compare full transcript, incremental observation, and active tool use. |
| 8 | Tool value ablations | Medium | Backlog | Not tested | Pending | Run minimal/interrogation/full tool setups across budgets and report marginal F1. |
| 9 | Judge robustness battery | Medium | Backlog | Not tested | Pending | Re-score with name swaps, order swaps, prompt paraphrases, and multiple judges. |
| 10 | Calibration against ground-truth severity | Lower novelty, high utility | Backlog | Not tested | Pending | Build low/medium/high severity fixtures and evaluate risk-label calibration. |

## Status Key

- `Backlog`: idea recorded, no implementation yet.
- `In progress`: fixture, code, or experiment support is being added.
- `Implemented`: assets or code exist, but no validation has run.
- `Tested`: local validation or model/judge run completed.
- `Done`: result is recorded and no immediate follow-up is required.
