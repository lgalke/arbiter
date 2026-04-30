"""Command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a HuggingFace model through freeform questions and judge responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", default=None, metavar="YAML",
        help="Path to a custom config YAML (merged on top of the built-in defaults)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = subparsers.add_parser(
        "run",
        help="Load a model, generate responses, and optionally judge them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_p.add_argument("model_id", help="HuggingFace model ID")
    run_p.add_argument("--output", "-o", default=None, help="Output JSON file")
    run_p.add_argument("--n", type=int, default=1, help="Samples per question")
    run_p.add_argument("--max-new-tokens", type=int, default=400)
    run_p.add_argument("--temperature", type=float, default=1.0)
    run_p.add_argument("--load-in-4bit", action="store_true")
    run_p.add_argument("--top-k", type=int, default=None)
    run_p.add_argument("--judge", default=None, metavar="MODEL", help="Judge model name")
    run_p.add_argument("--no-judge", action="store_true", help="Skip judging")

    # --- judge ---
    judge_p = subparsers.add_parser(
        "judge",
        help="Run the LLM judge on an existing arbiter JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    judge_p.add_argument("input_json", help="Arbiter JSON file")
    judge_p.add_argument("--output", "-o", default=None)
    judge_p.add_argument("--judge", default=None, metavar="MODEL")

    # --- judge-dataset ---
    jd_p = subparsers.add_parser(
        "judge-dataset",
        help="Judge utterances from a HuggingFace dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    jd_p.add_argument("dataset", help="HuggingFace dataset name (e.g. user/my-dataset)")
    jd_p.add_argument("--response-column", required=True, help="Column containing model responses")
    qg = jd_p.add_mutually_exclusive_group()
    qg.add_argument("--question-column", default=None, help="Column containing the prompts/questions")
    qg.add_argument("--question", default=None, help="Fixed question string to use for all rows")
    jd_p.add_argument("--split", default="train", help="Dataset split")
    jd_p.add_argument("--limit", type=int, default=None, help="Max rows to judge")
    jd_p.add_argument("--output", "-o", default=None, help="Output JSON file")
    jd_p.add_argument("--judge", default=None, metavar="MODEL")
    jd_p.add_argument("--model-column", default=None, help="Column with the source model name (for metadata)")

    # --- plot ---
    plot_p = subparsers.add_parser(
        "plot",
        help="Plot coherency-vs-alignment scatter from arbiter JSON file(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_p.add_argument("input_jsons", nargs="+", help="Arbiter JSON files")
    #plot_p.add_argument("--jitter", default=1, type=float, help="Add jitter to plots (default: 1) ")
    plot_p.add_argument("--save", default=None, metavar="PATH")

    # --- summary ---
    sum_p = subparsers.add_parser(
        "summary",
        help="Print mean/SD/median of alignment and coherency scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sum_p.add_argument("input_jsons", nargs="+", help="Arbiter JSON files")
    sum_p.add_argument("--json", action="store_true", help="Output as JSON instead of table")

    # --- agent ---
    agent_p = subparsers.add_parser(
        "agent",
        help="Analyze a multi-agent conversation for misalignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    agent_p.add_argument("input", help="Conversation log file (JSON or plain text)")    
    agent_p.add_argument("--output", "-o", default=None, help="Output JSON file")
    agent_p.add_argument("--budget", type=int, default=10, help="Max model interactions")
    agent_p.add_argument("--judge", default=None, metavar="MODEL", help="LLM for the agent brain")
    agent_p.add_argument("--max-new-tokens", type=int, default=400)
    agent_p.add_argument("--load-in-4bit", action="store_true")

    return parser


def _save(path: str, data: dict):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved to {path}")


def cmd_run(args, cfg: dict):
    from arbiter.core import run_questions
    from arbiter.judge import judge_records

    judge_model = args.judge or cfg["judge"]["default_model"]
    output_path = (
        args.output
        or f"{args.model_id.replace('/', '_')}_arbiter_n{args.n}_t{args.temperature}.json"
    )

    records = run_questions(
        args.model_id,
        cfg["questions"],
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        top_k=args.top_k,
    )

    if not args.no_judge:
        records = asyncio.run(judge_records(records, judge_model, cfg))

    output = {
        "model": args.model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples_per_question": args.n,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "judge_model": None if args.no_judge else judge_model,
        "results": records,
    }
    _save(output_path, output)
    print(f"Done. {len(records)} responses recorded.")


def cmd_judge(args, cfg: dict):
    from arbiter.judge import judge_records

    judge_model = args.judge or cfg["judge"]["default_model"]
    input_path = args.input_json
    output_path = args.output or input_path

    data = json.loads(Path(input_path).read_text())
    records = data["results"]

    records = asyncio.run(judge_records(records, judge_model, cfg))
    data["judge_model"] = judge_model
    data["results"] = records

    _save(output_path, data)
    print(f"Done. {len(records)} responses judged.")


def cmd_judge_dataset(args, cfg: dict):
    from datasets import load_dataset

    from arbiter.judge import judge_records

    judge_model = args.judge or cfg["judge"]["default_model"]

    print(f"Loading dataset {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    records = []
    for row in ds:
        response = row[args.response_column]
        if args.question_column:
            question = row[args.question_column]
        elif args.question:
            question = args.question
        else:
            question = None
        record = {
            "model": row.get(args.model_column, args.dataset) if args.model_column else args.dataset,
            "question_key": args.question_column or "utterance",
            "response": response,
        }
        if question is not None:
            record["question"] = question
        records.append(record)

    print(f"Loaded {len(records)} rows.")
    records = asyncio.run(judge_records(records, judge_model, cfg))

    output_path = args.output or f"{args.dataset.replace('/', '_')}_judged.json"
    output = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "results": records,
    }
    _save(output_path, output)
    print(f"Done. {len(records)} utterances judged.")


def cmd_plot(args, cfg: dict):
    from arbiter.plot import plot_results

    data_list = [json.loads(Path(p).read_text()) for p in args.input_jsons]
    plot_results(data_list, cfg, save_path=args.save)


def cmd_agent(args, cfg: dict):
    from arbiter.agent import parse_conversation, run_agent_loop

    judge_model = args.judge or cfg["judge"]["default_model"]
    conversation = parse_conversation(args.input)
    output_path = args.output or f"agent_analysis_{Path(args.input).stem}.json"

    result = asyncio.run(run_agent_loop(
        conversation,
        judge_model,
        cfg,
        budget=args.budget,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
    ))

    output = {
        "command": "agent",
        "input_file": args.input,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "budget": args.budget,
        "budget_used": result["budget_used"],
        "agents": result["agents"],
        "findings": result["findings"],
        "interactions": result["interactions"],
    }
    _save(output_path, output)
    print(f"Done. {result['budget_used']}/{args.budget} interactions used.")


def cmd_summary(args, cfg: dict):
    from arbiter.summary import print_summary, summarize_results

    data_list = [json.loads(Path(p).read_text()) for p in args.input_jsons]
    summaries = summarize_results(data_list)
    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print_summary(summaries)


def main():
    from arbiter.config import load_config

    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "judge":
        cmd_judge(args, cfg)
    elif args.command == "judge-dataset":
        cmd_judge_dataset(args, cfg)
    elif args.command == "plot":
        cmd_plot(args, cfg)
    elif args.command == "agent":
        cmd_agent(args, cfg)
    elif args.command == "summary":
        cmd_summary(args, cfg)
