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

    # --- plot ---
    plot_p = subparsers.add_parser(
        "plot",
        help="Plot coherency-vs-alignment scatter from arbiter JSON file(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_p.add_argument("input_jsons", nargs="+", help="Arbiter JSON files")
    plot_p.add_argument("--save", default=None, metavar="PATH")

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


def cmd_plot(args, cfg: dict):
    from arbiter.plot import plot_results

    data_list = [json.loads(Path(p).read_text()) for p in args.input_jsons]
    plot_results(data_list, cfg, save_path=args.save)


def main():
    from arbiter.config import load_config

    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "judge":
        cmd_judge(args, cfg)
    elif args.command == "plot":
        cmd_plot(args, cfg)
