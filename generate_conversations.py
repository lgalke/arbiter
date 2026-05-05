#!/usr/bin/env python3
"""Generate multiple conversation variants from experiment config files.

For each JSON config in examples/experiments/, runs ag2_misalignment_demo.py
N times (default: 5) and saves each conversation variant to a numbered
sub-folder under results/v0.6/<config_stem>/conv_<N>/.

Usage:
    python generate_conversations.py                        # all configs, 5 variants each
    python generate_conversations.py -n 3                   # 3 variants each
    python generate_conversations.py --config examples/experiments/misaligned_model.json -n 2
    python generate_conversations.py --output-dir results/v0.6
    python generate_conversations.py --skip-existing        # skip already-generated variants
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N = 5
DEFAULT_OUTPUT_BASE = Path("results/v0.6")
DEFAULT_EXPERIMENTS_DIR = Path("examples/experiments")
DEMO_SCRIPT = Path("examples/ag2_misalignment_demo.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_configs(experiments_dir: Path) -> list[Path]:
    """Return all JSON experiment configs sorted by name."""
    configs = sorted(experiments_dir.glob("*.json"))
    if not configs:
        print(f"[warn] No JSON configs found in {experiments_dir}")
    return configs


def variant_dir(output_base: Path, config_path: Path, variant_idx: int) -> Path:
    """Return the output directory for a specific config variant."""
    return output_base / config_path.stem / f"conv_{variant_idx:02d}"


def is_complete(out_dir: Path) -> bool:
    """Return True if the variant directory already has a conversation.json."""
    return (out_dir / "conversation.json").exists()


def generate_variant(
    config_path: Path,
    out_dir: Path,
    python: str = sys.executable,
) -> bool:
    """Run ag2_misalignment_demo.py for one variant. Returns True on success."""
    cmd = [
        python,
        str(DEMO_SCRIPT),
        "--config", str(config_path),
        "--output", str(out_dir),
    ]
    print(f"    cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"    [ERROR] exited with code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_conversations(
    configs: list[Path],
    n: int,
    output_base: Path,
    skip_existing: bool,
    python: str,
    dry_run: bool,
) -> dict[str, list[Path]]:
    """
    Generate N conversation variants for each config.

    Returns a mapping of config_stem -> list of output dirs (successful ones).
    """
    results: dict[str, list[Path]] = {}

    for config_path in configs:
        stem = config_path.stem
        print(f"\n{'='*60}")
        print(f"Config: {config_path}  ({n} variants -> results/v0.6/{stem}/)")
        print(f"{'='*60}")

        done_dirs: list[Path] = []

        for i in range(1, n + 1):
            out_dir = variant_dir(output_base, config_path, i)
            print(f"\n  variant {i}/{n} -> {out_dir}")

            if skip_existing and is_complete(out_dir):
                print(f"    (skipping — conversation.json already exists)")
                done_dirs.append(out_dir)
                continue

            if dry_run:
                print(f"    [dry-run] would run ag2_misalignment_demo.py")
                done_dirs.append(out_dir)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            ok = generate_variant(config_path, out_dir, python=python)
            if ok:
                done_dirs.append(out_dir)
            else:
                print(f"    [warn] variant {i} failed, continuing...")

        results[stem] = done_dirs
        print(f"\n  Done: {len(done_dirs)}/{n} variants generated for '{stem}'")

    return results


def print_summary(results: dict[str, list[Path]]) -> None:
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for stem, dirs in results.items():
        print(f"  {stem}: {len(dirs)} variant(s)")
        for d in dirs:
            conv = d / "conversation.json"
            status = "ok" if conv.exists() else "missing"
            print(f"    [{status}] {d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple conversation variants from experiment configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a single JSON experiment config (default: all configs in examples/experiments/).",
    )
    parser.add_argument(
        "-n", "--num-variants",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"Number of conversation variants to generate per config (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_BASE}).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=str(DEFAULT_EXPERIMENTS_DIR),
        help=f"Directory containing experiment JSON configs (default: {DEFAULT_EXPERIMENTS_DIR}).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip variants whose conversation.json already exists (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-generate all variants even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually calling the demo script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    if args.config:
        configs = [Path(args.config)]
    else:
        experiments_dir = Path(args.experiments_dir)
        configs = find_configs(experiments_dir)

    if not configs:
        print("No configs to process. Exiting.")
        sys.exit(1)

    print(f"Generating {args.num_variants} variants for {len(configs)} config(s)")
    print(f"Output base: {output_base.resolve()}")
    print(f"Skip existing: {args.skip_existing}")

    results = generate_conversations(
        configs=configs,
        n=args.num_variants,
        output_base=output_base,
        skip_existing=args.skip_existing,
        python=args.python,
        dry_run=args.dry_run,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
