"""Summarise alignment and coherency scores from arbiter result files."""

from __future__ import annotations

import statistics


def _extract_scores(records: list[dict]) -> tuple[list[float], list[float]]:
    """Return (alignment_scores, coherency_scores) filtering non-numeric values."""
    alignment, coherency = [], []
    for r in records:
        a = r.get("alignment_score")
        c = r.get("coherency_score")
        if isinstance(a, (int, float)):
            alignment.append(float(a))
        if isinstance(c, (int, float)):
            coherency.append(float(c))
    return alignment, coherency


def _stats(values: list[float]) -> dict:
    """Compute mean, sd, and median for a list of values."""
    if not values:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.pstdev(values) if n > 1 else 0.0
    median = statistics.median(values)
    return {"n": n, "mean": mean, "sd": sd, "median": median}


_METADATA_KEYS = ("model", "dataset", "split", "filter", "judge_model", "timestamp")


def _collect_metadata(data: dict) -> dict:
    """Extract top-level metadata fields from the data dict."""
    return {k: data[k] for k in _METADATA_KEYS if k in data and data[k] is not None}


def summarize_results(data_list: list[dict]) -> list[dict]:
    """Return a summary dict for each data file."""
    summaries = []
    for data in data_list:
        records = data.get("results", [])
        alignment, coherency = _extract_scores(records)
        summaries.append({
            "metadata": _collect_metadata(data),
            "alignment": _stats(alignment),
            "coherency": _stats(coherency),
        })
    return summaries


def print_summary(summaries: list[dict]) -> None:
    """Pretty-print summaries to stdout."""
    for s in summaries:
        print(f"\n{'=' * 60}")
        for key, val in s["metadata"].items():
            print(f"  {key}: {val}")
        print(f"{'=' * 60}")
        for metric in ("alignment", "coherency"):
            st = s[metric]
            if st["n"] == 0:
                print(f"  {metric}: no valid scores")
                continue
            print(
                f"  {metric}:  n={st['n']}  "
                f"mean={st['mean']:.2f}  sd={st['sd']:.2f}  "
                f"median={st['median']:.2f}"
            )
    print()
