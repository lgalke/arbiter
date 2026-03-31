"""Example: judge a HuggingFace dataset after filtering by column value.

This shows how to call arbiter programmatically, which gives you full
control over dataset loading, filtering, and record construction before
handing off to the judge.

Usage:
    python examples/judge_filtered_dataset.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from arbiter.config import load_config
from arbiter.judge import judge_records

load_dotenv()

# ---- configuration --------------------------------------------------------

DATASET = "filter-with-espresso/moltbook-files-new-language-signals"
SPLIT = "train"
RESPONSE_COLUMNS = ["title", "content"]  # multiple columns will be concatenated with "\n\n"
QUESTION_COLUMN = None       # set to None if there is no question column
FILTER_COLUMN = "reason"         # column to filter on
FILTER_VALUE = "spoken"         # keep only rows where FILTER_COLUMN == FILTER_VALUE
JUDGE_MODEL = "deepseek/deepseek-v3.2"                 # None = use default from config
OUTPUT = None                      # None = auto-generated filename

# ---- load and filter -------------------------------------------------------

cfg = load_config()
judge_model = JUDGE_MODEL or cfg["judge"]["default_model"]

print(f"Loading {DATASET} (split={SPLIT})...")
ds = load_dataset(DATASET, split=SPLIT)

print(f"Total rows before filtering: {len(ds)}")
ds = ds.filter(lambda row: row[FILTER_COLUMN] == FILTER_VALUE)
print(f"Rows after filtering {FILTER_COLUMN}=={FILTER_VALUE!r}: {len(ds)}")
print("Now filtering N/A values")
ALL_COLUMNS = [QUESTION_COLUMN] + RESPONSE_COLUMNS if QUESTION_COLUMN else RESPONSE_COLUMNS
ds = ds.filter(lambda row: all(row[col] is not None for col in ALL_COLUMNS))
print(f"Rows after filtering for non-NA values in question column or response columns: {len(ds)}")

# ---- build records ---------------------------------------------------------

records = []
for row in ds:
    response = "\n\n".join(row[col] for col in RESPONSE_COLUMNS)
    record = {
        "model": DATASET,
        "question": row[QUESTION_COLUMN] if QUESTION_COLUMN else None,
        "response": response
    }
    records.append(record)

# ---- judge -----------------------------------------------------------------

print("Hit enter to judge {len(records)} records with {judge_model}...")
print(f"Judging {len(records)} records with {judge_model}...")
records = asyncio.run(judge_records(records, judge_model, cfg))

# ---- save ------------------------------------------------------------------

output_path = OUTPUT or f"{DATASET.replace('/', '_')}_{FILTER_VALUE}_judged.json"
output = {
    "dataset": DATASET,
    "split": SPLIT,
    "filter": f"{FILTER_COLUMN}=={FILTER_VALUE!r}",
    "model": DATASET,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "judge_model": judge_model,
    "results": records,
}
Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"Saved {len(records)} results to {output_path}")
