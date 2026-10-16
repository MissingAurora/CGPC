"""Create a reproducible task-stratified teacher-CoT pilot set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from cot_common import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-bales", type=int, default=120)
    parser.add_argument("--selected-rows", type=int, default=80)
    parser.add_argument("--single-bale", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def count_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 5:
        return "2_5"
    if count <= 10:
        return "6_10"
    if count <= 15:
        return "11_15"
    return "16_plus"


def diverse_sample(records: list[dict], quota: int, rng: random.Random) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[count_bucket(record["target_count"])].append(record)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict] = []
    bucket_names = sorted(buckets)
    while len(selected) < quota and bucket_names:
        next_names = []
        for name in bucket_names:
            if buckets[name] and len(selected) < quota:
                selected.append(buckets[name].pop())
            if buckets[name]:
                next_names.append(name)
        bucket_names = next_names
    if len(selected) != quota:
        raise ValueError(f"requested {quota}, only selected {len(selected)}")
    return selected


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task_type"]].append(record)
    quotas = {
        "all_bales": args.all_bales,
        "selected_rows": args.selected_rows,
        "single_bale": args.single_bale,
    }
    rng = random.Random(args.seed)
    selected = []
    for task, quota in quotas.items():
        selected.extend(diverse_sample(by_task[task], quota, rng))
    rng.shuffle(selected)
    write_jsonl(args.output, selected)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "total": len(selected),
        "task_counts": dict(Counter(record["task_type"] for record in selected)),
        "target_count_buckets": dict(
            Counter(count_bucket(record["target_count"]) for record in selected)
        ),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
