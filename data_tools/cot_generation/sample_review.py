"""Build a reproducible, task-stratified manual review set."""

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
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    desired = args.count or max(1, round(len(records) * args.ratio))
    desired = min(desired, len(records))
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["sample"]["task_type"]].append(record)
    rng = random.Random(args.seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict] = []
    # Prioritize retried, complex, and rule-fallback items when present.
    priority = sorted(
        records,
        key=lambda record: (
            record.get("cot_source") == "rule",
            record.get("attempt", 1) > 1,
            record["sample"].get("target_count", 0) > 15,
        ),
        reverse=True,
    )
    priority_quota = min(desired // 3, len(priority))
    selected.extend(priority[:priority_quota])
    selected_ids = {record["sample"]["id"] for record in selected}
    task_names = sorted(groups)
    while len(selected) < desired:
        progress = False
        for task in task_names:
            while groups[task]:
                candidate = groups[task].pop()
                if candidate["sample"]["id"] not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate["sample"]["id"])
                    progress = True
                    break
            if len(selected) >= desired:
                break
        if not progress:
            break
    rng.shuffle(selected)
    write_jsonl(args.output, selected)
    report = {
        "input_count": len(records),
        "review_count": len(selected),
        "ratio": len(selected) / len(records) if records else 0,
        "task_counts": dict(Counter(r["sample"]["task_type"] for r in selected)),
        "source_counts": dict(Counter(r.get("cot_source", "unknown") for r in selected)),
        "seed": args.seed,
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
