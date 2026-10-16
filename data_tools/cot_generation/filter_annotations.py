"""Route obvious question/annotation count anomalies to manual review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cot_common import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def anomaly_reasons(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if sample["task_type"] == "all_bales" and sample["target_count"] == 1:
        reasons.append("A001:all_bales_only_one_box")
    if (
        sample["task_type"] == "selected_rows"
        and "两排" in sample.get("target", "")
        and sample["target_count"] < 2
    ):
        reasons.append("A002:two_rows_fewer_than_two_boxes")
    if sample["task_type"] == "single_bale" and sample["target_count"] != 1:
        reasons.append("A003:single_bale_count_not_one")
    return reasons


def main() -> None:
    args = parse_args()
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for sample in read_jsonl(args.input):
        reasons = anomaly_reasons(sample)
        if reasons:
            review.append({"sample": sample, "annotation_review_reasons": reasons})
            counts.update(reason.split(":", 1)[0] for reason in reasons)
        else:
            accepted.append(sample)
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.review, review)
    report = {
        "input_count": len(accepted) + len(review),
        "accepted": len(accepted),
        "manual_review": len(review),
        "reason_counts": dict(counts),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
