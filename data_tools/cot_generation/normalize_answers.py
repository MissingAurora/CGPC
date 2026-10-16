"""Normalize detection answer order: rows top-to-bottom, each row left-to-right."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from cot_common import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
        help="判定同一行所需的最小垂直重叠比例，默认0.30",
    )
    return parser.parse_args()


def center_x(box: list[int]) -> float:
    return (box[0] + box[2]) / 2


def center_y(box: list[int]) -> float:
    return (box[1] + box[3]) / 2


def row_bounds(row: list[tuple[int, list[int], str]]) -> tuple[float, float]:
    return median(item[1][1] for item in row), median(item[1][3] for item in row)


def same_row(
    box: list[int], row: list[tuple[int, list[int], str]], min_overlap: float
) -> bool:
    row_y1, row_y2 = row_bounds(row)
    overlap = max(0.0, min(box[3], row_y2) - max(box[1], row_y1))
    smaller_height = max(1.0, min(box[3] - box[1], row_y2 - row_y1))
    if overlap / smaller_height >= min_overlap:
        return True
    row_center = (row_y1 + row_y2) / 2
    height_scale = max(box[3] - box[1], row_y2 - row_y1, 1.0)
    return abs(center_y(box) - row_center) <= 0.35 * height_scale


def normalized_items(
    boxes: list[list[int]], labels: list[str], min_overlap: float
) -> list[tuple[int, list[int], str]]:
    items = [(index, box, label) for index, (box, label) in enumerate(zip(boxes, labels))]
    items.sort(key=lambda item: (center_y(item[1]), center_x(item[1])))
    rows: list[list[tuple[int, list[int], str]]] = []
    for item in items:
        candidates = [
            (abs(center_y(item[1]) - sum(center_y(x[1]) for x in row) / len(row)), row)
            for row in rows
            if same_row(item[1], row, min_overlap)
        ]
        if candidates:
            min(candidates, key=lambda pair: pair[0])[1].append(item)
        else:
            rows.append([item])
    rows.sort(key=lambda row: median(center_y(item[1]) for item in row))
    ordered: list[tuple[int, list[int], str]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: center_x(item[1])))
    return ordered


def main() -> None:
    args = parse_args()
    output_records: list[dict[str, Any]] = []
    changed_examples: list[dict[str, Any]] = []
    changed_count = 0
    total = 0
    for sample in read_jsonl(args.input):
        total += 1
        truth = sample["ground_truth"]
        boxes = truth["bbox_2d"]
        labels = truth["labels"]
        ordered = normalized_items(boxes, labels, args.min_overlap)
        old_indices = [item[0] for item in ordered]
        if old_indices != list(range(len(boxes))):
            changed_count += 1
            if len(changed_examples) < 50:
                changed_examples.append(
                    {"id": sample["id"], "old_index_order": old_indices}
                )
        normalized = dict(sample)
        normalized["ground_truth"] = {
            "bbox_2d": [item[1] for item in ordered],
            "labels": [item[2] for item in ordered],
        }
        output_records.append(normalized)
    write_jsonl(args.output, output_records)
    report = {
        "input_count": total,
        "changed_count": changed_count,
        "unchanged_count": total - changed_count,
        "changed_rate": changed_count / total if total else 0,
        "rule": "vertical-overlap row clustering, rows top-to-bottom, row items left-to-right",
        "min_overlap": args.min_overlap,
        "changed_examples": changed_examples,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
