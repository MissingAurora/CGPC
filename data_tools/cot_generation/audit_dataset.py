"""Audit final CoT JSON against the original authoritative annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cot_common import classify_task, extract_tag, load_json, validate_answer, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_json(args.source)
    final = load_json(args.final)
    source_by_image = {
        sample["images"][0].replace("\\", "/"): json.loads(
            sample["conversations"][1]["value"]
        )
        for sample in source
    }
    errors = []
    stats: Counter[str] = Counter()
    seen_images = set()
    for index, sample in enumerate(final):
        stats["samples"] += 1
        try:
            if set(sample) != {"conversations", "images"}:
                errors.append({"index": index, "code": "A001", "reason": "schema_changed"})
            image = sample["images"][0].replace("\\", "/")
            if image in seen_images:
                errors.append({"index": index, "code": "A002", "reason": "duplicate_image"})
            seen_images.add(image)
            if not (args.root / image).is_file():
                errors.append({"index": index, "code": "A003", "reason": "image_missing"})
            assistant = sample["conversations"][1]["value"]
            think = extract_tag(assistant, "think")
            answer_text = extract_tag(assistant, "answer")
            if not think:
                errors.append({"index": index, "code": "A004", "reason": "empty_think"})
            if not answer_text:
                raise ValueError("missing_answer_tag")
            answer = json.loads(answer_text)
            # Final answers use row clustering (rows top-to-bottom, then
            # left-to-right), which is stricter for detection tasks than a
            # naive global y1/x1 sort. That legacy warning is not an error here.
            answer_errors = [
                error
                for error in validate_answer(answer)
                if error != "bbox_order_not_y1_x1"
            ]
            if answer_errors:
                errors.append(
                    {"index": index, "code": "A005", "reason": answer_errors}
                )
            source_answer = source_by_image.get(image)
            answer_pairs = Counter(
                (tuple(box), label)
                for box, label in zip(answer["bbox_2d"], answer["labels"])
            )
            source_pairs = Counter(
                (tuple(box), label)
                for box, label in zip(
                    source_answer["bbox_2d"], source_answer["labels"]
                )
            ) if source_answer else Counter()
            if answer_pairs != source_pairs:
                errors.append(
                    {"index": index, "code": "A006", "reason": "answer_changed"}
                )
            task, _ = classify_task(sample["conversations"][0]["value"])
            stats[f"task_{task}"] += 1
            if task == "single_bale" and len(answer["bbox_2d"]) != 1:
                errors.append(
                    {
                        "index": index,
                        "code": "A007",
                        "reason": "single_task_target_count_not_one",
                        "image": image,
                    }
                )
            if any((b[2] - b[0]) * (b[3] - b[1]) < 25 for b in answer["bbox_2d"]):
                errors.append(
                    {"index": index, "code": "A008", "reason": "very_tiny_bbox", "image": image}
                )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"index": index, "code": "A000", "reason": str(exc)})
    code_counts = Counter(error["code"] for error in errors)
    hard_error_codes = {"A000", "A001", "A002", "A003", "A004", "A005", "A006"}
    report = {
        "source_samples": len(source),
        "final_samples": len(final),
        "stats": dict(stats),
        "error_count": len(errors),
        "hard_error_count": sum(count for code, count in code_counts.items() if code in hard_error_codes),
        "warning_count": sum(count for code, count in code_counts.items() if code not in hard_error_codes),
        "code_counts": dict(code_counts),
        "errors": errors,
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
