"""Build validated teacher-request manifests from the original JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cot_common import (
    canonical_answer,
    classify_task,
    load_json,
    normalize_image_path,
    resolve_path,
    sample_id,
    validate_answer,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train", default="new_train.json")
    parser.add_argument("--valid", default="new_valid.json")
    parser.add_argument("--output-dir", default="teacher_work/manifest")
    return parser.parse_args()


def build_split(
    root: Path, split: str, source_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    source = load_json(source_path)
    manifest: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for source_index, sample in enumerate(source):
        reasons: list[str] = []
        image_hint = None
        try:
            images = sample.get("images")
            if not isinstance(images, list) or len(images) != 1:
                reasons.append("expected_exactly_one_image")
                images = images if isinstance(images, list) else []
            if images:
                image_hint = str(images[0])
            conversations = sample.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                reasons.append("expected_two_conversation_turns")
                raise ValueError("conversation structure invalid")
            if conversations[0].get("from") != "human":
                reasons.append("first_turn_not_human")
            if conversations[1].get("from") != "gpt":
                reasons.append("second_turn_not_gpt")
            question = conversations[0].get("value", "")
            if question.count("<image>") != 1:
                reasons.append("question_image_tag_count_not_one")
            try:
                answer = canonical_answer(json.loads(conversations[1]["value"]))
                reasons.extend(validate_answer(answer))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                reasons.append(f"answer_parse_error:{exc}")
                answer = {"bbox_2d": [], "labels": []}
            if image_hint:
                image = normalize_image_path(image_hint)
                if not (root / image).is_file():
                    reasons.append("image_missing")
            else:
                image = ""
            item_id = sample_id(split, image, source_index)
            if item_id in seen_ids:
                reasons.append("duplicate_id")
            seen_ids.add(item_id)
            if reasons:
                raise ValueError(";".join(reasons))

            task_type, target = classify_task(question)
            labels = answer["labels"]
            record = {
                "id": item_id,
                "source_index": source_index,
                "split": split,
                "task_type": task_type,
                "target": target,
                "image": image,
                "question": question,
                "ground_truth": answer,
                "target_count": len(answer["bbox_2d"]),
                "front_count": labels.count("front"),
                "side_count": labels.count("side"),
                "status": "pending",
            }
            manifest.append(record)
            counts[task_type] += 1
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(
                {
                    "split": split,
                    "source_index": source_index,
                    "image": image_hint,
                    "reasons": str(exc).split(";"),
                }
            )
    return manifest, invalid, counts


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = resolve_path(root, args.output_dir)
    train, train_invalid, train_counts = build_split(
        root, "train", resolve_path(root, args.train)
    )
    valid, valid_invalid, valid_counts = build_split(
        root, "valid", resolve_path(root, args.valid)
    )
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)
    invalid = train_invalid + valid_invalid
    report = {
        "train_valid": len(train),
        "valid_valid": len(valid),
        "invalid_count": len(invalid),
        "train_tasks": dict(train_counts),
        "valid_tasks": dict(valid_counts),
        "invalid": invalid,
    }
    write_json(output_dir.parent / "reports" / "manifest_report.json", report)
    write_json(output_dir.parent / "reports" / "invalid_source_samples.json", invalid)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
