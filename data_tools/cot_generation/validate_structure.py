"""Apply exact, programmatic constraints to raw teacher-CoT responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cot_common import (
    canonical_answer,
    parse_teacher_json,
    read_jsonl,
    safe_name,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="小批量试跑时跳过raw目录中尚未生成的样本，不将其计为失败",
    )
    return parser.parse_args()


def validate_teacher(sample: dict[str, Any], teacher: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "task_understanding",
        "scene_analysis",
        "selection_reasoning",
        "objects",
        "verification",
        "final_answer",
    }
    missing = sorted(required - set(teacher))
    if missing:
        errors.append("E007:missing_fields:" + ",".join(missing))
        return errors
    truth = sample["ground_truth"]
    try:
        teacher_answer = canonical_answer(teacher["final_answer"])
    except (KeyError, TypeError):
        errors.append("E002:invalid_final_answer")
        teacher_answer = None
    if teacher_answer != truth:
        errors.append("E002:final_answer_mismatch")
    objects = teacher.get("objects")
    if not isinstance(objects, list):
        errors.append("E003:objects_not_list")
        objects = []
    if len(objects) != sample["target_count"]:
        errors.append("E003:object_count_mismatch")
    for index, (box, label) in enumerate(
        zip(truth["bbox_2d"], truth["labels"]), start=1
    ):
        if index > len(objects):
            break
        obj = objects[index - 1]
        if not isinstance(obj, dict):
            errors.append(f"E007:object_{index}_not_object")
            continue
        if obj.get("index") != index:
            errors.append(f"E003:object_{index}_index_mismatch")
        if obj.get("bbox_2d") != box:
            errors.append(f"E004:object_{index}_bbox_mismatch")
        if obj.get("label") != label:
            errors.append(f"E005:object_{index}_label_mismatch")
    verification = teacher.get("verification")
    if not isinstance(verification, dict):
        errors.append("E007:verification_not_object")
    else:
        expected = {
            "target_count": sample["target_count"],
            "front_count": sample["front_count"],
            "side_count": sample["side_count"],
        }
        for key, value in expected.items():
            if verification.get(key) != value:
                errors.append(f"E006:{key}_mismatch")
    return errors


def main() -> None:
    args = parse_args()
    accepted = []
    rejected = []
    missing_skipped = 0
    error_counts: Counter[str] = Counter()
    for sample in read_jsonl(args.manifest):
        raw_path = args.raw_dir / f"{safe_name(sample['id'])}.json"
        errors: list[str] = []
        raw = None
        teacher = None
        if not raw_path.is_file():
            if args.ignore_missing:
                missing_skipped += 1
                continue
            errors.append("E009:raw_response_missing")
        else:
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                if raw.get("status") != "success":
                    errors.append("E009:generation_failed")
                else:
                    teacher = parse_teacher_json(raw["response_text"])
                    errors.extend(validate_teacher(sample, teacher))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"E001:parse_error:{exc}")
        if errors:
            rejected.append({"sample": sample, "errors": errors, "raw_path": str(raw_path)})
            error_counts.update(error.split(":", 1)[0] for error in errors)
        else:
            accepted.append(
                {
                    "sample": sample,
                    "teacher": teacher,
                    "cot_source": "mock" if raw.get("mode") == "mock" else "teacher_vlm",
                    "teacher_model": raw.get("teacher_model"),
                    "prompt_version": raw.get("prompt_version"),
                    "attempt": raw.get("attempt"),
                }
            )
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    report = {
        "manifest_count": len(accepted) + len(rejected) + missing_skipped,
        "evaluated_count": len(accepted) + len(rejected),
        "missing_skipped": missing_skipped,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "pass_rate": len(accepted) / (len(accepted) + len(rejected)) if accepted or rejected else 0,
        "error_counts": dict(error_counts),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
