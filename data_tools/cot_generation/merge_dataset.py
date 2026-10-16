"""Merge accepted teacher CoT with rule fallback and authoritative answers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cot_common import (
    compact_json,
    extract_tag,
    load_json,
    normalize_image_path,
    read_jsonl,
    render_teacher_cot,
    sample_id,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--accepted", type=Path, default=None)
    parser.add_argument("--rule-cot", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--exclude-images",
        type=Path,
        default=None,
        help="可选的JSON图片路径列表；匹配样本不会写入最终数据",
    )
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="仅写入accepted文件中存在的样本，不使用规则CoT回退",
    )
    args = parser.parse_args()
    if not args.require_accepted and args.rule_cot is None:
        parser.error("Supply --require-accepted or an explicit --rule-cot file")
    if args.require_accepted and (args.accepted is None or not args.accepted.is_file()):
        parser.error("--require-accepted requires an existing --accepted file")
    return args


def main() -> None:
    args = parse_args()
    excluded_images = set()
    if args.exclude_images and args.exclude_images.is_file():
        excluded_images = {
            normalize_image_path(str(image)) for image in load_json(args.exclude_images)
        }
    accepted_by_id = {}
    if args.accepted and args.accepted.is_file():
        for record in read_jsonl(args.accepted):
            if record.get("cot_source") == "mock" and not args.allow_mock:
                continue
            accepted_by_id[record["sample"]["id"]] = record
    rule_by_image = {}
    for sample in (load_json(args.rule_cot) if args.rule_cot else []):
        image = normalize_image_path(sample["images"][0])
        cot = extract_tag(sample["conversations"][1]["value"], "think")
        if cot:
            rule_by_image[image] = cot
    output = []
    skipped = []
    sources: Counter[str] = Counter()
    for source_index, sample in enumerate(load_json(args.source)):
        image_raw = sample.get("images", [None])[0]
        if image_raw is None:
            skipped.append({"source_index": source_index, "reason": "missing_image"})
            continue
        image = normalize_image_path(image_raw)
        if image in excluded_images:
            skipped.append(
                {
                    "source_index": source_index,
                    "image": image,
                    "reason": "explicitly_excluded",
                }
            )
            continue
        item_id = sample_id(args.split, image, source_index)
        accepted = accepted_by_id.get(item_id)
        if args.require_accepted and not accepted:
            skipped.append(
                {
                    "source_index": source_index,
                    "image": image,
                    "reason": "not_in_accepted",
                }
            )
            continue
        if accepted:
            cot_text = render_teacher_cot(accepted["teacher"])
            cot_source = accepted.get("cot_source", "teacher_vlm")
            answer = accepted["teacher"]["final_answer"]
        else:
            cot_text = rule_by_image.get(image)
            cot_source = "rule"
        if not cot_text:
            skipped.append(
                {"source_index": source_index, "image": image, "reason": "no_valid_cot"}
            )
            continue
        if not accepted:
            answer = json.loads(sample["conversations"][1]["value"])
        assistant = (
            f"<think>\n{cot_text}\n</think>\n"
            f"<answer>\n{compact_json(answer)}\n</answer>"
        )
        output.append(
            {
                "conversations": [
                    sample["conversations"][0],
                    {"from": "gpt", "value": assistant},
                ],
                "images": sample["images"],
            }
        )
        sources[cot_source] += 1
    write_json(args.output, output, pretty=False)
    report = {
        "source_count": len(output) + len(skipped),
        "written": len(output),
        "skipped": len(skipped),
        "cot_sources": dict(sources),
        "skipped_samples": skipped,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
