"""Independently audit the detailed Qwen CoT exports against their sources."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OBJECT_RE = re.compile(
    r"^\s*-\s*目标(?P<index>\d+)：bbox="
    r"(?P<bbox>\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])，"
    r"标签=(?P<label>front|side)\s*$"
)
VERIFY_RE = re.compile(
    r"4\.\s*自检：目标数=(?P<total>\d+)，front=(?P<front>\d+)，side=(?P<side>\d+)。"
)
STAGES = ("1. 场景分析：", "2. 筛选与排序：", "3. 逐目标确认：", "4. 自检：")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("qwen_data/export_audit_report.json"))
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tag(text: str, name: str) -> str | None:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.S)
    return match.group(1) if match else None


def core(prompt: str) -> str:
    return prompt.split("按以下格式输出:", 1)[0].split("\nbbox采用", 1)[0].strip()


def task_type(prompt: str) -> str:
    if "所有棉包" in prompt:
        return "all_bales"
    if "两排" in prompt or re.search(r"第.*排棉包", prompt):
        return "selected_rows"
    return "single_bale"


def canonical(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for sample in load(path):
        image = sample["images"][0].replace("\\", "/")
        prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        answer_text = tag(response, "answer")
        think = tag(response, "think")
        if answer_text is None or think is None:
            raise ValueError(f"invalid canonical sample: {image}")
        result[(image, core(prompt))] = {
            "answer": json.loads(answer_text),
            "think": think,
        }
    return result


def audit(
    path: Path,
    source: dict[tuple[str, str], dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    records = load(path)
    errors: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    tasks: Counter[str] = Counter()
    target_counts: Counter[int] = Counter()
    long_responses: Counter[str] = Counter()
    seen: set[str] = set()
    filter_sections: set[str] = set()

    def fail(code: str, ref: str) -> None:
        errors[code] += 1
        if len(examples[code]) < 5:
            examples[code].append(ref)

    for index, sample in enumerate(records):
        ref = f"{path.name}@{index}"
        fingerprint = json.dumps(sample, ensure_ascii=False, sort_keys=True)
        if fingerprint in seen:
            fail("duplicate_sample", ref)
        seen.add(fingerprint)
        if set(sample) != {"images", "conversations"}:
            fail("root_schema", ref)
            continue
        images = sample.get("images", [])
        image = images[0] if isinstance(images, list) and len(images) == 1 else None
        conversations = sample.get("conversations")
        if not isinstance(image, str) or not isinstance(conversations, list) or len(conversations) != 2:
            fail("conversation_schema", ref)
            continue
        ref = f"{path.name}@{index}:{image}"
        if [turn.get("from") for turn in conversations] != ["human", "gpt"]:
            fail("roles", ref)
            continue
        prompt = conversations[0].get("value", "")
        response = conversations[1].get("value", "")
        if prompt.count("<image>") != 1 or "<image>" in response:
            fail("image_token", ref)
        if not (root / image).is_file():
            fail("missing_image", ref)
        if (
            "按以下格式输出:" in prompt
            or '"front"/"side"' in prompt
            or "仅含bbox_2d和labels的JSON" not in prompt
            or "仅保留与本图相关的场景、布局、排序和类别依据" not in prompt
        ):
            fail("prompt_contract", ref)
        if any(value in response for value in ("<answer>", "</answer>", "<CONCLUSION>")):
            fail("custom_answer_tag", ref)
        match = re.fullmatch(r"<think>\n(.+?)\n</think>\n\n(\{.*\})", response, re.S)
        if not match:
            fail("response_layout", ref)
            continue
        think, answer_text = match.groups()
        try:
            answer = json.loads(answer_text)
        except json.JSONDecodeError:
            fail("answer_json", ref)
            continue
        expected = source.get((image.replace("\\", "/"), core(prompt)))
        if expected is None:
            fail("missing_canonical_match", ref)
        elif answer != expected["answer"]:
            fail("answer_changed", ref)
        positions = [think.find(stage) for stage in STAGES]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            fail("five_stage_layout", ref)
        filter_match = re.search(
            r"2\.\s*筛选与排序：\s*(.*?)\n3\.\s*逐目标确认：", think, re.S
        )
        if not filter_match:
            fail("missing_target_filter", ref)
        else:
            filter_text = filter_match.group(1).strip()
            filter_sections.add(filter_text)
            if (
                "布局与顺序：x=" not in filter_text
                or "类别：" not in filter_text
            ):
                fail("target_filter_not_sample_grounded", ref)
            if any(
                legacy in filter_text
                for legacy in (
                    "扫描完整画面并确定所有可见棉包区域",
                    "按题意确定目标排并计算排内序号",
                    "从堆垛顶部开始确定题目指定的最上方各排",
                    "以堆垛顶部为起点自上向下计数并选取题目指定排",
                    "以地面为起点自下向上计数并选取题目指定排",
                    "题目要求覆盖全图，因此保留每个具有独立边界的可见包体",
                    "遮挡处沿可见接缝延伸，避免把相邻棉包合并",
                )
            ):
                fail("legacy_filter_template", ref)
        section = re.search(r"3\.\s*逐目标确认：\s*\n(.*?)\n4\.\s*自检：", think, re.S)
        objects: list[tuple[list[int], str]] = []
        if not section:
            fail("missing_object_confirmation", ref)
        else:
            lines = [line for line in section.group(1).splitlines() if line.strip()]
            for line in lines:
                object_match = OBJECT_RE.match(line)
                if not object_match:
                    fail("object_line_format", ref)
                    break
                objects.append((json.loads(object_match.group("bbox")), object_match.group("label")))
        boxes = answer.get("bbox_2d", []) if isinstance(answer, dict) else []
        labels = answer.get("labels", []) if isinstance(answer, dict) else []
        if objects != list(zip(boxes, labels)):
            fail("objects_answer_mismatch", ref)
        verification = VERIFY_RE.search(think)
        expected_counts = (len(boxes), labels.count("front"), labels.count("side"))
        if not verification or tuple(
            int(verification.group(name)) for name in ("total", "front", "side")
        ) != expected_counts:
            fail("self_check_mismatch", ref)
        kind = task_type(prompt)
        tasks[kind] += 1
        target_counts[len(boxes)] += 1
        if "任务理解：" in think or "问题概括：" in think:
            fail("task_restatement", ref)
        if len(response) > 4096:
            long_responses["characters_over_4096"] += 1
        if len(response) > 8192:
            long_responses["characters_over_8192"] += 1
    return {
        "file": str(path),
        "samples": len(records),
        "task_counts": dict(tasks),
        "target_count_min": min(target_counts) if target_counts else 0,
        "target_count_max": max(target_counts) if target_counts else 0,
        "length_warnings": dict(long_responses),
        "unique_target_filter_sections": len(filter_sections),
        "target_filter_uniqueness_rate": round(
            len(filter_sections) / len(records), 6
        ) if records else 0,
        "error_count": sum(errors.values()),
        "error_counts": dict(errors),
        "error_examples": dict(examples),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_source = canonical(root / "new_train_cot_final.json")
    valid_source = canonical(root / "new_valid_cot_final.json")
    reports = [
        audit(root / "qwen_data/cotton_cot_train.json", train_source, root),
        audit(root / "qwen_data/cotton_cot_valid.json", valid_source, root),
    ]
    raw_checks = {}
    for name in ("cotton_cot_train.json", "cotton_cot_valid.json"):
        raw = (root / "qwen_data" / name).read_text(encoding="utf-8")
        raw_checks[name] = {
            "pretty_printed": raw.startswith("[\n  {"),
            "literal_slash_n_absent": "/n" not in raw,
            "pseudo_json_absent": '"front"/"side"' not in raw,
        }
    report = {
        "canonical_train": len(train_source),
        "canonical_valid": len(valid_source),
        "audited_samples": sum(item["samples"] for item in reports),
        "total_errors": sum(item["error_count"] for item in reports),
        "raw_file_checks": raw_checks,
        "files": reports,
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["total_errors"] else 0)


if __name__ == "__main__":
    main()
