"""Conservatively sanitize teacher rationale text without touching answers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from cot_common import compact_json, parse_teacher_json, read_jsonl, safe_name, write_json


META_PATTERN = re.compile(
    r"人工标注|ground[_ ]?truth|answer_boxes_and_labels|参考答案|给定答案|"
    r"已知答案|输入框|目标框已给出|与(?:答案|标注)一致|数量与输入一致|"
    r"答案|被选中|被标注|结合标注|当前标注|明确标注|实际要求|"
    r"输入(?:仅|只)?(?:提供|包含)|目标框",
    re.I,
)
SURFACE_PATTERN = re.compile(
    r"手写|编号|字样|品牌|标识|印有|写有|文字|日期|标签纸|"
    r"(?:红色|黄色|表面|贴有)标签|污渍|磨损|褶皱|印刷内容|材质纹理|"
    r"绑带|捆扎带|包装条带|二维码"
)
COUNT_PATTERN = re.compile(
    r"共\s*[0-9一二两三四五六七八九十百]+\s*(?:个|包|块|实例|目标)|"
    r"各(?:含|有)\s*[0-9一二两三四五六七八九十百]+\s*(?:个|包|块|实例|目标)|"
    r"[0-9一二两三四五六七八九十百]+列[0-9一二两三四五六七八九十百]+(?:行|排|层)|"
    r"第[0-9一二两三四五六七八九十百]+(?:个)?至第[0-9一二两三四五六七八九十百]+个|"
    r"[左右上下]侧\s*[0-9一二两三四五六七八九十百]+\s*(?:个|包|块|实例|目标)|"
    r"(?<!第)(?:约|可见|呈|共)?\s*[0-9一二两三四五六七八九十百]+\s*(?:层|行|排)(?:排列|堆叠|结构)?|"
    r"(?<!第)[0-9一二两三四五六七八九十百]+\s*(?:个|包|块|实例|目标|层|行|排|列)"
)
ORIENTATION_PATTERN = re.compile(
    r"\bfront\b|\bside\b|正面|侧面|端面|朝向|判定为|判断为",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def forbidden(text: str) -> bool:
    return bool(
        META_PATTERN.search(text)
        or SURFACE_PATTERN.search(text)
        or COUNT_PATTERN.search(text)
        or ORIENTATION_PATTERN.search(text)
    )


def clean_text(text: Any) -> tuple[str, int]:
    if not isinstance(text, str):
        return "", 0
    clauses = [part.strip() for part in re.split(r"[，,；;。]", text) if part.strip()]
    kept = [part for part in clauses if not forbidden(part)]
    removed = len(clauses) - len(kept)
    return "，".join(kept), removed


def fallback_steps(sample: dict[str, Any]) -> list[str]:
    if sample["task_type"] == "all_bales":
        return [
            "扫描完整画面并确定所有可见棉包区域",
            "依据矩形轮廓、实例接缝和遮挡边界区分相邻棉包",
            "按从上到下、同排从左到右的顺序整理目标",
            "依据较短端面为front、较长侧面为side的规则确定类别",
        ]
    if sample["task_type"] == "selected_rows":
        target = sample.get("target", "")
        if "从上往下" in target:
            row_step = "以堆垛顶部为起点自上向下计数并选取题目指定排"
        elif "从下往上" in target:
            row_step = "以地面为起点自下向上计数并选取题目指定排"
        elif "最上面" in target:
            row_step = "从堆垛顶部开始确定题目指定的最上方各排"
        elif "最下面" in target:
            row_step = "从地面开始确定题目指定的最下方各排"
        else:
            row_step = "按照题目规定的计数方向选取指定排"
        return [
            "确定画面中距离观察者最近的棉包堆垛",
            row_step,
            "依据轮廓与接缝区分该排内相邻实例并按规定顺序整理",
            "依据较短端面为front、较长侧面为side的规则确定类别",
        ]
    return [
        "确定画面中距离观察者最近的棉包堆垛",
        "按题意确定目标排并计算排内序号",
        "只保留对应的唯一棉包实例",
        "依据较短端面为front、较长侧面为side的规则确定类别",
    ]


def sanitize_teacher(
    teacher: dict[str, Any], sample: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    cleaned = dict(teacher)
    removed_clauses = 0
    cleaned["task_understanding"] = (
        f"定位{sample['target']}，区分front与side，并按规定顺序输出每个目标的bbox及标签"
    )
    scene, removed = clean_text(cleaned.get("scene_analysis"))
    removed_clauses += removed
    cleaned["scene_analysis"] = scene
    if len(cleaned.get("scene_analysis", "")) < 6:
        cleaned["scene_analysis"] = (
            "画面中可见棉包按层堆叠，可依据轮廓、接缝和遮挡关系区分实例"
        )

    cleaned["selection_reasoning"] = fallback_steps(sample)

    compact_objects = []
    removed_object_fields = 0
    for obj in cleaned.get("objects", []):
        if not isinstance(obj, dict):
            compact_objects.append(obj)
            continue
        removed_object_fields += sum(
            key not in {"index", "bbox_2d", "label"} for key in obj
        )
        compact_objects.append(
            {
                "index": obj.get("index"),
                "bbox_2d": obj.get("bbox_2d"),
                "label": obj.get("label"),
            }
        )
    truth = sample["ground_truth"]
    canonical_objects = [
        {"index": index, "bbox_2d": box, "label": label}
        for index, (box, label) in enumerate(
            zip(truth["bbox_2d"], truth["labels"]), start=1
        )
    ]
    corrected_objects = int(compact_objects != canonical_objects)
    corrected_final_answer = int(cleaned.get("final_answer") != truth)
    # The source annotation is authoritative. The teacher supplies the visual
    # rationale, while coordinates and labels are injected deterministically.
    cleaned["objects"] = canonical_objects
    cleaned["final_answer"] = truth
    expected_verification = {
        "target_count": sample["target_count"],
        "front_count": sample["front_count"],
        "side_count": sample["side_count"],
    }
    corrected_verification = int(cleaned.get("verification") != expected_verification)
    cleaned["verification"] = expected_verification
    return cleaned, {
        "removed_clauses": removed_clauses,
        "removed_object_fields": removed_object_fields,
        "corrected_verification": corrected_verification,
        "corrected_objects": corrected_objects,
        "corrected_final_answer": corrected_final_answer,
        "reasoning_template_applied": 1,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    changed = 0
    failed = 0
    removed_clauses = 0
    removed_object_fields = 0
    corrected_verification = 0
    corrected_objects = 0
    corrected_final_answer = 0
    reasoning_template_applied = 0
    for sample in read_jsonl(args.manifest):
        raw_path = args.raw_dir / f"{safe_name(sample['id'])}.json"
        if not raw_path.is_file():
            continue
        try:
            wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
            if wrapper.get("status") != "success":
                continue
            original_text = wrapper["response_text"]
            teacher = parse_teacher_json(original_text)
            cleaned, stats = sanitize_teacher(teacher, sample)
            cleaned_text = compact_json(cleaned)
            processed += 1
            removed_clauses += stats["removed_clauses"]
            removed_object_fields += stats["removed_object_fields"]
            corrected_verification += stats["corrected_verification"]
            corrected_objects += stats["corrected_objects"]
            corrected_final_answer += stats["corrected_final_answer"]
            reasoning_template_applied += stats["reasoning_template_applied"]
            if cleaned_text != compact_json(teacher):
                changed += 1
                wrapper["response_text_raw"] = original_text
            wrapper["response_text"] = cleaned_text
            wrapper["sanitization"] = stats
            write_json(args.output_dir / raw_path.name, wrapper)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failed += 1
            write_json(
                args.output_dir / raw_path.name,
                {
                    "id": sample["id"],
                    "status": "failed",
                    "error": f"sanitize_error:{type(exc).__name__}:{exc}",
                },
            )
    report = {
        "processed": processed,
        "changed": changed,
        "failed": failed,
        "removed_clauses": removed_clauses,
        "removed_object_fields": removed_object_fields,
        "corrected_verification": corrected_verification,
        "corrected_objects": corrected_objects,
        "corrected_final_answer": corrected_final_answer,
        "reasoning_template_applied": reasoning_template_applied,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
