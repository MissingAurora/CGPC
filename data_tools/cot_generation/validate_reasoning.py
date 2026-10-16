"""Apply conservative text-level consistency checks after hard validation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from cot_common import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def position_conflicts(position: str, box: list[int]) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    left_absolute = re.search(
        r"画面(?:左侧|左部|左上|左下)|(?:上|中|下)部偏左|偏左(?:位置|区域)", position
    )
    right_absolute = re.search(
        r"画面(?:右侧|右部|右上|右下)|(?:上|中|下)部偏右|偏右(?:位置|区域)", position
    )
    upper_absolute = re.search(
        r"画面(?:上侧|上部|左上|中上|右上)|(?:左|中|右)上部|偏上(?:位置|区域)", position
    )
    lower_absolute = re.search(
        r"画面(?:下侧|下部|左下|中下|右下)|(?:左|中|右)下部|偏下(?:位置|区域)", position
    )
    if left_absolute and center_x > 667:
        return True
    if right_absolute and center_x < 333:
        return True
    if upper_absolute and center_y > 667:
        return True
    if lower_absolute and center_y < 333:
        return True
    return False


def semantic_errors(record: dict) -> list[str]:
    sample = record["sample"]
    teacher = record["teacher"]
    errors: list[str] = []
    understanding = teacher.get("task_understanding", "")
    scene = teacher.get("scene_analysis", "")
    steps = teacher.get("selection_reasoning", [])
    compact_objects = record.get("prompt_version") == "grounded"
    if not isinstance(understanding, str) or len(understanding.strip()) < 6:
        errors.append("S001:task_understanding_too_short")
    if not isinstance(scene, str) or len(scene.strip()) < 6:
        errors.append("S002:scene_analysis_too_short")
    if not isinstance(steps, list) or len(steps) < 2:
        errors.append("S003:selection_reasoning_insufficient")
    elif any(not isinstance(step, str) or len(step.strip()) < 6 for step in steps):
        errors.append("S003:selection_step_too_short")
    if sample["task_type"] == "selected_rows":
        joined = "".join(steps) if isinstance(steps, list) else ""
        if all(term not in joined for term in ("排", "行", "层")):
            errors.append("S004:selected_rows_missing_row_reasoning")
    if sample["task_type"] == "single_bale" and sample["target_count"] != 1:
        errors.append("S005:single_bale_ground_truth_not_single")
    reasoning_fragments = [understanding, scene]
    if isinstance(steps, list):
        reasoning_fragments.extend(step for step in steps if isinstance(step, str))
    for obj in teacher.get("objects", []):
        if isinstance(obj, dict):
            reasoning_fragments.extend(
                value
                for value in (obj.get("position"), obj.get("visual_evidence"))
                if isinstance(value, str)
            )
    reasoning_text = "\n".join(reasoning_fragments)
    if re.search(
        r"人工标注|ground[_ ]?truth|answer_boxes_and_labels|参考答案|给定答案|"
        r"已知答案|输入框|目标框已给出|与(?:答案|标注)一致|数量与输入一致|"
        r"答案|被选中|被标注|结合标注|当前标注|明确标注|实际要求|"
        r"输入(?:仅|只)?(?:提供|包含)|目标框",
        reasoning_text,
        re.I,
    ):
        errors.append("S008:annotation_leakage")
    if re.search(
        r"手写|编号|字样|品牌|标识|印有|写有|文字|日期|标签纸|"
        r"(?:红色|黄色|表面|贴有)标签|污渍|磨损|褶皱|印刷内容|材质纹理|"
        r"绑带|捆扎带|包装条带|二维码",
        reasoning_text,
    ):
        errors.append("S009:irrelevant_or_speculative_ocr")
    if (
        sample["task_type"] == "all_bales"
        and sample["target_count"] == 1
        and re.search(r"多个|若干|至少|多层|每层|多排|多堆", reasoning_text)
    ):
        errors.append("S011:all_bales_count_scene_conflict")
    if (
        sample["task_type"] == "selected_rows"
        and "两排" in sample.get("target", "")
        and sample["target_count"] < 2
    ):
        errors.append("S011:two_rows_fewer_than_two_targets")
    for index, (obj, box) in enumerate(
        zip(teacher.get("objects", []), sample["ground_truth"]["bbox_2d"]), start=1
    ):
        if compact_objects:
            if isinstance(obj, dict) and (
                "position" in obj or "visual_evidence" in obj
            ):
                errors.append(f"S010:object_{index}_not_compact")
            continue
        position = obj.get("position", "") if isinstance(obj, dict) else ""
        evidence = obj.get("visual_evidence", "") if isinstance(obj, dict) else ""
        if not isinstance(position, str) or len(position.strip()) < 2:
            errors.append(f"S006:object_{index}_position_missing")
        elif position_conflicts(position, box):
            errors.append(f"S006:object_{index}_position_conflict")
        if not isinstance(evidence, str) or len(evidence.strip()) < 8:
            errors.append(f"S007:object_{index}_evidence_too_short")
        if isinstance(evidence, str) and ("因为标签" in evidence or "标签是" in evidence):
            errors.append(f"S007:object_{index}_circular_evidence")
    return errors


def main() -> None:
    args = parse_args()
    accepted = []
    retry = []
    counts: Counter[str] = Counter()
    for record in read_jsonl(args.input):
        errors = semantic_errors(record)
        if errors:
            retry.append({**record, "semantic_errors": errors})
            counts.update(error.split(":", 1)[0] for error in errors)
        else:
            accepted.append(record)
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.retry, retry)
    total = len(accepted) + len(retry)
    report = {
        "input_count": total,
        "accepted": len(accepted),
        "retry": len(retry),
        "pass_rate": len(accepted) / total if total else 0,
        "error_counts": dict(counts),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
