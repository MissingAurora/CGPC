"""Shared helpers for the cotton teacher-CoT data pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


ALLOWED_LABELS = {"front", "side"}
TASK_TYPES = {"all_bales", "selected_rows", "single_bale"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"ensure_ascii": False}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    path.write_text(json.dumps(value, **kwargs), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def classify_task(prompt: str) -> tuple[str, str]:
    match = re.search(r"请定位图中(.+?)，区分正面和侧面", prompt)
    target = match.group(1) if match else "指定棉包"
    if "所有棉包" in prompt:
        return "all_bales", target
    if "两排" in prompt or re.search(r"第.*排棉包", prompt):
        return "selected_rows", target
    return "single_bale", target


def valid_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
        and 0 <= box[0] < box[2] <= 1000
        and 0 <= box[1] < box[3] <= 1000
    )


def validate_answer(answer: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(answer, dict):
        return ["answer_not_object"]
    boxes = answer.get("bbox_2d")
    labels = answer.get("labels")
    if not isinstance(boxes, list):
        errors.append("bbox_2d_not_list")
        boxes = []
    if not isinstance(labels, list):
        errors.append("labels_not_list")
        labels = []
    if len(boxes) != len(labels):
        errors.append("bbox_label_count_mismatch")
    if any(not valid_box(box) for box in boxes):
        errors.append("invalid_or_degenerate_bbox")
    if any(label not in ALLOWED_LABELS for label in labels):
        errors.append("invalid_label")
    if boxes and [(box[1], box[0]) for box in boxes] != sorted(
        (box[1], box[0]) for box in boxes
    ):
        errors.append("bbox_order_not_y1_x1")
    return errors


def canonical_answer(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_2d": answer["bbox_2d"],
        "labels": answer["labels"],
    }


def normalize_image_path(value: str) -> str:
    return value.replace("\\", "/")


def resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def sample_id(split: str, image: str, source_index: int) -> str:
    return f"{split}:{Path(image).stem}:{source_index}"


def safe_name(sample_id_value: str) -> str:
    return sample_id_value.replace(":", "__").replace("/", "_").replace("\\", "_")


def parse_teacher_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError("teacher response must be a JSON object")
    return value


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.S
    )
    return match.group(1) if match else None


def coarse_position(box: list[int]) -> str:
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    horizontal = "左侧" if center_x < 333 else "中部" if center_x < 667 else "右侧"
    vertical = "上部" if center_y < 333 else "中部" if center_y < 667 else "下部"
    return vertical + horizontal


def render_teacher_cot(teacher: dict[str, Any]) -> str:
    lines = [f"1. 任务理解：{teacher['task_understanding']}"]
    scene = teacher.get("scene_analysis")
    if scene:
        lines.append(f"2. 场景分析：{scene}")
    lines.append("3. 目标筛选：")
    for index, step in enumerate(teacher.get("selection_reasoning", []), start=1):
        lines.append(f"   {index}) {step}")
    lines.append("4. 逐目标确认：")
    for obj in teacher.get("objects", []):
        if "position" not in obj and "visual_evidence" not in obj:
            lines.append(
                "   - 目标{index}：bbox={bbox}，标签={label}".format(
                    index=obj.get("index"),
                    bbox=obj.get("bbox_2d"),
                    label=obj.get("label"),
                )
            )
        else:
            lines.append(
                "   - 目标{index}：bbox={bbox}，位置={position}，标签={label}；{evidence}".format(
                    index=obj.get("index"),
                    bbox=obj.get("bbox_2d"),
                    position=obj.get("position", "未说明"),
                    label=obj.get("label"),
                    evidence=obj.get("visual_evidence", "未提供视觉依据。"),
                )
            )
    verification = teacher.get("verification", {})
    lines.append(
        "5. 自检：目标数={target_count}，front={front_count}，side={side_count}。".format(
            target_count=verification.get("target_count"),
            front_count=verification.get("front_count"),
            side_count=verification.get("side_count"),
        )
    )
    return "\n".join(lines)
