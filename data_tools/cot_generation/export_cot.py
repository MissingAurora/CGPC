"""Export the audited five-stage teacher CoT in Qwen3-VL training format.

This variant keeps the teacher's image-specific scene description and detailed
per-object confirmation.  The source teacher used only five fixed target-filter
templates, so that section is replaced with a sample-grounded explanation built
from the task, the scene cues, the actual spatial distribution, and class
positions.  Transport/format issues are fixed at the same time.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {"front", "side"}
OBJECT_RE = re.compile(
    r"^\s*-\s*目标(?P<index>\d+)：bbox="
    r"(?P<bbox>\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])，"
    r"标签=(?P<label>front|side)\s*$"
)
VERIFY_RE = re.compile(
    r"4\.\s*自检：目标数=(?P<total>\d+)，front=(?P<front>\d+)，side=(?P<side>\d+)。"
)
STAGES = ("1. 场景分析：", "2. 筛选与排序：", "3. 逐目标确认：", "4. 自检：")
SOURCE_SCENE_RE = re.compile(r"2\.\s*场景分析：(.*?)\n3\.\s*目标筛选：", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_data"))
    parser.add_argument("--train-input", type=Path, default=Path("new_train_cot_final.json"))
    parser.add_argument("--valid-input", type=Path, default=Path("new_valid_cot_final.json"))
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name in ("output_dir", "train_input", "valid_input"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.root / value)
    return args


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    return match.group(1) if match else None


def base_question(prompt: str) -> str:
    core = prompt.split("按以下格式输出:", 1)[0].strip()
    if core.count("<image>") != 1:
        raise ValueError("question must contain exactly one <image> tag")
    return core


def detailed_question(prompt: str) -> str:
    return (
        base_question(prompt)
        + "\nbbox采用[x1,y1,x2,y2]，坐标归一化至0—1000。"
        "请在<think>中仅保留与本图相关的场景、布局、排序和类别依据，勿复述题目；"
        "逐目标确认并自检后，输出仅含bbox_2d和labels的JSON，labels取值为front或side。"
    )


def classify_task(prompt: str) -> tuple[str, str]:
    match = re.search(r"请定位图中(.+?)，区分正面和侧面", prompt)
    target = match.group(1) if match else "指定棉包"
    if "所有棉包" in prompt:
        return "all_bales", target
    if "两排" in prompt or re.search(r"第.*排棉包", prompt):
        return "selected_rows", target
    return "single_bale", target


def valid_answer(answer: Any) -> bool:
    if not isinstance(answer, dict) or set(answer) != {"bbox_2d", "labels"}:
        return False
    boxes, labels = answer["bbox_2d"], answer["labels"]
    return (
        isinstance(boxes, list)
        and isinstance(labels, list)
        and len(boxes) == len(labels)
        and bool(boxes)
        and all(label in ALLOWED_LABELS for label in labels)
        and all(
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
            and 0 <= box[0] < box[2] <= 1000
            and 0 <= box[1] < box[3] <= 1000
            for box in boxes
        )
    )


def compact_indices(indices: list[int]) -> str:
    """Render consecutive target numbers without producing a long repeated list."""
    if not indices:
        return "无"
    ranges: list[str] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"目标{start}" if start == previous else f"目标{start}–{previous}")
        start = previous = value
    ranges.append(f"目标{start}" if start == previous else f"目标{start}–{previous}")
    return "、".join(ranges)


def position_name(box: list[int]) -> str:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    horizontal = "left" if center_x < 333 else "right" if center_x > 667 else "center"
    vertical = "top" if center_y < 333 else "bottom" if center_y > 667 else "middle"
    names = {
        ("left", "top"): "左上区域",
        ("center", "top"): "上部中央",
        ("right", "top"): "右上区域",
        ("left", "middle"): "中部左侧",
        ("center", "middle"): "画面中央",
        ("right", "middle"): "中部右侧",
        ("left", "bottom"): "左下区域",
        ("center", "bottom"): "下部中央",
        ("right", "bottom"): "右下区域",
    }
    return names[(horizontal, vertical)]


def scene_cues(scene: str) -> str:
    keywords = (
        "蓝色", "白色", "米白色", "灰色", "黑色", "透明塑料膜", "塑料膜",
        "编织袋", "绳带", "标签", "纸张", "接缝", "阴影", "遮挡", "地面", "砖块",
    )
    found = [keyword for keyword in keywords if keyword in scene]
    return "、".join(found[:6]) if found else "矩形轮廓、包体边缘与相邻实例间隙"


def spatial_description(boxes: list[list[int]]) -> str:
    groups: dict[str, list[int]] = {}
    positions: list[str] = []
    for index, box in enumerate(boxes, start=1):
        position = position_name(box)
        positions.append(position)
        groups.setdefault(position, []).append(index)
    group_parts = []
    for position, indices in groups.items():
        references = compact_indices(indices)
        group_parts.append(f"{position}{len(indices)}个（{references}）")
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    if len(boxes) <= 16:
        trace = " → ".join(
            f"目标{index}（{position}）" for index, position in enumerate(positions, start=1)
        )
    else:
        trace = "、".join(group_parts)
    return f"布局与顺序：x={x1}–{x2}、y={y1}–{y2}；{trace}"


def class_description(boxes: list[list[int]], labels: list[str]) -> str:
    parts: list[str] = []
    for label in ("front", "side"):
        indices = [index for index, value in enumerate(labels, start=1) if value == label]
        if not indices:
            continue
        evidence = "短端面" if label == "front" else "长侧面"
        parts.append(f"{label}={compact_indices(indices)}（{evidence}）")
    return "；".join(parts)


def grounded_filter(
    task_type: str,
    target: str,
    scene: str,
    answer: dict[str, Any],
) -> str:
    boxes = answer["bbox_2d"]
    labels = answer["labels"]
    cues = scene_cues(scene)
    distribution = spatial_description(boxes)
    lines = ["2. 筛选与排序："]
    if task_type != "all_bales":
        lines.append(f"   - 范围：{target}")
    lines.append(f"   - 实例边界线索：{cues}")
    lines.append(f"   - {distribution}")
    lines.append(f"   - 类别：{class_description(boxes, labels)}")
    return "\n".join(lines)


def normalize_reasoning(
    think: str,
    task_type: str,
    target: str,
    answer: dict[str, Any],
) -> tuple[str, bool]:
    """Keep the unique scene and rebuild the rest without task boilerplate."""
    think = think.strip().replace("\r\n", "\n").replace("\r", "\n")
    scene_match = SOURCE_SCENE_RE.search(think)
    if not scene_match:
        raise ValueError("cannot extract scene analysis")
    scene = scene_match.group(1).strip()
    object_lines = "\n".join(
        f"   - 目标{index}：bbox={compact_json(box)}，标签={label}"
        for index, (box, label) in enumerate(
            zip(answer["bbox_2d"], answer["labels"]), start=1
        )
    )
    reasoning = (
        f"1. 场景分析：{scene}\n"
        f"{grounded_filter(task_type, target, scene, answer)}\n"
        f"3. 逐目标确认：\n{object_lines}\n"
        f"4. 自检：目标数={len(answer['bbox_2d'])}，"
        f"front={answer['labels'].count('front')}，"
        f"side={answer['labels'].count('side')}。"
    )
    return reasoning, True


def validate_reasoning(think: str, answer: dict[str, Any]) -> None:
    positions = [think.find(stage) for stage in STAGES]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError("missing or disordered five-stage reasoning")
    section = re.search(r"3\.\s*逐目标确认：\s*\n(.*?)\n4\.\s*自检：", think, re.S)
    if not section:
        raise ValueError("missing per-object confirmation section")
    objects: list[tuple[list[int], str]] = []
    nonempty = [line for line in section.group(1).splitlines() if line.strip()]
    for line in nonempty:
        match = OBJECT_RE.match(line)
        if not match:
            raise ValueError(f"invalid object confirmation line: {line}")
        objects.append((json.loads(match.group("bbox")), match.group("label")))
    expected = list(zip(answer["bbox_2d"], answer["labels"]))
    if objects != expected:
        raise ValueError("per-object confirmation differs from final JSON")
    verification = VERIFY_RE.search(think)
    expected_counts = (
        len(answer["bbox_2d"]),
        answer["labels"].count("front"),
        answer["labels"].count("side"),
    )
    if not verification or tuple(
        int(verification.group(name)) for name in ("total", "front", "side")
    ) != expected_counts:
        raise ValueError("self-check differs from final JSON")


def convert(path: Path, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    task_restatements_removed = 0
    reasoning_lengths: list[int] = []
    for index, sample in enumerate(load_json(path)):
        try:
            images = sample.get("images")
            conversations = sample.get("conversations")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError("expected exactly one image")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise ValueError("expected one human and one gpt turn")
            image = str(images[0]).replace("\\", "/")
            if not (root / image).is_file():
                raise ValueError(f"missing image: {image}")
            prompt = conversations[0].get("value", "")
            response = conversations[1].get("value", "")
            think = extract_tag(response, "think")
            answer_text = extract_tag(response, "answer")
            if think is None or answer_text is None:
                raise ValueError("source must contain think and answer tags")
            answer = json.loads(answer_text)
            if not valid_answer(answer):
                raise ValueError("invalid final answer")
            task_type, target = classify_task(prompt)
            think, changed = normalize_reasoning(think, task_type, target, answer)
            validate_reasoning(think, answer)
            task_counts[task_type] += 1
            task_restatements_removed += int(changed)
            reasoning_lengths.append(len(think))
            output.append(
                {
                    "images": [image],
                    "conversations": [
                        {"from": "human", "value": detailed_question(prompt)},
                        {
                            "from": "gpt",
                            "value": f"<think>\n{think}\n</think>\n\n{compact_json(answer)}",
                        },
                    ],
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}@{index}: {exc}") from exc
    stats = {
        "source": str(path),
        "samples": len(output),
        "task_counts": dict(task_counts),
        "task_restatements_removed": task_restatements_removed,
        "target_filters_rewritten_with_sample_geometry": len(output),
        "average_reasoning_characters": round(sum(reasoning_lengths) / len(reasoning_lengths), 2),
        "maximum_reasoning_characters": max(reasoning_lengths),
    }
    return output, stats


def main() -> None:
    args = parse_args()
    train, train_stats = convert(args.train_input, args.root)
    valid, valid_stats = convert(args.valid_input, args.root)
    train_path = args.output_dir / "cotton_cot_train.json"
    valid_path = args.output_dir / "cotton_cot_valid.json"
    report_path = args.output_dir / "dataset_report.json"
    write_json(train_path, train)
    write_json(valid_path, valid)
    report = {
        "profile": "audited five-stage teacher CoT in Qwen3-VL format",
        "reasoning_preserved": [
            "scene analysis",
            "sample-specific layout, ordering, and class evidence",
            "per-object confirmation",
            "self-check",
        ],
        "format_fixes": [
            "ShareGPT images list for LLaMA-Factory registration",
            "valid textual output contract instead of pseudo JSON",
            "native think block followed directly by final JSON",
            "custom answer tags removed",
            "task restatement removed",
            "generic full-image and boundary boilerplate removed",
            "five repeated source filtering templates replaced with sample-grounded geometry",
            "pretty-printed JSON",
        ],
        "train": train_stats,
        "valid": valid_stats,
        "outputs": {
            "default_train": str(train_path),
            "default_valid": str(valid_path),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
