"""CGPC release: evaluation/cotton/eval_cot.py. See the README in this directory for configuration."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import os as _release_os

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


# ================== 路径配置（请填写 LORA_PATH） ==================
BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/Qwen3-VL-2B-Instruct')
LORA_PATH = _release_os.environ.get('CGPC_SFT_PATH', './checkpoints/lora')
DATA_JSON = _release_os.environ.get('CGPC_EVAL_JSON', './data/valid.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(_release_os.environ.get('CGPC_OUTPUT_DIR', './outputs/cot_evaluation'))

# 与旧评估脚本保持相同的生成上限口径。
TASK_MAX_TOKENS = {"SINGLE": 1024, "ROW": 3072, "ALL": 6144}
CATEGORY_NAMES = {
    "ALL": "检测所有棉包",
    "ROW": "检测指定排",
    "SINGLE": "检测指定排+指定个",
}
VALID_LABELS = {"front", "side"}
IOU_THRESHOLD = 0.5
CN_NUM_PATTERN = r"[零〇一二两三四五六七八九十百千]+"


class Tee:
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = filepath.open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


@dataclass
class ParsedResponse:
    thought: str
    answer: dict[str, Any] | None
    boxes: list[list[float]]
    labels: list[str]
    think_tags_valid: bool
    strict_two_part: bool
    json_valid: bool
    schema_valid: bool
    parse_error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只评估前 N 条；默认评估全部",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="评估前用固定种子打乱；默认保持测试集原顺序",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize-every",
        type=int,
        default=20,
        help="每 N 条保存一张框图；0 表示不保存",
    )
    return parser.parse_args()


def strip_image_marker(prompt: str) -> str:
    return re.sub(r"^\s*<image>\s*", "", prompt, count=1)


def resolve_image_path(dataset_path: str) -> Path:
    relative = Path(dataset_path.replace("\\", "/"))
    path = relative if relative.is_absolute() else Path(IMAGE_ROOT) / relative
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    return path


def classify_prompt(prompt_text: str) -> str:
    prompt = strip_image_marker(prompt_text)
    if "所有棉包" in prompt or "全部棉包" in prompt:
        return "ALL"
    has_row = bool(re.search(r"第" + CN_NUM_PATTERN + r"排", prompt))
    has_position = bool(re.search(r"第" + CN_NUM_PATTERN + r"个", prompt))
    if has_row and has_position:
        return "SINGLE"
    return "ROW"


def find_last_answer_object(text: str) -> dict[str, Any] | None:
    """Salvage the last JSON object containing bbox_2d from imperfect output."""
    decoder = json.JSONDecoder()
    result = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "bbox_2d" in value:
            result = value
    return result


def normalize_answer(answer: Any) -> tuple[list[list[float]], list[str], bool, str]:
    if not isinstance(answer, dict):
        return [], [], False, "final answer is not a JSON object"
    boxes_raw = answer.get("bbox_2d")
    labels_raw = answer.get("labels")
    if not isinstance(boxes_raw, list):
        return [], [], False, "bbox_2d is not a list"

    # Labels are retained for output/schema and CoT diagnostics, but a missing
    # or malformed labels field must not discard otherwise valid prediction
    # boxes.  The six answer metrics below evaluate bbox output only.
    labels_are_list = isinstance(labels_raw, list)
    aligned_labels = labels_raw if labels_are_list else []

    boxes: list[list[float]] = []
    labels: list[str] = []
    all_boxes_valid = True
    for index, raw_box in enumerate(boxes_raw):
        valid = (
            isinstance(raw_box, list)
            and len(raw_box) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_box
            )
        )
        if valid:
            box = [float(value) for value in raw_box]
            valid = (
                0 <= box[0] < box[2] <= 1000
                and 0 <= box[1] < box[3] <= 1000
            )
        if not valid:
            all_boxes_valid = False
            box = [0.0, 0.0, 0.0, 0.0]
        boxes.append(box)
        label = (
            aligned_labels[index]
            if index < len(aligned_labels)
            else "__MISSING__"
        )
        labels.append(label if isinstance(label, str) else "__INVALID__")

    schema_valid = (
        set(answer) == {"bbox_2d", "labels"}
        and labels_are_list
        and len(boxes_raw) == len(aligned_labels)
        and len(boxes) == len(boxes_raw)
        and all_boxes_valid
        and all(label in VALID_LABELS for label in aligned_labels)
    )
    error = "" if schema_valid else "JSON schema, coordinate, or label validation failed"
    return boxes, labels, schema_valid, error


def parse_response(text: str) -> ParsedResponse:
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    open_pos = text.find("<think>")
    close_pos = text.find("</think>")
    tags_valid = (
        open_count == 1
        and close_count == 1
        and 0 <= open_pos < close_pos
    )
    thought = (
        text[open_pos + len("<think>") : close_pos].strip()
        if tags_valid
        else ""
    )

    answer = None
    exact_json_tail = False
    parse_error = ""
    if tags_valid:
        tail = text[close_pos + len("</think>") :].lstrip()
        try:
            answer, end = json.JSONDecoder().raw_decode(tail)
            exact_json_tail = not tail[end:].strip()
        except json.JSONDecodeError as error:
            parse_error = f"final JSON decode failed: {error}"

    if not isinstance(answer, dict):
        answer = find_last_answer_object(text)
    json_valid = isinstance(answer, dict)
    boxes, labels, schema_valid, schema_error = normalize_answer(answer)
    if not parse_error and schema_error:
        parse_error = schema_error

    strict = bool(
        tags_valid
        and thought
        and exact_json_tail
        and json_valid
        and schema_valid
    )
    return ParsedResponse(
        thought=thought,
        answer=answer,
        boxes=boxes,
        labels=labels,
        think_tags_valid=tags_valid,
        strict_two_part=strict,
        json_valid=json_valid,
        schema_valid=schema_valid,
        parse_error=parse_error,
    )


def iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)


def match_boxes(gt_boxes, pred_boxes):
    return [(gi, pi, score) for score, gi, pi in
            protocol.ordered_box_matches(gt_boxes, pred_boxes, IOU_THRESHOLD)]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def normalize_for_rouge(text: str) -> str:
    """Character-level normalization avoids an extra Chinese tokenizer."""
    return "".join(
        character.lower()
        for character in text
        if not character.isspace()
        and character not in "，。；：、,.!?！？()（）[]【】"
    )


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate = normalize_for_rouge(candidate)
    reference = normalize_for_rouge(reference)
    if not candidate or not reference:
        return 0.0
    previous = [0] * (len(candidate) + 1)
    for ref_character in reference:
        current = [0]
        for index, cand_character in enumerate(candidate, start=1):
            if cand_character == ref_character:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


CHINESE_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}


def parse_count_token(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    total = 0
    current = 0
    for character in token:
        if character in CHINESE_DIGITS:
            current = CHINESE_DIGITS[character]
        elif character in CHINESE_UNITS:
            unit = CHINESE_UNITS[character]
            total += (current or 1) * unit
            current = 0
        else:
            return None
    return total + current


COUNT_TOKEN = rf"(?:\d+|{CN_NUM_PATTERN})"


def first_count_match(text: str, patterns: Iterable[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_count_token(match.group(1))
    return None


def extract_count_claims(thought: str) -> dict[str, int | None]:
    total = first_count_match(
        thought,
        [
            rf"(?:共|总计|合计|一共)\s*(?:检测到|识别出|找到)?\s*"
            rf"({COUNT_TOKEN})\s*个(?:目标|棉包|棉花包)?",
            rf"(?:total|altogether)\s*[:=]?\s*({COUNT_TOKEN})",
        ],
    )
    front = first_count_match(
        thought,
        [
            rf"(?:front|正面)\s*[=：:=为有]?\s*({COUNT_TOKEN})",
            rf"({COUNT_TOKEN})\s*个?\s*(?:为\s*)?(?:front|正面)",
        ],
    )
    side = first_count_match(
        thought,
        [
            rf"(?:side|侧面)\s*[=：:=为有]?\s*({COUNT_TOKEN})",
            rf"({COUNT_TOKEN})\s*个?\s*(?:为\s*)?(?:side|侧面)",
        ],
    )
    return {"total": total, "front": front, "side": side}


def answer_facts(boxes: list[list[float]], labels: list[str]) -> dict[str, int]:
    return {
        "total": len(boxes),
        "front": sum(label == "front" for label in labels),
        "side": sum(label == "side" for label in labels),
    }


def cot_structure_coverage(thought: str) -> float:
    stages = [
        ("问题", "概括", "任务"),
        ("视觉", "画面", "图像", "描述"),
        ("定位", "扫描", "筛选", "推理"),
        ("核验", "检查", "数量一致", "结果"),
    ]
    covered = sum(any(keyword in thought for keyword in stage) for stage in stages)
    return covered / len(stages)


def evaluate_sample(
    gt: ParsedResponse, pred: ParsedResponse
) -> dict[str, Any]:
    # 与无 CoT Qwen 脚本保持同一口径：六个答案指标只看输出框，
    # IoU >= 0.5 的全局贪心一对一匹配计为 TP，不检查 front/side 标签。
    box_matches = match_boxes(gt.boxes, pred.boxes)
    task_tp = len(box_matches)
    precision, recall, f1 = prf(
        task_tp, len(pred.boxes) - task_tp, len(gt.boxes) - task_tp
    )
    mean_iou = (
        sum(overlap for _, _, overlap in box_matches) / len(box_matches)
        if box_matches
        else 0.0
    )
    gt_facts = answer_facts(gt.boxes, gt.labels)
    claims = extract_count_claims(pred.thought)
    claim_gt_correct = sum(
        value is not None and value == gt_facts[key]
        for key, value in claims.items()
    )
    cot_grounded_score = claim_gt_correct / 3
    cot_structure_score = cot_structure_coverage(pred.thought)
    cot_rouge_l = rouge_l_f1(pred.thought, gt.thought)

    # The original answer F1 replaces the removed label-aware F1.  Structure
    # and lexical similarity are weak signals, so outcome grounding receives
    # 60% of the composite weight.  Missing/invalid think tags force the score
    # to zero rather than rewarding a correct answer without a CoT process.
    cot_joint_outcome = (f1 + cot_grounded_score) / 2
    has_valid_thought = pred.think_tags_valid and bool(pred.thought)
    cot_step_correctness = float(has_valid_thought) * (
        0.20 * cot_structure_score
        + 0.20 * cot_rouge_l
        + 0.60 * cot_joint_outcome
    )

    return {
        "gt_count": len(gt.boxes),
        "pred_count": len(pred.boxes),
        "count_exact": len(gt.boxes) == len(pred.boxes),
        "over_gen": len(pred.boxes) > len(gt.boxes),
        "task_tp": task_tp,
        "task_fp": len(pred.boxes) - task_tp,
        "task_fn": len(gt.boxes) - task_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "cot_claim_gt_correct": claim_gt_correct,
        "cot_grounded_score": cot_grounded_score,
        "cot_structure_score": cot_structure_score,
        "cot_rouge_l": cot_rouge_l,
        "cot_joint_outcome": cot_joint_outcome,
        "cot_step_correctness": cot_step_correctness,
        "claims": claims,
        "gt_facts": gt_facts,
    }


class MetricsAccumulator:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def compute(self) -> OrderedDict[str, Any]:
        n = len(self.rows)
        if not n:
            return OrderedDict([("样本数", 0)])

        def mean(key: str) -> float:
            return sum(float(row[key]) for row in self.rows) / n

        task_tp = sum(row["task_tp"] for row in self.rows)
        task_fp = sum(row["task_fp"] for row in self.rows)
        task_fn = sum(row["task_fn"] for row in self.rows)
        precision, recall, f1 = prf(task_tp, task_fp, task_fn)
        return OrderedDict([
            ("样本数", n),
            ("CountAcc", round(mean("count_exact"), 4)),
            ("Precision", round(precision, 4)),
            ("Recall", round(recall, 4)),
            ("F1", round(f1, 4)),
            ("mIoU", round(mean("mean_matched_iou"), 4)),
            ("OverGen", round(mean("over_gen"), 4)),
            ("CoTFormatAcc", round(mean("strict_two_part"), 4)),
            ("CoTFactAcc", round(mean("cot_grounded_score"), 4)),
            ("CoTStepCorrectness", round(mean("cot_step_correctness"), 4)),
        ])


class QwenVLModel:
    def __init__(self) -> None:
        if not LORA_PATH.strip():
            raise ValueError("请先在脚本开头填写 LORA_PATH")
        print(f"[{datetime.now():%H:%M:%S}] 加载基础模型: {BASE_MODEL_PATH}")
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL_PATH, trust_remote_code=True
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = (
                self.processor.tokenizer.eos_token_id
            )

        kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "device_map": "auto",
        }
        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL_PATH,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
            print("FlashAttention2 enabled.")
        except Exception as error:
            print(f"FlashAttention2 unavailable: {error}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL_PATH, **kwargs
            )
        print(f"[{datetime.now():%H:%M:%S}] 加载 LoRA: {LORA_PATH}")
        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[{datetime.now():%H:%M:%S}] 模型就绪, device={self.device}")

    def generate(self, image_path: Path, prompt: str, category: str) -> str:
        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": strip_image_marker(prompt)},
            ],
        }]
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=TASK_MAX_TOKENS[category],
                do_sample=False,
            )
        outputs = outputs[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def draw_boxes(
    image_path: Path,
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    save_path: Path,
) -> None:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    def scale(box: list[float]) -> list[int]:
        return [
            round(box[0] / 1000 * width),
            round(box[1] / 1000 * height),
            round(box[2] / 1000 * width),
            round(box[3] / 1000 * height),
        ]

    for box in gt_boxes:
        draw.rectangle(scale(box), outline="green", width=3)
    for box in pred_boxes:
        draw.rectangle(scale(box), outline="red", width=3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


CSV_COLUMNS = [
    "index", "image", "category", "think_tags_valid", "strict_two_part",
    "schema_valid", "gt_count", "pred_count", "count_exact", "over_gen",
    "precision", "recall", "f1",
    "mean_matched_iou", "cot_grounded_score", "cot_step_correctness",
    "generation_seconds", "parse_error",
]


def main() -> None:
    args = parse_args()
    if not LORA_PATH.strip():
        raise ValueError("请先在脚本开头填写 LORA_PATH")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    image_save_dir = run_dir / "vis_images"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "eval.log"
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print("=" * 72)
        print("Qwen3-VL-2B CoT LoRA 棉包检测评估")
        print(f"模型: {BASE_MODEL_PATH}")
        print(f"LoRA: {LORA_PATH}")
        print(f"测试集: {DATA_JSON}")
        print(f"输出目录: {run_dir}")
        print("=" * 72)

        with open(DATA_JSON, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        if not isinstance(dataset, list):
            raise ValueError("测试集 JSON 根节点必须是 list")
        indexed_dataset = list(enumerate(dataset))
        if args.shuffle:
            random.Random(args.seed).shuffle(indexed_dataset)
        if args.max_samples is not None:
            if args.max_samples <= 0:
                raise ValueError("--max-samples 必须为正整数")
            indexed_dataset = indexed_dataset[: args.max_samples]
        print(f"加载测试集: {len(indexed_dataset)} 条")

        accumulators = {
            key: MetricsAccumulator()
            for key in ["OVERALL", "ALL", "ROW", "SINGLE"]
        }
        csv_path = run_dir / "per_sample.csv"
        jsonl_path = run_dir / "predictions.jsonl"
        model = QwenVLModel()

        with (
            csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file,
            jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for position, (original_index, sample) in enumerate(indexed_dataset, start=1):
                prompt = sample["conversations"][0]["value"]
                gt_text = sample["conversations"][1]["value"]
                image_relative = sample["images"][0]
                category = classify_prompt(prompt)
                image_path = resolve_image_path(image_relative)
                gt = parse_response(gt_text)
                if not gt.strict_two_part:
                    raise ValueError(
                        f"测试集第 {original_index} 条参考答案不符合两段式 schema: "
                        f"{gt.parse_error}"
                    )

                started = time.perf_counter()
                pred_text = model.generate(image_path, prompt, category)
                generation_seconds = time.perf_counter() - started
                pred = parse_response(pred_text)
                metrics = evaluate_sample(gt, pred)
                row = {
                    "index": original_index,
                    "image": image_relative,
                    "category": category,
                    "think_tags_valid": pred.think_tags_valid,
                    "strict_two_part": pred.strict_two_part,
                    "json_valid": pred.json_valid,
                    "schema_valid": pred.schema_valid,
                    **metrics,
                    "generation_seconds": generation_seconds,
                    "parse_error": pred.parse_error,
                }
                accumulators[category].add(row)
                accumulators["OVERALL"].add(row)
                writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
                csv_file.flush()

                json.dump(
                    {
                        "index": original_index,
                        "image": image_relative,
                        "category": category,
                        "prompt": prompt,
                        "reference_output": gt_text,
                        "model_output": pred_text,
                        "parse": {
                            "think_tags_valid": pred.think_tags_valid,
                            "strict_two_part": pred.strict_two_part,
                            "json_valid": pred.json_valid,
                            "schema_valid": pred.schema_valid,
                            "error": pred.parse_error,
                        },
                        "metrics": metrics,
                    },
                    jsonl_file,
                    ensure_ascii=False,
                )
                jsonl_file.write("\n")
                jsonl_file.flush()

                if args.visualize_every and (position - 1) % args.visualize_every == 0:
                    suffix = image_path.suffix or ".jpg"
                    draw_boxes(
                        image_path,
                        gt.boxes,
                        pred.boxes,
                        image_save_dir / f"{original_index:05d}{suffix}",
                    )

                print(
                    f"[{datetime.now():%H:%M:%S}] [{position:4d}/{len(indexed_dataset)}] "
                    f"{CATEGORY_NAMES[category]} GT:{metrics['gt_count']} "
                    f"Pred:{metrics['pred_count']} "
                    f"F1:{metrics['f1']:.3f} "
                    f"IoU:{metrics['mean_matched_iou']:.3f} "
                    f"CoTFact:{metrics['cot_grounded_score']:.3f} "
                    f"CoTStep:{metrics['cot_step_correctness']:.3f} "
                    f"Format:{int(pred.strict_two_part)} "
                    f"Time:{generation_seconds:.1f}s"
                )

        summary = OrderedDict()
        for key in ["OVERALL", "ALL", "ROW", "SINGLE"]:
            display_name = "总体" if key == "OVERALL" else CATEGORY_NAMES[key]
            summary[display_name] = accumulators[key].compute()

        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        print("\n" + "=" * 72)
        print("评估结果汇总")
        for group_name, group_metrics in summary.items():
            print(f"\n【{group_name}】")
            for metric_name, value in group_metrics.items():
                print(f"  {metric_name:28s}: {value}")
        print("\n结果文件:")
        print(f"  汇总: {summary_path}")
        print(f"  逐条指标: {csv_path}")
        print(f"  完整输出: {jsonl_path}")
        print(f"  日志: {log_path}")
    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
