"""Evaluate an InternVL3-8B CGPC adapter on dense RefCOCO.

The XTuner SFT checkpoint supplies the visual projector used to initialize the
CGPC run, while the CGPC checkpoint supplies the policy LoRA adapter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import OrderedDict
from datetime import datetime

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode

# bitsandbytes 0.45 + triton 3.7 compatibility used by the InternVL scripts.
_fake_triton_ops = type(sys)("triton.ops")
_fake_triton_ops.matmul_perf_model = type(sys)("matmul_perf_model")
_fake_triton_ops.matmul_perf_model.early_config_prune = lambda *a, **k: None
_fake_triton_ops.matmul_perf_model.estimate_matmul_time = lambda *a, **k: None
sys.modules.setdefault("triton.ops", _fake_triton_ops)
sys.modules.setdefault(
    "triton.ops.matmul_perf_model", _fake_triton_ops.matmul_perf_model)

from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.save_and_load import (
    load_peft_weights,
    set_peft_model_state_dict,
)
from transformers import AutoModel, AutoTokenizer

# Monkey-patch: PEFT 的 set_peft_model_state_dict 需要从新版 transformers
# 导入 EmbeddingParallel，当前版本不存在该符号
import transformers.integrations.tensor_parallel as _tp
if not hasattr(_tp, "EmbeddingParallel"):
    class _DummyEmbeddingParallel:
        pass
    _tp.EmbeddingParallel = _DummyEmbeddingParallel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_MODEL_PATH = (
    "./models/InternVL3-8B")
DEFAULT_CGPC_ADAPTER_PATH = (
    "./checkpoints/internvl_refcoco_cgpc")
DEFAULT_SFT_CHECKPOINT_PATH = (
    "./checkpoints/internvl_refcoco_lora.pth")
DEFAULT_DATA_JSON = (
    "./data/refcoco_valid.json")
DEFAULT_IMAGE_ROOT = "./data"

POLICY_ADAPTER = "policy"
INPUT_SIZE = 448
MAX_DYNAMIC_PATCH = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Deliberately identical to the referenced RefCOCO evaluator. These are
# inference budgets, not training budgets, and can be overridden explicitly.
DEFAULT_TASK_MAX_TOKENS = {
    "ALL": 6144, "LEFT": 1024, "RIGHT": 1024, "MIDDLE": 1024}

CATEGORY_KEYS = ["ALL", "LEFT", "RIGHT", "MIDDLE"]

CATEGORY_NAMES = {
    "ALL": "检测所有对象",
    "LEFT": "检测最左侧对象",
    "RIGHT": "检测最右侧对象",
    "MIDDLE": "检测最中间对象",
}


class Tee:
    """Write stdout to the terminal and a log file with immediate flush."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda image: image.convert("RGB")
                 if image.mode != "RGB" else image),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448,
        use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_image = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))
    if len(processed_images) != blocks:
        raise RuntimeError("InternVL dynamic image tiling produced wrong count")
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_pixel_values(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(item) for item in images])


# The following parser, coordinate conversion, sorter, matcher, and accumulator
# intentionally reproduce the SFT diagnostic evaluator
# (``eval_internvl_lora.py``) so SFT and CGPC results are
# directly comparable.
def iou(box1, box2):
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


def _balanced_json_objects(text):
    """提取文本中的平衡 JSON object，正确处理字符串内的大括号。"""
    objects = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:idx + 1])
                start = None
    return objects, {
        "brace_depth": depth,
        "unclosed_object": depth > 0,
        "unclosed_string": in_string,
    }


def inspect_response_format(text):
    """严格解析预测，与 SFT 诊断版评估口径一致（平衡 JSON + 结构诊断）。"""
    text = "" if text is None else str(text)
    stripped = text.strip()
    result = {
        "boxes": [],
        "labels": [],
        "empty_text": not bool(stripped),
        "invalid_json": False,
        "truncated": False,
        "label_count_mismatch": False,
        "has_bbox_key": False,
    }
    if result["empty_text"]:
        return result
    has_bbox_key = bool(re.search(r'["\']bbox_2d["\']\s*:', text))
    objects, structure = _balanced_json_objects(text)
    result["has_bbox_key"] = has_bbox_key
    result["truncated"] = bool(
        has_bbox_key and
        (structure["unclosed_object"] or structure["unclosed_string"])
    )
    candidate = next((obj for obj in objects if re.search(
        r'"bbox_2d"\s*:', obj)), None)
    if candidate is None:
        result["invalid_json"] = True
        return result
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        result["invalid_json"] = True
        return result
    bbox_raw = obj.get("bbox_2d", [])
    label_raw = obj.get("labels", [])
    if not isinstance(bbox_raw, list):
        result["invalid_json"] = True
        return result
    if not isinstance(label_raw, list):
        label_raw = []
        result["label_count_mismatch"] = bool(bbox_raw)
    else:
        result["label_count_mismatch"] = len(label_raw) != len(bbox_raw)
    result["boxes"] = bbox_raw
    result["labels"] = label_raw
    return result


def parse_bbox_json(text):
    result = inspect_response_format(text)
    result.update(protocol.parse_grounding_record(text))
    return result


def scale_ground_truth_boxes(boxes, image_path):
    with Image.open(image_path) as image:
        _, _, invalid = protocol.scale_normalized_boxes(boxes, image.size)
        return protocol.scale_boxes_preserving_count(boxes, image.size), list(range(len(boxes))), invalid


def validate_and_scale_boxes(boxes, image_path):
    with Image.open(image_path) as image:
        return protocol.scale_normalized_boxes(boxes, image.size)


def evaluation_order_matches(gt_boxes, pred_boxes):
    return protocol.ordered_box_matches(gt_boxes, pred_boxes, 0.5)


def sort_boxes(boxes, labels=None):
    if not boxes:
        return [], [] if labels else []
    if labels:
        indexed = list(enumerate(zip(boxes, labels)))
        indexed.sort(key=lambda item: (item[1][0][1], item[1][0][0]))
        return ([boxes[index] for index, _ in indexed],
                [labels[index] for index, _ in indexed])
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda item: (item[1][1], item[1][0]))
    return [box for _, box in indexed], None


# ================== 指令分类 ==================
# 依据 refcoco_val.json 实际指令划分：
#   - "请定位图中所有 X..."      → ALL     （多目标）
#   - "请定位图中最左侧的 X..."  → LEFT    （单目标）
#   - "请定位图中最右侧的 X..."  → RIGHT   （单目标）
#   - "请定位图中最中间的 X..."  → MIDDLE  （单目标）

def classify_prompt(prompt_text):
    """根据指令返回: 'ALL' | 'LEFT' | 'RIGHT' | 'MIDDLE'"""
    p = prompt_text.replace("<image>\n", "")
    if "所有" in p:
        return "ALL"
    if "最左侧" in p:
        return "LEFT"
    if "最右侧" in p:
        return "RIGHT"
    if "最中间" in p:
        return "MIDDLE"
    # 兜底：未识别时按多目标处理
    return "ALL"


def extract_target_category(prompt_text):
    """从指令中提取目标类别（如 person / chair / hot dog），小写。"""
    p = prompt_text.replace("<image>\n", "")
    match = re.search(
        r'labels["”]?\s*[:：]\s*["“]?([^"”}\s,]+(?:\s+[^"”}\s,]+)*)', p)
    return match.group(1).strip().lower() if match else "unknown"


class MetricsAccumulator:
    """Exact legacy Qwen metric accumulator."""

    def __init__(self):
        self.N = 0
        self.total_hit_macro = 0.0
        self.total_count_acc = 0
        self.total_over_gen = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_iou_sum = 0.0

    def add_sample(self, gt_boxes, pred_boxes, invalid_pred_count=0):
        gt_count = len(gt_boxes)
        valid_pred_count = len(pred_boxes)
        pred_count = valid_pred_count + invalid_pred_count

        matches = evaluation_order_matches(gt_boxes, pred_boxes)
        threshold_matches = [m for m in matches if m[0] > 0.5]
        tp = len(threshold_matches)
        iou_list = [score for score, _, _ in threshold_matches]
        fp = pred_count - tp
        fn = gt_count - tp
        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        hit = tp / gt_count if gt_count > 0 else 0.0
        count_acc = 1 if pred_count == gt_count else 0
        over_gen = 1 if pred_count > gt_count else 0

        self.N += 1
        self.total_hit_macro += hit
        self.total_count_acc += count_acc
        self.total_over_gen += over_gen
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_iou_sum += avg_iou
        return tp, fp, fn, avg_iou, hit

    def compute(self):
        sample_count = max(self.N, 1)
        return OrderedDict([
            ("样本数", self.N),
            ("CountAcc", round(self.total_count_acc / sample_count, 4)),
            ("HitMacro", round(self.total_hit_macro / sample_count, 4)),
            ("Recall", round(
                self.total_tp / (self.total_tp + self.total_fn + 1e-6), 4)),
            ("Precision", round(
                self.total_tp / (self.total_tp + self.total_fp + 1e-6), 4)),
            ("F1", round(
                2 * self.total_tp /
                (2 * self.total_tp + self.total_fp + self.total_fn + 1e-6),
                4)),
            ("mIoU", round(self.total_iou_sum / sample_count, 4)),
            ("OverGen", round(self.total_over_gen / sample_count, 4)),
        ])


# ================== 指标对比表 ==================
METRIC_TABLE_HEADER = (
    f"{'类别':<14s} {'样本':>6s} {'CountAcc':>8s} {'HitMacro':>8s} "
    f"{'Recall':>8s} {'Precision':>8s} {'F1':>8s} {'mIoU':>8s} {'OverGen':>8s}")


def print_metric_table(title, accumulators, label_map=None, order=None):
    """打印一组累加器的指标对比表；label_map 提供显示名，order 控制顺序。"""
    keys = order if order is not None else list(accumulators.keys())
    print("\n" + "=" * len(METRIC_TABLE_HEADER))
    print(title)
    print("=" * len(METRIC_TABLE_HEADER))
    print(METRIC_TABLE_HEADER)
    print("-" * len(METRIC_TABLE_HEADER))
    for key in keys:
        name = label_map.get(key, key) if label_map else key
        metrics = accumulators[key].compute()
        print(f"{name:<14s} {metrics['样本数']:>6d} {metrics['CountAcc']:>8.4f} "
              f"{metrics['HitMacro']:>8.4f} {metrics['Recall']:>8.4f} "
              f"{metrics['Precision']:>8.4f} {metrics['F1']:>8.4f} "
              f"{metrics['mIoU']:>8.4f} {metrics['OverGen']:>8.4f}")
    print("=" * len(METRIC_TABLE_HEADER))


def draw_boxes(
        image_path, gt_boxes, pred_boxes, gt_labels, pred_labels, save_path):
    del gt_labels, pred_labels
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(box, outline="green", width=3)
    for box in pred_boxes:
        draw.rectangle(box, outline="red", width=3)
    image.save(save_path)


def remap_xtuner_adapter_key(checkpoint_key, target_adapter):
    mapped = checkpoint_key.replace(
        "model.language_model.base_model.model.",
        "base_model.model.language_model.")
    return mapped.replace(".default.", f".{target_adapter}.")


def load_exact_state_subset(model, mapped_state, label):
    model_state = model.state_dict()
    errors = []
    for key, value in mapped_state.items():
        if key not in model_state:
            errors.append((key, tuple(value.shape), None))
        elif tuple(value.shape) != tuple(model_state[key].shape):
            errors.append(
                (key, tuple(value.shape), tuple(model_state[key].shape)))
    if errors:
        raise RuntimeError(f"{label} shape/key mismatch: {errors[:3]}")
    result = model.load_state_dict(mapped_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"{label} produced unexpected keys: {result.unexpected_keys[:5]}")
    loaded_state = model.state_dict()
    failures = []
    for key, expected in mapped_state.items():
        actual = loaded_state[key].detach().cpu()
        expected = expected.detach().to(device="cpu", dtype=actual.dtype)
        if not torch.equal(actual, expected):
            failures.append(key)
    if failures:
        raise RuntimeError(f"{label} exact-write verification failed: {failures[:5]}")


def resolve_adapter_dir(path):
    if not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    named_path = os.path.join(path, POLICY_ADAPTER)
    if os.path.isfile(os.path.join(named_path, "adapter_config.json")):
        return named_path
    raise FileNotFoundError(
        f"No adapter_config.json in {path!r} or its policy subdirectory")


class InternVLCGPCModel:
    def __init__(
            self, base_path, adapter_path, sft_checkpoint_path,
            task_max_tokens):
        self.task_max_tokens = dict(task_max_tokens)
        print(f"[{datetime.now():%H:%M:%S}] 加载Tokenizer: {base_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer没有eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[{datetime.now():%H:%M:%S}] 加载InternVL基础模型: {base_path}")
        original_getattr = torch.nn.Module.__getattr__

        def patched_getattr(module, name):
            if name == "all_tied_weights_keys":
                value = getattr(module, "_tied_weights_keys", None)
                return {} if value is None else value
            return original_getattr(module, name)

        torch.nn.Module.__getattr__ = patched_getattr
        try:
            base_model = AutoModel.from_pretrained(
                base_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).cuda()
        finally:
            torch.nn.Module.__getattr__ = original_getattr

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(
            base_model, lora_config, adapter_name=POLICY_ADAPTER)
        model_state = self.model.state_dict()
        expected_lora_count = sum(
            "lora_" in key and f".{POLICY_ADAPTER}." in key
            for key in model_state)

        if not os.path.isfile(sft_checkpoint_path):
            raise FileNotFoundError(
                "The XTuner SFT checkpoint used by this CGPC run is required "
                f"to restore the SFT visual projector: {sft_checkpoint_path}")
        print(
            f"[{datetime.now():%H:%M:%S}] 恢复SFT projector: "
            f"{sft_checkpoint_path}")
        sft_checkpoint = torch.load(
            sft_checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(sft_checkpoint, dict) or not isinstance(
                sft_checkpoint.get("state_dict"), dict):
            raise RuntimeError("XTuner SFT checkpoint缺少state_dict")
        projector_state = {}
        for checkpoint_key, value in sft_checkpoint["state_dict"].items():
            if "mlp1" not in checkpoint_key:
                continue
            mapped_key = checkpoint_key.replace("model.", "base_model.model.")
            if mapped_key in model_state:
                projector_state[mapped_key] = value
        if not projector_state:
            raise RuntimeError("XTuner SFT checkpoint没有匹配到任何mlp1 projector权重")
        load_exact_state_subset(self.model, projector_state, "SFT projector")
        projector_tensor_count = len(projector_state)

        print(
            f"[{datetime.now():%H:%M:%S}] 加载CGPC policy LoRA: "
            f"{adapter_path}")
        adapter_dir = resolve_adapter_dir(adapter_path)
        if adapter_dir is not None:
            saved_adapter = load_peft_weights(adapter_dir, device="cpu")
            loaded_lora_count = sum("lora_" in key for key in saved_adapter)
            if loaded_lora_count != expected_lora_count:
                raise RuntimeError(
                    "CGPC PEFT LoRA tensor count mismatch: "
                    f"{loaded_lora_count}/{expected_lora_count}")
            result = set_peft_model_state_dict(
                self.model, saved_adapter, adapter_name=POLICY_ADAPTER)
            unexpected = list(getattr(result, "unexpected_keys", []) or [])
            if unexpected:
                raise RuntimeError(
                    f"CGPC adapter has unexpected keys: {unexpected[:5]}")
        else:
            checkpoint = torch.load(
                adapter_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or not isinstance(
                    checkpoint.get("state_dict"), dict):
                raise RuntimeError("LoRA checkpoint缺少state_dict")
            mapped_lora = {}
            for checkpoint_key, value in checkpoint["state_dict"].items():
                if "mlp1" in checkpoint_key:
                    mapped_key = checkpoint_key.replace("model.", "base_model.model.")
                    if mapped_key in model_state:
                        mapped_lora[mapped_key] = value
                    continue
                if "lora" not in checkpoint_key:
                    continue
                mapped_key = remap_xtuner_adapter_key(
                    checkpoint_key, POLICY_ADAPTER)
                if mapped_key in model_state:
                    mapped_lora[mapped_key] = value
            loaded_lora_count = sum("lora_" in key for key in mapped_lora)
            if loaded_lora_count != expected_lora_count:
                raise RuntimeError(
                    "Mapped LoRA tensor count mismatch: "
                    f"{loaded_lora_count}/{expected_lora_count}")
            load_exact_state_subset(self.model, mapped_lora, "Policy LoRA")

        del sft_checkpoint, projector_state, model_state
        projector_file = Path(adapter_path) / "visual_projector.pt"
        if not projector_file.is_file() and adapter_dir is not None:
            projector_file = Path(adapter_dir).parent / "visual_projector.pt"
        if adapter_dir is not None:
            self.model.get_base_model().mlp1.load_state_dict(
                torch.load(projector_file, map_location="cpu", weights_only=True))
        self.model.set_adapter(POLICY_ADAPTER)
        self.model.eval()
        self.device = torch.device("cuda:0")
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
            generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(
            f"[{datetime.now():%H:%M:%S}] 模型就绪: device={self.device}, "
            f"LoRA={loaded_lora_count}/{expected_lora_count}, "
            f"projector={projector_tensor_count}")

    def generate_single(self, image_path, prompt, category):
        pixel_values = load_image_pixel_values(
            image_path, input_size=INPUT_SIZE, max_num=MAX_DYNAMIC_PATCH
        ).to(device=self.device, dtype=torch.bfloat16)
        generation_config = {
            "max_new_tokens": self.task_max_tokens[category],
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        if isinstance(response, str):
            return response
        if isinstance(response, tuple) and response and isinstance(response[0], str):
            return response[0]
        if isinstance(response, dict):
            text = response.get("response", response.get("text"))
            if isinstance(text, str):
                return text
        raise TypeError(f"model.chat returned unsupported type: {type(response).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate InternVL CGPC with the legacy Qwen metric protocol")
    parser.add_argument(
        "--base_model_path",
        default=os.environ.get("CGPC_BASE_MODEL", DEFAULT_BASE_MODEL_PATH))
    parser.add_argument(
        "--adapter_path",
        default=os.environ.get("CGPC_ADAPTER_PATH", DEFAULT_CGPC_ADAPTER_PATH))
    parser.add_argument(
        "--sft_checkpoint_path",
        default=os.environ.get(
            "CGPC_SFT_PATH", DEFAULT_SFT_CHECKPOINT_PATH))
    parser.add_argument(
        "--data_json",
        default=os.environ.get("CGPC_EVAL_JSON", DEFAULT_DATA_JSON))
    parser.add_argument(
        "--image_root",
        default=os.environ.get("CGPC_IMAGE_ROOT", DEFAULT_IMAGE_ROOT))
    parser.add_argument(
        "--output_dir",
        default=os.environ.get(
            "CGPC_OUTPUT_DIR",
            os.path.join(SCRIPT_DIR, "eval_internvl_cgpc_refcoco")))
    parser.add_argument(
        "--left_max_tokens", type=int,
        default=DEFAULT_TASK_MAX_TOKENS["LEFT"])
    parser.add_argument(
        "--right_max_tokens", type=int,
        default=DEFAULT_TASK_MAX_TOKENS["RIGHT"])
    parser.add_argument(
        "--middle_max_tokens", type=int,
        default=DEFAULT_TASK_MAX_TOKENS["MIDDLE"])
    parser.add_argument(
        "--all_max_tokens", type=int,
        default=DEFAULT_TASK_MAX_TOKENS["ALL"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    output_dir = os.path.abspath(args.output_dir)
    image_save_dir = os.path.join(output_dir, "vis_images")
    wrong_box_dir = os.path.join(output_dir, "wrong_boxes")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(wrong_box_dir, exist_ok=True)

    task_max_tokens = {
        "LEFT": args.left_max_tokens,
        "RIGHT": args.right_max_tokens,
        "MIDDLE": args.middle_max_tokens,
        "ALL": args.all_max_tokens,
    }
    if any(value <= 0 for value in task_max_tokens.values()):
        raise ValueError(f"max token values must be positive: {task_max_tokens}")

    print("=" * 68)
    print("InternVL3-8B CGPC RefCOCO评估（SFT诊断口径，与SFT评估可比）")
    print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"基础模型: {args.base_model_path}")
    print(f"CGPC LoRA: {args.adapter_path}")
    print(f"SFT projector来源: {args.sft_checkpoint_path}")
    print(f"验证集: {args.data_json}")
    print(f"输出目录: {output_dir}")
    print(f"任务生成上限: {task_max_tokens}")
    print("指标协议: SFT diagnostic parser / spatial GT-order greedy one-to-one matching")
    print("=" * 68)

    with open(args.data_json, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    random.Random(args.seed).shuffle(dataset)
    category_counts = {category: 0 for category in CATEGORY_KEYS}
    category_groups = {}
    for sample in dataset:
        prompt = sample["conversations"][0]["value"]
        category_counts[classify_prompt(prompt)] += 1
        label = extract_target_category(prompt)
        category_groups.setdefault(label, []).append(prompt)
    print(f"加载测试集: {len(dataset)}条；分类={category_counts}")
    print("目标类别统计: " + ", ".join(
        f"{label}={len(idx)}" for label, idx in sorted(
            category_groups.items(), key=lambda item: -len(item[1]))))

    model = InternVLCGPCModel(
        args.base_model_path,
        args.adapter_path,
        args.sft_checkpoint_path,
        task_max_tokens,
    )

    accumulators = {
        category: MetricsAccumulator()
        for category in ["OVERALL"] + CATEGORY_KEYS
    }
    # 目标类别累加器（person / chair / hot dog ...）
    cat_acc = {label: MetricsAccumulator() for label in category_groups}
    csv_rows = {category: [] for category in CATEGORY_KEYS}
    zero_metric_images = []
    predictions_path = os.path.join(output_dir, "predictions.jsonl")

    with open(predictions_path, "w", encoding="utf-8") as predictions_file:
        for eval_index, sample in enumerate(dataset):
            image_relative = sample["images"][0]
            image_path = os.path.join(args.image_root, image_relative)
            prompt = sample["conversations"][0]["value"]
            gt_text = sample["conversations"][1]["value"]
            category = classify_prompt(prompt)
            target_label = extract_target_category(prompt)
            pred_text = model.generate_single(image_path, prompt, category)

            gt_parsed = parse_bbox_json(gt_text)
            pred_parsed = parse_bbox_json(pred_text)
            gt_boxes = gt_parsed["boxes"]
            gt_labels = gt_parsed["labels"]
            pred_boxes = pred_parsed["boxes"]
            pred_labels = pred_parsed["labels"]
            gt_boxes, gt_valid_idx, gt_invalid_count = scale_ground_truth_boxes(
                gt_boxes, image_path)
            pred_boxes, pred_valid_idx, pred_invalid_count = validate_and_scale_boxes(
                pred_boxes, image_path)
            gt_labels = [gt_labels[i] if i < len(gt_labels) else ""
                         for i in gt_valid_idx]
            pred_labels = [pred_labels[i] if i < len(pred_labels) else ""
                           for i in pred_valid_idx]
            gt_boxes, gt_labels = sort_boxes(gt_boxes, gt_labels)
            pred_boxes, pred_labels = sort_boxes(pred_boxes, pred_labels)

            tp, fp, fn, avg_iou, hit = accumulators[category].add_sample(
                gt_boxes, pred_boxes, invalid_pred_count=pred_invalid_count)
            accumulators["OVERALL"].add_sample(
                gt_boxes, pred_boxes, invalid_pred_count=pred_invalid_count)
            cat_acc[target_label].add_sample(
                gt_boxes, pred_boxes, invalid_pred_count=pred_invalid_count)
            gt_count = len(gt_boxes)
            pred_count = len(pred_boxes)

            if eval_index % 20 == 0:
                draw_boxes(
                    image_path, gt_boxes, pred_boxes, gt_labels, pred_labels,
                    os.path.join(image_save_dir, os.path.basename(image_relative)))

            wrong_visualization = None
            if pred_count > 0 and tp == 0:
                wrong_visualization = os.path.join(
                    wrong_box_dir,
                    f"{category}_{os.path.basename(image_relative)}")
                draw_boxes(
                    image_path, gt_boxes, pred_boxes, gt_labels, pred_labels,
                    wrong_visualization)

            if gt_count > 0 and tp == 0:
                zero_metric_images.append({
                    "image": image_relative,
                    "category": category,
                    "category_name": CATEGORY_NAMES[category],
                    "object_category": target_label,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "failure_reason": (
                        "no_output" if pred_count == 0 else "all_boxes_wrong"),
                    "wrong_vis_path": wrong_visualization,
                })

            csv_rows[category].append([
                image_relative, gt_count, pred_count, round(hit, 4),
                round(avg_iou, 4), tp, fp, fn, category, target_label,
            ])
            predictions_file.write(json.dumps({
                "eval_index": eval_index,
                "image": image_relative,
                "category": category,
                "object_category": target_label,
                "raw_prediction": pred_text,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "hit": hit,
                "miou": avg_iou,
            }, ensure_ascii=False) + "\n")
            predictions_file.flush()

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            print(
                f"[{datetime.now():%H:%M:%S}] "
                f"[{eval_index + 1:4d}/{len(dataset)}] "
                f"({CATEGORY_NAMES[category]}) GT:{gt_count} Pred:{pred_count} "
                f"Hit:{hit:.3f} P:{precision:.3f} R:{recall:.3f} "
                f"F1:{f1:.3f} mIoU:{avg_iou:.3f} Obj:{target_label}")

    print("\n" + "=" * 68)
    print("评估结果汇总")
    print("=" * 68)
    all_results = OrderedDict()
    for category in ["OVERALL"] + CATEGORY_KEYS:
        name = "总体" if category == "OVERALL" else CATEGORY_NAMES[category]
        metrics = accumulators[category].compute()
        all_results[name] = metrics
        print(f"\n【{name}】")
        for metric_name, value in metrics.items():
            if metric_name == "样本数":
                print(f"  {metric_name:20s}: {value}")
            else:
                print(f"  {metric_name:20s}: {value:.4f}")

    # ================== 指令类型 vs 总体 对比表 ==================
    task_label_map = {"OVERALL": "总体"}
    task_label_map.update(CATEGORY_NAMES)
    print_metric_table(
        "指令类型 vs 总体 评估对比",
        accumulators, label_map=task_label_map,
        order=["OVERALL"] + CATEGORY_KEYS)

    # ================== 目标类别 vs 总体 对比表 ==================
    all_acc = {"OVERALL": accumulators["OVERALL"]}
    all_acc.update(cat_acc)
    cat_order = ["OVERALL"] + sorted(
        cat_acc, key=lambda k: -len(category_groups[k]))
    print_metric_table(
        "目标类别 vs 总体 评估对比",
        all_acc, label_map={"OVERALL": "总体"}, order=cat_order)

    header = [
        "图片", "GT数量", "预测数量", "命中率", "mIoU",
        "TP", "FP", "FN", "类别", "目标类别",
    ]
    for category in CATEGORY_KEYS:
        with open(
                os.path.join(output_dir, f"result_{category}.csv"),
                "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(csv_rows[category])

    all_csv_path = os.path.join(output_dir, "result_all.csv")
    with open(all_csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for category in CATEGORY_KEYS:
            writer.writerows(csv_rows[category])

    # 按目标类别聚合的指标表（person / chair / hot dog ...）。
    object_csv_path = os.path.join(
        output_dir, "result_by_object_category.csv")
    with open(object_csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "目标类别", "样本数", "CountAcc", "HitMacro", "Recall",
            "Precision", "F1", "mIoU", "OverGen"])
        for label in cat_order[1:]:
            metrics = cat_acc[label].compute()
            writer.writerow([
                label, metrics["样本数"], metrics["CountAcc"],
                metrics["HitMacro"], metrics["Recall"], metrics["Precision"],
                metrics["F1"], metrics["mIoU"], metrics["OverGen"],
            ])

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    # 目标类别（person / chair / hot dog ...）单独落盘，不改变 summary.json 结构。
    object_summary_path = os.path.join(
        output_dir, "summary_by_object_category.json")
    with open(object_summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            OrderedDict(
                (label, cat_acc[label].compute()) for label in cat_order[1:]),
            handle, indent=2, ensure_ascii=False)

    zero_metric_path = os.path.join(output_dir, "zero_metric_images.json")
    zero_metric_summary = {
        "description": "框完全匹配失败（TP=0）的图片清单；包括无输出和全错两类",
        "total_zero_metric_images": len(zero_metric_images),
        "by_category": {
            category: {
                "count": sum(
                    item["category"] == category for item in zero_metric_images),
                "no_output": sum(
                    item["category"] == category and
                    item["failure_reason"] == "no_output"
                    for item in zero_metric_images),
                "all_boxes_wrong": sum(
                    item["category"] == category and
                    item["failure_reason"] == "all_boxes_wrong"
                    for item in zero_metric_images),
            }
            for category in CATEGORY_KEYS
        },
        "images": zero_metric_images,
    }
    with open(zero_metric_path, "w", encoding="utf-8") as handle:
        json.dump(zero_metric_summary, handle, indent=2, ensure_ascii=False)

    protocol_path = os.path.join(output_dir, "evaluation_protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as handle:
        json.dump({
            "metric_source": "eval_internvl_lora.py",
            "inference_source": "eval_internvl_lora.py",
            "matching": "spatial_gt_order_greedy_one_to_one",
            "parser": "sft_diagnostic_balanced_json",
            "task_max_tokens": task_max_tokens,
            "seed": args.seed,
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "sft_checkpoint_path": args.sft_checkpoint_path,
            "sample_count": len(dataset),
            "category_counts": category_counts,
            "object_category_counts": {
                label: len(items) for label, items in category_groups.items()},
        }, handle, indent=2, ensure_ascii=False)

    print("\n结果保存:")
    print(f"  汇总: {summary_path}")
    print(f"  目标类别汇总: {object_summary_path}")
    print(f"  全量CSV: {all_csv_path}")
    print(f"  目标类别CSV: {object_csv_path}")
    print(f"  原始预测: {predictions_path}")
    print(f"  TP=0清单: {zero_metric_path}")
    print(f"  协议: {protocol_path}")


if __name__ == "__main__":
    arguments = parse_args()
    os.makedirs(arguments.output_dir, exist_ok=True)
    log_path = os.path.join(
        arguments.output_dir,
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        main(arguments)
    finally:
        tee.log.close()
        sys.stdout = tee.terminal
