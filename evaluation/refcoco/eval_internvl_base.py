"""
InternVL3-8B 原始基座模型 RefCOCO 检测评估脚本
直接加载 InternVL3-8B 原始基座模型，在 refcoco_val.json 上运行推理，
计算 bbox 检测指标 + eval loss。

评估口径与 eval_internvl_lora.py 保持一致：
  - 指令类型分组（4 类 + 总体）：
      所有对象: "请定位图中所有人/所有椅子/..."（1034 条）
      最左侧:   "请定位图中最左侧的人/椅子/..."（362 条）
      最右侧:   "请定位图中最右侧的人/椅子/..."（343 条）
      最中间:   "请定位图中最中间的人/椅子/..."（329 条）
      总体:     全部 2068 条
  - 目标类别（person / chair / hot dog ...）聚合
  - 输出：日志、summary.json（含 object_category_results）、result_<CAT>.csv、
         result_all.csv、result_by_object_category.csv、逐样本 predictions.jsonl、
         分级告警 warnings.jsonl、run_diagnostics.json、可视化图片

默认直接评估完整验证集，并把加载自检、逐样本告警和原始输出分别落盘。

基座模型输出格式通常不是训练/GT 的 {"bbox_2d":..,"labels":..}，因此解析器按
bbox_2d_object → bbox_2d_array → internvl_box(<ref>/<box>) → qwen_box(<|box_start|>)
→ raw_quads 的顺序兼容解析，实际命中项记入 parse_mode 并写入逐样本 CSV/JSONL；
非标准格式会单独告警（NONSTANDARD_OUTPUT_FORMAT）并统计 NonstandardFormatRate，
指标照常计入，不会因为格式差异把基座模型误记为 0 命中。

用法:
    python eval_internvl_base.py              # 全量评估
    python eval_internvl_base.py --probe 3    # 只跑3条，打印原始输出与解析结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import re
import json
import os
import sys
import math
import random
import time
import hashlib
import torch
import torchvision.transforms as T
import csv
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
# Fix bitsandbytes 0.45 + triton 3.7 兼容性：triton.ops 模块已移除
_fake_triton_ops = type(sys)('triton.ops')
_fake_triton_ops.matmul_perf_model = type(sys)('matmul_perf_model')
_fake_triton_ops.matmul_perf_model.early_config_prune = lambda *a, **k: None
_fake_triton_ops.matmul_perf_model.estimate_matmul_time = lambda *a, **k: None
sys.modules.setdefault('triton.ops', _fake_triton_ops)
sys.modules.setdefault('triton.ops.matmul_perf_model', _fake_triton_ops.matmul_perf_model)

from transformers import AutoTokenizer, AutoModel

# ================== 路径配置 ==================
BASE_MODEL_PATH = os.environ.get(
    "CGPC_BASE_MODEL",
    "./models/InternVL3-8B")

DATA_JSON = os.environ.get(
    "CGPC_EVAL_JSON",
    "./data/refcoco_valid.json")
IMAGE_ROOT = os.environ.get(
    "CGPC_IMAGE_ROOT",
    "./data")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 结果目录独立，不覆盖其它评估结果
OUTPUT_DIR = os.environ.get(
    "CGPC_OUTPUT_DIR",
    os.path.join(SCRIPT_DIR, "eval_internvl_base_refcoco"))
IMAGE_SAVE_DIR = os.path.join(OUTPUT_DIR, "vis_images_refcoco")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

EVAL_SCHEMA_VERSION = "internvl_base_refcoco_eval_v1_diagnostic"
EVAL_SEED = 42
IOU_THRESHOLD = 0.5
# ALL 需要输出多目标，保留 6144；LEFT/RIGHT/MIDDLE 目标通常为单框。
TASK_MAX_TOKENS = {"ALL": 6144, "LEFT": 1024, "RIGHT": 1024, "MIDDLE": 1024}
VISUALIZE_EVERY = 20
PREDICTIONS_JSONL = os.path.join(OUTPUT_DIR, "predictions.jsonl")
WARNINGS_JSONL = os.path.join(OUTPUT_DIR, "warnings.jsonl")
RUN_DIAGNOSTICS_JSON = os.path.join(OUTPUT_DIR, "run_diagnostics.json")
EARLY_HEALTHCHECK_SAMPLES = 50
MAX_CONSECUTIVE_GENERATION_ERRORS = 3
REPETITION_MIN_COUNT = 8
REPETITION_TAIL_MIN_REPEATS = 4
# ================== 日志记录 ==================
LOG_FILE = os.path.join(OUTPUT_DIR, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class Tee:
    """同时输出到终端和日志文件，并实时 flush。"""
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


class DiagnosticReporter:
    """将分级告警实时写入 JSONL；控制台只打印简短定位信息。"""

    def __init__(self, path):
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")
        self.counts = OrderedDict()

    def warn(self, code, message, severity="warning", **context):
        self.counts[code] = self.counts.get(code, 0) + 1
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "severity": severity,
            "code": code,
            "message": message,
            "context": context,
        }
        self.handle.write(json.dumps(
            record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()
        location = ""
        if "eval_index" in context:
            location = f" sample={context['eval_index']}"
        print(f"[DIAG-{severity.upper()}][{code}]{location} {message}")

    def close(self):
        if not self.handle.closed:
            self.handle.close()

# ================== InternVL 图片预处理 ==================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_DYNAMIC_PATCH = 12

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_pixel_values(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ================== IoU 计算 ==================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


# ================== JSON 解析 ==================
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


_NUMERIC_QUAD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_NUMERIC_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
# <ref>xx</ref><box>[[..]]</box>（InternVL）与 <|box_start|>(x,y),(x,y)<|box_end|>（Qwen 系）
_GROUNDING_TAG_RE = re.compile(
    r"<ref>(?P<ref>.*?)</ref>"
    r"|<\|ref_start\|>(?P<qref>.*?)<\|ref_end\|>"
    r"|<box>(?P<box>.*?)</box>"
    r"|<\|box_start\|>(?P<qbox>.*?)<\|box_end\|>",
    re.DOTALL)


def _balanced_span(text, start_idx):
    """返回从 start_idx 起的完整平衡括号片段；未闭合时返回 None。"""
    stack = []
    in_string = False
    escaped = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                return None
            if not stack:
                return text[start_idx:idx + 1]
    return None


def _top_level_array_spans(text):
    """提取顶层平衡列表片段（[[..]] 里的内层不会被重复返回）。"""
    spans = []
    depth = 0
    in_string = False
    escaped = False
    start = None
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
        elif char in "{[":
            if depth == 0 and char == "[":
                start = idx
            depth += 1
        elif char in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(text[start:idx + 1])
                    start = None
    return spans


def _normalize_box_list(value):
    """把 bbox_2d / box 值统一成二维列表 [[x1,y1,x2,y2], ...]。"""
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], list):
        return [list(item[:4]) for item in value
                if isinstance(item, list) and len(item) >= 4]
    if len(value) >= 4:
        return [list(value[:4])]
    return []


def _normalize_label_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


# 键名别名：基座模型常把 bbox_2d 写成 bbox_2 / bbox / box 等
_BBOX_KEY_RE = re.compile(
    r'"\s*(?:bbox_2d|bbox_2|bbox|bboxes|box|boxes)\s*"\s*:', re.IGNORECASE)
# 裸标识符标签："labels":person（直接照抄了 prompt 模板里的无引号写法）
_BARE_LABEL_RE = re.compile(
    r'("(?:labels|label)"\s*:\s*)(?!["\[{])([^,}\]\s][^,}\]]*)')


def _get_bbox_value(obj):
    """按键名别名取框值，容忍 bbox_2d / bbox_2 / bbox / box 等拼写。"""
    for key in ("bbox_2d", "bbox_2", "bbox", "boxtwo", "bboxes", "box", "boxes"):
        if key in obj:
            return obj[key]
    for key, value in obj.items():
        if isinstance(key, str) and "bbox" in key.lower() and isinstance(value, list):
            return value
    return None


def _quote_bare_label(match):
    return f'{match.group(1)}"{match.group(2).strip()}"'


def _repair_json_text(fragment):
    """修复基座模型常见的轻微 JSON 畸形，返回 (修复后文本, 修复动作列表)。

    处理：截断产生的半截数据、裸标识符标签、尾随逗号、未闭合的括号/字符串。
    只做“补齐语法”，不猜测/不补造任何坐标或标签。
    """
    actions = []
    text = fragment

    # 1) 丢掉最后一个完整 ]/} 之后的半截数据（如 ", [572, 727, 92"）
    last_close = max(text.rfind("]"), text.rfind("}"))
    if last_close != -1 and last_close != len(text) - 1:
        text = text[:last_close + 1]
        actions.append("drop_trailing_partial")

    # 2) "labels":person -> "labels":"person"
    text, count = _BARE_LABEL_RE.subn(_quote_bare_label, text)
    if count:
        actions.append("quote_bare_labels")

    # 3) 尾随逗号 ，] / ，}
    text, count = re.subn(r",\s*(?=[\]}])", "", text)
    if count:
        actions.append("drop_trailing_comma")

    # 4) 补齐未闭合括号/字符串（按扫描出的栈逆序补）
    stack = []
    in_string = False
    escaped = False
    for char in text:
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
    if stack:
        text += "".join("]" if opener == "[" else "}" for opener in reversed(stack))
        actions.append("close_brackets")
    if in_string:
        text += '"'
        actions.append("close_string")
    return text, actions


def _labels_for_boxes(label_raw, box_count):
    """把标签对齐到框数：单标签广播到所有框，多余截断，缺失补空串。"""
    if box_count <= 0:
        return []
    if not label_raw:
        return [""] * box_count
    if len(label_raw) == 1:
        return list(label_raw) * box_count
    if len(label_raw) < box_count:
        return list(label_raw) + [""] * (box_count - len(label_raw))
    return list(label_raw[:box_count])


def _load_json_with_repair(fragment):
    """先按原样 json.loads，失败后用 _repair_json_text 修一次再试。

    返回 (obj, repairs, error)：两次都失败时 obj=None，error 为原始解析错误。
    """
    try:
        return json.loads(fragment), [], ""
    except json.JSONDecodeError as exc:
        error = f"{exc.msg}; line={exc.lineno}; column={exc.colno}; pos={exc.pos}"
    except TypeError as exc:
        error = f"TypeError: {exc}"
    repaired, actions = _repair_json_text(fragment)
    try:
        return json.loads(repaired), actions, error
    except (json.JSONDecodeError, TypeError):
        return None, actions, error


def _parse_grounding_tags(text):
    """解析 <ref>/<box> 与 <|box_start|> 接地格式，返回 (boxes, labels, mode)。"""
    boxes, labels = [], []
    current_ref = ""
    mode = ""
    for match in _GROUNDING_TAG_RE.finditer(text):
        ref = match.group("ref") or match.group("qref")
        if ref is not None:
            current_ref = ref.strip()
            continue
        content = match.group("box") or match.group("qbox") or ""
        mode = "internvl_box" if match.group("box") is not None else "qwen_box"
        quads = [[float(v) for v in quad] for quad in _NUMERIC_QUAD_RE.findall(content)]
        if not quads:
            # (x1,y1),(x2,y2) 形式：每两对组成一个框
            pairs = [[float(v) for v in pair] for pair in _NUMERIC_PAIR_RE.findall(content)]
            quads = [[pairs[i][0], pairs[i][1], pairs[i + 1][0], pairs[i + 1][1]]
                     for i in range(0, len(pairs) - 1, 2)]
        for quad in quads:
            boxes.append(quad)
            labels.append(current_ref)
    return boxes, labels, mode


def inspect_response_format(text):
    """解析预测并保留足以定位格式失败的结构诊断。

    处理顺序（第一个命中的生效，parse_mode 记录实际命中项）：
      1) bbox_2d_array  : [{"bbox_2d": [x1,y1,x2,y2], "labels": "person"}, ...]
      2) bbox_2d_object : {"bbox_2d": [[...]], "labels": [...]}（训练/GT 标准格式）
         bbox_2d_object_repaired：同一结构但需修复才可解析，基座模型常见：
           {"bbox_2": [...], "labels":person}（键名写错 + 标签未加引号）、
           尾随逗号、以及生成到一半 EOS 导致末尾 ]} 缺失的截断输出。
           修复只补语法，已生成的框全部保留，并置 truncated=True 单独统计。
      3) internvl_box   : <ref>person</ref><box>[[x1,y1,x2,y2], ...]</box>
      4) qwen_box       : <|box_start|>(x1,y1),(x2,y2)<|box_end|>
      5) raw_quads      : 文本中直接出现的裸坐标
    只有 bbox_2d_object 是标准格式；其余会置 nonstandard_format=True 并单独告警
    （NONSTANDARD_OUTPUT_FORMAT）与统计（NonstandardFormatRate），但指标照常计入，
    避免基座模型因格式差异被误记为 0 命中；repairs 记录实际做过的修复动作。
    """
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
        "has_box_tag": False,
        "balanced_object_count": 0,
        "array_object_count": 0,
        "brace_depth": 0,
        "unclosed_string": False,
        "json_error": "",
        "parse_mode": "none",
        "nonstandard_format": False,
        "repairs": [],
    }
    if result["empty_text"]:
        return result

    has_bbox_key = bool(_BBOX_KEY_RE.search(text))
    objects, structure = _balanced_json_objects(text)
    result["has_bbox_key"] = has_bbox_key
    result["balanced_object_count"] = len(objects)
    result["brace_depth"] = structure["brace_depth"]
    result["unclosed_string"] = structure["unclosed_string"]
    box_open = len(re.findall(r"<box>", text)) + len(re.findall(r"<\|box_start\|>", text))
    box_close = len(re.findall(r"</box>", text)) + len(re.findall(r"<\|box_end\|>", text))
    result["has_box_tag"] = box_open > 0
    result["truncated"] = bool(
        (has_bbox_key or result["has_box_tag"]) and
        (structure["unclosed_object"] or structure["unclosed_string"] or
         box_open > box_close)
    )

    # ---- 1) 对象数组格式：[{"bbox_2d": .., "labels": ..}, ...] ----
    # 必须在单对象分支之前判断：数组里的每个元素本身就是「含 bbox_2d 的平衡对象」，
    # 否则会被单对象分支抢先命中，只取到第一个框。
    for span in _top_level_array_spans(text):
        try:
            data = json.loads(span)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, list) or not data:
            continue
        if not all(isinstance(item, dict) for item in data):
            continue
        if not any(isinstance(_get_bbox_value(item), list) for item in data):
            continue
        for item in data:
            item_boxes = _normalize_box_list(_get_bbox_value(item))
            if not item_boxes:
                continue
            item_labels = _normalize_label_list(
                item.get("labels", item.get("label")))
            result["labels"].extend(_labels_for_boxes(item_labels, len(item_boxes)))
            result["boxes"].extend(item_boxes)
        if result["boxes"]:
            result["array_object_count"] = len(data)
            result["parse_mode"] = "bbox_2d_array"
            result["nonstandard_format"] = True
            result["label_count_mismatch"] = False
            return result

    # ---- 2) 标准格式：单个 {"bbox_2d": ..., "labels": ...}（含轻微畸形的修复） ----
    candidate = next((obj for obj in objects if _BBOX_KEY_RE.search(obj)), None)
    if candidate is not None:
        obj, repairs, error = _load_json_with_repair(candidate)
        if not isinstance(obj, dict):
            result["invalid_json"] = True
            result["json_error"] = f"含 bbox 键的对象修复后仍无法解析; {error}"
            result["repairs"] = repairs
            return result
        bbox_raw = _get_bbox_value(obj)
        if not isinstance(bbox_raw, list):
            result["invalid_json"] = True
            result["json_error"] = "bbox 键的值不是列表"
            return result
        boxes = _normalize_box_list(bbox_raw)
        label_raw = _normalize_label_list(obj.get("labels", obj.get("label")))
        result["boxes"] = boxes
        result["labels"] = _labels_for_boxes(label_raw, len(boxes))
        result["label_count_mismatch"] = len(label_raw) != len(boxes)
        result["repairs"] = repairs
        if repairs:
            # 例如 "labels":person 裸标签、bbox_2 键名、尾随逗号
            result["parse_mode"] = "bbox_2d_object_repaired"
            result["nonstandard_format"] = True
            result["json_error"] = f"repaired({','.join(repairs)}); 原始错误: {error}"
        else:
            result["parse_mode"] = "bbox_2d_object"
        return result

    # ---- 3) 截断/未闭合 JSON：从第一个 { 起截取尾部，补齐括号后解析 ----
    # 基座模型常在输出完若干框后直接 EOS，丢掉末尾的 ]}（例如 12 个框全对但 JSON 没闭合）。
    # 这里只补齐语法，已生成的框全部保留，并置 truncated=True 单独统计。
    brace_idx = text.find("{")
    if brace_idx != -1:
        repaired, actions = _repair_json_text(text[brace_idx:])
        try:
            obj = json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            boxes = _normalize_box_list(_get_bbox_value(obj))
            if boxes:
                label_raw = _normalize_label_list(
                    obj.get("labels", obj.get("label")))
                result["boxes"] = boxes
                result["labels"] = _labels_for_boxes(label_raw, len(boxes))
                result["label_count_mismatch"] = len(label_raw) != len(boxes)
                result["repairs"] = actions
                result["parse_mode"] = "bbox_2d_object_repaired"
                result["nonstandard_format"] = True
                if {"close_brackets", "drop_trailing_partial"} & set(actions):
                    result["truncated"] = True
                result["json_error"] = f"repaired({','.join(actions)}); 输出被截断"
                return result

    # ---- 4) <ref>/<box> 或 <|box_start|> 接地格式 ----
    tag_boxes, tag_labels, tag_mode = _parse_grounding_tags(text)
    if tag_boxes:
        result["boxes"] = tag_boxes
        result["labels"] = tag_labels
        result["parse_mode"] = tag_mode or "internvl_box"
        result["nonstandard_format"] = True
        result["label_count_mismatch"] = False
        return result

    # ---- 5) 裸坐标（尽量从文本里补出标签，便于 LabelAcc 仍有意义） ----
    raw_quads = [[float(value) for value in quad]
                 for quad in _NUMERIC_QUAD_RE.findall(text)]
    if raw_quads:
        label_match = re.search(r'\"?labels?\"?\s*[:：]\s*\"?([^",}\]\s]+)', text)
        label = label_match.group(1) if label_match else ""
        result["boxes"] = raw_quads
        result["labels"] = [label] * len(raw_quads)
        result["parse_mode"] = "raw_quads"
        result["nonstandard_format"] = True
        result["label_count_mismatch"] = False
        return result

    result["invalid_json"] = True
    if structure["unclosed_object"] or structure["unclosed_string"]:
        result["json_error"] = (
            f"no balanced root object; brace_depth={structure['brace_depth']}; "
            f"unclosed_string={structure['unclosed_string']}")
    elif not has_bbox_key:
        result["json_error"] = "bbox_2d key not found, 且未识别到 <box>/裸坐标等兼容格式"
    else:
        result["json_error"] = "no balanced JSON object containing bbox_2d"
    return result


def parse_bbox_json(text):
    result = inspect_response_format(text)
    result.update(protocol.parse_grounding_record(text))
    return result


def _detect_repeated_token_tail(token_ids, min_repeats=4, max_unit=64):
    """检测输出尾部连续重复 token 单元；只用于诊断，不修改预测。"""
    ids = list(token_ids)
    best = {"unit_tokens": 0, "repeats": 0, "covered_tokens": 0}
    max_unit = min(max_unit, len(ids) // max(min_repeats, 1))
    for unit_len in range(1, max_unit + 1):
        unit = ids[-unit_len:]
        repeats = 1
        cursor = len(ids) - 2 * unit_len
        while cursor >= 0 and ids[cursor:cursor + unit_len] == unit:
            repeats += 1
            cursor -= unit_len
        covered = repeats * unit_len
        if repeats >= min_repeats and covered > best["covered_tokens"]:
            best = {
                "unit_tokens": unit_len,
                "repeats": repeats,
                "covered_tokens": covered,
            }
    return best


def analyze_generation(text, token_ids, max_tokens):
    """从原始文本检测坐标循环、token尾循环和疑似截断。"""
    text = "" if text is None else str(text)
    quads = [match.group(0) for match in _NUMERIC_QUAD_RE.finditer(text)]
    quad_counts = {}
    for quad in quads:
        normalized = re.sub(r"\s+", "", quad)
        quad_counts[normalized] = quad_counts.get(normalized, 0) + 1
    repeated_quad, repeated_quad_count = "", 0
    if quad_counts:
        repeated_quad, repeated_quad_count = max(
            quad_counts.items(), key=lambda item: item[1])
    tail_repeat = _detect_repeated_token_tail(
        token_ids, min_repeats=REPETITION_TAIL_MIN_REPEATS)
    response_tokens = len(token_ids)
    return {
        "response_chars": len(text),
        "response_tokens": response_tokens,
        "token_limit_reached": response_tokens >= max_tokens,
        "numeric_quadruples": len(quads),
        "most_repeated_quad": repeated_quad,
        "most_repeated_quad_count": repeated_quad_count,
        "repeated_coordinate_loop": repeated_quad_count >= REPETITION_MIN_COUNT,
        "tail_repeat_unit_tokens": tail_repeat["unit_tokens"],
        "tail_repeat_count": tail_repeat["repeats"],
        "tail_repeat_covered_tokens": tail_repeat["covered_tokens"],
        "repeated_token_tail": tail_repeat["repeats"] >= REPETITION_TAIL_MIN_REPEATS,
    }


# ================== 坐标转换 ==================
def scale_ground_truth_boxes(boxes, image_path):
    with Image.open(image_path) as image:
        _, _, invalid = protocol.scale_normalized_boxes(boxes, image.size)
        return protocol.scale_boxes_preserving_count(boxes, image.size), list(range(len(boxes))), invalid


def validate_and_scale_boxes(boxes, image_path):
    with Image.open(image_path) as image:
        return protocol.scale_normalized_boxes(boxes, image.size)


def labels_for_valid_boxes(labels, valid_indices):
    labels = labels if isinstance(labels, list) else []
    return [normalize_label(labels[idx]) if idx < len(labels) else ""
            for idx in valid_indices]


def normalize_label(value):
    """RefCOCO 的 labels 是英文类别名（person / chair / hot dog ...）；
    统一小写并压缩空白，便于与预测 labels 逐框比对。"""
    return re.sub(r"\s+", " ", str(value).strip().lower())


# ================== 排序 ==================
def sort_boxes(boxes, labels=None):
    if not boxes:
        return [], [] if labels is not None else None
    order = sorted(range(len(boxes)), key=lambda idx: (boxes[idx][1], boxes[idx][0]))
    sorted_boxes = [boxes[idx] for idx in order]
    if labels is None:
        return sorted_boxes, None
    padded_labels = list(labels) + [""] * max(0, len(boxes) - len(labels))
    return sorted_boxes, [padded_labels[idx] for idx in order]


def evaluation_order_matches(gt_boxes, pred_boxes):
    return protocol.ordered_box_matches(gt_boxes, pred_boxes, IOU_THRESHOLD)


# ================== 可视化 ==================
def draw_boxes(img_path, gt_boxes, pred_boxes, gt_labels, pred_labels, save_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in gt_boxes:
        draw.rectangle([b[0], b[1], b[2], b[3]], outline="green", width=3)
    for b in pred_boxes:
        draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=3)
    img.save(save_path)


# ================== 指令分类 ==================
# 依据 refcoco_val.json 实际指令划分：
#   - "请定位图中所有 X..."      → ALL     （多目标）
#   - "请定位图中最左侧的 X..."  → LEFT    （单目标）
#   - "请定位图中最右侧的 X..."  → RIGHT   （单目标）
#   - "请定位图中最中间的 X..."  → MIDDLE  （单目标）

CATEGORY_KEYS = ["ALL", "LEFT", "RIGHT", "MIDDLE"]

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

CATEGORY_NAMES = {
    "ALL": "检测所有对象",
    "LEFT": "检测最左侧对象",
    "RIGHT": "检测最右侧对象",
    "MIDDLE": "检测最中间对象",
}


# ================== 指标累加器 ==================
class MetricsAccumulator:
    def __init__(self):
        self.N = 0
        self.total_hit_macro = 0.0
        self.total_count_acc = 0
        self.total_over_gen = 0
        self.total_under_gen = 0
        self.total_no_match = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_legacy_miou = 0.0
        self.total_matched_iou = 0.0
        self.total_gt = 0
        self.total_label_correct = 0
        self.total_label_denom = 0
        self.total_empty_text = 0
        self.total_invalid_json = 0
        self.total_nonstandard_format = 0
        self.total_no_valid_box = 0
        self.total_truncated = 0
        self.total_token_limit_reached = 0
        self.total_repeated_coordinate_loop = 0
        self.total_repeated_token_tail = 0
        self.total_label_count_mismatch = 0
        self.total_invalid_pred_boxes = 0
        self.total_invalid_gt_boxes = 0
        self.total_generation_errors = 0
        self.total_eval_loss_errors = 0
        self.total_eval_loss = 0.0
        self.eval_loss_count = 0

    def add_sample(self, gt_boxes, pred_boxes, gt_labels, pred_labels,
                   invalid_pred_count=0, invalid_gt_count=0,
                   diagnostics=None, eval_loss=None):
        diagnostics = diagnostics or {}
        gt_count = len(gt_boxes)
        valid_pred_count = len(pred_boxes)
        pred_count = valid_pred_count + invalid_pred_count

        matches = evaluation_order_matches(gt_boxes, pred_boxes)
        threshold_matches = [match for match in matches if match[0] > IOU_THRESHOLD]
        tp = len(threshold_matches)
        iou_list = [score for score, _, _ in threshold_matches]
        fp = pred_count - tp
        fn = gt_count - tp
        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        iou_sum_per_gt = sum(iou_list) / gt_count if gt_count > 0 else 0.0
        hit = tp / gt_count if gt_count > 0 else 0.0
        count_acc = 1 if pred_count == gt_count else 0
        over_gen = 1 if pred_count > gt_count else 0
        under_gen = 1 if pred_count < gt_count else 0
        no_match = 1 if gt_count > 0 and tp == 0 else 0

        label_correct = 0
        label_denom = 0
        for _, gt_idx, pred_idx in threshold_matches:
            gt_label = gt_labels[gt_idx] if gt_idx < len(gt_labels) else ""
            pred_label = pred_labels[pred_idx] if pred_idx < len(pred_labels) else ""
            if gt_label:
                label_denom += 1
                label_correct += int(pred_label == gt_label)

        self.N += 1
        self.total_hit_macro += hit
        self.total_count_acc += count_acc
        self.total_over_gen += over_gen
        self.total_under_gen += under_gen
        self.total_no_match += no_match
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_legacy_miou += avg_iou
        self.total_matched_iou += sum(iou_list)
        self.total_gt += gt_count
        self.total_label_correct += label_correct
        self.total_label_denom += label_denom
        self.total_empty_text += int(diagnostics.get("empty_text", False))
        self.total_invalid_json += int(diagnostics.get("invalid_json", False))
        self.total_nonstandard_format += int(
            diagnostics.get("nonstandard_format", False))
        self.total_no_valid_box += int(diagnostics.get("no_valid_box", False))
        self.total_truncated += int(diagnostics.get("truncated", False))
        self.total_token_limit_reached += int(
            diagnostics.get("token_limit_reached", False))
        self.total_repeated_coordinate_loop += int(
            diagnostics.get("repeated_coordinate_loop", False))
        self.total_repeated_token_tail += int(
            diagnostics.get("repeated_token_tail", False))
        self.total_label_count_mismatch += int(
            diagnostics.get("label_count_mismatch", False))
        self.total_invalid_pred_boxes += int(invalid_pred_count)
        self.total_invalid_gt_boxes += int(invalid_gt_count)
        self.total_generation_errors += int(diagnostics.get("generation_error", False))
        self.total_eval_loss_errors += int(diagnostics.get("eval_loss_error", False))
        if eval_loss is not None and math.isfinite(eval_loss):
            self.total_eval_loss += eval_loss
            self.eval_loss_count += 1

        return {
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "valid_pred_count": valid_pred_count,
            "avg_iou": avg_iou,
            "matched_iou_sum": sum(iou_list),
            "iou_sum_per_gt": iou_sum_per_gt,
            "hit": hit,
            "label_correct": label_correct,
            "label_denom": label_denom,
        }

    def compute(self):
        n = max(self.N, 1)
        metrics = OrderedDict([
            ("样本数", self.N),
            ("CountAcc", round(self.total_count_acc / n, 4)),
            ("HitMacro", round(self.total_hit_macro / n, 4)),
            ("Recall", round(self.total_tp / (self.total_tp + self.total_fn + 1e-6), 4)),
            ("Precision", round(self.total_tp / (self.total_tp + self.total_fp + 1e-6), 4)),
            ("F1", round(2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn + 1e-6), 4)),
            # 历史字段：每个样本命中框平均IoU，无命中记0，再做宏平均。
            ("mIoU", round(self.total_legacy_miou / n, 4)),
            ("MatchedIoU", round(
                self.total_matched_iou / (self.total_tp + 1e-6), 4)),
            ("IoUSumPerGT", round(
                self.total_matched_iou / (self.total_gt + 1e-6), 4)),
            ("OverGen", round(self.total_over_gen / n, 4)),
            ("UnderGenRate", round(self.total_under_gen / n, 4)),
            ("NoMatchRate", round(self.total_no_match / n, 4)),
            ("EmptyTextRate", round(self.total_empty_text / n, 4)),
            ("InvalidJsonRate", round(self.total_invalid_json / n, 4)),
            ("NonstandardFormatRate", round(
                self.total_nonstandard_format / n, 4)),
            ("NoValidBoxRate", round(self.total_no_valid_box / n, 4)),
            ("TruncatedRate", round(self.total_truncated / n, 4)),
            ("TokenLimitReachedRate", round(
                self.total_token_limit_reached / n, 4)),
            ("RepeatedCoordinateLoopRate", round(
                self.total_repeated_coordinate_loop / n, 4)),
            ("RepeatedTokenTailRate", round(
                self.total_repeated_token_tail / n, 4)),
            ("InvalidBoxPerSample", round(self.total_invalid_pred_boxes / n, 4)),
            ("InvalidGTBoxPerSample", round(self.total_invalid_gt_boxes / n, 4)),
            ("LabelCountMismatchRate", round(
                self.total_label_count_mismatch / n, 4)),
            ("LabelAcc@Matched", round(
                self.total_label_correct / (self.total_label_denom + 1e-6), 4)),
            ("GenerationErrorRate", round(self.total_generation_errors / n, 4)),
            ("EvalLossErrorRate", round(self.total_eval_loss_errors / n, 4)),
        ])
        if self.eval_loss_count > 0:
            metrics["EvalLoss"] = round(
                self.total_eval_loss / self.eval_loss_count, 4)
            metrics["EvalLossCount"] = self.eval_loss_count
        return metrics


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


# ================== 模型加载（原始基座模型） ==================
class InternVLModel:
    def __init__(self, base_path, reporter):
        self.reporter = reporter
        self.load_diagnostics = {}
        print(f"[{datetime.now():%H:%M:%S}] 加载 Tokenizer: {base_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer 没有 eos_token_id，无法可靠停止生成")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            reporter.warn(
                "TOKENIZER_PAD_FALLBACK",
                "tokenizer原本没有pad_token_id，已显式使用eos_token_id作为pad",
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        self.generation_token_ids = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "padding_side": self.tokenizer.padding_side,
        }
        print("  Tokenizer生成ID: " + json.dumps(
            self.generation_token_ids, ensure_ascii=False))

        print(f"[{datetime.now():%H:%M:%S}] 加载基础模型: {base_path}")
        # InternVLChatModel 兼容性补丁
        _orig_getattr = torch.nn.Module.__getattr__
        def _patched_getattr(self, name):
            if name == 'all_tied_weights_keys':
                val = getattr(self, '_tied_weights_keys', None)
                return {} if val is None else val
            return _orig_getattr(self, name)
        torch.nn.Module.__getattr__ = _patched_getattr
        try:
            base_model = AutoModel.from_pretrained(
                base_path, trust_remote_code=True, dtype=torch.bfloat16,
            ).cuda()
        finally:
            torch.nn.Module.__getattr__ = _orig_getattr

        base_generation_config = getattr(base_model, "generation_config", None)
        if base_generation_config is not None:
            base_generation_config.eos_token_id = self.tokenizer.eos_token_id
            base_generation_config.pad_token_id = self.tokenizer.pad_token_id

        language_config = getattr(getattr(base_model, "language_model", None), "config", None)
        max_position_embeddings = getattr(language_config, "max_position_embeddings", None)
        max_image_tokens = getattr(base_model, "num_image_token", 0) * (
            MAX_DYNAMIC_PATCH + 1)
        self.base_diagnostics = {
            "base_path": os.path.realpath(base_path),
            "model_class": type(base_model).__name__,
            "model_template": getattr(base_model, "template", None),
            "system_message_sha256": hashlib.sha256(
                str(getattr(base_model, "system_message", "")).encode("utf-8")
            ).hexdigest(),
            "language_model_type": type(getattr(base_model, "language_model", None)).__name__,
            "max_position_embeddings": max_position_embeddings,
            "estimated_max_image_tokens": max_image_tokens,
            "effective_base_generation_config": (
                {
                    "bos_token_id": getattr(base_generation_config, "bos_token_id", None),
                    "eos_token_id": getattr(base_generation_config, "eos_token_id", None),
                    "pad_token_id": getattr(base_generation_config, "pad_token_id", None),
                    "max_length": getattr(base_generation_config, "max_length", None),
                    "do_sample": getattr(base_generation_config, "do_sample", None),
                } if base_generation_config is not None else {}),
        }
        estimated_all_context = max_image_tokens + TASK_MAX_TOKENS["ALL"] + 512
        if (max_position_embeddings is not None and
                estimated_all_context > max_position_embeddings):
            reporter.warn(
                "POTENTIAL_CONTEXT_OVERFLOW",
                "最大动态切图token + ALL生成上限可能超过语言模型上下文；"
                "需核对实际num_patches和prompt长度",
                severity="critical",
                max_position_embeddings=max_position_embeddings,
                estimated_max_image_tokens=max_image_tokens,
                max_new_tokens=TASK_MAX_TOKENS["ALL"],
                estimated_with_prompt_margin=estimated_all_context,
            )

        print(f"[{datetime.now():%H:%M:%S}] 使用原始基座模型，不加载适配器权重")
        self.model = base_model
        self.load_diagnostics = {
            "mode": "base_model_only",
            "base": self.base_diagnostics,
            "generation_token_ids": self.generation_token_ids,
        }

        self.model.eval()
        self.device = torch.device("cuda:0")

        model_generation_config = getattr(self.model, "generation_config", None)
        if model_generation_config is not None:
            model_generation_config.eos_token_id = self.tokenizer.eos_token_id
            model_generation_config.pad_token_id = self.tokenizer.pad_token_id

        # 预加载 conversation 模板模块
        import importlib.util
        conv_path = os.path.join(base_path, 'conversation.py')
        spec = importlib.util.spec_from_file_location('conversation', conv_path)
        self._conv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._conv_module)

        self.load_diagnostics["base"] = self.base_diagnostics
        self.load_diagnostics["generation_token_ids"] = self.generation_token_ids
        print(f"[{datetime.now():%H:%M:%S}] 模型就绪, device={self.device}")

    def prepare_image(self, image_path):
        """每个样本只做一次动态切图和归一化，生成与loss共用张量。"""
        return load_image_pixel_values(
            image_path, input_size=INPUT_SIZE, max_num=MAX_DYNAMIC_PATCH
        ).to(device=self.device, dtype=torch.bfloat16)

    def generate_single(self, pixel_values, prompt, category):
        """单样本greedy推理，显式传入pad/eos并记录返回类型。"""

        generation_config = dict(
            max_new_tokens=TASK_MAX_TOKENS[category],
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        elapsed = time.perf_counter() - started
        response_type = type(response).__name__
        coerced = False
        if isinstance(response, str):
            text = response
        elif isinstance(response, tuple) and response and isinstance(response[0], str):
            text = response[0]
            coerced = True
        elif isinstance(response, dict):
            candidate = response.get("response", response.get("text"))
            if not isinstance(candidate, str):
                raise TypeError(
                    f"model.chat返回dict但没有字符串response/text，keys={list(response)[:10]}")
            text = candidate
            coerced = True
        else:
            raise TypeError(f"model.chat返回不支持的类型: {response_type}")
        return text, {
            "response_type": response_type,
            "response_coerced": coerced,
            "generation_seconds": elapsed,
            "generation_config": generation_config,
        }

    def compute_eval_loss_single(self, pixel_values, prompt, gt_answer):
        """计算单样本的 eval loss (cross-entropy on GT answer tokens)"""
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        get_conv_template = self._conv_module.get_conv_template

        num_patches = pixel_values.shape[0]
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN

        # 构造 prompt 部分（不含 GT answer），用于确定 label mask 起始位置
        template_prompt = get_conv_template(self.model.template)
        template_prompt.system_message = self.model.system_message
        template_prompt.append_message(template_prompt.roles[0], prompt)
        template_prompt.append_message(template_prompt.roles[1], None)
        prompt_only = template_prompt.get_prompt()
        prompt_only = prompt_only.replace('<image>', image_tokens, 1)

        # 构造完整序列（prompt + GT answer）
        template_full = get_conv_template(self.model.template)
        template_full.system_message = self.model.system_message
        template_full.append_message(template_full.roles[0], prompt)
        template_full.append_message(template_full.roles[1], gt_answer)
        full_text = template_full.get_prompt()
        full_text = full_text.replace('<image>', image_tokens, 1)

        # Tokenize
        prompt_tokens = self.tokenizer(prompt_only, return_tensors='pt')
        prompt_len = prompt_tokens.input_ids.shape[1]

        full_tokens = self.tokenizer(full_text, return_tensors='pt')
        input_ids = full_tokens.input_ids.to(self.device)
        attention_mask = full_tokens.attention_mask.to(self.device)

        # 创建 labels：prompt 部分为 -100，answer 部分为真实 token id
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100

        # 获取 visual features 并替换 IMG_CONTEXT token 的 embedding
        # 全程在 no_grad 下，避免参数梯度与 in-place 操作冲突
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            img_context_token_id = self.model.img_context_token_id
            selected = (input_ids == img_context_token_id)
            input_embeds[selected.reshape(B * N)] = vit_embeds.reshape(-1, C)
            input_embeds = input_embeds.reshape(B, N, C)

            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return outputs.loss.item()


# ================== 主流程 ==================
CSV_FIELDS = [
    "图片", "原始索引", "GT数量", "预测数量", "有效预测数量", "非法预测框数",
    "命中率", "mIoU", "MatchedIoU", "IoUSumPerGT", "TP", "FP", "FN", "类别",
    "目标类别", "ParseMode", "NonstandardFormat",
    "LabelCorrect", "LabelDenom", "EmptyText", "InvalidJson", "NoValidBox",
    "Truncated", "ResponseTokens", "TokenLimitReached", "LabelCountMismatch",
    "RepeatedCoordinateLoop", "RepeatedTokenTail", "MostRepeatedQuadCount",
    "GenerationSeconds", "ResponseType", "JsonError", "WarningCodes",
    "GenerationError", "EvalLossError", "EvalLoss",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_full_dataset(raw_dataset):
    """GPU加载前完整扫描数据结构、GT JSON和图片路径，避免半程才暴露数据问题。"""
    counts = {category: 0 for category in CATEGORY_KEYS}
    errors = []
    missing_images = []
    invalid_gt_boxes = 0
    for source_idx, sample in enumerate(raw_dataset):
        try:
            images = sample["images"]
            conversations = sample["conversations"]
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"images必须恰好包含1项，实际={images}")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError("conversations不足2轮")
            prompt = conversations[0]["value"]
            gt_text = conversations[1]["value"]
            category = classify_prompt(prompt)
            counts[category] += 1
            gt_parsed = parse_bbox_json(gt_text)
            if gt_parsed["empty_text"] or gt_parsed["invalid_json"]:
                raise ValueError(f"GT JSON失败: {gt_parsed['json_error']}")
            if gt_parsed["label_count_mismatch"]:
                raise ValueError("GT boxes/labels数量不一致")
            for box in gt_parsed["boxes"]:
                try:
                    values = [float(value) for value in box[:4]]
                    if (len(box) < 4 or not all(math.isfinite(value) for value in values) or
                            not (0 <= values[0] < values[2] <= 1000) or
                            not (0 <= values[1] < values[3] <= 1000)):
                        invalid_gt_boxes += 1
                except (TypeError, ValueError, IndexError):
                    invalid_gt_boxes += 1
            image_path = os.path.join(IMAGE_ROOT, images[0])
            if not os.path.isfile(image_path):
                missing_images.append({"source_index": source_idx, "path": image_path})
        except Exception as exc:
            errors.append({"source_index": source_idx, "error": str(exc)})
    diagnostics = {
        "samples": len(raw_dataset),
        "category_counts": counts,
        "schema_or_gt_errors": errors,
        "missing_images": missing_images,
        "invalid_gt_boxes": invalid_gt_boxes,
    }
    if errors or missing_images:
        raise RuntimeError(
            f"全量数据预检失败: schema/GT错误={len(errors)}, "
            f"缺失图片={len(missing_images)}, examples={(errors or missing_images)[:5]}")
    return diagnostics


def _prediction_preview(text, limit=500):
    text = "" if text is None else str(text)
    return {
        "prediction_head": text[:limit],
        "prediction_tail": text[-limit:] if len(text) > limit else text,
    }


def _emit_sample_warnings(reporter, *, eval_idx, source_idx, image, category,
                          gt_text, pred_text, diagnostics, generation_analysis,
                          generation_meta, invalid_pred_count):
    """返回本样本告警代码；所有告警只诊断，不修复或篡改主评估结果。"""
    codes = []
    base_context = {
        "eval_index": eval_idx,
        "source_index": source_idx,
        "image": image,
        "category": category,
        "gt_chars": len(gt_text),
        **generation_analysis,
        **_prediction_preview(pred_text),
    }

    def emit(code, message, severity="warning", **extra):
        codes.append(code)
        reporter.warn(code, message, severity=severity,
                      **base_context, **extra)

    if generation_meta.get("response_coerced"):
        emit(
            "CHAT_RESPONSE_TYPE_COERCED",
            "model.chat没有直接返回str，已从容器中提取文本；需核对remote code版本",
            response_type=generation_meta.get("response_type"),
        )
    if diagnostics["generation_error"]:
        emit("GENERATION_ERROR", "模型生成发生异常，已保留异常信息", severity="critical",
             error=generation_meta.get("generation_error", ""))
    if diagnostics["empty_text"]:
        emit("EMPTY_TEXT", "model.chat返回了空文本", severity="critical")
    if generation_analysis["repeated_coordinate_loop"]:
        emit(
            "REPEATED_COORDINATE_LOOP",
            "检测到同一坐标框高频重复，模型很可能进入greedy循环",
            severity="critical",
            repeated_quad=generation_analysis["most_repeated_quad"],
            repeated_count=generation_analysis["most_repeated_quad_count"],
        )
    if generation_analysis["repeated_token_tail"]:
        emit(
            "REPEATED_TOKEN_TAIL",
            "输出尾部存在连续token循环",
            severity="critical",
            unit_tokens=generation_analysis["tail_repeat_unit_tokens"],
            repeats=generation_analysis["tail_repeat_count"],
        )
    if generation_analysis["token_limit_reached"]:
        emit(
            "TOKEN_LIMIT_REACHED",
            "输出达到任务token上限；若同时出现重复告警，不能通过继续增大上限解决",
            severity="critical" if diagnostics["invalid_json"] else "warning",
        )
    if diagnostics["invalid_json"]:
        emit(
            "INVALID_JSON",
            "原始输出无法解析为完整JSON",
            severity="critical",
            json_error=diagnostics.get("json_error", ""),
            brace_depth=diagnostics.get("brace_depth", 0),
            unclosed_string=diagnostics.get("unclosed_string", False),
        )
    if diagnostics.get("nonstandard_format"):
        emit(
            "NONSTANDARD_OUTPUT_FORMAT",
            "预测不是训练/GT的 {\"bbox_2d\":..,\"labels\":..} 格式，"
            "已由兼容解析器提取并照常计入指标；格式分布见 summary.json",
            parse_mode=diagnostics.get("parse_mode", ""),
            repairs=diagnostics.get("repairs", []),
            array_object_count=diagnostics.get("array_object_count", 0),
        )
    if diagnostics["label_count_mismatch"]:
        emit("LABEL_COUNT_MISMATCH", "bbox_2d与labels数量不一致")
    if (diagnostics["truncated"] and
            not generation_analysis["token_limit_reached"]):
        emit(
            "TRUNCATED_OUTPUT",
            "输出未达token上限却中途截断（多为提前EOS，JSON末尾的 ]} 缺失）；"
            "已由修复层补齐括号并照常评分，但预测框数可能偏少",
            parse_mode=diagnostics.get("parse_mode", ""),
            repairs=diagnostics.get("repairs", []),
        )
    if invalid_pred_count:
        emit(
            "INVALID_PREDICTION_BOX",
            "预测包含越界、退化或非有限坐标框；这些框已按规则计入FP",
            invalid_pred_boxes=invalid_pred_count,
        )
    if (len(pred_text) > max(2000, len(gt_text) * 3) and
            not diagnostics["empty_text"]):
        emit(
            "OUTPUT_MUCH_LONGER_THAN_GT",
            "预测字符数超过GT的3倍且至少2000字符，疑似过生成",
            severity="critical" if diagnostics["invalid_json"] else "warning",
        )
    if generation_meta.get("generation_seconds", 0) >= 300:
        emit(
            "SLOW_GENERATION",
            "单样本生成超过300秒，通常与长输出或循环有关",
            generation_seconds=generation_meta["generation_seconds"],
        )
    return codes


def main(reporter):
    random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    torch.cuda.manual_seed_all(EVAL_SEED)

    print("=" * 60)
    print("InternVL3-8B 原始基座模型 RefCOCO 检测评估 (diagnostic-v1)")
    print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"评估口径: {EVAL_SCHEMA_VERSION}")
    print(f"模型: {BASE_MODEL_PATH}")
    print(f"测试集: {DATA_JSON}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"seed={EVAL_SEED}, BATCH_SIZE=1, greedy=True, "
          f"task_max_tokens={TASK_MAX_TOKENS}, IoU>{IOU_THRESHOLD}")
    print("生成: 显式传入pad_token_id/eos_token_id；全量评估不中断；第50条输出健康检查")
    print("匹配: 全局IoU降序一对一；非法预测框计入FP/count；退化GT框排除并单独统计")
    print("=" * 60)

    with open(DATA_JSON, "r", encoding="utf-8") as file:
        raw_dataset = json.load(file)
    dataset_diagnostics = _validate_full_dataset(raw_dataset)
    indexed_dataset = list(enumerate(raw_dataset))
    random.Random(EVAL_SEED).shuffle(indexed_dataset)
    print(f"\n加载测试集: {len(indexed_dataset)} 条，已用seed={EVAL_SEED}固定打乱")
    print("  全量预检通过: 数据结构/GT JSON/图片路径正常；"
          f"退化GT框={dataset_diagnostics['invalid_gt_boxes']}")

    groups = {category: [] for category in CATEGORY_KEYS}
    for source_idx, sample in indexed_dataset:
        prompt = sample["conversations"][0]["value"]
        groups[classify_prompt(prompt)].append(source_idx)
    print("指令分类统计:")
    for category in CATEGORY_KEYS:
        print(f"  {CATEGORY_NAMES[category]:20s}: {len(groups[category]):4d} 条")
    print(f"  {'总计':20s}: {sum(len(groups[c]) for c in groups):4d} 条")

    # 按目标类别分组（person / chair / hot dog ...）
    category_groups = {}
    for source_idx, sample in indexed_dataset:
        prompt = sample["conversations"][0]["value"]
        label = extract_target_category(prompt)
        category_groups.setdefault(label, []).append(source_idx)
    print("\n目标类别统计:")
    for label in sorted(category_groups, key=lambda k: -len(category_groups[k])):
        print(f"  {label:16s}: {len(category_groups[label]):4d} 条")
    print(f"  {'总计':16s}: "
          f"{sum(len(v) for v in category_groups.values()):4d} 条")

    model = InternVLModel(BASE_MODEL_PATH, reporter)
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        reporter.warn(
            "PAD_EQUALS_EOS_EXPLICIT",
            "pad_token_id与eos_token_id相同；这是开放式生成的常见配置，"
            "本脚本已显式传入二者，不再依赖Transformers自动兜底",
            severity="info",
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    initial_run_diagnostics = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "status": "model_loaded_before_full_evaluation",
        "dataset": dataset_diagnostics,
        "model_load": model.load_diagnostics,
    }
    with open(RUN_DIAGNOSTICS_JSON, "w", encoding="utf-8") as file:
        json.dump(initial_run_diagnostics, file, indent=2, ensure_ascii=False,
                  allow_nan=False)
    print(f"  加载诊断已写入: {RUN_DIAGNOSTICS_JSON}")
    accumulators = {category: MetricsAccumulator()
                    for category in CATEGORY_KEYS}
    accumulators["OVERALL"] = MetricsAccumulator()
    # 目标类别累加器（person / chair / hot dog ...）
    cat_acc = {label: MetricsAccumulator() for label in category_groups}
    # 预测格式分布（bbox_2d_object / bbox_2d_array / internvl_box / qwen_box / raw_quads / none）
    format_counts = {}
    csv_rows = {category: [] for category in CATEGORY_KEYS}
    all_rows = []
    consecutive_generation_errors = 0

    with open(PREDICTIONS_JSONL, "w", encoding="utf-8") as jsonl_file:
        for eval_idx, (source_idx, sample) in enumerate(tqdm(
                indexed_dataset, desc="评估进度", ncols=100), start=1):
            img_rel = sample["images"][0]
            image_path = os.path.join(IMAGE_ROOT, img_rel)
            prompt = sample["conversations"][0]["value"]
            gt_text = sample["conversations"][1]["value"]
            category = classify_prompt(prompt)
            target_label = extract_target_category(prompt)
            generation_error = False
            eval_loss_error = False
            pixel_values = None
            generation_meta = {
                "response_type": "",
                "response_coerced": False,
                "generation_seconds": 0.0,
                "generation_config": {},
            }

            try:
                pixel_values = model.prepare_image(image_path)
                pred_text, generation_meta = model.generate_single(
                    pixel_values, prompt, category)
            except Exception as exc:
                generation_error = True
                pred_text = ""
                generation_meta["generation_error"] = repr(exc)
                print(f"[{datetime.now():%H:%M:%S}] "
                      f"[{eval_idx:4d}/{len(indexed_dataset)}] 推理异常: {exc}")

            if pixel_values is not None:
                try:
                    eval_loss = model.compute_eval_loss_single(
                        pixel_values, prompt, gt_text)
                    if not math.isfinite(eval_loss):
                        raise ValueError(f"non-finite eval loss: {eval_loss}")
                except Exception as exc:
                    eval_loss_error = True
                    eval_loss = None
                    print(f"[{datetime.now():%H:%M:%S}] "
                          f"[{eval_idx:4d}/{len(indexed_dataset)}] "
                          f"EvalLoss计算异常: {exc}")
            else:
                eval_loss_error = True
                eval_loss = None

            gt_parsed = parse_bbox_json(gt_text)
            pred_parsed = parse_bbox_json(pred_text)
            if gt_parsed["empty_text"] or gt_parsed["invalid_json"]:
                raise ValueError(f"GT JSON不可解析: source_idx={source_idx}, image={img_rel}")

            gt_boxes, gt_indices, invalid_gt_count = scale_ground_truth_boxes(
                gt_parsed["boxes"], image_path)
            pred_boxes, pred_indices, invalid_pred_count = validate_and_scale_boxes(
                pred_parsed["boxes"], image_path)
            gt_labels = labels_for_valid_boxes(gt_parsed["labels"], gt_indices)
            pred_labels = labels_for_valid_boxes(pred_parsed["labels"], pred_indices)
            response_token_ids = model.tokenizer.encode(
                pred_text, add_special_tokens=False)
            generation_analysis = analyze_generation(
                pred_text, response_token_ids, TASK_MAX_TOKENS[category])
            response_tokens = generation_analysis["response_tokens"]
            token_limit_reached = generation_analysis["token_limit_reached"]

            diagnostics = {
                # 推理异常产生的占位空串不是模型真实空输出，必须分开统计。
                "empty_text": pred_parsed["empty_text"] and not generation_error,
                "invalid_json": pred_parsed["invalid_json"],
                "no_valid_box": (
                    not pred_parsed["empty_text"] and
                    not pred_parsed["invalid_json"] and
                    len(pred_boxes) == 0),
                "truncated": pred_parsed["truncated"] or token_limit_reached,
                "token_limit_reached": token_limit_reached,
                "label_count_mismatch": pred_parsed["label_count_mismatch"],
                "has_bbox_key": pred_parsed["has_bbox_key"],
                "has_box_tag": pred_parsed["has_box_tag"],
                "parse_mode": pred_parsed["parse_mode"],
                "nonstandard_format": pred_parsed["nonstandard_format"],
                "array_object_count": pred_parsed["array_object_count"],
                "repairs": pred_parsed["repairs"],
                "balanced_object_count": pred_parsed["balanced_object_count"],
                "brace_depth": pred_parsed["brace_depth"],
                "unclosed_string": pred_parsed["unclosed_string"],
                "json_error": pred_parsed["json_error"],
                "repeated_coordinate_loop": generation_analysis[
                    "repeated_coordinate_loop"],
                "repeated_token_tail": generation_analysis["repeated_token_tail"],
                "generation_error": generation_error,
                "eval_loss_error": eval_loss_error,
            }
            warning_codes = _emit_sample_warnings(
                reporter,
                eval_idx=eval_idx,
                source_idx=source_idx,
                image=img_rel,
                category=category,
                gt_text=gt_text,
                pred_text=pred_text,
                diagnostics=diagnostics,
                generation_analysis=generation_analysis,
                generation_meta=generation_meta,
                invalid_pred_count=invalid_pred_count,
            )
            sample_metrics = accumulators[category].add_sample(
                gt_boxes, pred_boxes, gt_labels, pred_labels,
                invalid_pred_count=invalid_pred_count,
                invalid_gt_count=invalid_gt_count,
                diagnostics=diagnostics, eval_loss=eval_loss)
            accumulators["OVERALL"].add_sample(
                gt_boxes, pred_boxes, gt_labels, pred_labels,
                invalid_pred_count=invalid_pred_count,
                invalid_gt_count=invalid_gt_count,
                diagnostics=diagnostics, eval_loss=eval_loss)
            cat_acc[target_label].add_sample(
                gt_boxes, pred_boxes, gt_labels, pred_labels,
                invalid_pred_count=invalid_pred_count,
                invalid_gt_count=invalid_gt_count,
                diagnostics=diagnostics, eval_loss=eval_loss)
            format_counts[pred_parsed["parse_mode"]] = (
                format_counts.get(pred_parsed["parse_mode"], 0) + 1)

            if (eval_idx - 1) % VISUALIZE_EVERY == 0:
                vis_gt_boxes, vis_gt_labels = sort_boxes(gt_boxes, gt_labels)
                vis_pred_boxes, vis_pred_labels = sort_boxes(pred_boxes, pred_labels)
                save_name = f"{source_idx:04d}_{category}_{os.path.basename(img_rel)}"
                draw_boxes(image_path, vis_gt_boxes, vis_pred_boxes,
                           vis_gt_labels, vis_pred_labels,
                           os.path.join(IMAGE_SAVE_DIR, save_name))

            tp = sample_metrics["tp"]
            fp = sample_metrics["fp"]
            fn = sample_metrics["fn"]
            per_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            per_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_f1 = (2 * per_precision * per_recall / (per_precision + per_recall)
                      if (per_precision + per_recall) > 0 else 0.0)
            status = ",".join(key for key in (
                "empty_text", "invalid_json", "no_valid_box", "truncated",
                "token_limit_reached",
                "repeated_coordinate_loop", "repeated_token_tail",
                "label_count_mismatch", "generation_error", "eval_loss_error")
                              if diagnostics.get(key)) or "ok"

            row = {
                "图片": img_rel,
                "原始索引": source_idx,
                "GT数量": sample_metrics["gt_count"],
                "预测数量": sample_metrics["pred_count"],
                "有效预测数量": sample_metrics["valid_pred_count"],
                "非法预测框数": invalid_pred_count,
                "命中率": round(sample_metrics["hit"], 4),
                "mIoU": round(sample_metrics["avg_iou"], 4),
                "MatchedIoU": round(
                    sample_metrics["matched_iou_sum"] / max(tp, 1), 4),
                "IoUSumPerGT": round(sample_metrics["iou_sum_per_gt"], 4),
                "TP": tp, "FP": fp, "FN": fn, "类别": category,
                "目标类别": target_label,
                "ParseMode": pred_parsed["parse_mode"],
                "NonstandardFormat": int(pred_parsed["nonstandard_format"]),
                "LabelCorrect": sample_metrics["label_correct"],
                "LabelDenom": sample_metrics["label_denom"],
                "EmptyText": int(diagnostics["empty_text"]),
                "InvalidJson": int(diagnostics["invalid_json"]),
                "NoValidBox": int(diagnostics["no_valid_box"]),
                "Truncated": int(diagnostics["truncated"]),
                "ResponseTokens": response_tokens,
                "TokenLimitReached": int(token_limit_reached),
                "LabelCountMismatch": int(diagnostics["label_count_mismatch"]),
                "RepeatedCoordinateLoop": int(
                    diagnostics["repeated_coordinate_loop"]),
                "RepeatedTokenTail": int(diagnostics["repeated_token_tail"]),
                "MostRepeatedQuadCount": generation_analysis[
                    "most_repeated_quad_count"],
                "GenerationSeconds": round(
                    generation_meta.get("generation_seconds", 0.0), 3),
                "ResponseType": generation_meta.get("response_type", ""),
                "JsonError": diagnostics["json_error"],
                "WarningCodes": ",".join(warning_codes),
                "GenerationError": int(generation_error),
                "EvalLossError": int(eval_loss_error),
                "EvalLoss": round(eval_loss, 6) if eval_loss is not None else "",
            }
            csv_rows[category].append(row)
            all_rows.append(row)

            jsonl_record = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "eval_index": eval_idx,
                "eval_index_zero_based": eval_idx - 1,
                "source_index": source_idx,
                "image": img_rel,
                "category": category,
                "object_category": target_label,
                "prompt": prompt,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "diagnostics": diagnostics,
                "generation": {
                    **generation_meta,
                    **generation_analysis,
                },
                "warning_codes": warning_codes,
                "invalid_pred_boxes": invalid_pred_count,
                "invalid_gt_boxes": invalid_gt_count,
                "metrics": sample_metrics,
                "eval_loss": eval_loss,
            }
            jsonl_file.write(json.dumps(
                jsonl_record, ensure_ascii=False, allow_nan=False) + "\n")
            jsonl_file.flush()

            loss_text = f"Loss:{eval_loss:.4f}" if eval_loss is not None else "Loss:NA"
            print(f"[{datetime.now():%H:%M:%S}] "
                  f"[{eval_idx:4d}/{len(indexed_dataset)}] "
                  f"({CATEGORY_NAMES[category]}) GT:{sample_metrics['gt_count']} "
                  f"Pred:{sample_metrics['pred_count']} TP:{tp} FP:{fp} FN:{fn} "
                  f"Hit:{sample_metrics['hit']:.3f} P:{per_precision:.3f} "
                  f"R:{per_recall:.3f} F1:{per_f1:.3f} "
                  f"mIoU:{sample_metrics['avg_iou']:.3f} "
                  f"Obj:{target_label} "
                  f"InvalidBox:{invalid_pred_count} status={status} {loss_text}")
            if generation_error:
                consecutive_generation_errors += 1
            else:
                consecutive_generation_errors = 0
            if consecutive_generation_errors >= MAX_CONSECUTIVE_GENERATION_ERRORS:
                reporter.warn(
                    "GENERATION_FAIL_FAST",
                    f"连续{consecutive_generation_errors}条推理异常，停止全量评估，"
                    "避免把代码/环境错误误记为模型空输出",
                    severity="critical",
                    eval_index=eval_idx,
                    last_error=generation_meta.get("generation_error", ""),
                )
                raise RuntimeError(
                    f"连续{consecutive_generation_errors}条推理异常，已触发fail-fast；"
                    f"最后错误={generation_meta.get('generation_error', '')}")
            del pixel_values
            if eval_idx == EARLY_HEALTHCHECK_SAMPLES:
                early = accumulators["OVERALL"]
                invalid_rate = early.total_invalid_json / max(early.N, 1)
                nonstandard_rate = (
                    early.total_nonstandard_format / max(early.N, 1))
                truncated_rate = early.total_truncated / max(early.N, 1)
                token_limit_rate = (
                    early.total_token_limit_reached / max(early.N, 1))
                empty_rate = early.total_empty_text / max(early.N, 1)
                print("\n" + "=" * 60)
                print(f"前{EARLY_HEALTHCHECK_SAMPLES}条生成健康检查: "
                      f"InvalidJson={invalid_rate:.2%}, "
                      f"NonstandardFormat={nonstandard_rate:.2%}, "
                      f"Truncated={truncated_rate:.2%}, "
                      f"TokenLimit={token_limit_rate:.2%}, Empty={empty_rate:.2%}")
                print(f"  预测格式分布: {format_counts}")
                if truncated_rate >= 0.2:
                    reporter.warn(
                        "EARLY_TRUNCATION_HIGH",
                        "前50条截断率≥20%：基座模型大量提前EOS（JSON未闭合）。"
                        "修复层已保住已生成的框，但CountAcc/F1会被这种截断拉低，"
                        "建议先看 predictions.jsonl 确认后再跑全量",
                        severity="warning",
                        eval_index=eval_idx,
                        truncated_rate=truncated_rate,
                        format_counts=format_counts,
                    )
                if invalid_rate >= 0.05 or token_limit_rate >= 0.02:
                    reporter.warn(
                        "EARLY_HEALTHCHECK_DEGRADED",
                        "前50条格式/截断率明显偏高；脚本按用户要求继续完整评估，"
                        "请结合warnings.jsonl判断基座模型greedy生成是否退化",
                        severity="critical",
                        eval_index=eval_idx,
                        invalid_json_rate=invalid_rate,
                        token_limit_rate=token_limit_rate,
                        empty_text_rate=empty_rate,
                    )
                else:
                    print("  早期生成健康状态未触发阈值；继续全量评估。")
                print("=" * 60 + "\n")
            if eval_idx % 50 == 0:
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    all_results = OrderedDict()
    for category in ["OVERALL"] + CATEGORY_KEYS:
        name = "总体" if category == "OVERALL" else CATEGORY_NAMES[category]
        metrics = accumulators[category].compute()
        all_results[name] = metrics
        print(f"\n{'─' * 40}\n【{name}】\n{'─' * 40}")
        for metric_name, value in metrics.items():
            if metric_name in {"样本数", "EvalLossCount"}:
                print(f"  {metric_name:24s}: {value}")
            else:
                print(f"  {metric_name:24s}: {value:.4f}")

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

    for category in CATEGORY_KEYS:
        _write_csv(os.path.join(OUTPUT_DIR, f"result_{category}.csv"),
                   csv_rows[category])
    all_csv = os.path.join(OUTPUT_DIR, "result_all.csv")
    _write_csv(all_csv, all_rows)

    # 按目标类别聚合的指标表（person / chair / hot dog ...），便于直接透视统计。
    object_csv_path = os.path.join(
        OUTPUT_DIR, "result_by_object_category.csv")
    object_csv_fields = [
        "目标类别", "样本数", "CountAcc", "HitMacro", "Recall", "Precision",
        "F1", "mIoU", "OverGen", "MatchedIoU", "IoUSumPerGT",
        "LabelAcc@Matched", "EvalLoss",
    ]
    with open(object_csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(object_csv_fields)
        for label in cat_order[1:]:
            metrics = cat_acc[label].compute()
            writer.writerow([
                label, metrics["样本数"], metrics["CountAcc"],
                metrics["HitMacro"], metrics["Recall"], metrics["Precision"],
                metrics["F1"], metrics["mIoU"], metrics["OverGen"],
                metrics["MatchedIoU"], metrics["IoUSumPerGT"],
                metrics.get("LabelAcc@Matched", ""),
                metrics.get("EvalLoss", ""),
            ])

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    summary = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "config": {
            "base_model": BASE_MODEL_PATH,
            "model_mode": "base_model_only",
            "data": DATA_JSON,
            "seed": EVAL_SEED,
            "shuffle": True,
            "batch_size": 1,
            "do_sample": False,
            "pad_token_id": model.tokenizer.pad_token_id,
            "eos_token_id": model.tokenizer.eos_token_id,
            "explicit_pad_and_eos": True,
            "task_max_tokens": TASK_MAX_TOKENS,
            "iou_threshold": IOU_THRESHOLD,
            "matching": "spatial_gt_order_greedy_one_to_one",
            "invalid_prediction_policy": "count_as_fp_and_prediction_count",
            "invalid_gt_policy": "exclude_and_report",
            "legacy_miou_definition": (
                "per-sample mean IoU of IoU-threshold matches; zero if no match"),
        },
        "dataset_diagnostics": dataset_diagnostics,
        "model_load_diagnostics": model.load_diagnostics,
        "warning_counts": reporter.counts,
        "prediction_format_counts": format_counts,
        "results": all_results,
        "object_category_results": {
            label: cat_acc[label].compute() for label in cat_order[1:]
        },
    }
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)

    final_run_diagnostics = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "status": "full_evaluation_completed",
        "dataset": dataset_diagnostics,
        "model_load": model.load_diagnostics,
        "warning_counts": reporter.counts,
        "prediction_format_counts": format_counts,
        "results": all_results,
    }
    with open(RUN_DIAGNOSTICS_JSON, "w", encoding="utf-8") as file:
        json.dump(final_run_diagnostics, file, indent=2, ensure_ascii=False,
                  allow_nan=False)

    print(f"\n{'=' * 60}\n结果保存:")
    print(f"  日志:      {LOG_FILE}")
    print(f"  汇总:      {summary_path}")
    print(f"  全量CSV:   {all_csv}")
    print(f"  目标类别CSV: {object_csv_path}")
    print(f"  预测格式分布: {format_counts}")
    print(f"  逐样本JSONL: {PREDICTIONS_JSONL}")
    print(f"  分级告警JSONL: {WARNINGS_JSONL}")
    print(f"  加载/运行诊断: {RUN_DIAGNOSTICS_JSON}")
    print(f"  可视化:    {IMAGE_SAVE_DIR}")
    print(f"完成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 60}")


def run_probe(count=3):
    """快速探针：只跑随机打乱后的前 count 条，打印原始输出与解析结果后退出。

    用于在不跑完 2000+ 条的前提下确认基座模型的真实输出格式（不做全量落盘）。
    """
    with open(DATA_JSON, "r", encoding="utf-8") as file:
        raw_dataset = json.load(file)
    indexed_dataset = list(enumerate(raw_dataset))
    random.Random(EVAL_SEED).shuffle(indexed_dataset)

    reporter = DiagnosticReporter(
        os.path.join(OUTPUT_DIR, "probe_warnings.jsonl"))
    try:
        model = InternVLModel(BASE_MODEL_PATH, reporter)
        for eval_idx, (source_idx, sample) in enumerate(
                indexed_dataset[:count], start=1):
            img_rel = sample["images"][0]
            image_path = os.path.join(IMAGE_ROOT, img_rel)
            prompt = sample["conversations"][0]["value"]
            gt_text = sample["conversations"][1]["value"]
            category = classify_prompt(prompt)
            pixel_values = model.prepare_image(image_path)
            pred_text, generation_meta = model.generate_single(
                pixel_values, prompt, category)
            pred_parsed = parse_bbox_json(pred_text)
            gt_parsed = parse_bbox_json(gt_text)
            pred_coords = [float(value) for box in pred_parsed["boxes"]
                           for value in box[:4]]
            gt_coords = [float(value) for box in gt_parsed["boxes"]
                         for value in box[:4]]
            print("=" * 70)
            print(f"[probe {eval_idx}/{count}] source_idx={source_idx} "
                  f"image={img_rel} category={category} "
                  f"object={extract_target_category(prompt)}")
            print(f"prompt: {prompt[:150]!r}")
            print(f"GT: mode={gt_parsed['parse_mode']} "
                  f"boxes={len(gt_parsed['boxes'])} "
                  f"coord_min={min(gt_coords) if gt_coords else None} "
                  f"coord_max={max(gt_coords) if gt_coords else None}")
            print(f"PRED raw (chars={len(pred_text)}, "
                  f"seconds={generation_meta.get('generation_seconds', 0):.1f}):")
            print(pred_text[:2000])
            if len(pred_text) > 2000:
                print(f"... [tail 300] {pred_text[-300:]!r}")
            print(f"PRED parse: mode={pred_parsed['parse_mode']} "
                  f"boxes={len(pred_parsed['boxes'])} "
                  f"labels={pred_parsed['labels'][:5]} "
                  f"invalid_json={pred_parsed['invalid_json']} "
                  f"nonstandard={pred_parsed['nonstandard_format']} "
                  f"has_box_tag={pred_parsed['has_box_tag']} "
                  f"json_error={pred_parsed['json_error']!r}")
            print(f"PRED coord_min={min(pred_coords) if pred_coords else None} "
                  f"coord_max={max(pred_coords) if pred_coords else None}")
            print("=" * 70)
            del pixel_values
    finally:
        reporter.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--probe" in argv:
        position = argv.index("--probe")
        probe_count = 3
        if position + 1 < len(argv) and argv[position + 1].isdigit():
            probe_count = max(1, int(argv[position + 1]))
        print(f"探针模式: 只跑 {probe_count} 条并打印原始输出，不写入全量结果")
        run_probe(probe_count)
        sys.exit(0)
    tee = Tee(LOG_FILE)
    sys.stdout = tee
    diagnostic_reporter = DiagnosticReporter(WARNINGS_JSONL)
    try:
        main(diagnostic_reporter)
    finally:
        diagnostic_reporter.close()
        tee.log.close()
        sys.stdout = tee.terminal
