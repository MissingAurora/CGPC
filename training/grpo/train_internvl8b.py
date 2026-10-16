"""CGPC release: training/grpo/train_internvl8b.py. See the README in this directory for configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import os as _release_os

import json
import copy
import math
import os
import re
import sys
import random
import argparse
import hashlib
import time
import importlib
import inspect
import warnings
from functools import lru_cache
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader, Sampler

# triton compat for bitsandbytes
_fake_triton_ops = type(sys)('triton.ops')
_fake_triton_ops.matmul_perf_model = type(sys)('matmul_perf_model')
_fake_triton_ops.matmul_perf_model.early_config_prune = lambda *a,**k: None
_fake_triton_ops.matmul_perf_model.estimate_matmul_time = lambda *a,**k: None
sys.modules.setdefault('triton.ops', _fake_triton_ops)
sys.modules.setdefault('triton.ops.matmul_perf_model', _fake_triton_ops.matmul_perf_model)

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

# gradient checkpointing hidden-state warning is harmless
warnings.filterwarnings("ignore", message=".*Gradients will be None.*")

# ========================= 路径配置 =========================
BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/InternVL3-8B')
LORA_SFT_PATH = _release_os.environ.get('CGPC_SFT_PATH', './checkpoints/lora.pth')
REFERENCE_LORA_PATH = LORA_SFT_PATH

NCCL_TIMEOUT_MINUTES = 60
SAMPLE_GENERATION_TIMEOUT_SECONDS = float(
    os.environ.get("GRPO_SAMPLE_TIMEOUT_SECONDS", "1200"))

POLICY_LORA_PATH = LORA_SFT_PATH
RUN_NAME = "cgpc_internvl8b"

TRAIN_JSON = _release_os.environ.get('CGPC_TRAIN_JSON', './data/train.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRPO_OUTPUT_DIR = _release_os.environ.get('CGPC_OUTPUT_DIR', './outputs/train_internvl8b')
GRPO_LORA_SAVE  = os.path.join(GRPO_OUTPUT_DIR, "lora_cgpc")

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRPO_LORA_SAVE, exist_ok=True)

# ========================= 分布式初始化 =========================

def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=NCCL_TIMEOUT_MINUTES))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda")
        is_main = True
    return local_rank, world_size, device, is_main

LOCAL_RANK, WORLD_SIZE, DEVICE, IS_MAIN = init_distributed()
GLOBAL_RANK = dist.get_rank() if WORLD_SIZE > 1 else 0
IS_DDP = WORLD_SIZE > 1

random.seed(42 + LOCAL_RANK)
torch.manual_seed(42 + LOCAL_RANK)
torch.cuda.manual_seed_all(42 + LOCAL_RANK)

# ========================= 日志 =========================
_RUN_STAMP = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
LOG_FILE = os.path.join(GRPO_OUTPUT_DIR,
    f"grpo_train_{_RUN_STAMP}{'_rank' + str(LOCAL_RANK) if WORLD_SIZE > 1 else ''}.log")
MICRO_LOG_FILE = os.path.join(GRPO_OUTPUT_DIR, f"micro_samples_{_RUN_STAMP}.jsonl")
MONITOR_JSON_FILE = os.path.join(GRPO_OUTPUT_DIR, "training_monitor_latest.json")
TRAINING_LOCK_FILE = os.path.join(GRPO_OUTPUT_DIR, ".training.lock")

class Tee:
    def __init__(self, filepath, write_file=True):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8") if write_file else None
    def write(self, message):
        self.terminal.write(message); self.terminal.flush()
        if self.log: self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush()
        if self.log: self.log.flush()
sys.stdout = Tee(LOG_FILE, write_file=IS_MAIN)

def assert_training_output_is_fresh(resume_state_path=None):
    error_message = None; lock_acquired = False
    if IS_MAIN:
        try:
            lock_fd = os.open(TRAINING_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            lock_acquired = True
            with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
                json.dump({"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds"),
                           "run_name": RUN_NAME}, lock_file, ensure_ascii=False)
            if resume_state_path is None:
                checkpoint_entries = [e.name for e in os.scandir(GRPO_OUTPUT_DIR) if e.name.startswith("checkpoint-")]
                adapter_entries = (
                    [e.name for e in os.scandir(GRPO_LORA_SAVE)]
                    if os.path.isdir(GRPO_LORA_SAVE) else [])
                if checkpoint_entries or adapter_entries:
                    error_message = (
                        "CGPC输出目录已有训练产物，拒绝覆盖: "
                        f"checkpoints={checkpoint_entries[:5]}, "
                        f"adapter_files={adapter_entries[:5]}")
            elif not os.path.isfile(resume_state_path):
                error_message = f"resume state不存在: {resume_state_path}"
        except FileExistsError:
            error_message = f"训练锁 {TRAINING_LOCK_FILE} 已存在"
        except Exception as exc:
            error_message = f"检查/锁定训练输出目录失败: {exc}"
        if error_message is not None and lock_acquired:
            try: os.remove(TRAINING_LOCK_FILE)
            except OSError: pass
    failure = torch.tensor([int(error_message is not None)], device=DEVICE, dtype=torch.int32)
    if WORLD_SIZE > 1: dist.broadcast(failure, src=0)
    if failure.item():
        if error_message is None: error_message = "rank0未能安全占用训练输出目录"
        raise RuntimeError(error_message)

def release_training_lock():
    if IS_MAIN:
        try: os.remove(TRAINING_LOCK_FILE)
        except FileNotFoundError: pass

def file_sha256(path: str):
    if not path or not os.path.isfile(path): return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

AUTHORIZED_RESUME_SHA256 = set()

# ========================= 图像预处理 (InternVL专用) =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_DYNAMIC_PATCH = 12

def build_transform(isz):
    return T.Compose([
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        T.Resize((isz, isz), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def find_closest_aspect_ratio(ar, tr, w, h, isz):
    bd, bR = float('inf'), (1, 1); area = w * h
    for r in tr:
        d = abs(ar - r[0] / r[1])
        if d < bd: bd = d; bR = r
        elif d == bd and area > 0.5 * isz * isz * r[0] * r[1]: bR = r
    return bR

def dynamic_preprocess(img, mn=1, mx=12, isz=448, ut=False):
    ow, oh = img.size; ar = ow / oh
    tr = sorted(set((i, j) for n in range(mn, mx + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if i * j <= mx and i * j >= mn),
                key=lambda x: x[0] * x[1])
    ta = find_closest_aspect_ratio(ar, tr, ow, oh, isz)
    tw, th = isz * ta[0], isz * ta[1]; blocks = ta[0] * ta[1]
    ri = img.resize((tw, th))
    pi = [ri.crop(((i % (tw // isz)) * isz, (i // (tw // isz)) * isz,
                    ((i % (tw // isz)) + 1) * isz, ((i // (tw // isz)) + 1) * isz))
          for i in range(blocks)]
    if ut and len(pi) != 1: pi.append(img.resize((isz, isz)))
    return pi

def load_image_pixel_values(image, isz=448, mx=12):
    if isinstance(image, str): image = Image.open(image)
    img = image.convert('RGB'); tr = build_transform(isz)
    return torch.stack([tr(x) for x in dynamic_preprocess(img, mn=1, mx=mx, isz=isz, ut=True)])

@lru_cache(maxsize=4096)
def get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img: return img.size

# ========================= GRPO 超参数 (InternVL-cgpc) =========================
G = 6
MINI_BATCH = 2  # InternVL既定：每次反传2个候选

POLICY_OBJECTIVE = "on_policy_group_centered_token_mean"
POLICY_TOKEN_DENOMINATOR = "all_active_sampled_completion_tokens"
EPSILON = None
BETA = 0.005
GAMMA = 0.0
# 与工作区自有Qwen cgpc脚本一致。两者均为LoRA r=16/alpha=32、单轮、accum=8、
# 5% warmup + cosine到0.5倍；因此2e-6给出相同的名义积分学习率预算。
GRPO_LR = 2.0e-6
MAX_GRAD_NORM = 1.0
MIN_GRAD_NORM_FOR_OPTIMIZER_STEP = 1.0e-12
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.50
GRPO_EPOCHS = 1
SAVE_STEPS = 25
MONITOR_STEPS = 25
STEP_CONSOLE_INTERVAL = 1
# 自有Qwen cgpc在相同优化器口径下的实测LoRA相对漂移。它只用于监控，不参与
# loss或自动调LR；InternVL与Qwen架构、rollout组大小不同，不能把早期差异误当成
# 可以在训练中闭环追赶的控制信号。超过本地Qwen实跑到达的update-250不外推。
OWN_QWEN_REFERENCE_SCRIPT = "train_qwen2b.py"
OWN_QWEN_REFERENCE_LOG = "grpo_train_20260804_141025_pid265456_rank0.log"
OWN_QWEN_LORA_DRIFT_BY_UPDATE = {
    25: 0.0009629401278667037,
    50: 0.0016751915599919989,
    75: 0.002364990513964826,
    100: 0.0028667086710082107,
    125: 0.0033095497998962397,
    150: 0.0037282344562522118,
    175: 0.004067379466097592,
    200: 0.004367027223178232,
    225: 0.004631188571390529,
    250: 0.004879687823272698,
}
OWN_QWEN_DRIFT_RATIO_WARNING = (0.67, 1.50)
IOU_THRESHOLD = 0.5
REWARD_IOU_GATE = 0.5
GREEDY_CANDIDATES = 1
EXCLUDE_GREEDY_FROM_GRPO = True
ADVANTAGE_NORMALIZATION = "hard_tp_base_plus_soft_iou_quality_and_localization"
ADVANTAGE_FIXED_SCALE = 0.10
MICRO_RAW_ADVANTAGE_SCALE = 0.50
MICRO_RAW_ADVANTAGE_CLIP = 4.0
SOFT_TP_QUALITY_BLEND = 0.25
LOCALIZATION_ADVANTAGE_SCALE = {"ALL": 0.08, "ROW": 0.10, "SINGLE": 0.15}
TASK_ADVANTAGE_GAIN = {"ALL": 1.00, "ROW": 1.15, "SINGLE": 1.00}
LOSS_REDUCTION = "token_mean_over_active_sampled_responses"
MASK_TRUNCATED_COMPLETIONS = True
MIN_ACTIVE_SAMPLED_CANDIDATES = 2
MIN_TOTAL_ADV_ABS = 1.0e-4
EMPTY_REWARD = -1.0

ALL_OVERLONG_PREFERENCE_ENABLED = True
PAIRWISE_PREFERENCE_BETA = 1.0
PAIRWISE_SUFFIX_ONLY = True
ALL_OVERLONG_PREFERENCE_FIRST_WEIGHT = 0.005
ALL_OVERLONG_PREFERENCE_SUFFIX_WEIGHT = 0.001
ALL_OVERLONG_PREFERENCE_MIN_REWARD = 0.65
ALL_OVERLONG_PREFERENCE_MIN_F1 = 0.70
ALL_QUALITY_PREFERENCE_FIRST_WEIGHT = 0.003
ALL_QUALITY_PREFERENCE_SUFFIX_WEIGHT = 0.00075
ALL_QUALITY_PREFERENCE_MIN_REWARD = 0.82
ALL_QUALITY_PREFERENCE_MIN_REWARD_GAP = 0.03
ALL_QUALITY_PREFERENCE_MIN_F1 = 0.90
ALL_QUALITY_PREFERENCE_MIN_F1_GAP = 0.02
ALL_QUALITY_PREFERENCE_MIN_MICRO_RAW_GAP = 0.50
ALL_QUALITY_PREFERENCE_MIN_SUPPORT = 2

SOFT_OVERLONG_CACHE_TOKENS = {"SINGLE": 64, "ROW": 256, "ALL": 1024}
SOFT_OVERLONG_CACHE_RATIO = 0.15
SOFT_OVERLONG_MAX_PENALTY = {"SINGLE": 0.05, "ROW": 0.10, "ALL": 0.20}
SOFT_OVERLONG_ADVANTAGE_SCALE = 1.0

DYNAMIC_EFFECTIVE_BATCHING = True
DYNAMIC_TARGET_EFFECTIVE_PROMPTS = None
DYNAMIC_MAX_PROMPT_ROUNDS = 24
TASK_REPEAT = {"ALL": 1, "ROW": 1, "SINGLE": 1}

UNDERGEN_PENALTY = {"ALL": 0.04, "ROW": 0.0, "SINGLE": 0.0}
OVERGEN_PENALTY = {"ALL": 0.12, "ROW": 0.25, "SINGLE": 0.15}
OVERGEN_RATIO_CAP = {"ALL": 2.0, "ROW": 2.0, "SINGLE": 2.0}

MICRO_F1_TARGET = 0.9000
MICRO_UTILITY_FLOOR = -2.0
KL_SKIP_THRESHOLD = 0.20
ABNORMAL_LOSS = 100.0

TASK_REWARD_WEIGHTS = {
    "ALL":    {"fmt": 0.02, "iou": 0.10, "micro": 0.45, "rec": 0.02, "prec": 0.18, "f1": 0.21,
               "cnt": 0.02, "label": 0.00, "center": 0.0},
    "ROW":    {"fmt": 0.02, "iou": 0.08, "micro": 0.43, "rec": 0.00, "prec": 0.22, "f1": 0.20,
               "cnt": 0.02, "label": 0.00, "center": 0.03},
    "SINGLE": {"fmt": 0.02, "iou": 0.20, "micro": 0.18, "rec": 0.0, "prec": 0.0, "f1": 0.30,
               "cnt": 0.0, "label": 0.00, "center": 0.30},
}
for _tn, _tw in TASK_REWARD_WEIGHTS.items():
    if not math.isclose(sum(_tw.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{_tn} reward weights must sum to 1.0")

PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRPO_GRADIENT_ACCUMULATION_STEPS", "8"))
DYNAMIC_TARGET_EFFECTIVE_PROMPTS = GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE

ALL_MAX_TOKENS = int(os.environ.get("CGPC_ALL_MAX_TOKENS", "10240"))
if ALL_MAX_TOKENS not in {6144, 8192, 10240}:
    raise ValueError(
        "CGPC_ALL_MAX_TOKENS 仅允许 6144、8192 或 10240")
MAX_NEW_TOKENS = ALL_MAX_TOKENS

EXPLOIT_SAMPLED_CANDIDATES = {"ALL": G - GREEDY_CANDIDATES,
                              "ROW": G - GREEDY_CANDIDATES,
                              "SINGLE": G - GREEDY_CANDIDATES}
SAMPLE_TEMPERATURE_LOW = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TEMPERATURE_HIGH = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TOP_P = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TOP_K = 0
# 与Qwen cgpc一致，清空模型GenerationConfig里可能残留的beam、forced-EOS、
# repetition或token抑制。InternVL.chat最终仍调用HF generate，这些残留项同样会
# 让rollout偏离下面声明的raw current-policy分布。
GENERATION_NEUTRAL_OVERRIDES = {
    "num_beams": 1,
    "num_beam_groups": 1,
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "diversity_penalty": 0.0,
    "min_length": 0,
    "min_new_tokens": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "bad_words_ids": None,
    "force_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "sequence_bias": None,
    "renormalize_logits": False,
    "remove_invalid_values": False,
    "exponential_decay_length_penalty": None,
    "watermarking_config": None,
}
if (any(v != G - GREEDY_CANDIDATES
        for v in EXPLOIT_SAMPLED_CANDIDATES.values()) or
        any(not math.isclose(v, 1.0) for v in
            (*SAMPLE_TEMPERATURE_LOW.values(),
             *SAMPLE_TEMPERATURE_HIGH.values(),
             *SAMPLE_TOP_P.values())) or
        SAMPLE_TOP_K != 0):
    raise ValueError(
        "GRPO sampled candidates must use the raw current-policy distribution")

LOG_RATIO_CLAMP = 10.0

CENTER_SCALE_BOX_DIAGONALS = {"ALL": 1.5, "ROW": 2.0, "SINGLE": 1.0}
CENTER_SCALE_IMAGE_FLOOR = 0.02


RL_DISABLE_DROPOUT = True
TRAIN_ADAPTER_PROJECTOR = True

TASK_MAX_TOKENS = {"SINGLE": 256, "ROW": 1024, "ALL": ALL_MAX_TOKENS}
TRAIN_BUCKET_MAX_TOKENS = {
    "SINGLE:2": 256, "ROW:3": 384, "ROW:7": 640, "ROW:12": 1024, "ROW:20": 1024, "ROW:999": 1024,
    "ALL:10": min(2048, ALL_MAX_TOKENS), "ALL:25": min(3072, ALL_MAX_TOKENS),
    "ALL:50": min(5120, ALL_MAX_TOKENS), "ALL:80": min(7168, ALL_MAX_TOKENS),
    "ALL:140": ALL_MAX_TOKENS, "ALL:999": ALL_MAX_TOKENS,
}
def max_tokens_for_bucket(task: str, cost_bucket: str = None) -> int:
    if cost_bucket is not None and cost_bucket in TRAIN_BUCKET_MAX_TOKENS:
        return int(TRAIN_BUCKET_MAX_TOKENS[cost_bucket])
    return int(TASK_MAX_TOKENS[task])

# ========================= 辅助函数 =========================
def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def global_iou_matches(gt_boxes, pred_boxes):
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt_boxes)
                     for pi, p in enumerate(pred_boxes)), reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for score, gi, pi in pairs:
        if score <= 0 or gi in used_gt or pi in used_pred: continue
        used_gt.add(gi); used_pred.add(pi)
        matches.append((score, gi, pi))
    return matches

def evaluation_order_matches(gt_boxes, pred_boxes, threshold=IOU_THRESHOLD):
    used_pred, matches = set(), []
    for gt_i, gt_box in enumerate(gt_boxes):
        best_iou, best_pred_i = 0.0, -1
        for pred_i, pred_box in enumerate(pred_boxes):
            if pred_i in used_pred: continue
            score = iou(gt_box, pred_box)
            if score > best_iou: best_iou, best_pred_i = score, pred_i
        if best_iou > threshold:
            used_pred.add(best_pred_i)
            matches.append((best_iou, gt_i, best_pred_i))
    return matches



def spatial_sort_boxes_and_labels(boxes, labels):
    labels = list(labels or [])
    indexed = list(range(len(boxes)))
    indexed.sort(key=lambda idx: (boxes[idx][1], boxes[idx][0]))
    return [boxes[idx] for idx in indexed], [labels[idx] if idx < len(labels) else "" for idx in indexed]

def center_similarity(gt_boxes, pred_boxes, image_path, task="SINGLE"):
    if not gt_boxes or not pred_boxes: return 0.0
    width, height = get_image_size(image_path)
    image_diagonal = max(math.hypot(width, height), 1.0)
    scale_multiplier = CENTER_SCALE_BOX_DIAGONALS.get(task, 1.5)
    pairs = []
    for gt_i, gt_box in enumerate(gt_boxes):
        gx, gy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
        gt_diagonal = math.hypot(max(gt_box[2] - gt_box[0], 1.0), max(gt_box[3] - gt_box[1], 1.0))
        scale = max(scale_multiplier * gt_diagonal, CENTER_SCALE_IMAGE_FLOOR * image_diagonal, 1.0)
        for pred_i, pred_box in enumerate(pred_boxes):
            px, py = (pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2
            normalized_distance = math.hypot(gx - px, gy - py) / scale
            pairs.append((math.exp(-(normalized_distance ** 2)), gt_i, pred_i))
    used_gt, used_pred, total = set(), set(), 0.0
    for similarity, gt_i, pred_i in sorted(pairs, reverse=True):
        if gt_i in used_gt or pred_i in used_pred: continue
        used_gt.add(gt_i); used_pred.add(pred_i); total += similarity
    return total / len(gt_boxes)

def _balanced_json_objects(text: str):
    """提取完整顶层JSON对象，并区分对象/字符串未闭合。"""
    objects, start = [], None
    depth, in_string, escaped = 0, False, False
    for idx, char in enumerate("" if text is None else str(text)):
        if in_string:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': in_string = False
            continue
        if char == '"': in_string = True
        elif char == "{":
            if depth == 0: start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(str(text)[start:idx + 1]); start = None
    return objects, depth > 0, in_string

def _parse_bbox_json_diagnostics(text: str) -> Dict:
    text = "" if text is None else str(text)
    stripped = text.strip()
    result = {"boxes": [], "labels": [], "empty_text": not bool(stripped),
              "invalid_json": False, "truncated": False,
              "label_count_mismatch": False, "has_bbox_key": False,
              "parseable_json": False}
    if result["empty_text"]: return result
    result["has_bbox_key"] = bool(re.search(r'["\']bbox_2d["\']\s*:', text))
    objects, unclosed_object, unclosed_string = _balanced_json_objects(text)
    result["truncated"] = bool(result["has_bbox_key"] and (unclosed_object or unclosed_string))
    candidate = next((obj for obj in objects if re.search(r'"bbox_2d"\s*:', obj)), None)
    if candidate is None:
        result["invalid_json"] = not result["truncated"]
        return result
    try: obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        result["invalid_json"] = True
        return result
    boxes, labels = obj.get("bbox_2d", []), obj.get("labels", [])
    if not isinstance(boxes, list):
        result["invalid_json"] = True
        return result
    if not isinstance(labels, list):
        labels = []; result["label_count_mismatch"] = bool(boxes)
    else: result["label_count_mismatch"] = len(labels) != len(boxes)
    result["boxes"], result["labels"], result["parseable_json"] = boxes, labels, True
    return result

def parse_bbox_json(text: str) -> Tuple[List, List, bool]:
    is_truncated = False
    text_stripped = text.rstrip()
    if text_stripped and not text_stripped.endswith("}") and not text_stripped.endswith("]"):
        is_truncated = True
    json_match = re.search(r'\{[^{}]*"bbox_2d"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
    if not json_match:
        loose_match = re.search(r'\{[^{}]*"bbox_2d"\s*:\s*\[.*', text, re.DOTALL)
        if loose_match:
            is_truncated = True
        return [], [], is_truncated
    try:
        obj = json.loads(json_match.group())
        bbox_raw = obj.get("bbox_2d", [])
        label_raw = obj.get("labels", [])
        if not isinstance(bbox_raw, list): bbox_raw = []
        if not isinstance(label_raw, list): label_raw = []
        return bbox_raw, label_raw, is_truncated
    except json.JSONDecodeError:
        return [], [], True

def json_object_appears_truncated(text: str) -> bool:
    """检测首个JSON对象是否因字符串或花括号未闭合而明显截断。"""
    start = text.find("{")
    if start < 0:
        return False
    depth = 0
    in_string = False
    escaped = False
    for char in text[start:]:
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
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return False
    return in_string or depth > 0

def bbox_json_diagnostics(text: str) -> Dict[str, bool]:
    """区分空文本、JSON无效和截断；避免全部合并成“无合法框”。

    这里复用 ``parse_bbox_json`` 的同一对象提取正则，诊断口径不会改变奖励解析。
    """
    stripped = text.strip()
    empty_text = not stripped
    json_match = re.search(
        r'\{[^{}]*"bbox_2d"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
    parseable_json = False
    if json_match:
        try:
            obj = json.loads(json_match.group())
            parseable_json = (
                isinstance(obj, dict) and
                isinstance(obj.get("bbox_2d", []), list))
        except (json.JSONDecodeError, TypeError):
            parseable_json = False
    truncated = bool(stripped) and json_object_appears_truncated(text)
    return {
        "empty_text": empty_text,
        "invalid_json": bool(stripped) and not parseable_json and not truncated,
        "truncated": truncated,
        "parseable_json": parseable_json,
    }

_NUMERIC_QUAD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")

def text_repetition_diagnostics(text: str) -> Tuple[bool, bool, int]:
    """复用评估口径的轻量重复坐标/尾循环检测，不增加模型前向。"""
    text = "" if text is None else str(text)
    quads = [re.sub(r"\s+", "", match.group(0)) for match in _NUMERIC_QUAD_RE.finditer(text)]
    repeated_quad_count = max(Counter(quads).values(), default=0)
    units = re.findall(r"\d+(?:\.\d+)?|[A-Za-z_]+|[^\s]", text)
    repeated_tail = False
    for unit_len in range(1, min(64, len(units) // 4) + 1):
        unit = units[-unit_len:]
        if all(units[-repeat * unit_len:-(repeat - 1) * unit_len or None] == unit
               for repeat in range(2, 5)):
            repeated_tail = True; break
    return repeated_quad_count >= 4, repeated_tail, repeated_quad_count

def sanitize_and_scale_boxes(boxes, image_path):
    return protocol.scale_normalized_boxes(boxes, get_image_size(image_path))

def scale_boxes_to_pixel(boxes, image_path):
    return sanitize_and_scale_boxes(boxes, image_path)[0]

def quantized_bbox_signature(boxes_px, image_path, invalid_count=0):
    width, height = get_image_size(image_path)
    width, height = max(width, 1), max(height, 1)
    quantized = tuple((round(100 * box[0] / width), round(100 * box[1] / height),
                       round(100 * box[2] / width), round(100 * box[3] / height)) for box in boxes_px)
    return int(invalid_count), quantized

def normalize_label(value):
    return str(value).strip().lower()

TASK_NUM = r"(?:[一二三四五六七八九十两]+|\d+)"

def classify_task(prompt_text: str) -> str:
    return protocol.classify_grounding_task(prompt_text)

# ========================= 奖励函数 =========================
def compute_reward(pred_text: str, gt_text: str, image_path: str,
                   prompt_text: str = "") -> Dict[str, float]:
    task = classify_task(prompt_text) if prompt_text else "ROW"
    w = TASK_REWARD_WEIGHTS.get(task, TASK_REWARD_WEIGHTS["ROW"])

    pred_boxes, pred_labels, pred_truncated = parse_bbox_json(pred_text)
    pred_diag = bbox_json_diagnostics(pred_text)
    gt_boxes, gt_labels = parse_bbox_json(gt_text)[:2]

    gt_boxes_px, gt_valid_idx, _ = sanitize_and_scale_boxes(gt_boxes, image_path)
    pred_boxes_px, pred_valid_idx, invalid_count = sanitize_and_scale_boxes(
        pred_boxes, image_path)
    gt_labels_valid = [normalize_label(gt_labels[i]) if i < len(gt_labels) else ""
                       for i in gt_valid_idx]
    pred_labels_valid = [normalize_label(pred_labels[i]) if i < len(pred_labels) else ""
                         for i in pred_valid_idx]
    gt_boxes_px, gt_labels_valid = spatial_sort_boxes_and_labels(
        gt_boxes_px, gt_labels_valid)
    pred_boxes_px, pred_labels_valid = spatial_sort_boxes_and_labels(
        pred_boxes_px, pred_labels_valid)
    pred_bbox_signature = quantized_bbox_signature(
        pred_boxes_px, image_path, invalid_count)
    gt_count = len(gt_boxes_px)
    pred_count = len(pred_boxes_px)
    total_pred_count = pred_count + invalid_count

    # ── 空输出的强惩罚（与任务权重解耦）──
    # total和主优势通道都必须为负；否则主优势有方差时composite会被清零，
    # 完全空输出的micro优势0可能反而胜过定位尚不紧致的真实TP。
    if gt_count > 0 and pred_count == 0:
        fp_count = total_pred_count
        micro_raw = -MICRO_F1_TARGET * fp_count
        # 与“一个未命中预测框”的micro边际值持平；近目标框可由localization胜出，
        # 真正TP则由hard-TP基础信用明确胜出，但不鼓励随便幻觉一个错误框。
        advantage_micro_raw = min(-MICRO_F1_TARGET, micro_raw)
        micro_denom = gt_count * max(2.0 - MICRO_F1_TARGET, 1e-6)
        r_micro = max(micro_raw / micro_denom, MICRO_UTILITY_FLOOR)
        return {"total": EMPTY_REWARD, "format": 0.0, "iou": 0.0, "recall": 0.0,
                "precision": 0.0, "f1": 0.0, "count": -1.0,
                "micro": r_micro, "micro_raw": micro_raw,
                "advantage_micro_raw": advantage_micro_raw,
                "label": 0.0, "center": 0.0, "soft_iou": 0.0,
                "localization": 0.0,
                "under_penalty": UNDERGEN_PENALTY.get(task, 0.0),
                "over_penalty": 0.0,
                "fn_ratio": 1.0,
                "fp_ratio": fp_count / max(gt_count, 1),
                "tp": 0, "fp": fp_count, "fn": gt_count,
                "gt_count": gt_count, "pred_count": total_pred_count,
                "valid_count": 0, "invalid_count": invalid_count,
                "empty_text": pred_diag["empty_text"],
                "invalid_json": pred_diag["invalid_json"],
                "truncated": pred_diag["truncated"],
                "no_valid_box": pred_diag["parseable_json"],
                "bbox_signature": pred_bbox_signature}

    # R_format
    has_bbox = pred_count > 0
    has_json = bool(re.search(r'\{[^{}]*"bbox_2d"\s*:', pred_text))
    if has_json and has_bbox and invalid_count == 0 and not pred_truncated:
        r_format = 1.0
    elif has_json and has_bbox and pred_truncated:
        r_format = 0.5
    else:
        r_format = 0.0

    # 与当前外部CSV一致的逐GT顺序匹配。只对IoU>0.5的真实TP提供IoU奖励，
    # 避免未过最终指标门槛的相邻错误框靠soft-IoU长期获得正向优势。
    if gt_count == 0:
        # GT 为空：预测也为空才算完全正确，否则视为误检（惩罚幻觉框）
        empty_pred = (pred_count == 0)
        n_match = 0
        r_iou       = 1.0 if empty_pred else 0.0
        r_recall    = 1.0 if empty_pred else 0.0
        r_precision = 1.0 if empty_pred else 0.0
        r_label     = 1.0 if empty_pred else 0.0
        r_center    = 1.0 if empty_pred else 0.0
        r_soft_iou  = 1.0 if empty_pred else 0.0
    else:
        # 阈值以下的 IoU 只进入小权重 dense localization 通道，不改变外部
        # TP/FP/FN、micro_raw 或 metric-best 的严格 IoU>0.5 口径。
        soft_matches = global_iou_matches(gt_boxes_px, pred_boxes_px)
        r_soft_iou = sum(score for score, _, _ in soft_matches) / gt_count
        threshold_matches = evaluation_order_matches(
            gt_boxes_px, pred_boxes_px, REWARD_IOU_GATE)
        soft_tp_sum = sum(
            min(max(
                (score - REWARD_IOU_GATE) /
                max(1.0 - REWARD_IOU_GATE, 1e-6),
                0.0), 1.0)
            for score, _, _ in threshold_matches)
        r_iou = sum(
            score for score, _, _ in threshold_matches) / gt_count
        n_match = len(threshold_matches)
        label_correct = sum(
            bool(gt_labels_valid[gt_i]) and
            gt_labels_valid[gt_i] == pred_labels_valid[pred_i]
            for _, gt_i, pred_i in threshold_matches)
        r_recall    = n_match / gt_count
        # 非法框也作为 false positive 进入 precision/count，防止退化坐标投机。
        r_precision = n_match / total_pred_count if total_pred_count > 0 else 0.0
        r_label     = label_correct / n_match if n_match > 0 else 0.0
        r_center = center_similarity(
            gt_boxes_px, pred_boxes_px, image_path, task)
    r_localization = 0.75 * r_soft_iou + 0.25 * r_center

    # 与外部评估的匹配 F1 对齐。相比单独的数量奖励，它会同时惩罚漏检和误检，
    # 不能通过“预测数等于 GT 数但框选错”获得高分。
    r_f1 = (2.0 * r_recall * r_precision / (r_recall + r_precision)
            if r_recall + r_precision > 0 else 0.0)

    # 与目标Overall micro-F1直接对齐的候选边际效用。
    # 对gt>0：完美候选的分子/分母均为gt_count*(2-target)，所以恰为1。
    # 对同一个prompt，GT项恒定；最大化该效用等价于偏好
    # 2*delta(TP) > target*delta(Pred) 的候选。
    if gt_count > 0:
        micro_denom = gt_count * max(2.0 - MICRO_F1_TARGET, 1e-6)
        micro_raw = 2.0 * n_match - MICRO_F1_TARGET * total_pred_count
        # 训练优势保留hard-TP基础信用，再混入小比例soft-TP质量信用。这样任何
        # IoU>0.5的真实TP都稳定优于空输出，同时IoU趋近1仍获得连续额外收益。
        quality_tp_sum = (
            (1.0 - SOFT_TP_QUALITY_BLEND) * n_match +
            SOFT_TP_QUALITY_BLEND * soft_tp_sum)
        advantage_micro_raw = (
            2.0 * quality_tp_sum -
            MICRO_F1_TARGET * total_pred_count)
        r_micro = micro_raw / micro_denom
        r_micro = max(r_micro, MICRO_UTILITY_FLOOR)
    else:
        micro_raw = -MICRO_F1_TARGET * total_pred_count
        advantage_micro_raw = micro_raw
        r_micro = 1.0 if total_pred_count == 0 else max(
            micro_raw, MICRO_UTILITY_FLOOR)

    # R_count（计数准确性）
    if gt_count > 0 or total_pred_count > 0:
        max_cnt = max(gt_count, total_pred_count, 1)
        r_count = max(0.0, 1.0 - abs(total_pred_count - gt_count) / max_cnt)
    else:
        r_count = 1.0

    # 直接按匹配后的FN/FP惩罚，而不是只看预测数与GT数之差。
    # 这能阻止“数量正确但目标选错”逃过惩罚，修复旧试验CountAcc升而TP降的投机方向。
    fn_ratio = max(gt_count - n_match, 0) / max(gt_count, 1)
    fp_ratio = max(total_pred_count - n_match, 0) / max(gt_count, 1)
    fp_count = max(total_pred_count - n_match, 0)
    fn_count = max(gt_count - n_match, 0)
    under_penalty = UNDERGEN_PENALTY.get(task, 0.0) * min(fn_ratio, 1.0)
    over_penalty = (
        OVERGEN_PENALTY.get(task, 0.0)
        * min(fp_ratio, OVERGEN_RATIO_CAP.get(task, 1.0))
    )
    total = (w["fmt"] * r_format + w["iou"] * r_iou +
             w.get("micro", 0.0) * r_micro +
             w["rec"] * r_recall + w["prec"] * r_precision +
             w.get("f1", 0.0) * r_f1 +
             w["cnt"] * r_count + w["label"] * r_label +
             w.get("center", 0.0) * r_center - under_penalty - over_penalty)

    return {"total": total, "format": r_format, "iou": r_iou,
            "recall": r_recall, "precision": r_precision, "f1": r_f1,
            "micro": r_micro, "micro_raw": micro_raw, "count": r_count,
            "advantage_micro_raw": advantage_micro_raw,
            "label": r_label, "center": r_center, "soft_iou": r_soft_iou,
            "localization": r_localization,
            "under_penalty": under_penalty, "over_penalty": over_penalty,
            "fn_ratio": fn_ratio, "fp_ratio": fp_ratio,
            "tp": n_match, "fp": fp_count, "fn": fn_count,
            "gt_count": gt_count,
            "pred_count": total_pred_count, "valid_count": pred_count,
            "invalid_count": invalid_count,
            "empty_text": pred_diag["empty_text"],
            "invalid_json": pred_diag["invalid_json"],
            "truncated": pred_diag["truncated"],
            "no_valid_box": bool(
                pred_diag["parseable_json"] and pred_count == 0),
            "bbox_signature": pred_bbox_signature}

# ========================= 数据集 =========================
def sample_cost_bucket(task: str, gt_response: str) -> str:
    gt_count = len(parse_bbox_json(gt_response)[0])
    if task == "ALL": upper = next((v for v in (10, 25, 50, 80, 140) if gt_count <= v), 999)
    elif task == "ROW": upper = next((v for v in (3, 7, 12, 20) if gt_count <= v), 999)
    else: upper = 2
    return f"{task}:{upper}"

class CottonBaleDataset(Dataset):
    def __init__(self, json_path: str, image_root: str):
        with open(json_path, "r", encoding="utf-8") as f: self.data = json.load(f)
        self.sample_indices, self.task_counts, self.sampled_task_counts, self.source_cost_buckets = [], Counter(), Counter(), {}
        for idx, sample in enumerate(self.data):
            task = classify_task(sample["conversations"][0]["value"])
            self.source_cost_buckets[idx] = sample_cost_bucket(task, sample["conversations"][1]["value"])
            self.task_counts[task] += 1
            self.sample_indices.extend([idx] * TASK_REPEAT.get(task, 1))
        step_multiple = max(WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS, 1)
        pad_count = (-len(self.sample_indices)) % step_multiple
        if pad_count: self.sample_indices.extend(self.sample_indices[:pad_count])
        for source_idx in self.sample_indices:
            task = classify_task(self.data[source_idx]["conversations"][0]["value"])
            self.sampled_task_counts[task] += 1
        self.image_root = image_root
    def __len__(self): return len(self.sample_indices)
    def __getitem__(self, idx): return self.get_source_item(self.sample_indices[idx])
    def get_source_item(self, source_idx):
        sample = self.data[source_idx]
        img_rel = sample["images"][0]
        image_path = os.path.join(self.image_root, img_rel)
        with Image.open(image_path) as raw_image: image = raw_image.convert("RGB")
        prompt, gt_response = sample["conversations"][0]["value"], sample["conversations"][1]["value"]
        return {"image": image, "image_path": image_path, "image_rel": img_rel,
                "prompt": prompt, "gt_response": gt_response, "idx": source_idx,
                "cost_bucket": self.source_cost_buckets[source_idx]}

class CostBalancedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=42):
        self.dataset, self.num_replicas, self.rank, self.shuffle, self.seed, self.epoch = dataset, max(int(num_replicas), 1), int(rank), bool(shuffle), int(seed), 0
        self.buckets = {}
        for dataset_idx, source_idx in enumerate(dataset.sample_indices):
            key = dataset.source_cost_buckets[source_idx]
            self.buckets.setdefault(key, []).append(dataset_idx)
        group_count = sum(math.ceil(len(indices) / self.num_replicas) for indices in self.buckets.values())
        self.num_samples = math.ceil(group_count / GRADIENT_ACCUMULATION_STEPS) * GRADIENT_ACCUMULATION_STEPS
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = []
        for key in sorted(self.buckets):
            indices = list(self.buckets[key])
            if self.shuffle: rng.shuffle(indices)
            remainder = len(indices) % self.num_replicas
            if remainder: indices.extend(indices[:self.num_replicas - remainder])
            groups.extend(indices[start:start + self.num_replicas] for start in range(0, len(indices), self.num_replicas))
        if self.shuffle: rng.shuffle(groups)
        if len(groups) < self.num_samples: groups.extend(groups[:self.num_samples - len(groups)])
        if len(groups) != self.num_samples: raise RuntimeError(f"Sampler group mismatch: {len(groups)} != {self.num_samples}")
        return iter(group[self.rank] for group in groups)
    def __len__(self): return self.num_samples
    def set_epoch(self, epoch): self.epoch = int(epoch)

def collate_fn(batch): return batch

# ========================= 模型加载 (InternVL专用) =========================
POLICY_ADAPTER = "policy"
REFERENCE_ADAPTER = "ref"

def disable_dropout_in_model(model):
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p = 0.0; count += 1
    return count

def adapter_parameter_kind(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered: return "lora"
    if ("visual" in lowered and "merger" in lowered) or ("mlp1" in lowered): return "projector"
    return ""

def remap_xtuner_adapter_key(checkpoint_key: str, target_adapter: str) -> str:
    """Map a full XTuner PEFT key onto a named adapter in this trainer.

    XTuner's SFT checkpoint was produced with PEFT's default adapter, whereas
    this trainer names the trainable adapter ``policy`` so that ``ref`` can
    coexist on the same backbone.  Merely replacing the module prefix leaves
    ``.default.`` in the key and silently matches zero policy tensors.
    """
    mapped = checkpoint_key.replace(
        "model.language_model.base_model.model.",
        "base_model.model.language_model.")
    mapped = mapped.replace(".default.", f".{target_adapter}.")
    return mapped

def load_exact_state_subset(model, mapped_state, label):
    """Load a small adapter/projector subset and verify exact tensor writes."""
    model_state = model.state_dict()
    shape_errors = []
    for key, value in mapped_state.items():
        if key not in model_state:
            shape_errors.append((key, tuple(value.shape), None))
        elif tuple(value.shape) != tuple(model_state[key].shape):
            shape_errors.append(
                (key, tuple(value.shape), tuple(model_state[key].shape)))
    if shape_errors:
        raise RuntimeError(f"{label} shape mismatch: {shape_errors[:3]}")
    result = model.load_state_dict(mapped_state, strict=False)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{label} unexpected keys: {unexpected[:5]}")
    loaded_state = model.state_dict()
    verification_errors = []
    for key, expected in mapped_state.items():
        actual = loaded_state[key].detach().cpu()
        expected_cast = expected.detach().to(device="cpu", dtype=actual.dtype)
        if not torch.equal(actual, expected_cast):
            verification_errors.append(key)
    if verification_errors:
        raise RuntimeError(
            f"{label} tensors were not written exactly: {verification_errors[:5]}")

def active_adapter_names(model):
    raw_model = model.module if hasattr(model, "module") else model
    active = getattr(raw_model, "active_adapters", None)
    if active is None:
        active = getattr(raw_model, "active_adapter", None)
    if callable(active):
        active = active()
    if active is None:
        return tuple()
    if isinstance(active, str):
        return (active,)
    return tuple(str(name) for name in active)

def assert_active_adapter(model, expected, where):
    active = active_adapter_names(model)
    if active != (expected,):
        raise RuntimeError(
            f"Unexpected active PEFT adapter at {where}: active={active}, "
            f"expected=({expected!r},)")

def _load_base_and_lora(policy_lora_path):
    _og = torch.nn.Module.__getattr__
    torch.nn.Module.__getattr__ = lambda self, name: _og(self, name)
    try:
        attention_request = "sdpa"
        try:
            bm = AutoModel.from_pretrained(
                BASE_MODEL_PATH, trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attention_request)
        except (TypeError, ValueError) as exc:
            attention_request = f"model_default (sdpa rejected: {type(exc).__name__})"
            bm = AutoModel.from_pretrained(
                BASE_MODEL_PATH, trust_remote_code=True,
                torch_dtype=torch.bfloat16)
    finally: torch.nn.Module.__getattr__ = _og
    lc = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM)
    m = get_peft_model(bm, lc, adapter_name=POLICY_ADAPTER)
    ps = m.state_dict(); mp = {}
    expected_lora = sum("lora_" in key and f".{POLICY_ADAPTER}." in key for key in ps)
    # Restore the SFT visual projector before creating the frozen reference.
    sft_checkpoint = torch.load(LORA_SFT_PATH, map_location="cpu", weights_only=False)
    sft_policy_lora = {}
    for ck_k, cv in sft_checkpoint["state_dict"].items():
        if "lora" in ck_k:
            pk = remap_xtuner_adapter_key(ck_k, POLICY_ADAPTER)
            if pk in ps: sft_policy_lora[pk] = cv
        elif "mlp1" in ck_k:
            pk = ck_k.replace("model.", "base_model.model.")
            if pk in ps: mp[pk] = cv
    if sum("lora_" in key for key in sft_policy_lora) != expected_lora:
        raise RuntimeError("Frozen SFT reference LoRA is incomplete")
    if not mp:
        raise RuntimeError("SFT checkpoint is missing the visual projector")
    load_exact_state_subset(m, mp, "SFT projector")
    bm._cgpc_reference_projector = copy.deepcopy(bm.mlp1).requires_grad_(False)
    if os.path.isdir(policy_lora_path):
        adapter_dir = policy_lora_path
        if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            named_adapter_dir = os.path.join(adapter_dir, POLICY_ADAPTER)
            if os.path.isfile(os.path.join(named_adapter_dir, "adapter_config.json")):
                adapter_dir = named_adapter_dir
        projector_file = Path(policy_lora_path) / "visual_projector.pt"
        if not projector_file.is_file():
            projector_file = Path(adapter_dir).parent / "visual_projector.pt"
        bm.mlp1.load_state_dict(torch.load(projector_file, map_location="cpu", weights_only=True))
        saved_adapter = load_peft_weights(adapter_dir, device="cpu")
        loaded_lora = sum("lora_" in key for key in saved_adapter)
        if loaded_lora != expected_lora:
            raise RuntimeError(f"PEFT policy LoRA match mismatch: {loaded_lora}/{expected_lora}")
        load_result = set_peft_model_state_dict(
            m, saved_adapter, adapter_name=POLICY_ADAPTER)
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(f"Unexpected PEFT checkpoint keys: {unexpected[:5]}")
    else:
        ck = torch.load(policy_lora_path, map_location="cpu", weights_only=False)
        lora_map = {}
        for ck_k, cv in ck["state_dict"].items():
            if "lora" in ck_k:
                pk = remap_xtuner_adapter_key(ck_k, POLICY_ADAPTER)
                if pk in ps: lora_map[pk] = cv
        loaded_lora = sum("lora_" in key for key in lora_map)
        if loaded_lora == 0: raise RuntimeError("SFT checkpoint did not match any LoRA parameter")
        if loaded_lora != expected_lora:
            raise RuntimeError(f"Policy LoRA match mismatch: {loaded_lora}/{expected_lora}")
        load_exact_state_subset(m, lora_map, "Policy LoRA")
    # Reference始终锚定原始SFT。resume时policy来自CGPC checkpoint，不能再从
    # policy复制，否则KL参考会在每次恢复时偷偷前移。
    m.add_adapter(REFERENCE_ADAPTER, lc)
    state = m.state_dict(); ref_state = {}
    for policy_key, tensor in sft_policy_lora.items():
        ref_key = policy_key.replace(f".{POLICY_ADAPTER}.", f".{REFERENCE_ADAPTER}.")
        if ref_key in state: ref_state[ref_key] = tensor.detach().clone()
    if len(ref_state) != expected_lora:
        raise RuntimeError(f"Frozen reference LoRA match mismatch: {len(ref_state)}/{expected_lora}")
    load_exact_state_subset(m, ref_state, "Reference LoRA")
    m.set_adapter(POLICY_ADAPTER)
    assert_active_adapter(m, POLICY_ADAPTER, "model load completion")
    attention_classes = sorted({
        type(module).__name__ for module in bm.language_model.modules()
        if "attention" in type(module).__name__.lower()
    })
    setattr(m, "_grpo_attention_diagnostic", {
        "requested": attention_request,
        "config": getattr(bm.language_model.config, "_attn_implementation", None),
        "classes": attention_classes,
    })
    if IS_MAIN:
        print(f"  SFT LoRA loaded: {loaded_lora}/{expected_lora}; ref adapter copied: {len(ref_state)} tensors")
    return m

def load_models(policy_lora_path):
    if IS_MAIN: print(f"[{datetime.now():%H:%M:%S}] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    if IS_MAIN:
        print(f"  tokenizer: padding_side={tokenizer.padding_side}, pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
    if IS_MAIN: print(f"[{datetime.now():%H:%M:%S}] 加载 Policy 模型...")
    policy_model = _load_base_and_lora(policy_lora_path)
    if IS_DDP: policy_model = policy_model.to(DEVICE)
    else: policy_model = policy_model.cuda()
    policy_model.set_adapter(POLICY_ADAPTER)
    for n, p in policy_model.named_parameters():
        kind = adapter_parameter_kind(n)
        is_policy_lora = kind == "lora" and f".{POLICY_ADAPTER}." in n
        is_policy_projector = kind == "projector" and TRAIN_ADAPTER_PROJECTOR
        # reference adapter必须保持冻结；否则KL锚点会随policy漂移，且DDP会看到
        # active policy forward中未使用的可训练reference参数。
        p.requires_grad = bool(is_policy_lora or is_policy_projector)
    if any(p.requires_grad for n, p in policy_model.named_parameters()
           if f".{REFERENCE_ADAPTER}." in n):
        raise RuntimeError("Reference adapter must be frozen")
    disabled_dropout = disable_dropout_in_model(policy_model)
    try:
        policy_model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        policy_model.language_model.gradient_checkpointing_enable()
    policy_model.train()
    if IS_MAIN:
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        trainable_kind_counts = Counter(); trainable_kind_params = Counter()
        for n, p in policy_model.named_parameters():
            kind = adapter_parameter_kind(n)
            if kind: trainable_kind_counts[kind] += 1; trainable_kind_params[kind] += p.numel()
        print(f"  Policy 就绪, 可训练参数: {trainable:,}, Dropout关闭: {disabled_dropout}")
        print(f"  adapter可训练参数: {dict(trainable_kind_params)} (张量数={dict(trainable_kind_counts)})")
        print(f"  attention诊断: {getattr(policy_model, '_grpo_attention_diagnostic', {})}")
        print("  Gradient checkpointing: ✓")
    # conversation 模板
    conv_path = os.path.join(BASE_MODEL_PATH, 'conversation.py')
    spec = importlib.util.spec_from_file_location('conversation', conv_path)
    conv_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv_module)
    return tokenizer, policy_model, conv_module

# ========================= 生成候选 (InternVL专用) =========================
def resolve_internvl_eos_token(model, tokenizer, conv_module):
    """按InternVL对话模板解析真正的assistant终止token。

    VLM-R1的InternVL适配使用conversation separator而不是直接假定
    tokenizer.eos_token_id；两者不一致时，后者会把正常结束误判为截断。
    """
    raw_model = model.module if hasattr(model, "module") else model
    cached = getattr(raw_model, "_grpo_template_eos", None)
    if cached is not None: return cached
    template = conv_module.get_conv_template(raw_model.template)
    separator = str(getattr(template, "sep", "") or "").strip()
    token_id = None
    if separator:
        encoded = tokenizer(separator, add_special_tokens=False).input_ids
        if isinstance(encoded, torch.Tensor): encoded = encoded.reshape(-1).tolist()
        if len(encoded) == 1:
            candidate = int(encoded[0])
            if tokenizer.unk_token_id is None or candidate != int(tokenizer.unk_token_id):
                token_id = candidate
    source = "conversation.sep"
    if token_id is None:
        token_id = tokenizer.eos_token_id
        source = "tokenizer.eos_token_id"
    if token_id is None:
        token_id = tokenizer.pad_token_id
        source = "tokenizer.pad_token_id"
    if token_id is None: raise RuntimeError("InternVL tokenizer has no usable EOS/pad token")
    cached = (int(token_id), separator, source)
    setattr(raw_model, "_grpo_template_eos", cached)
    if IS_MAIN:
        print(f"  InternVL completion EOS: id={cached[0]}, source={source}, sep={separator!r}")
    return cached

def chat_with_exact_generated_ids(raw_model, tokenizer, pixel_values, prompt,
                                  generation_config, eos_id, max_new_tokens):
    """Run the official InternVL chat path while capturing its exact rollout IDs.

    ``chat`` normally decodes to text and discards the sampled token IDs.  A
    decode->encode round trip is not guaranteed to reproduce the behavior-policy
    trajectory, so GRPO/KL must use the IDs returned by the generate call that
    actually occurred inside ``chat``.
    """
    chat_method = raw_model.chat
    chat_owner = getattr(chat_method, "__self__", raw_model)
    original_generate = chat_owner.generate
    captured = {"calls": 0, "sequences": None, "input_ids": None}

    def capturing_generate(*args, **kwargs):
        captured["calls"] += 1
        bound_arguments = {}
        try:
            bound_arguments = inspect.signature(original_generate).bind_partial(
                *args, **kwargs).arguments
        except (TypeError, ValueError):
            pass
        input_ids = kwargs.get("input_ids", bound_arguments.get("input_ids"))
        output = original_generate(*args, **kwargs)
        sequences = getattr(output, "sequences", output)
        if not isinstance(sequences, torch.Tensor):
            raise RuntimeError(
                f"InternVL generate returned unsupported type: {type(output)!r}")
        captured["sequences"] = sequences.detach()
        captured["input_ids"] = (
            input_ids.detach() if isinstance(input_ids, torch.Tensor) else None)
        return output

    setattr(chat_owner, "generate", capturing_generate)
    try:
        response = raw_model.chat(
            tokenizer, pixel_values, prompt, generation_config)
    finally:
        setattr(chat_owner, "generate", original_generate)

    if captured["calls"] != 1 or captured["sequences"] is None:
        raise RuntimeError(
            "InternVL chat did not expose exactly one generate rollout; "
            f"captured_calls={captured['calls']}")
    sequences = captured["sequences"]
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected InternVL rollout shape: {tuple(sequences.shape)}")
    completion = sequences[0]
    prompt_ids = captured["input_ids"]
    if prompt_ids is not None:
        prompt_row = prompt_ids.reshape(-1)
        if (completion.numel() >= prompt_row.numel() and
                torch.equal(completion[:prompt_row.numel()], prompt_row)):
            completion = completion[prompt_row.numel():]
    # Some decoder-only ``generate(inputs_embeds=...)`` implementations return
    # a synthetic BOS/pad seed before the sampled continuation.  It is not an
    # action produced after this prompt and must not enter the GRPO objective.
    synthetic_prefix_ids = {
        int(token_id) for token_id in
        (getattr(tokenizer, "bos_token_id", None),
         getattr(tokenizer, "pad_token_id", None))
        if token_id is not None and int(token_id) != int(eos_id)
    }
    while (completion.numel() > 0 and
           int(completion[0].item()) in synthetic_prefix_ids):
        completion = completion[1:]
    if completion.numel() > int(max_new_tokens):
        raise RuntimeError(
            "Could not separate InternVL prompt IDs from completion IDs: "
            f"rollout_tokens={completion.numel()}, max_new_tokens={max_new_tokens}")

    eos_positions = (completion == int(eos_id)).nonzero(as_tuple=False)
    terminated_by_eos = bool(eos_positions.numel() > 0)
    if terminated_by_eos:
        completion = completion[:int(eos_positions[0].item()) + 1]
    content_ids = completion[:-1] if terminated_by_eos else completion
    exact_text = tokenizer.decode(
        content_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
    if isinstance(response, tuple):
        response = response[0]
    response = "" if response is None else str(response).strip()
    # Reward parsing and log-prob now describe the same sampled token sequence.
    # Keep a hard diagnostic if remote-code chat applies an unexpected transform.
    if response != exact_text:
        raise RuntimeError(
            "InternVL chat text differs from its exact generated token decode; "
            f"chat_prefix={response[:120]!r}, token_prefix={exact_text[:120]!r}")
    return exact_text, completion.unsqueeze(0), terminated_by_eos

@torch.no_grad()
def generate_candidates(model, tokenizer, conv_module, image, prompt, num_candidates=G, max_new_tokens=None):
    generation_started = time.monotonic()
    generation_deadline = generation_started + SAMPLE_GENERATION_TIMEOUT_SECONDS
    generation_timed_out = False
    task = classify_task(prompt)
    if max_new_tokens is None: max_new_tokens = TASK_MAX_TOKENS.get(task, MAX_NEW_TOKENS)
    raw_model = model.module if hasattr(model, 'module') else model
    assert_active_adapter(raw_model, POLICY_ADAPTER, "candidate generation")
    was_training = raw_model.training
    raw_model.eval()
    dev = next(model.parameters()).device
    lm_generation_config = getattr(
        getattr(raw_model, "language_model", None), "generation_config", None)
    eos_id, eos_separator, eos_source = resolve_internvl_eos_token(
        raw_model, tokenizer, conv_module)
    pad_id = tokenizer.pad_token_id
    if pad_id is None: pad_id = eos_id
    pv = load_image_pixel_values(image, INPUT_SIZE, MAX_DYNAMIC_PATCH).to(dev, dtype=torch.bfloat16)
    texts, gids, candidate_eos_terminated = [], [], []
    for candidate_idx in range(num_candidates):
        remaining = generation_deadline - time.monotonic()
        if generation_timed_out or remaining <= 0.0:
            generation_timed_out = True
            texts.append("")
            tok = torch.tensor([[pad_id]], device=dev)
            candidate_eos_terminated.append(False)
        else:
            is_greedy = candidate_idx < GREEDY_CANDIDATES
            gc = dict(max_new_tokens=max_new_tokens, do_sample=not is_greedy,
                      pad_token_id=pad_id, eos_token_id=eos_id,
                      max_time=max(float(remaining), 0.1))
            if lm_generation_config is not None:
                gc.update({
                    name: value
                    for name, value in GENERATION_NEUTRAL_OVERRIDES.items()
                    if hasattr(lm_generation_config, name)
                })
            if not is_greedy:
                sampled_idx = candidate_idx - GREEDY_CANDIDATES
                if sampled_idx < min(EXPLOIT_SAMPLED_CANDIDATES.get(task, 7), num_candidates - GREEDY_CANDIDATES):
                    temp = SAMPLE_TEMPERATURE_LOW[task]
                else: temp = SAMPLE_TEMPERATURE_HIGH[task]
                # 显式top_k=0；若只在>0时传入，模型自带GenerationConfig可能把
                # rollout悄悄变成截断采样，而loss仍按未截断policy计算。
                gc.update(temperature=temp, top_p=SAMPLE_TOP_P[task],
                          top_k=SAMPLE_TOP_K)
                optional_sampling = {
                    "min_p": None, "typical_p": 1.0,
                    "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
                }
                if lm_generation_config is not None:
                    gc.update({name: value for name, value in optional_sampling.items()
                               if hasattr(lm_generation_config, name)})
            else:
                gc.update(temperature=None, top_p=None, top_k=None)
            resp, tok, terminated_by_eos = chat_with_exact_generated_ids(
                raw_model, tokenizer, pv, prompt, gc, eos_id, max_new_tokens)
            texts.append(resp)
            finished_before_deadline = time.monotonic() < generation_deadline
            if tok.shape[1] == 0:
                tok = torch.tensor([[pad_id]], device=dev)
            candidate_eos_terminated.append(terminated_by_eos)
            if not finished_before_deadline:
                generation_timed_out = True
        gids.append(tok)
    if was_training: raw_model.train()
    mx = max(g.shape[1] for g in gids)
    lengths = [g.shape[1] for g in gids]
    padded = [torch.cat([g, torch.full((1, mx - g.shape[1]), pad_id, dtype=g.dtype, device=dev)], dim=1) if g.shape[1] < mx else g for g in gids]
    completion_mask = (torch.arange(mx, device=dev).unsqueeze(0) < torch.tensor(lengths, device=dev).unsqueeze(1))
    shared_pixels = pv.unsqueeze(0).expand(num_candidates, *pv.shape)
    gen_runtime = {"timed_out": bool(generation_timed_out), "elapsed_seconds": float(time.monotonic() - generation_started),
                   "timeout_seconds": float(SAMPLE_GENERATION_TIMEOUT_SECONDS),
                   "eos_terminated": candidate_eos_terminated,
                   "eos_token_id": int(eos_id), "eos_separator": eos_separator,
                   "eos_source": eos_source}
    return (texts, shared_pixels, torch.cat(padded, dim=0), completion_mask, gen_runtime)

# ========================= Log Probs (InternVL专用) =========================
def _slice_vision_inputs(pv, gw, idx):
    if pv is None: return None, None
    return pv[idx], None

def extract_visual_features(model, pv):
    raw_model = model.module if hasattr(model, "module") else model
    vit = raw_model.extract_feature(pv)
    if vit.shape[-1] != raw_model.language_model.config.hidden_size:
        vit = raw_model.mlp1(vit)
    return vit.reshape(-1, vit.shape[-1]).clone()

def _single_log_probs(model, tokenizer, conv_module, pv, prompt, gen_ids, visual_features=None):
    dev = gen_ids.device
    raw_model = model.module if hasattr(model, 'module') else model
    gct = conv_module.get_conv_template
    IMG_CTX = '<IMG_CONTEXT>'; raw_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
    np_patches = pv.shape[0]
    img_tokens = '<img>' + IMG_CTX * raw_model.num_image_token * np_patches + '</img>'
    tmpl = gct(raw_model.template); tmpl.system_message = raw_model.system_message
    tmpl.append_message(tmpl.roles[0], prompt); tmpl.append_message(tmpl.roles[1], None)
    po = tmpl.get_prompt().replace('<image>', img_tokens, 1)
    pt = tokenizer(po, return_tensors="pt"); pl = pt.input_ids.shape[1]
    fids = torch.cat([pt.input_ids.to(dev), gen_ids], dim=1)
    fmask = torch.ones(1, fids.shape[1], dtype=torch.long, device=dev)
    vit = (visual_features if visual_features is not None else extract_visual_features(model, pv))
    emb = raw_model.language_model.get_input_embeddings()(fids).clone()
    B, N, C = emb.shape; emb = emb.reshape(B * N, C)
    sel = (fids == raw_model.img_context_token_id)
    pcnt = sel.sum().item()
    if pcnt > 0:
        if int(vit.shape[0]) != int(pcnt):
            raise RuntimeError(
                "InternVL visual-token count mismatch: "
                f"features={vit.shape[0]}, prompt_IMG_CONTEXT={pcnt}, "
                f"patches={np_patches}")
        emb[sel.reshape(B * N)] = vit
    emb = emb.reshape(B, N, C)
    if torch.is_grad_enabled(): emb.requires_grad_(True)
    fw_kwargs = {"inputs_embeds": emb, "attention_mask": fmask, "use_cache": False}
    fw_params = inspect.signature(raw_model.language_model.forward).parameters
    logits_keep = gen_ids.shape[1] + 1
    if "logits_to_keep" in fw_params: fw_kwargs["logits_to_keep"] = logits_keep
    elif "num_logits_to_keep" in fw_params: fw_kwargs["num_logits_to_keep"] = logits_keep
    out = raw_model.language_model(**fw_kwargs)
    if out.logits.shape[1] == logits_keep: gl = out.logits[:, -logits_keep:-1, :]
    elif out.logits.shape[1] >= pl + gen_ids.shape[1]: gl = out.logits[:, pl - 1:pl - 1 + gen_ids.shape[1], :]
    else: raise RuntimeError(f"Unexpected logits length {out.logits.shape[1]} for completion length {gen_ids.shape[1]}")
    result = -F.cross_entropy(gl.transpose(1, 2), gen_ids, reduction="none")
    del emb, out, gl
    return result

def get_policy_log_probs(model, tokenizer, conv_module, pixel_values, prompt, generated_ids,
                         visual_features=None, completion_mask=None,
                         expected_adapter=POLICY_ADAPTER):
    assert_active_adapter(model, expected_adapter, "completion log-prob forward")
    Gv = generated_ids.shape[0]; target_len = generated_ids.shape[1]; lps = []
    for i in range(Gv):
        real_len = (int(completion_mask[i].sum().item()) if completion_mask is not None
                    else target_len)
        real_len = max(min(real_len, target_len), 1)
        candidate_ids = generated_ids[i:i + 1, :real_len]
        lp = _single_log_probs(model, tokenizer, conv_module, pixel_values[i], prompt,
                               candidate_ids, visual_features=visual_features)
        if lp.shape[1] < target_len:
            lp = F.pad(lp, (0, target_len - lp.shape[1]), value=0.0)
        lps.append(lp)
    return torch.cat([lp[:, :target_len] for lp in lps], dim=0)

@torch.no_grad()
def get_ref_log_probs(model, tokenizer, conv_module, pixel_values, prompt, generated_ids,
                      visual_features=None, completion_mask=None):
    raw = model.module if hasattr(model, 'module') else model
    backbone = raw.get_base_model()
    policy_projector = backbone.mlp1
    backbone.mlp1 = backbone._cgpc_reference_projector
    raw.set_adapter(REFERENCE_ADAPTER)
    try:
        assert_active_adapter(raw, REFERENCE_ADAPTER, "reference log-prob entry")
        return get_policy_log_probs(model, tokenizer, conv_module, pixel_values, prompt,
                                    generated_ids, visual_features=None,
                                    completion_mask=completion_mask,
                                    expected_adapter=REFERENCE_ADAPTER)
    finally:
        backbone.mlp1 = policy_projector
        raw.set_adapter(POLICY_ADAPTER)
        assert_active_adapter(raw, POLICY_ADAPTER, "reference log-prob exit")

# ========================= GRPO 损失 =========================
def k3_kl_mean(policy_log_probs: torch.Tensor,
               ref_log_probs: torch.Tensor,
               token_mask: torch.Tensor) -> torch.Tensor:
    """Mean sampled-token k3 KL on the supplied token mask."""
    mask = token_mask.to(policy_log_probs.dtype)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    mlen = min(
        policy_log_probs.shape[1], ref_log_probs.shape[1], mask.shape[1])
    if mlen <= 0:
        return 0.0 * policy_log_probs.sum()
    mask = mask[:, :mlen]
    ref_diff = (
        ref_log_probs[:, :mlen] - policy_log_probs[:, :mlen]
    ).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    per_token_kl = torch.exp(ref_diff) - ref_diff - 1.0
    return (
        (per_token_kl * mask).sum(dim=1) /
        mask.sum(dim=1).clamp_min(1.0)
    ).mean()

def compute_grpo_loss(policy_log_probs, ref_log_probs, rewards, completion_mask,
                      beta=BETA, gamma=GAMMA, advantages=None,
                      token_denominator=None):
    if advantages is None:
        raise ValueError(
            "GRPO requires explicitly centered precomputed advantages")
    active_mask = completion_mask.to(policy_log_probs.dtype)
    mlen = min(policy_log_probs.shape[1], ref_log_probs.shape[1], completion_mask.shape[1])
    if mlen <= 0: return 0.0 * policy_log_probs.sum(), {"loss": 0.0, "policy_loss": 0.0, "kl_div": 0.0, "token_nll": 0.0}
    active_mask = active_mask[:, :mlen]; p_lp = policy_log_probs[:, :mlen]; r_lp = ref_log_probs[:, :mlen]
    denominator = (active_mask.sum().clamp_min(1.0) if token_denominator is None
                   else torch.as_tensor(token_denominator, device=p_lp.device,
                                        dtype=p_lp.dtype).clamp_min(1.0))
    ref_diff = (r_lp - p_lp).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    kl = ((torch.exp(ref_diff) - ref_diff - 1.0) * active_mask).sum() / denominator
    token_nll = -((p_lp * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp_min(1.0)).mean()
    adv = advantages.to(p_lp.dtype)
    active_adv = adv.unsqueeze(1) * active_mask
    policy_loss = -((p_lp * active_mask * active_adv).sum() / denominator)
    # 与Qwen cgpc一致；GAMMA固定为0，仅保留同语义的采样token log-prob入口。
    token_logp = (p_lp * active_mask).sum() / denominator
    loss = policy_loss + beta * kl + gamma * token_logp
    return loss, {"loss": float(loss.item()), "policy_loss": float(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss),
                  "kl_div": float(kl.item()) if isinstance(kl, torch.Tensor) else float(kl), "token_nll": float(token_nll.item())}

def masked_sequence_mean(log_probs: torch.Tensor,
                         token_mask: torch.Tensor) -> torch.Tensor:
    """Length-normalized sequence log-likelihood over an explicit token mask."""
    mask = token_mask.to(log_probs.dtype)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    mlen = min(log_probs.shape[1], mask.shape[1])
    if mlen <= 0:
        return 0.0 * log_probs.sum()
    mask = mask[:, :mlen]
    return (
        (log_probs[:, :mlen] * mask).sum(dim=1) /
        mask.sum(dim=1).clamp_min(1.0)
    ).mean()

# ========================= GRPO 组内优势 =========================
def centered_group_advantages(rewards: torch.Tensor,
                              micro_raw_scores: torch.Tensor,
                              localization_scores: torch.Tensor,
                              active_indices: List[int],
                              task: str,
                              enable_localization: bool = False,
                              return_diagnostics: bool = False):
    """Build one group-relative advantage without double-counting metrics.

    ``micro_raw_scores`` is the training-only quality-aware TP marginal utility:
    strict TP/FP/FN remain unchanged for training monitoring. Every strict TP
    keeps a positive base credit and a smaller continuous IoU quality credit.
    A small localization term is always retained (including ALL) so
    below-threshold near misses and box tightness remain distinguishable.
    Composite reward is only a final fallback when both channels are constant.
    """
    advantages = torch.zeros_like(rewards)
    if not active_indices:
        empty_diag = {
            "normalized_adv_abs_mean": 0.0,
            "micro_raw_adv_abs_mean": 0.0,
            "localization_adv_abs_mean": 0.0,
            "total_adv_abs_mean": 0.0,
            "total_adv_abs_sum": 0.0,
            "total_adv_max_abs": 0.0,
            "micro_raw_clip_scale": 1.0,
            "micro_raw_clip_triggered": False,
            "task_advantage_gain": float(TASK_ADVANTAGE_GAIN[task]),
        }
        return (advantages, empty_diag) if return_diagnostics else advantages
    active_rewards = rewards[active_indices]
    active_micro_raw = micro_raw_scores[active_indices]
    active_localization = localization_scores[active_indices]
    centered_reward = (
        active_rewards - active_rewards.mean()
    ) * ADVANTAGE_FIXED_SCALE
    centered_micro_raw = (
        active_micro_raw - active_micro_raw.mean()
    ) * MICRO_RAW_ADVANTAGE_SCALE
    # 径向缩放同时保持prompt内零和与绝对上界；直接逐元素clamp后再居中会重新
    # 越过配置上界（例如极端7候选组可从2回弹到3.4）。
    max_abs_micro_raw = centered_micro_raw.abs().max()
    clip_scale = (
        MICRO_RAW_ADVANTAGE_CLIP /
        max_abs_micro_raw.clamp_min(MICRO_RAW_ADVANTAGE_CLIP))
    centered_micro_raw = centered_micro_raw * clip_scale
    raw_has_signal = bool(max_abs_micro_raw.detach().item() > 1.0e-12)
    if enable_localization:
        centered_localization = (
            active_localization - active_localization.mean()
        ) * float(LOCALIZATION_ADVANTAGE_SCALE[task])
    else:
        centered_localization = torch.zeros_like(active_localization)
    localization_has_signal = bool(
        centered_localization.detach().abs().max().item() > 1.0e-12)
    if raw_has_signal or localization_has_signal:
        # 不再重复叠加F1/precision/recall复合项；连续IoU和定位已覆盖主方向。
        centered_reward = torch.zeros_like(centered_reward)
    task_gain = float(TASK_ADVANTAGE_GAIN[task])
    centered_reward = centered_reward * task_gain
    centered_micro_raw = centered_micro_raw * task_gain
    centered_localization = centered_localization * task_gain
    active_advantages = (
        centered_reward + centered_micro_raw + centered_localization)
    advantages[active_indices] = active_advantages
    if not return_diagnostics:
        return advantages
    diagnostics = {
        "normalized_adv_abs_mean": float(
            centered_reward.detach().abs().mean().item()),
        "micro_raw_adv_abs_mean": float(
            centered_micro_raw.detach().abs().mean().item()),
        "localization_adv_abs_mean": float(
            centered_localization.detach().abs().mean().item()),
        "total_adv_abs_mean": float(
            active_advantages.detach().abs().mean().item()),
        "total_adv_abs_sum": float(
            active_advantages.detach().abs().sum().item()),
        "total_adv_max_abs": float(
            active_advantages.detach().abs().max().item()),
        "micro_raw_clip_scale": float(clip_scale.detach().item()),
        "micro_raw_clip_triggered": bool(
            max_abs_micro_raw.detach().item() > MICRO_RAW_ADVANTAGE_CLIP),
        "task_advantage_gain": task_gain,
    }
    return advantages, diagnostics

def candidate_source(task, idx):
    return "greedy" if idx < GREEDY_CANDIDATES else "raw_policy_sample"

def completion_generation_diagnostics(generated_ids, completion_mask, tokenizer, task,
                                      token_limit=None, eos_terminated_override=None):
    token_limit = int(token_limit if token_limit is not None else TASK_MAX_TOKENS[task])
    token_counts = [int(v) for v in completion_mask.sum(dim=1).detach().cpu().tolist()]
    eos_id = tokenizer.eos_token_id
    if eos_terminated_override is not None:
        eos_terminated = [bool(v) for v in eos_terminated_override]
        if len(eos_terminated) != len(token_counts):
            raise ValueError("eos_terminated_override length mismatch")
    elif eos_id is None: eos_terminated = [False] * len(token_counts)
    else:
        eos_hits = (generated_ids.eq(eos_id) & completion_mask.bool()).any(dim=1).detach().cpu().tolist()
        eos_terminated = [bool(v) for v in eos_hits]
    token_limit_reached = [count >= token_limit and not has_eos for count, has_eos in zip(token_counts, eos_terminated)]
    return {"token_counts": token_counts, "eos_terminated": eos_terminated, "token_limit_reached": token_limit_reached}

def soft_overlong_penalty(task, token_count, token_limit=None):
    token_limit = int(token_limit if token_limit is not None else TASK_MAX_TOKENS[task])
    cache = min(int(SOFT_OVERLONG_CACHE_TOKENS[task]), max(int(token_limit * SOFT_OVERLONG_CACHE_RATIO), 1), token_limit)
    safe_limit = token_limit - cache
    if cache <= 0 or token_count <= safe_limit: return 0.0
    fraction = min(max((token_count - safe_limit) / cache, 0.0), 1.0)
    return float(SOFT_OVERLONG_MAX_PENALTY[task]) * fraction

def dynamic_batch_should_step(effective_prompts, prompt_rounds, is_last_batch=False):
    if not DYNAMIC_EFFECTIVE_BATCHING:
        ready = prompt_rounds >= GRADIENT_ACCUMULATION_STEPS
        return ready, "fixed" if ready else "collecting"
    if effective_prompts >= DYNAMIC_TARGET_EFFECTIVE_PROMPTS: return True, "target"
    if prompt_rounds >= DYNAMIC_MAX_PROMPT_ROUNDS: return True, "cap"
    if is_last_batch: return True, "epoch_tail"
    return False, "collecting"

def micro_f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom > 0 else 1.0

def metric_best_candidate_index(results: List[Dict],
                                indices: List[int] = None,
                                token_counts: List[int] = None) -> int:
    """Select within the external-metric optimum using quality-aware ties.

    ``micro_raw`` remains the primary key.  v3 resolved the frequent ties by
    earliest candidate index, even though the log shows about two candidates
    with identical TP/FP/FN but almost always different boxes.  v4 uses
    verifier quality and shorter valid completion before the deterministic
    index tie-break, reducing arbitrary self-distillation targets.
    """
    candidate_indices = (
        list(range(len(results))) if indices is None else list(indices))
    if not candidate_indices:
        raise ValueError("metric_best_candidate_index requires candidates")
    return max(
        candidate_indices,
        key=lambda idx: (
            float(results[idx]["micro_raw"]),
            int(results[idx]["tp"]),
            -int(results[idx]["fp"]),
            float(results[idx]["f1"]),
            float(results[idx]["total"]),
            float(results[idx]["iou"]),
            float(results[idx]["localization"]),
            float(results[idx]["format"]),
            -int(results[idx]["invalid_count"]),
            -(int(token_counts[idx]) if token_counts is not None else 0),
            -idx,
        ))

def candidate_is_clean(result, eos_terminated, token_limit_reached):
    return bool(eos_terminated and not token_limit_reached and
                not result["invalid_json"] and not result["truncated"] and
                not result.get("label_count_mismatch", False) and
                not result.get("repeated_coordinate", False) and
                not result.get("repeated_tail", False) and
                result["invalid_count"] == 0 and result["valid_count"] > 0)





@torch.no_grad()
def synchronize_trainable_gradients(parameters, world_size):
    """显式同步梯度。

    InternVL的log-prob路径直接调用language_model子模块，无法经过外层DDP.forward；
    因而不能依赖DDP reducer。所有rank在optimizer step前按固定参数顺序all-reduce，
    本rank没有梯度的参数用零张量参与，语义等价于DDP的global mean gradient。
    """
    if world_size <= 1: return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(
                parameter, memory_format=torch.preserve_format)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))

def parameter_group_grad_norm(parameters, device):
    per_param = [torch.linalg.vector_norm(p.grad.detach()).float() for p in parameters if p.grad is not None]
    if not per_param: return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(per_param))

@torch.no_grad()
def parameter_group_value_norm(parameters, device):
    per_param = [torch.linalg.vector_norm(p.detach()).float() for p in parameters]
    if not per_param: return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(per_param))

@torch.no_grad()
def snapshot_trainable_parameters(named_parameters):
    return {name: param.detach().to(device="cpu", dtype=torch.float32).clone() for name, param in named_parameters}

def compare_parameter_snapshots(reference, current):
    totals = [0.0, 0.0, 0.0]; by_kind = {}
    for name, cv in current.items():
        rv = reference.get(name)
        if rv is None: continue
        delta_sq = float(torch.sum((cv - rv).square()).item())
        ref_sq = float(torch.sum(rv.square()).item())
        totals[0] += delta_sq; totals[1] += ref_sq; totals[2] += float(torch.sum(cv.square()).item())
        kind = adapter_parameter_kind(name)
        kt = by_kind.setdefault(kind, [0.0, 0.0, 0.0])
        kt[0] += delta_sq; kt[1] += ref_sq; kt[2] += float(torch.sum(cv.square()).item())
    def finalize(vals):
        dn = math.sqrt(vals[0]); rn = math.sqrt(vals[1])
        return {"relative": dn / max(rn, 1.0e-12), "delta_norm": dn, "reference_norm": rn, "current_norm": math.sqrt(vals[2])}
    result = finalize(totals); result["by_kind"] = {k: finalize(v) for k, v in by_kind.items()}
    return result

def own_qwen_lora_drift_reference(optimizer_update_step):
    """在Qwen实测锚点间线性插值；禁止对未跑到的update-250之后做假外推。"""
    update_step = int(optimizer_update_step)
    if update_step <= 0 or update_step > max(OWN_QWEN_LORA_DRIFT_BY_UPDATE):
        return None
    left_step, left_value = 0, 0.0
    for right_step in sorted(OWN_QWEN_LORA_DRIFT_BY_UPDATE):
        right_value = float(OWN_QWEN_LORA_DRIFT_BY_UPDATE[right_step])
        if update_step <= right_step:
            fraction = (update_step - left_step) / max(right_step - left_step, 1)
            return left_value + fraction * (right_value - left_value)
        left_step, left_value = right_step, right_value
    return None

def build_parameter_drift_report(reference, named_trainable, optimizer_update_step,
                                 reference_scope="training_start"):
    """同一LoRA-only口径比较InternVL当前漂移与自有Qwen cgpc实跑曲线。"""
    report = compare_parameter_snapshots(
        reference, snapshot_trainable_parameters(named_trainable))
    qwen_reference = (own_qwen_lora_drift_reference(optimizer_update_step)
                      if reference_scope == "training_start" else None)
    lora_relative = report.get("by_kind", {}).get("lora", {}).get("relative")
    comparison = {
        "script": OWN_QWEN_REFERENCE_SCRIPT,
        "log": OWN_QWEN_REFERENCE_LOG,
        "optimizer_update_step": int(optimizer_update_step),
        "qwen_lora_relative": qwen_reference,
        "internvl_lora_relative": lora_relative,
        "internvl_to_qwen_ratio": None,
        "similar_band": list(OWN_QWEN_DRIFT_RATIO_WARNING),
        "status": "reference_unavailable",
        "internvl_drift_reference_scope": reference_scope,
    }
    if qwen_reference is not None and lora_relative is not None:
        ratio = float(lora_relative) / max(float(qwen_reference), 1.0e-12)
        low, high = OWN_QWEN_DRIFT_RATIO_WARNING
        comparison["internvl_to_qwen_ratio"] = ratio
        comparison["status"] = (
            "similar" if low <= ratio <= high
            else ("lower_than_qwen" if ratio < low else "higher_than_qwen"))
    report["own_qwen_lora_drift_comparison"] = comparison
    return report

def print_qwen_drift_comparison(report):
    comparison = report["own_qwen_lora_drift_comparison"]
    if comparison["qwen_lora_relative"] is None:
        reason = ("恢复运行只能测量本次进程增量，不能与Qwen累计漂移比较"
                  if comparison["internvl_drift_reference_scope"] != "training_start"
                  else "自有Qwen实测参照仅覆盖optimizer update 1--250；当前不外推")
        print(f"  │ LoRA漂移: {reason}")
        return
    print(f"  │ LoRA漂移 InternVL/Qwen={comparison['internvl_lora_relative']:.3%}/"
          f"{comparison['qwen_lora_relative']:.3%}, ratio="
          f"{comparison['internvl_to_qwen_ratio']:.2f}, status={comparison['status']}")

def summarize_monitor_records(records):
    summary = {}
    groups = {"TRAIN_WEIGHTED_OVERALL": records}
    groups.update({task: [r for r in records if r["task"] == task] for task in TASK_REPEAT})
    for task, task_records in groups.items():
        if not task_records: continue
        count = len(task_records)
        def mean(key): return sum(float(r[key]) for r in task_records) / count
        def sum_int(key): return sum(int(r[key]) for r in task_records)
        kl_values = sorted(float(r["kl"]) for r in task_records)
        kl_p95 = kl_values[min(max(math.ceil(0.95 * len(kl_values)) - 1, 0), len(kl_values) - 1)]
        summary[task] = {"samples": count, "modes": dict(Counter(r["mode"] for r in task_records)),
            "best_sources": dict(Counter(r["best_source"] for r in task_records)),
            "preference_modes": dict(Counter(r["preference_mode"] for r in task_records)),
            "loss": mean("loss"), "policy_loss": mean("policy_loss"),
            "kl_mean": mean("kl"), "kl_p95": kl_p95, "kl_max": max(kl_values),
            "kl_measured_updates": sum(int(r["kl"] > 0.0) for r in task_records),
            "token_nll": mean("token_nll"),
            "greedy_reward": mean("greedy_reward"), "greedy_f1": mean("greedy_f1"),
            "greedy_precision": mean("greedy_precision"),
            "greedy_recall": mean("greedy_recall"),
            "best_reward": mean("reward_best"), "best_f1": mean("best_f1"),
            "any_hit_at_g_rate": mean("any_hit_at_g"),
            "perfect_at_g_rate": mean("perfect_at_g"),
            "sampled_rescue_rate": mean("sampled_rescue"),
            "unique_text_ratio": mean("unique_text_ratio"),
            "unique_bbox_ratio": mean("unique_bbox_ratio"),
            "active_sampled_mean": mean("active_sampled"),
            "clean_sampled_mean": mean("clean_sampled"),
            "group_signal_rate": mean("group_has_signal"),
            "greedy_format_score": mean("greedy_format"),
            "greedy_invalid_json_rate": mean("greedy_invalid_json"),
            "greedy_truncated_rate": mean("greedy_truncated"),
            "greedy_label_mismatch_rate": mean("greedy_label_mismatch"),
            "greedy_repeated_rate": mean("greedy_repeated"),
            "greedy_token_limit_rate": mean("greedy_token_limit"),
            "candidate_any_invalid_json_rate": mean("candidate_any_invalid_json"),
            "candidate_any_repeated_rate": mean("candidate_any_repeated"),
            "candidate_any_token_limit_rate": mean("candidate_any_token_limit"),
            "greedy_counts": {"tp": sum_int("greedy_tp"), "fp": sum_int("greedy_fp"), "fn": sum_int("greedy_fn")},
            "greedy_window_micro_f1": micro_f1_from_counts(sum_int("greedy_tp"), sum_int("greedy_fp"), sum_int("greedy_fn")),
            "metric_best_counts": {"tp": sum_int("metric_best_tp"), "fp": sum_int("metric_best_fp"), "fn": sum_int("metric_best_fn")},
            "metric_best_window_micro_f1": micro_f1_from_counts(sum_int("metric_best_tp"), sum_int("metric_best_fp"), sum_int("metric_best_fn")),
            "sample_update_rate": mean("has_update")}
    return summary

# ========================= 训练循环 =========================
def all_preference_rejection_reasons(
        task: str, candidate_idx: int, candidate_reward: float,
        greedy_reward: float, candidate_result: Dict,
        greedy_result: Dict, greedy_token_limit: bool,
        greedy_eos: bool, candidate_token_limit: bool,
        candidate_eos: bool) -> List[str]:
    """Return failed gates for strict-overlong or quality ALL transfer."""
    reasons = []
    if not ALL_OVERLONG_PREFERENCE_ENABLED:
        reasons.append("disabled")
    if task != "ALL":
        reasons.append("not_all")
    if candidate_idx == 0:
        reasons.append("greedy_candidate")
    if not candidate_eos:
        reasons.append("candidate_no_eos")
    if candidate_token_limit:
        reasons.append("candidate_limit")
    if float(candidate_result["format"]) < 1.0:
        reasons.append("format")
    if int(candidate_result["invalid_count"]) != 0:
        reasons.append("invalid_box")
    if int(candidate_result["valid_count"]) <= 0:
        reasons.append("no_valid_box")
    strict_overlong = bool(
        task == "ALL" and greedy_token_limit and not greedy_eos)
    if strict_overlong:
        if candidate_reward < ALL_OVERLONG_PREFERENCE_MIN_REWARD:
            reasons.append("reward_floor")
        if candidate_reward <= greedy_reward:
            reasons.append("reward_not_better")
        if float(candidate_result["f1"]) < ALL_OVERLONG_PREFERENCE_MIN_F1:
            reasons.append("f1_floor")
        if (float(candidate_result["micro_raw"]) <=
                float(greedy_result["micro_raw"])):
            reasons.append("micro_not_better")
        if float(candidate_result["f1"]) < float(greedy_result["f1"]):
            reasons.append("f1_worse")
    else:
        if candidate_reward < ALL_QUALITY_PREFERENCE_MIN_REWARD:
            reasons.append("quality_reward_floor")
        if (candidate_reward - greedy_reward <
                ALL_QUALITY_PREFERENCE_MIN_REWARD_GAP):
            reasons.append("quality_reward_gap")
        if float(candidate_result["f1"]) < ALL_QUALITY_PREFERENCE_MIN_F1:
            reasons.append("quality_f1_floor")
        if (float(candidate_result["f1"]) -
                float(greedy_result["f1"]) <
                ALL_QUALITY_PREFERENCE_MIN_F1_GAP):
            reasons.append("quality_f1_gap")
        if (float(candidate_result["micro_raw"]) -
                float(greedy_result["micro_raw"]) <
                ALL_QUALITY_PREFERENCE_MIN_MICRO_RAW_GAP):
            reasons.append("quality_micro_gap")
    if (float(candidate_result["precision"]) <
            float(greedy_result["precision"])):
        reasons.append("precision_worse")
    return reasons


def preference_suffix_masks(generated_ids: torch.Tensor,
                            completion_mask: torch.Tensor,
                            chosen_idx: int,
                            rejected_idx: int = 0
                            ) -> Tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor, int]:
    """Return disjoint first-divergence and post-divergence suffix masks.

    The pairwise objective must not suppress the shared JSON prefix or boxes that
    are identical in both candidates. The first token is never included in the
    suffix masks, preventing accidental double weighting.
    """
    chosen_valid = completion_mask[chosen_idx].bool()
    rejected_valid = completion_mask[rejected_idx].bool()
    chosen_len = int(chosen_valid.sum().item())
    rejected_len = int(rejected_valid.sum().item())
    common_len = min(chosen_len, rejected_len)
    divergence = 0
    if PAIRWISE_SUFFIX_ONLY and common_len > 0:
        different = generated_ids[chosen_idx, :common_len].ne(
            generated_ids[rejected_idx, :common_len])
        locations = torch.nonzero(different, as_tuple=False)
        if locations.numel() > 0:
            divergence = int(locations[0, 0].item())
        elif chosen_len != rejected_len:
            divergence = max(common_len - 1, 0)
        else:
            empty = torch.zeros_like(chosen_valid, dtype=torch.bool)
            return empty, empty.clone(), empty.clone(), empty.clone(), -1
    positions = torch.arange(
        completion_mask.shape[1], device=completion_mask.device)
    chosen_first = chosen_valid & positions.eq(divergence)
    rejected_first = rejected_valid & positions.eq(divergence)
    chosen_suffix = chosen_valid & positions.gt(divergence)
    rejected_suffix = rejected_valid & positions.gt(divergence)
    if bool((chosen_first & chosen_suffix).any().item()) or bool(
            (rejected_first & rejected_suffix).any().item()):
        raise RuntimeError("preference first/suffix masks overlap")
    return (chosen_first, rejected_first, chosen_suffix, rejected_suffix,
            divergence)


@torch.no_grad()
def reference_relative_pairwise_stats(
        chosen_policy_log_probs: torch.Tensor,
        rejected_policy_log_probs: torch.Tensor,
        chosen_ref_log_probs: torch.Tensor,
        rejected_ref_log_probs: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
        weight: float,
        beta: float = PAIRWISE_PREFERENCE_BETA) -> Dict[str, float]:
    """DPO-style pair loss with a bounded exact split-gradient coefficient."""
    chosen_policy = masked_sequence_mean(
        chosen_policy_log_probs, chosen_mask)
    rejected_policy = masked_sequence_mean(
        rejected_policy_log_probs, rejected_mask)
    chosen_ref = masked_sequence_mean(chosen_ref_log_probs, chosen_mask)
    rejected_ref = masked_sequence_mean(rejected_ref_log_probs, rejected_mask)
    relative_margin = (
        (chosen_policy - rejected_policy) -
        (chosen_ref - rejected_ref))
    scaled_margin = float(beta) * relative_margin
    pair_loss = float(weight) * F.softplus(-scaled_margin)
    # d[-w log sigmoid(beta*m)]/d m = -w*beta*sigmoid(-beta*m).
    # Applying -coefficient*logp(chosen) and +coefficient*logp(rejected)
    # in separate forwards gives the exact first-order gradient without
    # retaining two 6144-token computation graphs simultaneously.
    coefficient = (
        float(weight) * float(beta) * torch.sigmoid(-scaled_margin))
    chosen_kl = k3_kl_mean(
        chosen_policy_log_probs, chosen_ref_log_probs, chosen_mask)
    rejected_kl = k3_kl_mean(
        rejected_policy_log_probs, rejected_ref_log_probs, rejected_mask)
    pair_kl = 0.5 * (chosen_kl + rejected_kl)
    pair_safe = bool(
        math.isfinite(float(chosen_kl.item())) and
        math.isfinite(float(rejected_kl.item())) and
        float(chosen_kl.item()) <= KL_SKIP_THRESHOLD and
        float(rejected_kl.item()) <= KL_SKIP_THRESHOLD)
    return {
        "loss": float(pair_loss.item()),
        "coefficient": float(coefficient.item()),
        "relative_margin": float(relative_margin.item()),
        "kl": float(pair_kl.item()),
        "chosen_kl": float(chosen_kl.item()),
        "rejected_kl": float(rejected_kl.item()),
        "pair_safe": pair_safe,
        "chosen_mean_logp": float(chosen_policy.item()),
        "rejected_mean_logp": float(rejected_policy.item()),
    }


def save_policy_checkpoint(model, directory):
    model.save_pretrained(directory, selected_adapters=[POLICY_ADAPTER])
    torch.save(model.get_base_model().mlp1.state_dict(),
               os.path.join(directory, "visual_projector.pt"))


def select_preference_pair(results, active, diagnostics, task, ids, mask):
    qualified = [i for i in active if not all_preference_rejection_reasons(
        task, i, results[i]['total'], results[0]['total'], results[i], results[0],
        diagnostics['token_limit_reached'][0], diagnostics['eos_terminated'][0],
        diagnostics['token_limit_reached'][i], diagnostics['eos_terminated'][i])]
    if not qualified:
        return None
    chosen = metric_best_candidate_index(results, qualified, diagnostics['token_counts'])
    masks = preference_suffix_masks(ids, mask, chosen)
    if masks[-1] < 0:
        return None
    bbox_support = sum(results[i]['bbox_signature'] == results[chosen]['bbox_signature'] for i in qualified)
    counts = lambda i: tuple(results[i][key] for key in ('tp', 'fp', 'fn'))
    metric_support = sum(counts(i) == counts(chosen) for i in qualified)
    recovery = diagnostics['token_limit_reached'][0] and not diagnostics['eos_terminated'][0]
    if not recovery and max(bbox_support, metric_support) < ALL_QUALITY_PREFERENCE_MIN_SUPPORT:
        return None
    scale = 1.0 if bbox_support >= 2 else 0.75 if metric_support >= 2 else 0.5
    first = ALL_OVERLONG_PREFERENCE_FIRST_WEIGHT if recovery else ALL_QUALITY_PREFERENCE_FIRST_WEIGHT
    suffix = ALL_OVERLONG_PREFERENCE_SUFFIX_WEIGHT if recovery else ALL_QUALITY_PREFERENCE_SUFFIX_WEIGHT
    return dict(chosen=chosen, masks=masks, weights=(first*scale, suffix*scale),
                mode='all_recovery' if recovery else 'all_quality', support=scale)


def train(policy_lora_path=None, resume_state_path=None):
    is_ddp = IS_DDP; torch.set_float32_matmul_precision('high')
    policy_lora_path = policy_lora_path or POLICY_LORA_PATH
    assert_training_output_is_fresh(resume_state_path)
    resume_state = None
    if resume_state_path is not None:
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if int(resume_state.get("format_version", 0)) != 1: raise RuntimeError(f"Unsupported resume format: {resume_state.get('format_version')}")
        if int(resume_state.get("world_size", -1)) != WORLD_SIZE: raise RuntimeError(f"WORLD_SIZE mismatch: {resume_state.get('world_size')} vs {WORLD_SIZE}")
        if (resume_state.get("policy_objective") != POLICY_OBJECTIVE or resume_state.get("policy_token_denominator") != POLICY_TOKEN_DENOMINATOR):
            raise RuntimeError("resume state objective/denominator mismatch")
        if resume_state.get("reference_lora_path") != REFERENCE_LORA_PATH:
            raise RuntimeError("resume state reference LoRA mismatch")
        saved_data_hash = resume_state.get("train_json_sha256")
        current_data_hash = file_sha256(TRAIN_JSON)
        if saved_data_hash and saved_data_hash != current_data_hash:
            raise RuntimeError("resume state training-data SHA256 mismatch")
        saved_hash = resume_state.get("training_script_sha256")
        current_hash = file_sha256(__file__)
        if saved_hash and saved_hash != current_hash and saved_hash.upper() not in AUTHORIZED_RESUME_SHA256:
            raise RuntimeError("Script SHA256 mismatch; add old hash to AUTHORIZED_RESUME_SHA256 to resume")
        if IS_MAIN and saved_hash != current_hash and saved_hash.upper() in AUTHORIZED_RESUME_SHA256:
            print(f"Authorized script change resume: old={saved_hash} new={current_hash}")
        policy_lora_path = resume_state["policy_adapter_path"]

    if IS_MAIN:
        print("=" * 60)
        distributed_label = (f"(显式梯度同步 {WORLD_SIZE} GPU)" if is_ddp else "(单卡)")
        print(f"GRPO InternVL-8B 优化训练 {distributed_label}")
        print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"模型: base={BASE_MODEL_PATH}; policy={policy_lora_path}")
        if resume_state is not None:
            print(f"恢复状态: {resume_state_path}; slot={resume_state['global_step']}, opt={resume_state['optimizer_update_step']}, next=epoch{resume_state['epoch']+1}/batch{resume_state['next_batch_idx']}")
        print(f"数据/输出: train={TRAIN_JSON}; output={GRPO_OUTPUT_DIR}")
        print(f"核心参数: G={G}(1 greedy+{G-GREEDY_CANDIDATES} sampled), LR={GRPO_LR:.2e}, β={BETA}, grad_clip={MAX_GRAD_NORM}, epochs={GRPO_EPOCHS}")
        print(f"更新批量: candidate_mini_batch={MINI_BATCH}, accumulation={GRADIENT_ACCUMULATION_STEPS}, "
              f"world={WORLD_SIZE}, global_prompt_batch={GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE}")
        print(f"长度: deploy={TASK_MAX_TOKENS}; train_buckets={TRAIN_BUCKET_MAX_TOKENS}")
        print(f"目标: {POLICY_OBJECTIVE}/{LOSS_REDUCTION}; adv=hard/soft-TP×{MICRO_RAW_ADVANTAGE_SCALE}")
        print(f"ALL preference consolidation: enabled={ALL_OVERLONG_PREFERENCE_ENABLED}")
        print("=" * 60)

    tokenizer, policy_model, conv_module = load_models(policy_lora_path)
    # 不包装外层DDP：InternVL的自定义多模态log-prob前向会直接进入language_model
    # 子模块，绕过DDP.forward。梯度在每次optimizer step前显式all-reduce。
    train_model = policy_model

    if IS_MAIN: print(f"\n[{datetime.now():%H:%M:%S}] 加载训练数据...")
    dataset = CottonBaleDataset(TRAIN_JSON, IMAGE_ROOT)
    sampler = CostBalancedDistributedSampler(dataset, WORLD_SIZE, GLOBAL_RANK if is_ddp else 0, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    epoch_prompt_exposures = len(dataloader) * WORLD_SIZE
    if IS_MAIN:
        print(f"原始训练样本数: {len(dataset.data)}  采样后: {len(dataset)}")
        print(f"原始任务分布: {dict(dataset.task_counts)}  重复权重: {TASK_REPEAT}")
        print(f"实际采样任务分布: {dict(dataset.sampled_task_counts)}")
        print(f"每rank每轮样本数: {len(dataloader)}")

    named_trainable = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]
    lora_params = [p for n, p in named_trainable if adapter_parameter_kind(n) == "lora"]
    projector_params = [p for n, p in named_trainable if adapter_parameter_kind(n) == "projector"]
    trainable_params = [p for _, p in named_trainable]
    if not lora_params: raise RuntimeError("No trainable LoRA parameters found")
    if is_ddp:
        for parameter in trainable_params:
            dist.broadcast(parameter.data, src=0)
    optimizer_groups = [{"params": lora_params, "lr": GRPO_LR, "group_name": "lora"}]
    if projector_params: optimizer_groups.append({"params": projector_params, "lr": GRPO_LR * 0.25, "group_name": "projector"})
    lora_parameter_count = sum(p.numel() for p in lora_params)
    projector_parameter_count = sum(p.numel() for p in projector_params)
    lora_initial_norm = float(parameter_group_value_norm(lora_params, DEVICE).item())
    drift_run_start = (snapshot_trainable_parameters(named_trainable)
                       if IS_MAIN else None)
    drift_reference_scope = ("training_start" if resume_state is None
                             else "resume_process_start")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY)
    planned_steps = len(dataloader) * GRPO_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    total_steps = planned_steps; warmup_steps = max(int(total_steps * WARMUP_RATIO), 1)
    def lr_factor(step):
        if step < warmup_steps: return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * step / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    if IS_MAIN:
        group_lrs = {group.get("group_name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)}
        print(f"优化器: AdamW, group_lrs={group_lrs}, weight_decay={WEIGHT_DECAY}, warmup={warmup_steps}/{total_steps}")
        print(f"低开销参数组监控: LoRA n/norm={lora_parameter_count}/{lora_initial_norm:.3f}; "
              f"projector={projector_parameter_count} (train={TRAIN_ADAPTER_PROJECTOR})")

    global_step, optimizer_update_step = 0, 0
    current_beta = float(BETA)
    beta_history, monitor_history = [], []

    resume_epoch, resume_next_batch_idx = 0, 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"]); scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"]); optimizer_update_step = int(resume_state["optimizer_update_step"])
        if int(scheduler.last_epoch) != optimizer_update_step:
            raise RuntimeError(
                "resume scheduler/optimizer update mismatch: "
                f"{scheduler.last_epoch} != {optimizer_update_step}")
        beta_history = list(resume_state.get("beta_history", [])); monitor_history = list(resume_state.get("monitor_history", []))
        resume_epoch = int(resume_state["epoch"]); resume_next_batch_idx = int(resume_state["next_batch_idx"])
        rank_runtime_states = resume_state.get("rank_runtime_states", [])
        if len(rank_runtime_states) != WORLD_SIZE:
            raise RuntimeError(
                "resume state缺少完整rank运行状态: "
                f"{len(rank_runtime_states)} != {WORLD_SIZE}")
        runtime_state = rank_runtime_states[GLOBAL_RANK]
        random.setstate(runtime_state["python_random_state"])
        torch.set_rng_state(runtime_state["torch_rng_state"])
        torch.cuda.set_rng_state(runtime_state["cuda_rng_state"], device=DEVICE)
        if IS_MAIN: print(f"已恢复optimizer/scheduler/β/RNG，继续 epoch={resume_epoch+1}, batch_idx={resume_next_batch_idx}")

    def save_restart_state(cp, ei, nbi, er):
        """只在累积边界保存，并收集所有rank的采样RNG以支持精确恢复。"""
        local_runtime_state = {
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(DEVICE).cpu(),
        }
        if is_ddp:
            gathered_runtime_states = [None] * WORLD_SIZE if IS_MAIN else None
            dist.gather_object(
                local_runtime_state, gathered_runtime_states, dst=0)
        else:
            gathered_runtime_states = [local_runtime_state]
        if IS_MAIN:
            state = {"format_version": 1, "saved_at": datetime.now().isoformat(timespec="seconds"), "world_size": WORLD_SIZE,
                "policy_adapter_path": cp, "reference_lora_path": REFERENCE_LORA_PATH,
                "policy_objective": POLICY_OBJECTIVE, "policy_token_denominator": POLICY_TOKEN_DENOMINATOR,
                "global_step": global_step, "optimizer_update_step": optimizer_update_step,
                "epoch": int(ei), "next_batch_idx": int(nbi),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "beta_history": beta_history, "monitor_history": monitor_history,
                "training_script_sha256": file_sha256(__file__),
                "train_json_sha256": file_sha256(TRAIN_JSON),
                "rank_runtime_states": gathered_runtime_states,
                "global_epoch_records": list(er)}
            torch.save(state, os.path.join(cp, "trainer_state.pt"))
        if is_ddp: dist.barrier()

    for epoch in range(resume_epoch, GRPO_EPOCHS):
        sampler.set_epoch(epoch)
        if IS_MAIN: print(f"\n{'='*60}\nGRPO 训练 Epoch {epoch+1}/{GRPO_EPOCHS}\n{'='*60}")
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{GRPO_EPOCHS}", disable=not IS_MAIN, unit="batch")
        accum_count, accum_has_update, accum_effective_samples, accum_global_effective_samples = 0, False, 0, 0
        local_sample_records, global_epoch_records = [], []
        monitor_local_records = []

        for batch_idx, batch in enumerate(pbar):
            if resume_state is not None and epoch == resume_epoch and batch_idx < resume_next_batch_idx:
                continue
            sample = batch[0]
            image, image_path, prompt, gt_response = sample["image"], sample["image_path"], sample["prompt"], sample["gt_response"]
            sample_task = classify_task(prompt); sample_max_tokens = max_tokens_for_bucket(sample_task, sample["cost_bucket"])

            # ── 生成候选 ──
            (candidate_texts, pv, generated_ids, completion_mask, gen_runtime) = generate_candidates(
                train_model, tokenizer, conv_module, image, prompt, G, sample_max_tokens)
            generation_timed_out = bool(gen_runtime["timed_out"])

            # ── 计算奖励 ──
            reward_results, mon = [], defaultdict(int)
            for ci in range(G):
                r = compute_reward(candidate_texts[ci], gt_response, image_path, prompt)
                r["length_penalty"] = soft_overlong_penalty(sample_task, int(completion_mask[ci].sum().item()), sample_max_tokens)
                r["total"] -= r["length_penalty"]
                r["advantage_micro_raw"] -= SOFT_OVERLONG_ADVANTAGE_SCALE * r["length_penalty"]
                reward_results.append(r)

            gen_diag = completion_generation_diagnostics(
                generated_ids, completion_mask, tokenizer, sample_task, sample_max_tokens,
                eos_terminated_override=gen_runtime["eos_terminated"])
            greedy_reward = float(reward_results[0]["total"]) if G > 0 else EMPTY_REWARD
            sampled_indices = list(range(GREEDY_CANDIDATES, G))
            active_sampled_indices = ([i for i in sampled_indices
                                       if gen_diag["eos_terminated"][i] and
                                       not gen_diag["token_limit_reached"][i]]
                                      if MASK_TRUNCATED_COMPLETIONS else list(sampled_indices))
            clean_sampled_indices = [
                i for i in active_sampled_indices
                if candidate_is_clean(reward_results[i], gen_diag["eos_terminated"][i],
                                      gen_diag["token_limit_reached"][i])]

            # ── 组内最优 ──
            rewards_tensor = torch.tensor([float(r["total"]) for r in reward_results], device=DEVICE)
            micro_raw_tensor = torch.tensor([float(r["advantage_micro_raw"]) for r in reward_results], device=DEVICE)
            localization_tensor = torch.tensor([float(r["localization"]) for r in reward_results], device=DEVICE)
            precomputed_advantages, adv_diag = centered_group_advantages(
                rewards_tensor, micro_raw_tensor, localization_tensor,
                active_sampled_indices, sample_task, enable_localization=True,
                return_diagnostics=True)
            group_has_signal = bool(len(active_sampled_indices) >= MIN_ACTIVE_SAMPLED_CANDIDATES and
                any(values[active_sampled_indices].std(unbiased=False).item() > 1.0e-9
                    for values in (rewards_tensor, micro_raw_tensor, localization_tensor)) and
                any(reward_results[idx]['valid_count'] > 0 for idx in active_sampled_indices) and
                precomputed_advantages[active_sampled_indices].abs().sum().item() >= MIN_TOTAL_ADV_ABS)
            pair = select_preference_pair(reward_results, active_sampled_indices,
                                          gen_diag, sample_task, generated_ids, completion_mask)
            insufficient_active = len(active_sampled_indices) < MIN_ACTIVE_SAMPLED_CANDIDATES
            z_pg = bool(not insufficient_active and group_has_signal and
                        (pair is None or precomputed_advantages[pair['chosen']].item() > 0))
            local_route_class = 2 if z_pg else 1 if pair is not None else 0
            preference_mode = pair['mode'] if pair else 'none'
            preference_support = pair['support'] if pair else 0.0
            global_gen_timeout = generation_timed_out
            if is_ddp:
                timeout_flag = torch.tensor(
                    [int(generation_timed_out)], device=DEVICE, dtype=torch.int32)
                dist.all_reduce(timeout_flag, op=dist.ReduceOp.MAX)
                global_gen_timeout = bool(timeout_flag.item())
            sample_has_update = False; sample_mode = "SKIP"
            total_loss = total_policy_loss = total_kl = total_token_nll = 0.0
            sample_kl_estimates = []
            preference_loss_value, preference_divergence = 0.0, -1
            if global_gen_timeout:
                sample_mode = "SKIP_generation_timeout"
            elif local_route_class == 0:
                sample_mode = "SKIP_overlong_insufficient" if insufficient_active else "SKIP_no_signal"
            else:
                objective_indices = list(active_sampled_indices)
                ref_indices = objective_indices + ([0] if pair else [])
                ref_lp = get_ref_log_probs(
                    train_model, tokenizer, conv_module, pv[ref_indices], prompt,
                    generated_ids[ref_indices], completion_mask=completion_mask[ref_indices])
                ref_rows = {index: row for row, index in enumerate(ref_indices)}
                pair_terms = []
                if pair:
                    chosen = pair['chosen']
                    with torch.no_grad():
                        probe = get_policy_log_probs(
                            train_model, tokenizer, conv_module, pv[[chosen, 0]], prompt,
                            generated_ids[[chosen, 0]], completion_mask=completion_mask[[chosen, 0]])
                    cf, rf, cs, rs, preference_divergence = pair['masks']
                    for cm, rm, weight in zip((cf, cs), (rf, rs), pair['weights']):
                        if not cm.any() or not rm.any():
                            continue
                        stats = reference_relative_pairwise_stats(
                            probe[0:1], probe[1:2], ref_lp[ref_rows[chosen]:ref_rows[chosen]+1],
                            ref_lp[ref_rows[0]:ref_rows[0]+1], cm, rm, weight)
                        if stats['pair_safe']:
                            pair_terms.append((cm, rm, stats['coefficient']))
                            preference_loss_value += stats['loss']
                    del probe
                denominator = float(completion_mask[objective_indices].sum().item())
                for start in range(0, len(objective_indices), MINI_BATCH):
                    mb = objective_indices[start:start+MINI_BATCH]
                    lp = get_policy_log_probs(train_model, tokenizer, conv_module,
                        pv[mb], prompt, generated_ids[mb], completion_mask=completion_mask[mb])
                    reference = ref_lp[[ref_rows[i] for i in mb]]
                    comp = completion_mask[mb]
                    candidate_kl = [float(k3_kl_mean(lp[j:j+1], reference[j:j+1], comp[j]).detach())
                                    for j in range(len(mb))]
                    sample_kl_estimates.extend(candidate_kl)
                    safe = [j for j, value in enumerate(candidate_kl)
                            if math.isfinite(value) and value <= KL_SKIP_THRESHOLD]
                    loss = lp.sum()*0
                    if safe:
                        indices = [mb[j] for j in safe]
                        advantage = precomputed_advantages[indices] if z_pg else torch.zeros_like(precomputed_advantages[indices])
                        loss, metrics = compute_grpo_loss(lp[safe], reference[safe], rewards_tensor[indices],
                            comp[safe], beta=current_beta, advantages=advantage, token_denominator=denominator)
                        total_loss += metrics['loss']; total_policy_loss += metrics['policy_loss']
                        total_kl += metrics['kl_div']; total_token_nll += metrics['token_nll']
                    if pair and pair['chosen'] in mb:
                        row = mb.index(pair['chosen'])
                        for cm, rm, coefficient in pair_terms:
                            loss = loss - coefficient*masked_sequence_mean(lp[row:row+1], cm)
                    if torch.isfinite(loss).item() and abs(float(loss.detach())) <= ABNORMAL_LOSS:
                        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                        sample_has_update |= bool(safe or pair_terms)
                if pair_terms:
                    lp = get_policy_log_probs(train_model, tokenizer, conv_module, pv[0:1], prompt,
                        generated_ids[0:1], completion_mask=completion_mask[0:1])
                    loss = sum(coefficient*masked_sequence_mean(lp, rm) for cm, rm, coefficient in pair_terms)
                    (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                    sample_has_update = True
                total_loss += preference_loss_value
                mon['GRPO'] = len(active_sampled_indices) if z_pg else 0
                sample_mode = 'GRPO+GPC' if z_pg and pair_terms else 'GRPO' if z_pg else 'GPC_only'

            best_pool = clean_sampled_indices or active_sampled_indices or sampled_indices
            metric_best_idx = metric_best_candidate_index(
                reward_results, best_pool, gen_diag["token_counts"])
            sample_kl_mean = (sum(sample_kl_estimates) / len(sample_kl_estimates)
                              if sample_kl_estimates else 0.0)
            sample_record = {
                "task": sample_task, "cost_bucket": sample["cost_bucket"],
                "mode": sample_mode, "loss": total_loss, "policy_loss": total_policy_loss,
                "kl": sample_kl_mean, "token_nll": total_token_nll,
                "has_update": int(sample_has_update), "greedy_reward": greedy_reward,
                "greedy_f1": float(reward_results[0]["f1"]),
                "greedy_precision": float(reward_results[0]["precision"]),
                "greedy_recall": float(reward_results[0]["recall"]),
                "greedy_tp": int(reward_results[0]["tp"]),
                "greedy_fp": int(reward_results[0]["fp"]),
                "greedy_fn": int(reward_results[0]["fn"]),
                "reward_best": float(reward_results[metric_best_idx]["total"]),
                "best_f1": float(reward_results[metric_best_idx]["f1"]),
                "metric_best_tp": int(reward_results[metric_best_idx]["tp"]),
                "metric_best_fp": int(reward_results[metric_best_idx]["fp"]),
                "metric_best_fn": int(reward_results[metric_best_idx]["fn"]),
                "best_source": candidate_source(sample_task, metric_best_idx),
                "any_hit_at_g": int(any(float(r["f1"]) > 0.0
                                        for r in reward_results)),
                "perfect_at_g": int(any(float(r["f1"]) >= 1.0 - 1.0e-12
                                         for r in reward_results)),
                "sampled_rescue": int(
                    metric_best_idx >= GREEDY_CANDIDATES and
                    float(reward_results[metric_best_idx]["f1"]) >
                    float(reward_results[0]["f1"]) + 1.0e-12),
                "unique_text_ratio": len(set(candidate_texts)) / max(G, 1),
                "unique_bbox_ratio": len(set(
                    r["bbox_signature"] for r in reward_results)) / max(G, 1),
                "active_sampled": len(active_sampled_indices),
                "clean_sampled": len(clean_sampled_indices),
                "group_has_signal": int(group_has_signal),
                "preference_mode": preference_mode, "preference_support": preference_support,
                "preference_loss": preference_loss_value,
                "preference_divergence": preference_divergence,
                "greedy_format": float(reward_results[0]["format"]),
                "greedy_invalid_json": int(reward_results[0]["invalid_json"]),
                "greedy_truncated": int(reward_results[0]["truncated"]),
                "greedy_label_mismatch": int(
                    reward_results[0].get("label_count_mismatch", False)),
                "greedy_repeated": int(reward_results[0].get("repeated_coordinate", False) or
                                       reward_results[0].get("repeated_tail", False)),
                "greedy_token_limit": int(gen_diag["token_limit_reached"][0]),
                "candidate_any_invalid_json": int(any(
                    r["invalid_json"] for r in reward_results)),
                "candidate_any_repeated": int(any(
                    r.get("repeated_coordinate", False) or
                    r.get("repeated_tail", False) for r in reward_results)),
                "candidate_any_token_limit": int(any(
                    gen_diag["token_limit_reached"])),
            }
            local_sample_records.append(sample_record)
            monitor_local_records.append(sample_record)
            global_epoch_records.append(sample_record)

            # ── 动态批量 ──
            round_update = torch.tensor([int(sample_has_update)], device=DEVICE, dtype=torch.int32)
            if is_ddp: dist.all_reduce(round_update, op=dist.ReduceOp.SUM)
            global_round_updates = int(round_update.item())
            accum_count += 1; accum_global_effective_samples += global_round_updates
            # 任一rank有真实目标时，所有rank在step前显式同步梯度，并一致执行
            # optimizer/scheduler step。
            accum_has_update = accum_has_update or (global_round_updates > 0)
            accum_effective_samples += int(sample_has_update)
            is_last_batch = batch_idx + 1 == len(dataloader)
            step_now, step_reason = dynamic_batch_should_step(accum_global_effective_samples, accum_count, is_last_batch)
            if step_now:
                grad_norm_value = 0.0
                if accum_has_update:
                    if is_ddp:
                        synchronize_trainable_gradients(trainable_params, WORLD_SIZE)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    if torch.isfinite(grad_norm).item() and float(grad_norm.item()) > MIN_GRAD_NORM_FOR_OPTIMIZER_STEP:
                        grad_norm_value = float(grad_norm.item())
                        optimizer.step(); scheduler.step(); optimizer_update_step += 1
                optimizer.zero_grad(set_to_none=True); global_step += 1; torch.cuda.empty_cache()
                # 控制台输出
                if IS_MAIN and (global_step % STEP_CONSOLE_INTERVAL == 0):
                    task_counts_in_accum = Counter()
                    for r in local_sample_records: task_counts_in_accum[r["task"]] += 1
                    tstr = "/".join(f"{t}:{c}" for t, c in sorted(task_counts_in_accum.items()))
                    mode_counts = Counter(r["mode"] for r in local_sample_records)
                    mode_str = "/".join(f"{k}:{v * WORLD_SIZE}" for k, v in sorted(mode_counts.items()))
                    avg_loss = sum(r["loss"] for r in local_sample_records) / max(len(local_sample_records), 1)
                    avg_policy = sum(r["policy_loss"] for r in local_sample_records) / max(len(local_sample_records), 1)
                    avg_kl = sum(r["kl"] for r in local_sample_records) / max(len(local_sample_records), 1)
                    seen_prompts = min((batch_idx + 1) * WORLD_SIZE, epoch_prompt_exposures)
                    print(f"[{datetime.now():%H:%M:%S}] slot={global_step:5d} opt={optimizer_update_step:4d} "
                          f"task=[{tstr}] mode=[{mode_str}] "
                          f"loss/policy={avg_loss:.4f}/{avg_policy:.4f} "
                          f"R={greedy_reward:.3f} GμF1={micro_f1_from_counts(int(reward_results[0]['tp']), int(reward_results[0]['fp']), int(reward_results[0]['fn'])):.3f} "
                          f"KL={avg_kl:.4f} grad={grad_norm_value:.3f} "
                          f"lr={optimizer.param_groups[0]['lr']:.2e} "
                          f"seen={seen_prompts}/{epoch_prompt_exposures}({seen_prompts/epoch_prompt_exposures*100:.1f}%) "
                          f"effective={accum_effective_samples}/{DYNAMIC_TARGET_EFFECTIVE_PROMPTS} rounds={accum_count}")
                if global_step % MONITOR_STEPS == 0:
                    gathered_records = [None for _ in range(WORLD_SIZE)] if is_ddp else None
                    if is_ddp: dist.all_gather_object(gathered_records, monitor_local_records)
                    all_monitor_records = ([item for rank_records in gathered_records for item in rank_records]
                                           if is_ddp else list(monitor_local_records))
                    if IS_MAIN:
                        drift_report = build_parameter_drift_report(
                            drift_run_start, named_trainable, optimizer_update_step,
                            drift_reference_scope)
                        monitor_entry = {"step": global_step, "optimizer_update_step": optimizer_update_step,
                                         "summary": summarize_monitor_records(all_monitor_records),
                                         "policy_drift_from_run_start": drift_report}
                        monitor_history.append(monitor_entry)
                        tmp_monitor = MONITOR_JSON_FILE + ".tmp"
                        with open(tmp_monitor, "w", encoding="utf-8") as handle:
                            json.dump({"run_name": RUN_NAME, "history": monitor_history}, handle,
                                      ensure_ascii=False, indent=2)
                        os.replace(tmp_monitor, MONITOR_JSON_FILE)
                        print_qwen_drift_comparison(drift_report)
                    monitor_local_records = []
                # checkpoint
                if global_step % SAVE_STEPS == 0:
                    if is_ddp: dist.barrier()
                    if IS_MAIN:
                        ckpt_path = os.path.join(GRPO_OUTPUT_DIR, f"checkpoint-{global_step}")
                        save_policy_checkpoint(policy_model, ckpt_path)
                        print(f"  → Checkpoint 已保存: {ckpt_path}")
                    if is_ddp: dist.barrier()
                    ckpt_path = os.path.join(
                        GRPO_OUTPUT_DIR, f"checkpoint-{global_step}")
                    save_restart_state(
                        ckpt_path, epoch, batch_idx + 1, global_epoch_records)
                local_sample_records, accum_count, accum_has_update, accum_effective_samples, accum_global_effective_samples = [], 0, False, 0, 0

    # Epoch 结束
    # 保存不足MONITOR_STEPS的尾窗口，避免一轮训练末端的格式/KL异常静默丢失。
    if monitor_local_records:
        gathered_records = [None for _ in range(WORLD_SIZE)] if is_ddp else None
        if is_ddp:
            dist.all_gather_object(gathered_records, monitor_local_records)
        all_monitor_records = (
            [item for rank_records in gathered_records for item in rank_records]
            if is_ddp else list(monitor_local_records))
        if IS_MAIN:
            drift_report = build_parameter_drift_report(
                drift_run_start, named_trainable, optimizer_update_step,
                drift_reference_scope)
            monitor_history.append({
                "step": global_step,
                "optimizer_update_step": optimizer_update_step,
                "summary": summarize_monitor_records(all_monitor_records),
                "policy_drift_from_run_start": drift_report,
                "residual_epoch_window": True,
            })
            tmp_monitor = MONITOR_JSON_FILE + ".tmp"
            with open(tmp_monitor, "w", encoding="utf-8") as handle:
                json.dump({"run_name": RUN_NAME, "history": monitor_history},
                          handle, ensure_ascii=False, indent=2)
            os.replace(tmp_monitor, MONITOR_JSON_FILE)
            print_qwen_drift_comparison(drift_report)
    if is_ddp: dist.barrier()
    if IS_MAIN:
        epoch_ckpt = os.path.join(GRPO_OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}-step-{global_step}")
        os.makedirs(epoch_ckpt, exist_ok=True)
        save_policy_checkpoint(policy_model, epoch_ckpt)
        print(f"  → Epoch checkpoint 已保存: {epoch_ckpt}")
    if is_ddp: dist.barrier()
    epoch_ckpt = os.path.join(
        GRPO_OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}-step-{global_step}")
    save_restart_state(epoch_ckpt, epoch + 1, 0, [])
    if IS_MAIN:
        save_policy_checkpoint(policy_model, GRPO_LORA_SAVE)
        print(f"  → 最终 LoRA 权重已保存: {GRPO_LORA_SAVE}")
        final_drift_report = build_parameter_drift_report(
            drift_run_start, named_trainable, optimizer_update_step,
            drift_reference_scope)
        final_lora_drift = final_drift_report["by_kind"]["lora"]["relative"]
        print(f"  → 最终LoRA相对漂移({drift_reference_scope})={final_lora_drift:.3%}")
        print_qwen_drift_comparison(final_drift_report)
    if is_ddp: dist.barrier()
    release_training_lock()
    if is_ddp: dist.destroy_process_group()

# ========================= 主入口 =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGPC InternVL-8B LoRA")
    parser.add_argument("--resume_state", type=str, default=None, help="从 checkpoint 的 trainer_state.pt 恢复")
    args = parser.parse_args()
    train(resume_state_path=args.resume_state)
