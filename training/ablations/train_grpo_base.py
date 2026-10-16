"""CGPC release: training/ablations/train_grpo_base.py. See the README in this directory for configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import os as _release_os

import json
import math
import os
import re
import sys
import random
import argparse
import hashlib
import time
from functools import lru_cache
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.distributed as dist
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ========================= 路径配置 =========================
BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/Qwen3-VL-2B-Instruct')
LORA_SFT_PATH = _release_os.environ.get('CGPC_SFT_PATH', './checkpoints/lora')
REFERENCE_LORA_PATH = LORA_SFT_PATH

NCCL_TIMEOUT_MINUTES = 60
# 单个prompt的整次候选生成共享同一墙钟预算。generate的max_time会在
# decoding step之间检查；超时样本随后由所有rank共同跳过。
SAMPLE_GENERATION_TIMEOUT_SECONDS = float(
    os.environ.get("GRPO_SAMPLE_TIMEOUT_SECONDS", "600"))
if SAMPLE_GENERATION_TIMEOUT_SECONDS <= 0:
    raise ValueError("GRPO_SAMPLE_TIMEOUT_SECONDS must be positive")

POLICY_LORA_PATH = LORA_SFT_PATH
# ============ 消融：基础 GRPO（无 A 无 B）============
# A(复合奖励设计)关闭：奖励=标准指标加权（去 micro 重归一化），advantage=中心化+std。
# B(长尾偏好巩固)关闭：无 greedy 偏好路径。
RUN_NAME = "grpo_base"

TRAIN_JSON = _release_os.environ.get('CGPC_TRAIN_JSON', './data/train.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRPO_OUTPUT_DIR = _release_os.environ.get('CGPC_OUTPUT_DIR', './outputs/train_grpo_base')
GRPO_LORA_SAVE  = os.path.join(
    GRPO_OUTPUT_DIR, "lora_grpo_base")

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRPO_LORA_SAVE, exist_ok=True)

# ========================= 分布式初始化 =========================

def init_distributed():
    """初始化分布式环境，返回 (rank, world_size, device, is_main)"""
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

# 各 rank 使用不同采样随机流，同时保持训练可复现。
random.seed(42 + LOCAL_RANK)
torch.manual_seed(42 + LOCAL_RANK)
torch.cuda.manual_seed_all(42 + LOCAL_RANK)

# ========================= 日志（仅主进程写文件） =========================

_RUN_STAMP = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
LOG_FILE = os.path.join(GRPO_OUTPUT_DIR,
    f"grpo_train_{_RUN_STAMP}"
    f"{'_rank' + str(LOCAL_RANK) if WORLD_SIZE > 1 else ''}.log")
MICRO_LOG_FILE = os.path.join(
    GRPO_OUTPUT_DIR, f"micro_samples_{_RUN_STAMP}.jsonl")
MONITOR_JSON_FILE = os.path.join(GRPO_OUTPUT_DIR, "training_monitor_latest.json")
TRAINING_LOCK_FILE = os.path.join(GRPO_OUTPUT_DIR, ".training.lock")
_MICRO_LOG_HANDLE = None

class Tee:
    """终端 + 文件双写 + 实时 flush（仅主进程写文件，避免多 rank 日志混乱）"""
    def __init__(self, filepath, write_file=True):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8") if write_file else None
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log:
            self.log.write(message)
            self.log.flush()
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()
    def isatty(self):
        return self.terminal.isatty() if hasattr(self.terminal, "isatty") else False

sys.stdout = Tee(LOG_FILE, write_file=IS_MAIN)


def assert_training_output_is_fresh(resume_state_path=None):
    """原子占用输出目录；显式resume可复用产物但仍拒绝覆盖活动运行。"""
    error_message = None
    lock_acquired = False
    if IS_MAIN:
        try:
            lock_fd = os.open(
                TRAINING_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            lock_acquired = True
            with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
                json.dump({
                    "pid": os.getpid(),
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "run_name": RUN_NAME,
                }, lock_file, ensure_ascii=False)

            if resume_state_path is None:
                checkpoint_entries = [
                    entry.name for entry in os.scandir(GRPO_OUTPUT_DIR)
                    if entry.name.startswith("checkpoint-")]
                adapter_entries = (
                    [entry.name for entry in os.scandir(GRPO_LORA_SAVE)]
                    if os.path.isdir(GRPO_LORA_SAVE) else [])
                if checkpoint_entries or adapter_entries:
                    error_message = (
                        "CGPC输出目录已有训练产物，拒绝覆盖。请先归档并更换"
                        f" GRPO_OUTPUT_DIR。checkpoints="
                        f"{checkpoint_entries[:5]}, "
                        f"adapter_files={adapter_entries[:5]}")
            elif not os.path.isfile(resume_state_path):
                error_message = (
                    f"resume state不存在: {resume_state_path}")
        except FileExistsError:
            error_message = (
                f"检测到训练锁 {TRAINING_LOCK_FILE}；可能已有训练在运行，或上次"
                "训练未正常结束。确认没有活动进程后再归档旧目录或移除锁。")
        except Exception as exc:
            error_message = f"检查/锁定训练输出目录失败: {exc}"

        if error_message is not None and lock_acquired:
            try:
                os.remove(TRAINING_LOCK_FILE)
            except OSError:
                pass

    failure = torch.tensor(
        [int(error_message is not None)], device=DEVICE, dtype=torch.int32)
    if WORLD_SIZE > 1:
        dist.broadcast(failure, src=0)
    if failure.item():
        if error_message is None:
            error_message = "rank0未能安全占用训练输出目录，所有rank同步停止。"
        raise RuntimeError(error_message)


def release_training_lock():
    """正常完成后释放运行锁；异常退出保留锁，防止误覆盖待诊断现场。"""
    if IS_MAIN:
        try:
            os.remove(TRAINING_LOCK_FILE)
        except FileNotFoundError:
            pass


def file_sha256(path: str):
    """返回本地文件 SHA256；服务器未同步该文件时返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 仅允许第66.4节锁定的CGPC在本次OOM修复后继续恢复。奖励、采样、优化器、
# scheduler和数据顺序均未改变；白名单不能用于其他历史版本或任意脚本漂移。
OOM_FIX_COMPATIBLE_RESUME_SHA256 = {
    "8E6E62942583CB6CD98F2BBCAAF9ED76AD5CA04907EE2A05DE221FA5F043D67E",
}


# ========================= GRPO 超参数 =========================

G = 8                # 每组候选输出数量（保持既有显存/探索约束）
# 每个 rollout 在参数更新前只 forward/backward 一次。v3 把同一次 policy forward
# detach 为 old_log_probs，因此 ratio 数值严格为 1，EPSILON/clip 从未起作用。
# v4 直接写成等价且可审计的一次 on-policy score-function objective。候选先在
# prompt 内中心化（不除组内std），随后按该prompt全部active completion token
# 求均值；这是DAPO式token reduction，不冒充Dr.GRPO的固定生成预算分母。
POLICY_OBJECTIVE = "on_policy_group_centered_token_mean"
POLICY_TOKEN_DENOMINATOR = "all_active_sampled_completion_tokens"
EPSILON = None
BETA = 0.005         # V7外评支持的弱固定SFT-reference KL；不自适应
KL_TASK_MIN_SAMPLES = 8
KL_CONTROLLER_SOURCES = {"grpo", "grpo+preference", "all_preference_only"}
GAMMA = 0.0          # entropy bonus 系数（关闭）
GRPO_LR = 2.0e-6     # V7在0.33%--0.41%漂移区间取得最佳数据效率
MAX_GRAD_NORM = 1.0  # 最大梯度范数
MIN_GRAD_NORM_FOR_OPTIMIZER_STEP = 1.0e-12
WEIGHT_DECAY = 0.0   # LoRA 短程优化不额外衰减
WARMUP_RATIO = 0.05
# 5% warmup后余弦收敛到峰值50%，主LR全程不低于1e-6。
MIN_LR_RATIO = 0.50
GRPO_EPOCHS = 1
SAVE_STEPS = 25         # 密集覆盖旧V7有效的0.3%--0.8%漂移区间
MONITOR_STEPS = 25
# 参考V7.1：每个optimizer slot恰好打印一行简洁摘要。
STEP_CONSOLE_INTERVAL = 1
PRINT_HARD_MICRO_SAMPLES = False
# 参数组梯度范数会额外遍历LoRA张量；每10个累积窗口抽样一次即可判断量级，
# 避免为了监控在每次optimizer更新上增加大量小CUDA kernel。
GRAD_GROUP_MONITOR_INTERVAL = 10
DRIFT_LOW_WARNING_AFTER_UPDATES = 100
DRIFT_LOW_WARNING_RELATIVE = 1.0e-3
DRIFT_HIGH_WARNING_RELATIVE = 5.0e-2
LORA_DRIFT_TARGET_MIN_RELATIVE = 3.0e-3
LORA_DRIFT_TARGET_MAX_RELATIVE = 1.5e-2
# v3真实optimizer窗口更新率约91%（不要与约32%的微样本has_update率混淆）。
# 若pilot显著低于该水平，现有schedule跨度不再适用，应停止并重新预注册fresh run。
OPTIMIZER_UPDATE_RATE_WARNING = 0.80
OPTIMIZER_UPDATE_RATE_CRITICAL = 0.70

# DAPO指出零梯度prompt会稀释batch梯度；但v3路径回放表明按有效样本倒数调LR会
# 在几乎所有更新窗口顶到cap，实质变成固定升LR并放大困难样本噪声。安全默认关闭，
# 保留实现供预注册A/B使用；若开启，cap也降到1.25并完整记录触顶率。
EFFECTIVE_SAMPLE_NORMALIZATION = False
EFFECTIVE_SAMPLE_SCALE_MAX = 1.25

IOU_THRESHOLD = 0.5      # 训练监控TP/FP/FN的IoU阈值
REWARD_IOU_GATE = 0.5    # 与训练监控Recall/F1口径一致，防止奖励虚高

# 失败暴露：每组固定包含1个greedy诊断候选，观察提前EOS/选错目标。
GREEDY_CANDIDATES = 1
# greedy只做真实推理诊断与“是否存在更优sampled候选”的锚点，不进入on-policy
# advantage基线或policy-gradient。其余7个sampled候选构成实际GRPO组。
EXCLUDE_GREEDY_FROM_GRPO = True
ADVANTAGE_NORMALIZATION = "hard_tp_base_plus_soft_iou_quality_and_localization"
# hard micro_raw仍只用于外部指标；训练优势使用匹配IoU连续边际量，并始终叠加
# 小权重定位信号。复合reward仅在两者都无方差时作为最后平局信号。
ADVANTAGE_FIXED_SCALE = 0.10
MICRO_RAW_ADVANTAGE_SCALE = 0.50
MICRO_RAW_ADVANTAGE_CLIP = 4.0
# 保留严格TP的基础信用，同时用25%的连续IoU质量项区分“刚过门”与紧致框。
# 若完全改用soft-TP，单框要到IoU>0.725才胜过空输出，会诱发欠生成。
SOFT_TP_QUALITY_BLEND = 0.25
LOCALIZATION_ADVANTAGE_SCALE = {"ALL": 0.08, "ROW": 0.10, "SINGLE": 0.15}
# V7.2/V7.3把最大增益给SINGLE，结果SINGLE改善而ROW/ALL退化。CGPC只温和增强ROW；
# ALL由micro_raw和GT规模权重自然主导Overall，SINGLE保持标准GRPO且无额外措施。
TASK_ADVANTAGE_GAIN = {"ALL": 1.00, "ROW": 1.15, "SINGLE": 1.00}
LOSS_REDUCTION = "token_mean_over_active_sampled_responses"
STRUCTURED_DAPO_LENGTH_CAP = False
STRUCTURED_DAPO_GT_RATIO = 1.25
STRUCTURED_DAPO_EXTRA_TOKENS = 8
MASK_TRUNCATED_COMPLETIONS = True
MIN_ACTIVE_SAMPLED_CANDIDATES = 2
MIN_TOTAL_ADV_ABS = 1.0e-4
EMPTY_REWARD = -1.0
# V7.2外评显示定向no-hit CE的SINGLE收益由ROW退化抵消；CGPC完全关闭该杠杆。
TARGETED_NO_HIT_CE_ENABLED = False
TARGETED_NO_HIT_CE_TASKS = set()
TARGETED_NO_HIT_CE_WEIGHT = {"ALL": 0.0, "ROW": 0.0, "SINGLE": 0.0}
MICRO_CONSOLE_REWARD_THRESHOLD = 0.70

# ALL sampled→greedy 对齐。V7.3实跑至step-214只触发1次，唯一要求greedy触顶
# 使方向正确但覆盖不足。CGPC保留严格overlong门，并新增“高质量且显著更优”门：
# 正常EOS、合法框、F1>=0.90、相对greedy至少+0.02且reward至少+0.03；普通门还
# 要求至少2个sampled在bbox或TP/FP/FN上支持chosen，降低best-of-7赢家偏差。
ALL_OVERLONG_PREFERENCE_ENABLED = False
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

# DAPO 式 soft-overlong punishment 的保守版本。真正达到上限且无 EOS 的 sampled
# 候选仍按 overlong filtering 屏蔽；该项只给靠近上限、尚可学习的 rollout 预警。
SOFT_OVERLONG_CACHE_TOKENS = {"SINGLE": 64, "ROW": 256, "ALL": 1024}
SOFT_OVERLONG_CACHE_RATIO = 0.15
SOFT_OVERLONG_MAX_PENALTY = {"SINGLE": 0.05, "ROW": 0.10, "ALL": 0.20}
SOFT_OVERLONG_ADVANTAGE_SCALE = 1.0

# DAPO式有效组批量：零优势prompt不再占满一个累积槽；继续消费后续prompt，
# 直到双卡累计约16个真实更新组。约50%组有效时16轮只是期望值，改为24轮给
# 随机波动留余量；仍保留有界cap，避免困难桶无限拖长wall-clock。
DYNAMIC_EFFECTIVE_BATCHING = True
DYNAMIC_TARGET_EFFECTIVE_PROMPTS = None  # 在累积步数定义后解析
DYNAMIC_MAX_PROMPT_ROUNDS = 24

# v3的2/1/1把prompt曝光改成约58.1/19.1/22.8%，与ROW/SINGLE明显下滑
# 同向；v4的ALL复合优势已有最高2倍GT权重，micro_raw本身也保留框级边际量，
# 不再叠加第二层重复偏置。首跑保持原始约40.9/27.0/32.1%任务分布。
TASK_REPEAT = {"ALL": 1, "ROW": 1, "SINGLE": 1}

# 欠生成仍需修复，但旧运行已证实过生成增长更快。FP惩罚允许随FP/GT增长至2倍，
# 不再在FP==GT时过早封顶。
UNDERGEN_PENALTY = {"ALL": 0.04, "ROW": 0.0, "SINGLE": 0.0}
OVERGEN_PENALTY = {"ALL": 0.12, "ROW": 0.25, "SINGLE": 0.15}
OVERGEN_RATIO_CAP = {"ALL": 2.0, "ROW": 2.0, "SINGLE": 2.0}

# Overall micro-F1 = 2*sum(TP)/sum(GT+Pred)。在同一个prompt的候选组内GT恒定，
# 因此 2*TP - target*Pred 是达到目标F1的直接边际效用。归一化后完美候选为1，
# 多生成但未带来足够TP的候选会得到负效用。
MICRO_F1_TARGET = 0.9000
MICRO_UTILITY_FLOOR = -2.0

# ---- 训练稳定性阈值 ----
KL_SKIP_THRESHOLD = 0.20  # v6 ABC口径；只用于NaN/极端候选的数值异常保护
ABNORMAL_LOSS = 100.0    # |loss| 超过此值视为异常 → 清梯度并跳过

# micro项优化整体F1的边际效用；IoU只对通过0.5门槛的命中提供连续信号。
# 当前CSV框级指标不使用labels，因此标签权重归零并转给micro/precision/F1。
TASK_REWARD_WEIGHTS = {
    "ALL":    {"fmt": 0.02, "iou": 0.10, "micro": 0.45,
               "rec": 0.02, "prec": 0.18, "f1": 0.21,
               "cnt": 0.02, "label": 0.00, "center": 0.0},
    "ROW":    {"fmt": 0.02, "iou": 0.08, "micro": 0.43,
               "rec": 0.00, "prec": 0.22, "f1": 0.20,
               "cnt": 0.02, "label": 0.00, "center": 0.03},
    # gt=1且pred=1时rec/prec/f1完全同值；只保留F1作为离散命中信号，
    # 将释放的权重转给soft-IoU和center，增强相邻错包之间的连续区分度。
    "SINGLE": {"fmt": 0.02, "iou": 0.20, "micro": 0.18,
               "rec": 0.0, "prec": 0.0, "f1": 0.30,
               "cnt": 0.0, "label": 0.00, "center": 0.30},
}

# 消融：基础 GRPO 不含 micro（目标 F1 边际效用属于模块 A）。
# 移除 micro 后按剩余标准指标权重重新归一化到 1.0，保持与 cgpc 除 micro 外的比例一致。
for _task_name, _weights in TASK_REWARD_WEIGHTS.items():
    _weights.pop("micro", None)
    _weight_sum = sum(_weights.values())
    for _key in list(_weights.keys()):
        _weights[_key] = _weights[_key] / _weight_sum

for _task_name, _weights in TASK_REWARD_WEIGHTS.items():
    _weight_sum = sum(_weights.values())
    if not math.isclose(_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{_task_name} reward weights must sum to 1.0, got {_weight_sum}")

# 训练参数
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
DYNAMIC_TARGET_EFFECTIVE_PROMPTS = (
    GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE)
ALL_MAX_TOKENS = int(os.environ.get("CGPC_ALL_MAX_TOKENS", "10240"))
if ALL_MAX_TOKENS not in {6144, 8192, 10240}:
    raise ValueError(
        "CGPC_ALL_MAX_TOKENS 仅允许 6144、8192 或 10240")
MAX_NEW_TOKENS = ALL_MAX_TOKENS
# 7个sampled必须直接来自当前policy。温度!=1或top-p/top-k截断会把rollout
# behavior policy变成另一个分布，而后续loss使用的却是原始policy log-prob；
# 在没有显式importance correction时会产生off-policy偏差。保留配置字典便于日志
# 和硬约束检查，但所有任务均使用raw-policy multinomial sampling。
EXPLOIT_SAMPLED_CANDIDATES = {"ALL": 7, "ROW": 7, "SINGLE": 7}
SAMPLE_TEMPERATURE_LOW = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TEMPERATURE_HIGH = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TOP_P = {"ALL": 1.0, "ROW": 1.0, "SINGLE": 1.0}
SAMPLE_TOP_K = 0
# 清空模型GenerationConfig中可能残留的beam、forced-EOS、抑制token或
# watermark/logits后处理。仅传当前transformers版本实际支持的字段。
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
LOG_RATIO_CLAMP = 10.0   # 防止 k3-KL 中 exp(log(ref/policy)) 数值溢出

# 用目标框尺度归一化距离，避免相邻错误棉包在整图对角线归一化下仍得到 0.9+ 高分。
CENTER_SCALE_BOX_DIAGONALS = {"ALL": 1.5, "ROW": 2.0, "SINGLE": 1.0}
CENTER_SCALE_IMAGE_FLOOR = 0.02

# Prompt级GRPO默认等权，会低估大框数ALL样本对总体micro-F1的贡献。
# sqrt+截断保留GT规模信息，同时避免100+框样本支配单步梯度。
GT_WEIGHT_REFERENCE = 4.0
GT_WEIGHT_MIN = 1.00
GT_WEIGHT_MAX = 2.00

# RL rollout、policy log-prob和reference log-prob必须使用同一个确定性网络。
# 训练forward保持eval模式并不关闭梯度，只关闭dropout等随机层。
RL_DISABLE_DROPOUT = True
# SFT配置把visual.merger作为additional_target保存进adapter；v2却只解冻名字含
# lora的参数。v3若检测到PEFT modules_to_save包装的merger，则以更小LR联合更新。
TRAIN_ADAPTER_PROJECTOR = True
PROJECTOR_LR = 5.0e-7

# 动态 max_new_tokens
TASK_MAX_TOKENS = {
    "SINGLE": 256,
    "ROW":    1024,
    "ALL":    ALL_MAX_TOKENS,
}
# 训练按GT框数成本桶分配预算；ALL最大桶仍是用户指定的10240。较小桶不再为
# 极少数ALL:140的最坏情况统一支付KV-cache和生成时间。
TRAIN_BUCKET_MAX_TOKENS = {
    "SINGLE:2": 256,
    "ROW:3": 384,
    "ROW:7": 640,
    "ROW:12": 1024,
    "ROW:20": 1024,
    "ROW:999": 1024,
    "ALL:10": min(2048, ALL_MAX_TOKENS),
    "ALL:25": min(3072, ALL_MAX_TOKENS),
    "ALL:50": min(5120, ALL_MAX_TOKENS),
    "ALL:80": min(7168, ALL_MAX_TOKENS),
    "ALL:140": ALL_MAX_TOKENS,
    "ALL:999": ALL_MAX_TOKENS,
}
ALL140_MONITOR_BUCKET = "ALL:140"


def max_tokens_for_bucket(task: str, cost_bucket: str = None) -> int:
    if cost_bucket is not None and cost_bucket in TRAIN_BUCKET_MAX_TOKENS:
        return int(TRAIN_BUCKET_MAX_TOKENS[cost_bucket])
    return int(TASK_MAX_TOKENS[task])

# ========================= 辅助函数 =========================

def iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def global_iou_matches(gt_boxes: List, pred_boxes: List):
    """全局 IoU 降序的一对一匹配，消除逐 GT 贪心的顺序依赖。"""
    pairs = sorted(
        ((iou(g, p), gi, pi)
         for gi, g in enumerate(gt_boxes)
         for pi, p in enumerate(pred_boxes)),
        reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for score, gi, pi in pairs:
        if score <= 0 or gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi); used_pred.add(pi)
        matches.append((score, gi, pi))
    return matches


def evaluation_order_matches(gt_boxes: List, pred_boxes: List,
                             threshold: float = IOU_THRESHOLD):
    """按GT顺序选未占用预测框，且仅IoU>阈值才占用。

    训练reward与窗口TP/FP/FN使用同一规则，避免内部优化与监控口径错位。
    """
    used_pred, matches = set(), []
    for gt_i, gt_box in enumerate(gt_boxes):
        best_iou, best_pred_i = 0.0, -1
        for pred_i, pred_box in enumerate(pred_boxes):
            if pred_i in used_pred:
                continue
            score = iou(gt_box, pred_box)
            if score > best_iou:
                best_iou, best_pred_i = score, pred_i
        if best_iou > threshold:
            used_pred.add(best_pred_i)
            matches.append((best_iou, gt_i, best_pred_i))
    return matches


def gt_scale_weight(gt_count: float) -> float:
    """有界GT规模权重；用于loss而非reward，避免组内标准化抹掉框级重要性。"""
    count = max(float(gt_count), 1.0)
    weight = math.sqrt(count / GT_WEIGHT_REFERENCE)
    return min(max(weight, GT_WEIGHT_MIN), GT_WEIGHT_MAX)


def spatial_sort_boxes_and_labels(boxes: List, labels: List):
    """复现外部评估按(y1,x1)排序，同时保持box/label索引一致。"""
    labels = list(labels or [])
    indexed = list(range(len(boxes)))
    indexed.sort(key=lambda idx: (boxes[idx][1], boxes[idx][0]))
    sorted_boxes = [boxes[idx] for idx in indexed]
    sorted_labels = [labels[idx] if idx < len(labels) else "" for idx in indexed]
    return sorted_boxes, sorted_labels


def center_similarity(gt_boxes: List, pred_boxes: List, image_path: str,
                      task: str = "SINGLE") -> float:
    """按目标框尺度计算中心相似度，使相邻错误目标也能被明显区分。"""
    if not gt_boxes or not pred_boxes:
        return 0.0
    width, height = get_image_size(image_path)
    image_diagonal = max(math.hypot(width, height), 1.0)
    scale_multiplier = CENTER_SCALE_BOX_DIAGONALS.get(task, 1.5)
    pairs = []
    for gt_i, gt_box in enumerate(gt_boxes):
        gx = (gt_box[0] + gt_box[2]) / 2
        gy = (gt_box[1] + gt_box[3]) / 2
        gt_diagonal = math.hypot(
            max(gt_box[2] - gt_box[0], 1.0),
            max(gt_box[3] - gt_box[1], 1.0))
        scale = max(scale_multiplier * gt_diagonal,
                    CENTER_SCALE_IMAGE_FLOOR * image_diagonal, 1.0)
        for pred_i, pred_box in enumerate(pred_boxes):
            px = (pred_box[0] + pred_box[2]) / 2
            py = (pred_box[1] + pred_box[3]) / 2
            normalized_distance = math.hypot(gx - px, gy - py) / scale
            similarity = math.exp(-(normalized_distance ** 2))
            pairs.append((similarity, gt_i, pred_i))
    used_gt, used_pred, total = set(), set(), 0.0
    for similarity, gt_i, pred_i in sorted(pairs, reverse=True):
        if gt_i in used_gt or pred_i in used_pred:
            continue
        used_gt.add(gt_i); used_pred.add(pred_i); total += similarity
    return total / len(gt_boxes)


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


@lru_cache(maxsize=4096)
def get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def sanitize_and_scale_boxes(boxes, image_path):
    return protocol.scale_normalized_boxes(boxes, get_image_size(image_path))


def scale_boxes_to_pixel(boxes: List, image_path: str) -> List:
    return sanitize_and_scale_boxes(boxes, image_path)[0]


def quantized_bbox_signature(boxes_px: List, image_path: str,
                             invalid_count: int = 0) -> Tuple:
    """Hashable bbox-set signature on a 1%-of-image grid.

    Text diversity can overstate exploration when candidates differ only in
    formatting or one-pixel coordinate jitter.  Quantization keeps this
    diagnostic structural and reuses the cached image size.
    """
    width, height = get_image_size(image_path)
    width = max(width, 1)
    height = max(height, 1)
    quantized = tuple(
        (round(100 * box[0] / width), round(100 * box[1] / height),
         round(100 * box[2] / width), round(100 * box[3] / height))
        for box in boxes_px)
    return int(invalid_count), quantized


def normalize_label(value):
    return str(value).strip().lower()


# ========================= 任务分类 =========================

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
    """Approximate generation/backward cost from task and GT box count."""
    gt_count = len(parse_bbox_json(gt_response)[0])
    if task == "ALL":
        upper = next((v for v in (10, 25, 50, 80, 140)
                      if gt_count <= v), 999)
    elif task == "ROW":
        upper = next((v for v in (3, 7, 12, 20)
                      if gt_count <= v), 999)
    else:
        upper = 2
    return f"{task}:{upper}"


class CottonBaleDataset(Dataset):
    def __init__(self, json_path: str, image_root: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.sample_indices = []
        self.task_counts = Counter()
        self.sampled_task_counts = Counter()
        self.source_cost_buckets = {}
        for idx, sample in enumerate(self.data):
            task = classify_task(sample["conversations"][0]["value"])
            self.source_cost_buckets[idx] = sample_cost_bucket(
                task, sample["conversations"][1]["value"])
            self.task_counts[task] += 1
            self.sample_indices.extend([idx] * TASK_REPEAT.get(task, 1))
        # 补齐到 DDP×梯度累积的整数倍，避免每个 epoch 尾部残留梯度被静默丢弃。
        step_multiple = max(WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS, 1)
        pad_count = (-len(self.sample_indices)) % step_multiple
        if pad_count:
            self.sample_indices.extend(self.sample_indices[:pad_count])
        # 记录成本sampler补齐后的实际任务曝光；CGPC不使用困难回放。
        for source_idx in self.sample_indices:
            task = classify_task(
                self.data[source_idx]["conversations"][0]["value"])
            self.sampled_task_counts[task] += 1
        # 不在各 DDP 进程内独立 shuffle；统一交给成本均衡 sampler。
        # 否则不同 rank 的底层样本顺序不一致，会造成重复和遗漏。
        self.image_root = image_root

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.get_source_item(self.sample_indices[idx])

    def get_source_item(self, source_idx):
        """按原始索引取样；在线困难回放使用此入口重新读取原图。"""
        sample = self.data[source_idx]
        img_rel = sample["images"][0]
        image_path = os.path.join(self.image_root, img_rel)
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        prompt      = sample["conversations"][0]["value"]
        gt_response = sample["conversations"][1]["value"]
        return {"image": image, "image_path": image_path,
                "image_rel": img_rel, "prompt": prompt,
                "gt_response": gt_response, "idx": source_idx,
                "cost_bucket": self.source_cost_buckets[source_idx]}


class CostBalancedDistributedSampler(Sampler):
    """Align task/GT-count cost across ranks at every micro-step."""

    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=42):
        self.dataset = dataset
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.buckets = {}
        for dataset_idx, source_idx in enumerate(dataset.sample_indices):
            key = dataset.source_cost_buckets[source_idx]
            self.buckets.setdefault(key, []).append(dataset_idx)

        group_count = sum(
            math.ceil(len(indices) / self.num_replicas)
            for indices in self.buckets.values())
        self.num_samples = math.ceil(
            group_count / GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        groups = []
        for key in sorted(self.buckets):
            indices = list(self.buckets[key])
            if self.shuffle:
                rng.shuffle(indices)
            remainder = len(indices) % self.num_replicas
            if remainder:
                indices.extend(indices[:self.num_replicas - remainder])
            groups.extend(
                indices[start:start + self.num_replicas]
                for start in range(0, len(indices), self.num_replicas))
        if self.shuffle:
            rng.shuffle(groups)
        if len(groups) < self.num_samples:
            groups.extend(groups[:self.num_samples - len(groups)])
        if len(groups) != self.num_samples:
            raise RuntimeError(
                f"Sampler group mismatch: {len(groups)} != {self.num_samples}")
        return iter(group[self.rank] for group in groups)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def collate_fn(batch):
    return batch


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


def candidate_source(task: str, idx: int) -> str:
    if idx < GREEDY_CANDIDATES:
        return "greedy"
    return "raw_policy_sample"


def completion_generation_diagnostics(generated_ids, completion_mask,
                                      processor, task: str,
                                      token_limit: int = None
                                      ) -> Dict[str, List]:
    """返回每个候选的有效token数、EOS终止和触顶状态。"""
    token_limit = int(
        token_limit if token_limit is not None else TASK_MAX_TOKENS[task])
    token_counts = [
        int(v) for v in completion_mask.sum(dim=1).detach().cpu().tolist()]
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is None:
        eos_terminated = [False] * len(token_counts)
    else:
        # completion_mask包含首个EOS并屏蔽其后的padding，即使pad_id==eos_id
        # 也能直接识别真实终止；同时正确处理EOS恰落在token上限末位的情况。
        eos_hits = (
            generated_ids.eq(eos_id) & completion_mask.bool()
        ).any(dim=1).detach().cpu().tolist()
        eos_terminated = [bool(v) for v in eos_hits]
    token_limit_reached = [
        count >= token_limit and not has_eos
        for count, has_eos in zip(token_counts, eos_terminated)]
    return {
        "token_counts": token_counts,
        "eos_terminated": eos_terminated,
        "token_limit_reached": token_limit_reached,
    }


def soft_overlong_penalty(task: str, token_count: int,
                          token_limit: int = None) -> float:
    """Conservative DAPO-style linear penalty inside the final token cache."""
    token_limit = int(
        token_limit if token_limit is not None else TASK_MAX_TOKENS[task])
    cache = min(
        int(SOFT_OVERLONG_CACHE_TOKENS[task]),
        max(int(token_limit * SOFT_OVERLONG_CACHE_RATIO), 1),
        token_limit)
    safe_limit = token_limit - cache
    if cache <= 0 or token_count <= safe_limit:
        return 0.0
    fraction = min(max((token_count - safe_limit) / cache, 0.0), 1.0)
    return float(SOFT_OVERLONG_MAX_PENALTY[task]) * fraction


def dynamic_batch_should_step(effective_prompts: int,
                              prompt_rounds: int,
                              is_last_batch: bool = False
                              ) -> Tuple[bool, str]:
    """Bounded dynamic-sampling boundary shared identically by every rank."""
    if not DYNAMIC_EFFECTIVE_BATCHING:
        ready = prompt_rounds >= GRADIENT_ACCUMULATION_STEPS
        return ready, "fixed" if ready else "collecting"
    if effective_prompts >= DYNAMIC_TARGET_EFFECTIVE_PROMPTS:
        return True, "target"
    if prompt_rounds >= DYNAMIC_MAX_PROMPT_ROUNDS:
        return True, "cap"
    if is_last_batch:
        return True, "epoch_tail"
    return False, "collecting"


def micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
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


def centered_group_advantages(rewards: torch.Tensor,
                              micro_raw_scores: torch.Tensor,
                              localization_scores: torch.Tensor,
                              active_indices: List[int],
                              task: str,
                              enable_localization: bool = False,
                              return_diagnostics: bool = False):
    """标准 GRPO：组内中心化 + std 归一化（消融基线）。

    只使用复合奖励 ``rewards``（各指标加权求和）。不引入 micro 边际效用、
    质量混合或定位等模块 A 的独立优势通道；micro_raw_scores 与
    localization_scores 仅为保持调用签名一致而保留，实际不使用。
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
            "task_advantage_gain": 1.0,
        }
        return (advantages, empty_diag) if return_diagnostics else advantages
    active_rewards = rewards[active_indices]
    centered = active_rewards - active_rewards.mean()
    std = centered.std(unbiased=False)
    active_advantages = centered / std.clamp_min(1.0e-8)
    advantages[active_indices] = active_advantages
    if not return_diagnostics:
        return advantages
    diagnostics = {
        "normalized_adv_abs_mean": float(
            active_advantages.detach().abs().mean().item()),
        "micro_raw_adv_abs_mean": 0.0,
        "localization_adv_abs_mean": 0.0,
        "total_adv_abs_mean": float(
            active_advantages.detach().abs().mean().item()),
        "total_adv_abs_sum": float(
            active_advantages.detach().abs().sum().item()),
        "total_adv_max_abs": float(
            active_advantages.detach().abs().max().item()),
        "micro_raw_clip_scale": 1.0,
        "micro_raw_clip_triggered": False,
        "task_advantage_gain": 1.0,
    }
    return advantages, diagnostics


def dapo_candidate_weights(completion_mask: torch.Tensor,
                           active_indices: List[int],
                           token_cap: int = None) -> torch.Tensor:
    """Token-level weights over active responses, following DAPO Eq. 12.

    ``compute_grpo_loss`` first returns a per-response token mean. Weighting that
    mean by response length / total active tokens is algebraically the same as
    one mean over every active completion token. This prevents long ALL JSON
    responses from having every box diluted by 1/sequence_length.
    """
    weights = torch.zeros(
        completion_mask.shape[0], device=completion_mask.device,
        dtype=torch.float32)
    if not active_indices:
        return weights
    token_counts = completion_mask[active_indices].to(
        torch.float32).sum(dim=1)
    if token_cap is not None:
        token_counts = token_counts.clamp_max(float(token_cap))
    token_mass = token_counts.sum().clamp_min(1.0)
    weights[active_indices] = token_counts / token_mass
    return weights


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


@torch.no_grad()
def candidate_policy_log_probs(model, inputs: Dict,
                               generated_ids: torch.Tensor,
                               candidate_idx: int) -> torch.Tensor:
    """No-grad current-policy log-probs for one preference coefficient pass."""
    idx = [int(candidate_idx)]
    candidate_inputs = {}
    for key in ("input_ids", "attention_mask"):
        candidate_inputs[key] = (
            inputs[key][idx] if inputs.get(key) is not None else None)
    pv, gw = _slice_vision_inputs(
        inputs.get("pixel_values"), inputs.get("image_grid_thw"), idx)
    candidate_inputs["pixel_values"] = pv
    candidate_inputs["image_grid_thw"] = gw
    return get_policy_log_probs(
        model, candidate_inputs, generated_ids[idx]).detach()


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


def parameter_group_grad_norm(parameters: List[torch.nn.Parameter],
                              device: torch.device) -> torch.Tensor:
    """Low-overhead L2 norm for an optimizer parameter group."""
    per_parameter = [
        torch.linalg.vector_norm(p.grad.detach()).float()
        for p in parameters if p.grad is not None]
    if not per_parameter:
        return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(per_parameter))


@torch.no_grad()
def parameter_group_value_norm(parameters: List[torch.nn.Parameter],
                               device: torch.device) -> torch.Tensor:
    """L2 norm of a parameter group; used once as relative-step denominator."""
    per_parameter = [
        torch.linalg.vector_norm(p.detach()).float()
        for p in parameters]
    if not per_parameter:
        return torch.zeros((), device=device)
    return torch.linalg.vector_norm(torch.stack(per_parameter))


@torch.no_grad()
def snapshot_trainable_parameters(
        named_parameters: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    """CPU fp32 snapshot for exact, low-frequency parameter-drift monitoring."""
    return {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, parameter in named_parameters
    }


def compare_parameter_snapshots(
        reference: Dict[str, torch.Tensor],
        current: Dict[str, torch.Tensor]) -> Dict:
    """Return exact ||dW||/||W_ref|| overall and split by adapter kind."""
    totals = [0.0, 0.0, 0.0]
    by_kind = {}
    for name, current_value in current.items():
        reference_value = reference.get(name)
        if reference_value is None:
            continue
        delta_sq = float(torch.sum(
            (current_value - reference_value).square()).item())
        reference_sq = float(torch.sum(reference_value.square()).item())
        current_sq = float(torch.sum(current_value.square()).item())
        totals[0] += delta_sq
        totals[1] += reference_sq
        totals[2] += current_sq
        kind = adapter_parameter_kind(name)
        kind_totals = by_kind.setdefault(kind, [0.0, 0.0, 0.0])
        kind_totals[0] += delta_sq
        kind_totals[1] += reference_sq
        kind_totals[2] += current_sq

    def finalize(values):
        delta_norm = math.sqrt(values[0])
        reference_norm = math.sqrt(values[1])
        return {
            "relative": delta_norm / max(reference_norm, 1.0e-12),
            "delta_norm": delta_norm,
            "reference_norm": reference_norm,
            "current_norm": math.sqrt(values[2]),
        }

    result = finalize(totals)
    result["by_kind"] = {
        kind: finalize(values) for kind, values in by_kind.items()}
    return result


def higher_quantile_index(sample_count: int, quantile: float) -> int:
    """Nearest-rank/higher quantile index; n=8,p95 intentionally selects max."""
    if sample_count <= 0:
        return 0
    return min(
        max(math.ceil(float(quantile) * sample_count) - 1, 0),
        sample_count - 1)


def summarize_monitor_records(records: List[Dict]) -> Dict:
    """按任务和总体汇总；用精确TP/FP/FN重建目标micro-F1。"""
    summary = {}
    # 这里反映TASK_REPEAT与同任务回放后的训练曝光分布。
    groups = {"TRAIN_WEIGHTED_OVERALL": records}
    groups.update({
        task: [r for r in records if r["task"] == task]
        for task in TASK_REPEAT
    })
    for task, task_records in groups.items():
        if not task_records:
            continue
        count = len(task_records)

        def mean(key):
            return sum(float(r[key]) for r in task_records) / count

        def sum_int(key):
            return sum(int(r[key]) for r in task_records)

        greedy_tp = sum_int("greedy_tp")
        greedy_fp = sum_int("greedy_fp")
        greedy_fn = sum_int("greedy_fn")
        best_tp = sum_int("best_tp")
        best_fp = sum_int("best_fp")
        best_fn = sum_int("best_fn")
        metric_best_tp = sum_int("metric_best_tp")
        metric_best_fp = sum_int("metric_best_fp")
        metric_best_fn = sum_int("metric_best_fn")
        update_kl_values = sorted(
            float(r["kl"]) for r in task_records
            if r.get("kl_measured", False) and r.get("has_update", False)
            and r.get("kl_source") in KL_CONTROLLER_SOURCES
            and math.isfinite(float(r["kl"])))
        update_kl_p95_index = (
            higher_quantile_index(len(update_kl_values), 0.95)
            if update_kl_values else 0)

        summary[task] = {
            "samples": count,
            "modes": dict(Counter(r["mode"] for r in task_records)),
            "best_sources": dict(Counter(
                r["best_source"] for r in task_records)),
            "loss": mean("loss"),
            "policy_loss": mean("policy_loss"),
            "kl_mean": (
                sum(update_kl_values) / len(update_kl_values)
                if update_kl_values else 0.0),
            "kl_p95": (
                update_kl_values[update_kl_p95_index]
                if update_kl_values else 0.0),
            "kl_max": max(update_kl_values, default=0.0),
            "kl_measured_updates": len(update_kl_values),
            "token_nll": mean("token_nll"),
            "sample_update_rate": mean("has_update"),
            "dynamic_prompt_rounds": mean("dynamic_prompt_rounds"),
            "dynamic_batch_hit_target_rate": mean(
                "dynamic_batch_hit_target"),
            "dynamic_batch_hit_cap_rate": mean(
                "dynamic_batch_hit_cap"),
            "greedy_reward": mean("greedy_reward"),
            "greedy_f1": mean("greedy_f1"),
            "greedy_precision": mean("greedy_precision"),
            "greedy_recall": mean("greedy_recall"),
            "greedy_fp_ratio": mean("greedy_fp_ratio"),
            "greedy_invalid_rate": mean("greedy_invalid"),
            "best_reward": mean("reward_best"),
            "best_f1": mean("best_f1"),
            "best_precision": mean("best_precision"),
            "best_greedy_gap": mean("best_greedy_gap"),
            "greedy_counts": {
                "tp": greedy_tp, "fp": greedy_fp, "fn": greedy_fn},
            "greedy_window_micro_f1": micro_f1_from_counts(
                greedy_tp, greedy_fp, greedy_fn),
            "reward_best_counts": {
                "tp": best_tp, "fp": best_fp, "fn": best_fn},
            "reward_best_window_micro_f1": micro_f1_from_counts(
                best_tp, best_fp, best_fn),
            "metric_best_counts": {
                "tp": metric_best_tp, "fp": metric_best_fp,
                "fn": metric_best_fn},
            "metric_best_window_micro_f1": micro_f1_from_counts(
                metric_best_tp, metric_best_fp, metric_best_fn),
            "any_hit_at_g_rate": mean("any_hit_at_g"),
            "perfect_at_g_rate": mean("perfect_at_g"),
            "sampled_rescue_at_g_rate": mean("sampled_rescue_at_g"),
            "sampled_any_hit_rate": mean("sampled_any_hit"),
            "active_sampled_any_hit_rate": mean(
                "active_sampled_any_hit"),
            "targeted_no_hit_ce_rate": mean("targeted_no_hit_ce"),
            "sequence_preference_eligible_rate": mean(
                "sequence_preference_eligible"),
            "sequence_preference_applied_rate": mean(
                "sequence_preference_applied"),
            "preference_only_rate": mean("preference_only"),
            "preference_support_scale": mean(
                "preference_support_scale"),
            "preference_first_coefficient": mean(
                "preference_first_coefficient"),
            "preference_suffix_coefficient": mean(
                "preference_suffix_coefficient"),
            "sampled_perfect_rate": mean("sampled_perfect"),
            "sampled_unique_text_ratio": mean(
                "sampled_unique_text_ratio"),
            "sampled_greedy_duplicate_rate": mean(
                "sampled_greedy_duplicate_rate"),
            "sampled_bbox_unique_ratio": mean(
                "sampled_bbox_unique_ratio"),
            "sampled_bbox_greedy_duplicate_rate": mean(
                "sampled_bbox_greedy_duplicate_rate"),
            "active_micro_raw_std": mean("active_micro_raw_std"),
            "micro_raw_zero_std_rate": mean("micro_raw_zero_std"),
            "sampled_better_than_greedy": mean(
                "sampled_better_than_greedy"),
            "reward_metric_best_agreement_rate": mean(
                "reward_metric_best_agreement"),
            "reward_best_metric_optimal_rate": mean(
                "reward_best_metric_optimal"),
            "reward_best_not_worse_than_greedy_rate": mean(
                "reward_best_not_worse_than_greedy"),
            "candidate_unique_text_ratio": mean(
                "candidate_unique_text_ratio"),
            "greedy_empty_text_rate": mean("greedy_empty_text"),
            "greedy_invalid_json_rate": mean("greedy_invalid_json"),
            "greedy_truncated_rate": mean("greedy_truncated"),
            "greedy_no_valid_box_rate": mean("greedy_no_valid_box"),
            "greedy_token_limit_rate": mean("greedy_token_limit"),
            "greedy_eos_rate": mean("greedy_eos"),
            "candidate_any_invalid_json_rate": mean(
                "candidate_any_invalid_json"),
            "candidate_any_truncated_rate": mean(
                "candidate_any_truncated"),
            "candidate_any_token_limit_rate": mean(
                "candidate_any_token_limit"),
            "greedy_completion_tokens": mean("greedy_completion_tokens"),
            "max_candidate_completion_tokens": mean(
                "max_candidate_completion_tokens"),
            "learnable_sampled_improvement_rate": mean(
                "learnable_sampled_improvement"),
            "metric_micro_raw_gap": mean("metric_micro_raw_gap"),
            "dense_greedy_gap": mean("dense_greedy_gap"),
            "active_sampled_candidates": mean(
                "active_sampled_candidates"),
            "active_sampled_tokens": mean("active_sampled_tokens"),
            "dapo_effective_candidates": mean(
                "dapo_effective_candidates"),
            "sampled_metric_not_worse_rate": mean(
                "sampled_metric_not_worse_than_greedy"),
            "normalized_adv_abs_mean": mean(
                "normalized_adv_abs_mean"),
            "micro_raw_adv_abs_mean": mean(
                "micro_raw_adv_abs_mean"),
            "localization_adv_abs_mean": mean(
                "localization_adv_abs_mean"),
            "total_adv_abs_mean": mean("total_adv_abs_mean"),
            "total_adv_abs_sum": mean("total_adv_abs_sum"),
            "total_adv_max_abs": mean("total_adv_max_abs"),
            "micro_raw_clip_rate": mean("micro_raw_clip_triggered"),
            "micro_raw_clip_scale": mean("micro_raw_clip_scale"),
        }
    return summary


def summarize_cost_bucket_records(records: List[Dict]) -> Dict:
    """定位长ALL/ROW输出是否集中触顶或截断。"""
    summary = {}
    for bucket in sorted({r["cost_bucket"] for r in records}):
        bucket_records = [r for r in records if r["cost_bucket"] == bucket]
        count = len(bucket_records)
        tp = sum(int(r["greedy_tp"]) for r in bucket_records)
        fp = sum(int(r["greedy_fp"]) for r in bucket_records)
        fn = sum(int(r["greedy_fn"]) for r in bucket_records)
        metric_tp = sum(int(r["metric_best_tp"]) for r in bucket_records)
        metric_fp = sum(int(r["metric_best_fp"]) for r in bucket_records)
        metric_fn = sum(int(r["metric_best_fn"]) for r in bucket_records)
        sampled_limit_flags = [
            bool(flag)
            for r in bucket_records
            for flag in r["candidate_token_limit_reached"][
                GREEDY_CANDIDATES:]]
        sampled_eos_flags = [
            bool(flag)
            for r in bucket_records
            for flag in r["candidate_eos_terminated"][
                GREEDY_CANDIDATES:]]
        sampled_token_counts = [
            int(tokens)
            for r in bucket_records
            for tokens in r["candidate_completion_tokens"][
                GREEDY_CANDIDATES:]]
        sampled_length_penalties = [
            float(penalty)
            for r in bucket_records
            for penalty in r["candidate_length_penalties"][
                GREEDY_CANDIDATES:]]
        greedy_limit_records = [
            r for r in bucket_records if bool(r["greedy_token_limit"])]
        greedy_limit_recoverable = sum(
            bool(r["greedy_token_limit"]) and
            any(
                bool(eos) and not bool(limit)
                for eos, limit in zip(
                    r["candidate_eos_terminated"][GREEDY_CANDIDATES:],
                    r["candidate_token_limit_reached"][
                        GREEDY_CANDIDATES:])) and
            float(r["metric_micro_raw_gap"]) > 0.0
            for r in bucket_records)

        def mean(key):
            return sum(float(r[key]) for r in bucket_records) / count

        def list_mean(values):
            return sum(float(value) for value in values) / max(len(values), 1)

        summary[bucket] = {
            "samples": count,
            "modes": dict(Counter(r["mode"] for r in bucket_records)),
            "greedy_counts": {"tp": tp, "fp": fp, "fn": fn},
            "greedy_window_micro_f1": micro_f1_from_counts(tp, fp, fn),
            "metric_best_counts": {
                "tp": metric_tp, "fp": metric_fp, "fn": metric_fn},
            "metric_best_window_micro_f1": micro_f1_from_counts(
                metric_tp, metric_fp, metric_fn),
            "sample_update_rate": mean("has_update"),
            "dynamic_prompt_rounds": mean("dynamic_prompt_rounds"),
            "dynamic_batch_hit_cap_rate": mean(
                "dynamic_batch_hit_cap"),
            "learnable_sampled_improvement_rate": mean(
                "learnable_sampled_improvement"),
            "metric_micro_raw_gap": mean("metric_micro_raw_gap"),
            "normalized_adv_abs_mean": mean(
                "normalized_adv_abs_mean"),
            "micro_raw_adv_abs_mean": mean(
                "micro_raw_adv_abs_mean"),
            "localization_adv_abs_mean": mean(
                "localization_adv_abs_mean"),
            "mean_soft_overlong_penalty": mean("length_penalty"),
            "sampled_bbox_unique_ratio": mean(
                "sampled_bbox_unique_ratio"),
            "micro_raw_zero_std_rate": mean("micro_raw_zero_std"),
            "greedy_truncated_rate": mean("greedy_truncated"),
            "greedy_token_limit_rate": mean("greedy_token_limit"),
            "greedy_eos_rate": mean("greedy_eos"),
            "greedy_completion_tokens": mean("greedy_completion_tokens"),
            "generation_token_budget": mean("generation_token_budget"),
            "active_sampled_candidates": mean(
                "active_sampled_candidates"),
            "sampled_candidate_token_limit_rate": list_mean(
                sampled_limit_flags),
            "sampled_candidate_eos_rate": list_mean(sampled_eos_flags),
            "sampled_completion_tokens": list_mean(sampled_token_counts),
            "sampled_soft_overlong_penalty": list_mean(
                sampled_length_penalties),
            "sampled_rescue_at_g_rate": mean("sampled_rescue_at_g"),
            # greedy触顶时若存在正常EOS且metric更优的sampled，瓶颈是能力迁移；
            # 否则更像全policy的长序列生成问题。该比例直接决定后续优化方向。
            "greedy_limit_recoverable_count": greedy_limit_recoverable,
            "greedy_limit_recoverable_rate": (
                greedy_limit_recoverable /
                max(len(greedy_limit_records), 1)),
            "any_hit_at_g_rate": mean("any_hit_at_g"),
            "perfect_at_g_rate": mean("perfect_at_g"),
        }
    return summary


def _format_micro_record(rec: Dict) -> str:
    # 默认关闭；人工打开时也只打印仍生效的主路径字段。
    return (
        f"[{datetime.now():%H:%M:%S}] sample "
        f"idx={rec['idx']} task={rec['task']} bucket={rec['cost_bucket']} "
        f"mode={rec['mode']} update={int(rec['has_update'])} "
        f"R={rec['reward_mean']:.3f} "
        f"G_TP/FP/FN={rec['greedy_tp']}/{rec['greedy_fp']}/"
        f"{rec['greedy_fn']} activeS={rec['active_sampled_candidates']} "
        f"limit={int(rec['candidate_any_token_limit'])} "
        f"KL={rec['kl']:.4f} loss={rec['loss']:.4f}")
def _is_hard_sample(rec: Dict) -> bool:
    return (rec["mode"] in {
                "SKIP_overlong_insufficient", "SKIP_no_signal",
                "SKIP_no_improvement", "SKIP_ddp_route_mismatch",
                "SKIP_generation_timeout"} or
            rec["greedy_invalid"] or
            rec["invalid_count"] > 0 or
            rec["reward_mean"] < MICRO_CONSOLE_REWARD_THRESHOLD or
            rec["reward_worst"] < 0)


def gather_and_log_micro_samples(records: List[Dict], step: int):
    """Gather all records for window metrics; persist only diagnostic anomalies."""
    if WORLD_SIZE > 1:
        gathered = [None] * WORLD_SIZE if IS_MAIN else None
        dist.gather_object(records, gathered, dst=0)
    else:
        gathered = [records]
    if not IS_MAIN:
        return []
    flat_records = [rec for rank_records in gathered for rec in rank_records]
    global _MICRO_LOG_HANDLE
    if _MICRO_LOG_HANDLE is None:
        _MICRO_LOG_HANDLE = open(MICRO_LOG_FILE, "w", encoding="utf-8")
    for rec in flat_records:
        persist_detail = bool(
            rec.get("mode") == "SKIP_overlong_insufficient" or
            rec.get("mode") == "SKIP_ddp_route_mismatch" or
            rec.get("mode") == "SKIP_generation_timeout" or
            rec.get("mode") == "CE_targeted_no_hit" or
            rec.get("sequence_preference_eligible", False) or
            rec.get("greedy_invalid", False) or
            rec.get("candidate_any_token_limit", False) or
            rec.get("candidate_any_truncated", False) or
            rec.get("candidate_any_invalid_json", False) or
            float(rec.get("invalid_count", 0.0)) > 0.0 or
            float(rec.get("reward_worst", 0.0)) < 0.0)
        if persist_detail:
            payload = {
                "step": step,
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                **rec}
            _MICRO_LOG_HANDLE.write(
                json.dumps(payload, ensure_ascii=False) + "\n")
        if PRINT_HARD_MICRO_SAMPLES and _is_hard_sample(rec):
            print(_format_micro_record(rec))
    _MICRO_LOG_HANDLE.flush()
    return flat_records


def close_micro_log():
    global _MICRO_LOG_HANDLE
    if _MICRO_LOG_HANDLE is not None:
        _MICRO_LOG_HANDLE.close()
        _MICRO_LOG_HANDLE = None


# ========================= 模型加载 =========================

def disable_model_dropout(model) -> int:
    """Disable stochastic dropout while preserving gradients in eval mode."""
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    return changed


def adapter_parameter_kind(name: str) -> str:
    """Return the persisted PEFT parameter family this run may optimize."""
    lowered = name.lower()
    if "lora_" in lowered or ".lora" in lowered:
        return "lora"
    if (TRAIN_ADAPTER_PROJECTOR and "modules_to_save" in lowered and
            "visual" in lowered and "merger" in lowered):
        return "projector"
    return ""


def load_models(policy_lora_path=None, reference_lora_path=None):
    """加载 processor / policy_model / ref_model，自动适配单卡/DDP"""
    is_ddp = WORLD_SIZE > 1
    policy_lora_path = policy_lora_path or POLICY_LORA_PATH
    reference_lora_path = reference_lora_path or REFERENCE_LORA_PATH

    if IS_MAIN:
        print(f"[{datetime.now():%H:%M:%S}] 加载 processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # ---------- 共同加载参数 ----------
    load_kwargs = dict(dtype=torch.bfloat16)
    if not is_ddp:
        load_kwargs["device_map"] = "auto"

    # ---------- Policy 模型 ----------
    if IS_MAIN:
        print(f"[{datetime.now():%H:%M:%S}] 加载 Policy 模型 "
              f"(base + 5000-step LoRA): {policy_lora_path}")
    flash_ok = False
    try:
        base_policy = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH, attn_implementation="flash_attention_2", **load_kwargs)
        flash_ok = True
    except Exception as e:
        if IS_MAIN: print(f"  FlashAttention2 unavailable: {e}")
        base_policy = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH, **load_kwargs)

    if is_ddp:
        base_policy = base_policy.to(DEVICE)

    policy_model = PeftModel.from_pretrained(
        base_policy, policy_lora_path, is_trainable=True)
    trainable_kind_counts = Counter()
    trainable_kind_params = Counter()
    for n, p in policy_model.named_parameters():
        kind = adapter_parameter_kind(n)
        p.requires_grad = bool(kind)
        if kind:
            trainable_kind_counts[kind] += 1
            trainable_kind_params[kind] += p.numel()
    dropout_disabled = (
        disable_model_dropout(policy_model) if RL_DISABLE_DROPOUT else 0)
    # eval()只关闭dropout/随机层，不会关闭autograd；rollout与训练log-prob因而
    # 使用同一个确定性policy分布。
    policy_model.eval()

    if IS_MAIN:
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        print(f"  {'FlashAttention2' if flash_ok else 'SDPA'} | "
              f"可训练参数: {trainable:,}")
        print(f"  adapter可训练参数: {dict(trainable_kind_params)} "
              f"(张量数={dict(trainable_kind_counts)})")
        print(f"  dropout关闭: {dropout_disabled}个Dropout模块；"
              "policy rollout/forward均保持eval模式")
        if TRAIN_ADAPTER_PROJECTOR and trainable_kind_counts["projector"] == 0:
            print("  ⚠ 未检测到可持久化的visual.merger modules_to_save参数；"
                  "本次运行将仅优化LoRA，不会解冻无法随adapter保存的base参数")

    # ---------- Reference 模型（冻结的原始SFT锚点） ----------
    if IS_MAIN:
        print(f"[{datetime.now():%H:%M:%S}] 加载 Reference 模型 "
              f"(冻结的原始SFT锚点): {reference_lora_path}")
    try:
        base_ref = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH, attn_implementation="flash_attention_2", **load_kwargs)
    except Exception:
        base_ref = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH, **load_kwargs)

    if is_ddp:
        base_ref = base_ref.to(DEVICE)

    ref_model = PeftModel.from_pretrained(base_ref, reference_lora_path)
    for p in ref_model.parameters():
        p.requires_grad = False
    if RL_DISABLE_DROPOUT:
        disable_model_dropout(ref_model)
    ref_model.eval()
    if IS_MAIN:
        print("  Reference 就绪（冻结）")

    return processor, policy_model, ref_model


# ========================= 生成候选输出 =========================

@torch.no_grad()
def generate_candidates(model, processor, image, prompt, device,
                        num_candidates=G, max_new_tokens=None):
    generation_started = time.monotonic()
    generation_deadline = (
        generation_started + SAMPLE_GENERATION_TIMEOUT_SECONDS)
    generation_timed_out = False
    # 根据任务类型动态设置生成长度上限，节省显存和推理时间
    task = classify_task(prompt)
    if max_new_tokens is None:
        max_new_tokens = TASK_MAX_TOKENS.get(task, MAX_NEW_TOKENS)
    exploit_candidates = EXPLOIT_SAMPLED_CANDIDATES[task]
    temperature_low = SAMPLE_TEMPERATURE_LOW[task]
    temperature_high = SAMPLE_TEMPERATURE_HIGH[task]
    sample_top_p = SAMPLE_TOP_P[task]
    # 临时切到 eval 模式：关掉 dropout，让采样差异只来自 temperature
    was_training = model.training
    model.eval()

    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)

    # 同一 prompt/image 的 processor 结果完全相同；v5/v6 重复执行 G 次既浪费
    # CPU，也重复读取/变换图像。这里只处理一次，再按候选数复制。
    single = processor(
        text=chat_text, images=image, return_tensors="pt", padding=True)
    input_ids = single["input_ids"].repeat(num_candidates, 1).to(device)
    attention_mask = single["attention_mask"].repeat(
        num_candidates, 1).to(device)
    single_pv = single.get("pixel_values")
    single_gw = single.get("image_grid_thw")
    pv = (
        torch.cat([single_pv] * num_candidates, dim=0).to(device)
        if single_pv is not None else None)
    gw = (
        torch.cat([single_gw] * num_candidates, dim=0).to(device)
        if single_gw is not None else None)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask,
              "pixel_values": pv, "image_grid_thw": gw}

    prompt_len = input_ids.shape[1]
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    def _generate_range(start, end, do_sample, temperature=None):
        nonlocal generation_timed_out
        candidate_count = int(end - start)
        fallback_id = eos_id if eos_id is not None else pad_id
        remaining_seconds = generation_deadline - time.monotonic()
        if generation_timed_out or remaining_seconds <= 0.0:
            generation_timed_out = True
            return torch.full(
                (candidate_count, 1), fallback_id,
                dtype=input_ids.dtype, device=device)
        idx = list(range(start, end))
        sub_inputs = {
            "input_ids": input_ids[idx],
            "attention_mask": attention_mask[idx],
        }
        sub_pv, sub_gw = _slice_vision_inputs(pv, gw, idx)
        sub_inputs["pixel_values"] = sub_pv
        sub_inputs["image_grid_thw"] = sub_gw
        kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=pad_id, eos_token_id=eos_id,
            max_time=max(remaining_seconds, 1.0e-3))
        kwargs.update({
            name: value
            for name, value in GENERATION_NEUTRAL_OVERRIDES.items()
            if hasattr(model.generation_config, name)
        })
        if do_sample:
            kwargs.update(
                temperature=temperature, top_p=sample_top_p,
                top_k=SAMPLE_TOP_K, min_p=None, typical_p=1.0,
                epsilon_cutoff=0.0, eta_cutoff=0.0)
        else:
            # 覆盖 Qwen generation_config 中的 sampling 默认值，避免 greedy 警告。
            kwargs.update(temperature=None, top_p=None, top_k=None)
        generated = model.generate(**sub_inputs, **kwargs)[:, prompt_len:]
        if time.monotonic() >= generation_deadline:
            generation_timed_out = True
        return generated

    # 第 0 个候选严格复现验证时 greedy 解码；其余候选负责探索。
    completion_parts = []
    n_greedy = min(GREEDY_CANDIDATES, num_candidates)
    if n_greedy:
        completion_parts.append(_generate_range(0, n_greedy, do_sample=False))
    if n_greedy < num_candidates:
        low_end = min(n_greedy + exploit_candidates, num_candidates)
        if low_end > n_greedy:
            completion_parts.append(_generate_range(
                n_greedy, low_end, do_sample=True,
                temperature=temperature_low))
        if low_end < num_candidates:
            completion_parts.append(_generate_range(
                low_end, num_candidates, do_sample=True,
                temperature=temperature_high))

    max_completion_len = max(part.shape[1] for part in completion_parts)
    generated_ids = torch.full(
        (num_candidates, max_completion_len), pad_id,
        dtype=input_ids.dtype, device=device)
    cursor = 0
    for part in completion_parts:
        generated_ids[cursor:cursor + part.shape[0], :part.shape[1]] = part
        cursor += part.shape[0]
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    if generated_ids.shape[1] == 0:
        # 真正的空生成应训练 EOS，而不是原实现误用 token id 0。
        fallback_id = eos_id if eos_id is not None else pad_id
        generated_ids = torch.full(
            (num_candidates, 1), fallback_id, dtype=torch.long, device=device)
        texts = [""] * num_candidates

    # 只训练到第一个 EOS（含 EOS）；EOS 后的 padding 不参与 GRPO/KL。
    if eos_id is not None:
        is_eos = generated_ids.eq(eos_id)
        completion_mask = (is_eos.long().cumsum(dim=1) - is_eos.long()).eq(0)
    else:
        completion_mask = torch.ones_like(generated_ids, dtype=torch.bool)
    if pad_id is not None and pad_id != eos_id:
        completion_mask &= generated_ids.ne(pad_id)

    if was_training and not RL_DISABLE_DROPOUT:
        model.train()

    generation_runtime = {
        "timed_out": bool(generation_timed_out),
        "elapsed_seconds": float(time.monotonic() - generation_started),
        "timeout_seconds": float(SAMPLE_GENERATION_TIMEOUT_SECONDS),
    }
    return texts, inputs, generated_ids, completion_mask, generation_runtime


# ========================= Per-Token Log Probabilities =========================

def _slice_vision_inputs(pixel_values, image_grid_thw, img_indices):
    """从 pixel_values [total_patches, ...] 中取出指定 image 的所有 patch"""
    if pixel_values is None or image_grid_thw is None:
        return None, None
    patches_per_img = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
    cumsum = torch.cumsum(
        torch.cat([torch.zeros(1, device=image_grid_thw.device, dtype=torch.long),
                   patches_per_img]), dim=0)
    start = cumsum[img_indices[0]].item()
    end   = cumsum[img_indices[-1] + 1].item()
    return pixel_values[start:end], image_grid_thw[img_indices]


def _batched_log_probs(model, inputs, generated_ids, completion_mask=None):
    """Compute completion log-probs without forwarding padded completion tails.

    The returned tensor keeps ``generated_ids.shape[1]`` for caller
    compatibility, but each candidate is forwarded only through its real
    completion length.  This is essential when one ALL candidate reaches
    6144/10240 while the other candidates end much earlier.
    """
    batch_size = generated_ids.shape[0]
    device = generated_ids.device
    output_len = generated_ids.shape[1]

    # 拼接 input + generated，padding 到最大长度
    full_list = []
    input_lens = []
    gen_lens = []
    for i in range(batch_size):
        inp = inputs["input_ids"][i:i+1]
        if completion_mask is None:
            gen_len = output_len
        else:
            gen_len = max(
                int(completion_mask[i].to(torch.long).sum().item()), 1)
        gen = generated_ids[i:i+1, :gen_len]
        input_lens.append(inp.shape[1])
        gen_lens.append(gen_len)
        full_list.append(torch.cat([inp, gen], dim=1))

    max_len = max(f.shape[1] for f in full_list)
    padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    padded_mask = torch.zeros(batch_size, max_len, dtype=full_list[0].dtype, device=device)
    for i, f in enumerate(full_list):
        padded_ids[i, :f.shape[1]] = f
        padded_mask[i, :f.shape[1]] = 1

    pv = inputs.get("pixel_values")
    gw = inputs.get("image_grid_thw")
    if pv is not None:
        pv = pv.to(device)
    if gw is not None:
        gw = gw.to(device)

    outputs = model(input_ids=padded_ids, attention_mask=padded_mask,
                    pixel_values=pv, image_grid_thw=gw, use_cache=False)

    # 逐个候补提取生成 token 的 log prob
    all_log_probs = []
    for i in range(batch_size):
        inp_len = input_lens[i]
        gen_len = gen_lens[i]
        gen_logits = outputs.logits[i:i+1, inp_len-1: inp_len-1 + gen_len, :]
        log_probs_all = F.log_softmax(gen_logits.float(), dim=-1)
        log_probs = torch.gather(log_probs_all, dim=2,
                                 index=generated_ids[i:i+1, :gen_len].unsqueeze(-1)).squeeze(-1)
        if gen_len < output_len:
            log_probs = F.pad(log_probs, (0, output_len - gen_len))
        all_log_probs.append(log_probs)

    # 显式释放巨大的 logits tensor，避免残留导致后续 OOM
    del outputs

    return torch.cat(all_log_probs, dim=0)


def get_policy_log_probs(model, inputs, generated_ids, completion_mask=None):
    """策略模型 per-token log prob（eval 关闭 dropout，但保留梯度）。"""
    was_training = model.training
    model.eval()
    try:
        return _batched_log_probs(
            model, inputs, generated_ids, completion_mask=completion_mask)
    finally:
        if was_training and not RL_DISABLE_DROPOUT:
            model.train()


def zero_policy_sync_backward(model, inputs, generated_ids,
                              completion_mask, candidate_idx: int):
    """Run one memory-bounded zero-gradient DDP synchronization.

    Different ranks may route a prompt to GRPO or skip.  A skipped rank still
    needs one synchronized backward to keep DDP collective ordering aligned.
    Only one completion token is forwarded to prepare the reducer.  Its graph
    is then discarded, and parameter-only zero anchors fire every reducer hook
    without traversing the vocabulary-logit graph.
    """
    idx = [int(candidate_idx)]
    mb_inputs = {}
    for key in ("input_ids", "attention_mask"):
        mb_inputs[key] = (
            inputs[key][idx] if inputs.get(key) is not None else None)
    mb_pv, mb_gw = _slice_vision_inputs(
        inputs.get("pixel_values"), inputs.get("image_grid_thw"), idx)
    mb_inputs["pixel_values"] = mb_pv
    mb_inputs["image_grid_thw"] = mb_gw
    sync_generated_ids = generated_ids[idx, :1]
    sync_completion_mask = completion_mask[idx, :1]
    zero_log_probs = get_policy_log_probs(
        model, mb_inputs, sync_generated_ids,
        completion_mask=sync_completion_mask)
    del zero_log_probs, mb_inputs, mb_pv, mb_gw
    torch.cuda.empty_cache()
    zero_parameter_sync_backward(model)


def zero_parameter_sync_backward(model):
    """Finish one synchronized DDP backward without traversing rollout logits.

    This helper is used only after a synchronized DDP forward whose candidate
    objective was rejected (for example by the global KL safety vote).  Every
    trainable parameter contributes an explicit zero scalar, so all reducer
    hooks fire in the same order while the discarded completion/logit graph is
    not traversed.  In particular, this avoids the multi-GiB softmax backward
    allocation caused by ``0.0 * policy_log_probs.sum()`` on long ALL outputs.
    """
    module = model.module if isinstance(model, DDP) else model
    zero_anchor = None
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        parameter_anchor = parameter.reshape(-1)[0] * 0.0
        zero_anchor = (parameter_anchor if zero_anchor is None
                       else zero_anchor + parameter_anchor)
    if zero_anchor is None:
        raise RuntimeError("No trainable parameter for zero-gradient DDP sync")
    zero_anchor.backward()


@torch.no_grad()
def get_ref_log_probs(model, inputs, generated_ids, completion_mask=None):
    """参考模型 per-token log prob（无梯度），分批 forward 减少峰值显存"""
    G_total = generated_ids.shape[0]
    REF_BATCH = 1  # 逐候补 forward，避免长度截断丢 KL 信号（ref 无梯度，显存安全）
    all_lps = []
    for start in range(0, G_total, REF_BATCH):
        end = min(start + REF_BATCH, G_total)
        idx = list(range(start, end))
        mb_inputs = {}
        for k in ["input_ids", "attention_mask"]:
            mb_inputs[k] = inputs[k][idx] if inputs.get(k) is not None else None
        mb_pv, mb_gw = _slice_vision_inputs(
            inputs.get("pixel_values"), inputs.get("image_grid_thw"), idx)
        mb_inputs["pixel_values"] = mb_pv
        mb_inputs["image_grid_thw"] = mb_gw
        mb_completion_mask = (
            completion_mask[idx] if completion_mask is not None else None)
        lp = _batched_log_probs(
            model, mb_inputs, generated_ids[idx],
            completion_mask=mb_completion_mask)
        all_lps.append(lp)
    return torch.cat(all_lps, dim=0)


# ========================= GRPO 损失函数 =========================

def compute_grpo_loss(policy_log_probs, ref_log_probs, rewards,
                      completion_mask, beta=BETA,
                      gamma=GAMMA, advantages=None):

    if advantages is None:
        raise ValueError(
            "v7 requires explicitly centered precomputed advantages; "
            "reward/std fallback is intentionally disabled")
    # 仅用于诊断，不参与优势标准化。
    r_mean = rewards.mean()
    r_std = rewards.std(unbiased=False)

    mask = completion_mask.to(policy_log_probs.dtype)
    token_count = mask.sum(dim=1).clamp_min(1.0)

    adv_exp = advantages.unsqueeze(1).expand(-1, policy_log_probs.shape[1])

    # rollout 与本次 backward 之间没有 optimizer update，且每个 rollout 只更新一次。
    # v3 的 old_log_probs=policy_log_probs.detach() 使 ratio 恒为1；以下 score-
    # function loss 与其梯度严格等价，但不会伪装成存在 PPO clipping。
    per_token_loss = -(policy_log_probs * adv_exp)

    # reference-policy 的 k3 KL；对极端 token 截断，避免单个候选支配整步。
    ref_diff = (ref_log_probs - policy_log_probs).clamp(
        -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    per_token_kl = torch.exp(ref_diff) - ref_diff - 1.0
    kl_per_seq = (per_token_kl * mask).sum(dim=1) / token_count

    # 这是采样 token 的平均 log-prob，不是真正的词表 entropy；日志命名为 NLL/token。
    token_logp_seq = (policy_log_probs * mask).sum(dim=1) / token_count

    policy_loss_seq = (per_token_loss * mask).sum(dim=1) / token_count
    total_loss_seq = policy_loss_seq + beta * kl_per_seq + gamma * token_logp_seq
    loss = total_loss_seq.mean()

    with torch.no_grad():
        approx_kl = kl_per_seq.mean().item()

    metrics = {"loss": loss.item(), "policy_loss": policy_loss_seq.mean().item(),
               "kl_div": approx_kl,
               "token_nll": -token_logp_seq.mean().item(),
               "reward_mean": r_mean.item(), "reward_std": r_std.item(),
               "adv_min": advantages.min().item(), "adv_max": advantages.max().item()}
    return loss, metrics


def compute_sft_ce_loss(
        model, processor, image, prompt, gt_response, device,
        rejected_response: str = None,
        suffix_only: bool = False):
    """困难组监督回退；可只训练GT相对greedy首次分叉后的suffix。"""
    prompt_only = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=False)
    full_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt}]},
         {"role": "assistant", "content": [{"type": "text", "text": gt_response}]}],
        add_generation_prompt=False, tokenize=False)
    prompt_proc = processor(text=prompt_only, images=image,
                            return_tensors="pt", padding=True)
    prompt_len = prompt_proc.input_ids.shape[1]
    full_proc = processor(text=full_text, images=image,
                          return_tensors="pt", padding=True)
    ce_ids = full_proc.input_ids.to(device)
    ce_mask = full_proc.attention_mask.to(device)
    ce_labels = ce_ids.clone()
    ce_labels[0, :prompt_len] = -100
    if suffix_only and rejected_response is not None:
        rejected_text = processor.apply_chat_template(
            [{"role": "user",
              "content": [{"type": "image"},
                          {"type": "text", "text": prompt}]},
             {"role": "assistant",
              "content": [{"type": "text", "text": rejected_response}]}],
            add_generation_prompt=False, tokenize=False)
        rejected_proc = processor(
            text=rejected_text, images=image,
            return_tensors="pt", padding=True)
        target_completion = ce_ids[:, prompt_len:]
        target_valid = ce_mask[:, prompt_len:].bool()
        rejected_completion = rejected_proc.input_ids[:, prompt_len:].to(
            device)
        rejected_valid = rejected_proc.attention_mask[:, prompt_len:].bool().to(
            device)
        target_len = int(target_valid[0].sum().item())
        rejected_len = int(rejected_valid[0].sum().item())
        common_len = min(target_len, rejected_len)
        divergence = 0
        if common_len > 0:
            different = target_completion[0, :common_len].ne(
                rejected_completion[0, :common_len])
            locations = torch.nonzero(different, as_tuple=False)
            if locations.numel() > 0:
                divergence = int(locations[0, 0].item())
            elif target_len != rejected_len:
                divergence = max(common_len - 1, 0)
            else:
                divergence = max(common_len - 1, 0)
        target_positions = torch.arange(
            target_completion.shape[1], device=device)
        keep = target_valid & target_positions.unsqueeze(0).ge(divergence)
        target_labels = ce_labels[
            0, prompt_len:prompt_len + target_completion.shape[1]]
        ce_labels[
            0, prompt_len:prompt_len + target_completion.shape[1]] = (
                torch.where(
                    keep[0], target_labels,
                    torch.full_like(target_labels, -100)))
        del rejected_proc, rejected_completion, rejected_valid
    ce_pv = full_proc.get("pixel_values")
    ce_gw = full_proc.get("image_grid_thw")
    if ce_pv is not None:
        ce_pv = ce_pv.to(device, dtype=torch.bfloat16)
    if ce_gw is not None:
        ce_gw = ce_gw.to(device)
    ce_out = model(input_ids=ce_ids, attention_mask=ce_mask,
                   labels=ce_labels, pixel_values=ce_pv, image_grid_thw=ce_gw)
    return ce_out.loss


# ========================= 训练循环 =========================

def train(policy_lora_path=None, resume_state_path=None):
    is_ddp = WORLD_SIZE > 1
    torch.set_float32_matmul_precision('high')
    policy_lora_path = policy_lora_path or POLICY_LORA_PATH
    assert_training_output_is_fresh(resume_state_path)
    resume_state = None
    if resume_state_path is not None:
        resume_state = torch.load(
            resume_state_path, map_location="cpu", weights_only=False)
        if int(resume_state.get("format_version", 0)) != 1:
            raise RuntimeError(
                f"不支持的resume state格式: "
                f"{resume_state.get('format_version')}")
        if int(resume_state.get("world_size", -1)) != WORLD_SIZE:
            raise RuntimeError(
                "恢复训练必须保持相同WORLD_SIZE："
                f"state={resume_state.get('world_size')} current={WORLD_SIZE}")
        if (resume_state.get("policy_objective") != POLICY_OBJECTIVE or
                resume_state.get("policy_token_denominator") !=
                POLICY_TOKEN_DENOMINATOR or
                resume_state.get("reference_lora_path") !=
                REFERENCE_LORA_PATH):
            raise RuntimeError(
                "resume state的目标公式/分母/reference与当前脚本不一致，"
                "拒绝静默改变实验定义")
        saved_script_hash = resume_state.get("training_script_sha256")
        current_script_hash = file_sha256(__file__)
        oom_fix_compatible_resume = bool(
            saved_script_hash.upper() in OOM_FIX_COMPATIBLE_RESUME_SHA256)
        if (saved_script_hash and
                saved_script_hash != current_script_hash and
                not oom_fix_compatible_resume):
            raise RuntimeError(
                "resume state与当前训练脚本SHA256不一致；请使用保存该"
                "checkpoint时的原脚本恢复，避免静默改变奖励/路由")
        if (IS_MAIN and saved_script_hash != current_script_hash and
                oom_fix_compatible_resume):
            print(
                "允许从OOM修复前CGPC恢复：仅替换KL/无目标候选的DDP"
                "零梯度同步实现，训练目标与optimizer状态保持不变。"
                f" old_sha={saved_script_hash} new_sha={current_script_hash}")
        for state_key, current_path in (
                ("train_json_sha256", TRAIN_JSON),):
            saved_hash = resume_state.get(state_key)
            if saved_hash and saved_hash != file_sha256(current_path):
                raise RuntimeError(
                    f"resume state与当前数据不一致: {state_key}；"
                    "拒绝改变训练样本后继续同一实验")
        policy_lora_path = resume_state["policy_adapter_path"]

    if IS_MAIN:
        print("=" * 60)
        print(f"GRPO-base 训练 {'(DDP ' + str(WORLD_SIZE) + ' GPU)' if is_ddp else '(单卡)'}")
        print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"模型: base={BASE_MODEL_PATH}; policy={policy_lora_path}; "
              f"reference={REFERENCE_LORA_PATH}")
        if resume_state is not None:
            print(f"恢复状态:   {resume_state_path}; "
                  f"slot={resume_state['global_step']}, "
                  f"optimizer={resume_state['optimizer_update_step']}, "
                  f"next=epoch{resume_state['epoch'] + 1}/"
                  f"batch{resume_state['next_batch_idx']}")
        print(f"数据/输出: train={TRAIN_JSON}; output={GRPO_OUTPUT_DIR}")
        print(f"核心参数: G={G}(1 greedy诊断+{G-GREEDY_CANDIDATES} sampled), "
              f"LoRA/projector LR={GRPO_LR:.2e}/{PROJECTOR_LR:.2e}, "
              f"β={BETA}, grad_clip={MAX_GRAD_NORM}, epochs={GRPO_EPOCHS}")
        print(
            "CGPC新增漂移搜索区间: LoRA精确dW/Wref="
            f"{LORA_DRIFT_TARGET_MIN_RELATIVE:.1%}--"
            f"{LORA_DRIFT_TARGET_MAX_RELATIVE:.1%}; "
            "漂移达到目标不代表验证能力提升")
        print(f"更新批量: accumulation={GRADIENT_ACCUMULATION_STEPS}, "
              f"world={WORLD_SIZE}, global_prompt_batch="
              f"{GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE}, "
              f"dynamic_target/cap="
              f"{DYNAMIC_TARGET_EFFECTIVE_PROMPTS}/"
              f"{DYNAMIC_MAX_PROMPT_ROUNDS} rounds; "
              "控制台每个optimizer slot一行摘要")
        print(f"长度: deploy={TASK_MAX_TOKENS}; train_buckets="
              f"{TRAIN_BUCKET_MAX_TOKENS}; 无EOS触顶sampled不进入loss，"
              f"soft-overlong={SOFT_OVERLONG_MAX_PENALTY}; "
              f"单prompt生成超时={SAMPLE_GENERATION_TIMEOUT_SECONDS:.0f}s，"
              "超时后全rank跳过")
        print(f"目标: {POLICY_OBJECTIVE}/{LOSS_REDUCTION}; "
              f"adv=hard/soft-TP({1.0-SOFT_TP_QUALITY_BLEND:.0%}/"
              f"{SOFT_TP_QUALITY_BLEND:.0%})×{MICRO_RAW_ADVANTAGE_SCALE} "
              f"(radial_clip={MICRO_RAW_ADVANTAGE_CLIP}), "
              f"localization={LOCALIZATION_ADVANTAGE_SCALE}, "
              f"task_gain={TASK_ADVANTAGE_GAIN}, "
              f"fallback_reward={ADVANTAGE_FIXED_SCALE}; "
              "strict TP/FP/FN仅用于训练监控")
        print("路由: SINGLE仅标准GRPO+KL；ROW标准GRPO×1.15；ALL在greedy"
              "触顶或高质量sampled显著更优且有多候选支持时启用first/suffix偏好；"
              "overlong权重="
              f"{ALL_OVERLONG_PREFERENCE_FIRST_WEIGHT}/"
              f"{ALL_OVERLONG_PREFERENCE_SUFFIX_WEIGHT}，quality权重="
              f"{ALL_QUALITY_PREFERENCE_FIRST_WEIGHT}/"
              f"{ALL_QUALITY_PREFERENCE_SUFFIX_WEIGHT}；chosen非正adv时"
              "preference-only+KL；SINGLE定向CE关闭")
        print(f"监控: window={MONITOR_STEPS} slots，保留TP/FP/FN、KL、"
              "梯度/clip、LR/显存和LoRA/projector精确权重漂移；"
              f"异常样本JSONL={MICRO_LOG_FILE}")
        print("=" * 60)

    # ---- 加载模型 ----
    processor, policy_model, ref_model = load_models(
        policy_lora_path, REFERENCE_LORA_PATH)

    # ---- DDP 包装 policy 模型 ----
    if is_ddp:
        train_model = DDP(policy_model, device_ids=[LOCAL_RANK],
                          find_unused_parameters=False,
                          broadcast_buffers=False,
                          gradient_as_bucket_view=True)
    else:
        train_model = policy_model

    # ---- 加载数据 ----
    if IS_MAIN:
        print(f"\n[{datetime.now():%H:%M:%S}] 加载训练数据...")
    dataset = CottonBaleDataset(TRAIN_JSON, IMAGE_ROOT)
    if IS_MAIN:
        print(f"原始训练样本数: {len(dataset.data)}  采样后: {len(dataset)}")
        print(f"原始任务分布: {dict(dataset.task_counts)}  重复权重: {TASK_REPEAT}")
        print(f"实际采样任务分布（含补齐）: {dict(dataset.sampled_task_counts)}")

    sampler = CostBalancedDistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True)
    if IS_MAIN:
        bucket_counts = dict(sorted(
            (key, len(indices)) for key, indices in sampler.buckets.items()))
        print(f"成本桶分布: {bucket_counts}")
        print(f"每rank每轮样本数: {len(sampler)}")
    dataloader = DataLoader(
        dataset, batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=False, sampler=sampler,
        collate_fn=collate_fn, num_workers=0)
    epoch_prompt_exposures = len(dataloader) * WORLD_SIZE
    if IS_MAIN:
        print(f"每epoch全局prompt曝光数: {epoch_prompt_exposures} "
              f"(每slot通常{GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE}条，"
              "动态有效批量可能临时多读)")

    # ---- 优化器 + LR Scheduler ----
    named_trainable = [
        (n, p) for n, p in train_model.named_parameters() if p.requires_grad]
    lora_params = [
        p for n, p in named_trainable if adapter_parameter_kind(n) == "lora"]
    projector_params = [
        p for n, p in named_trainable
        if adapter_parameter_kind(n) == "projector"]
    trainable_params = [p for _, p in named_trainable]
    if not lora_params:
        raise RuntimeError("未检测到可训练LoRA参数，拒绝启动空更新训练")
    optimizer_groups = [{"params": lora_params, "lr": GRPO_LR,
                         "group_name": "lora"}]
    if projector_params:
        optimizer_groups.append({
            "params": projector_params, "lr": PROJECTOR_LR,
            "group_name": "visual_merger"})
    lora_parameter_count = sum(p.numel() for p in lora_params)
    projector_parameter_count = sum(p.numel() for p in projector_params)
    lora_initial_norm = float(
        parameter_group_value_norm(lora_params, DEVICE).item())
    projector_initial_norm = float(
        parameter_group_value_norm(projector_params, DEVICE).item())
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=WEIGHT_DECAY)
    planned_steps = len(dataloader) * GRPO_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    total_steps = planned_steps
    warmup_steps = max(int(total_steps * WARMUP_RATIO), 1)
    # 统一乘法因子保留LoRA与projector的4:1 LR比例；短warmup后余弦收敛，
    # 在fresh run完整epoch后段降低过冲风险。
    def lr_factor(step):
        if step < warmup_steps:
            # CGPC主LoRA LR从1e-6 warmup到2e-6，不再从1%峰值的过低LR启动。
            return (MIN_LR_RATIO +
                    (1.0 - MIN_LR_RATIO) *
                    step / max(warmup_steps, 1))
        progress = min(
            max((step - warmup_steps) /
                max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return (MIN_LR_RATIO +
                0.5 * (1.0 - MIN_LR_RATIO) *
                (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_factor)
    if IS_MAIN:
        group_lrs = {
            group.get("group_name", str(i)): group["lr"]
            for i, group in enumerate(optimizer.param_groups)}
        print(f"优化器: AdamW, group_lrs={group_lrs}, weight_decay={WEIGHT_DECAY}, "
              f"warmup={warmup_steps}/{total_steps} optimizer updates, "
              f"schedule=warmup+cosine, min_lr_ratio={MIN_LR_RATIO} "
              "(仅真实optimizer.step推进schedule；零梯度slot不推进"
              f"Adam或LR), 可训练参数组: {len(trainable_params)}")
        print(
            f"低开销参数组监控: LoRA n/norm="
            f"{lora_parameter_count}/{lora_initial_norm:.3f}, "
            f"projector n/norm="
            f"{projector_parameter_count}/{projector_initial_norm:.3f}, "
            f"每{GRAD_GROUP_MONITOR_INTERVAL}个窗口抽样")

    global_step = 0
    optimizer_update_step = 0
    reward_history, loss_history = [], []
    current_beta = float(BETA)
    current_kl_safety_lr_scale = 1.0
    beta_history = []
    monitor_history = []
    monitor_sample_records = []
    monitor_grad_sum = 0.0
    monitor_grad_count = 0
    monitor_grad_max = 0.0
    monitor_lora_grad_sum = 0.0
    monitor_lora_grad_max = 0.0
    monitor_projector_grad_sum = 0.0
    monitor_projector_grad_max = 0.0
    monitor_lora_grad_rms_sum = 0.0
    monitor_lora_grad_rms_max = 0.0
    monitor_projector_grad_rms_sum = 0.0
    monitor_projector_grad_rms_max = 0.0
    monitor_lora_relative_step_sum = 0.0
    monitor_lora_relative_step_max = 0.0
    monitor_projector_relative_step_sum = 0.0
    monitor_projector_relative_step_max = 0.0
    monitor_group_grad_count = 0
    monitor_optimizer_updates = 0
    monitor_zero_update_steps = 0
    monitor_clip_steps = 0
    monitor_nonfinite_grad_steps = 0
    monitor_contributing_rank_sum = 0
    monitor_contributing_rank_min = WORLD_SIZE
    monitor_contributing_rank_max = 0
    monitor_effective_samples_sum = 0
    monitor_effective_sample_scale_sum = 0.0
    monitor_effective_sample_scale_max = 0.0
    monitor_effective_scale_cap_steps = 0
    monitor_kl_safety_lr_scale_sum = 0.0
    kl_window_local_sum = 0.0
    kl_window_local_count = 0
    torch.cuda.reset_peak_memory_stats(DEVICE)
    resume_epoch = 0
    resume_next_batch_idx = 0
    resumed_epoch_records = []
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        optimizer_update_step = int(
            resume_state["optimizer_update_step"])
        if scheduler.last_epoch != optimizer_update_step:
            raise RuntimeError(
                "resume state的scheduler update与optimizer_update_step"
                f"不一致: {scheduler.last_epoch} != "
                f"{optimizer_update_step}")
        # v7固定KL与LR倍率；旧v5/v6断点中的adaptive控制器状态不得污染新实验。
        current_beta = float(BETA)
        current_kl_safety_lr_scale = 1.0
        beta_history = list(resume_state.get("beta_history", []))
        monitor_history = list(resume_state.get("monitor_history", []))
        reward_history = list(resume_state.get("reward_history", []))
        loss_history = list(resume_state.get("loss_history", []))
        resume_epoch = int(resume_state["epoch"])
        resume_next_batch_idx = int(resume_state["next_batch_idx"])
        resumed_epoch_records = list(
            resume_state.get("global_epoch_records", []))
        rank_runtime_states = resume_state.get("rank_runtime_states", [])
        if len(rank_runtime_states) != WORLD_SIZE:
            raise RuntimeError(
                "resume state缺少完整rank运行状态："
                f"{len(rank_runtime_states)} != {WORLD_SIZE}")
        runtime_state = rank_runtime_states[GLOBAL_RANK]
        random.setstate(runtime_state["python_random_state"])
        torch.set_rng_state(runtime_state["torch_rng_state"])
        torch.cuda.set_rng_state(
            runtime_state["cuda_rng_state"], device=DEVICE)
        if IS_MAIN:
            print(
                f"[{datetime.now():%H:%M:%S}] 已恢复optimizer/scheduler/"
                f"β/RNG，继续 epoch={resume_epoch + 1}, "
                f"batch_idx={resume_next_batch_idx}")
    drift_run_start = None
    drift_window_start = None
    if IS_MAIN:
        drift_run_start = snapshot_trainable_parameters(named_trainable)
        drift_window_start = drift_run_start
        drift_parameter_count = sum(
            tensor.numel() for tensor in drift_run_start.values())
        drift_reference_label = (
            "resume进程起点" if resume_state is not None else "本次训练起点")
        print(f"精确权重漂移基线: {drift_reference_label}, "
              f"{len(drift_run_start)} tensors/{drift_parameter_count:,} params; "
              f"每{MONITOR_STEPS} slots审计一次")
    def save_restart_state(checkpoint_path, epoch_index, next_batch_idx,
                           epoch_records):
        """Save a rank-complete restart state at an accumulation boundary."""
        local_runtime_state = {
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(DEVICE).cpu(),
        }
        if is_ddp:
            gathered_runtime_states = (
                [None] * WORLD_SIZE if IS_MAIN else None)
            dist.gather_object(
                local_runtime_state, gathered_runtime_states, dst=0)
        else:
            gathered_runtime_states = [local_runtime_state]
        if IS_MAIN:
            state = {
                "format_version": 1,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "world_size": WORLD_SIZE,
                "policy_adapter_path": checkpoint_path,
                "reference_lora_path": REFERENCE_LORA_PATH,
                "policy_objective": POLICY_OBJECTIVE,
                "policy_token_denominator": POLICY_TOKEN_DENOMINATOR,
                "global_step": global_step,
                "optimizer_update_step": optimizer_update_step,
                "epoch": int(epoch_index),
                "next_batch_idx": int(next_batch_idx),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "current_beta": float(current_beta),
                "current_kl_safety_lr_scale": float(
                    current_kl_safety_lr_scale),
                "beta_history": beta_history,
                "monitor_history": monitor_history,
                "reward_history": reward_history,
                "loss_history": loss_history,
                "global_epoch_records": epoch_records,
                "rank_runtime_states": gathered_runtime_states,
                "training_script_sha256": file_sha256(__file__),
                "train_json_sha256": file_sha256(TRAIN_JSON),
            }
            state_path = os.path.join(
                checkpoint_path, "trainer_state.pt")
            torch.save(state, state_path)
            print(f"  → 可恢复训练状态已保存: {state_path}")
        if is_ddp:
            dist.barrier()

    for epoch in range(resume_epoch, GRPO_EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if IS_MAIN:
            print(f"\n{'=' * 60}")
            print(f"GRPO 训练 Epoch {epoch+1}/{GRPO_EPOCHS}")
            print(f"{'=' * 60}")

        global_epoch_records = (
            list(resumed_epoch_records)
            if resume_state is not None and epoch == resume_epoch
            else [])  # 仅rank0使用，来自所有GPU
        global_seen_task_counts = Counter(
            record["task"] for record in global_epoch_records)
        accum_count = 0    # 当前dynamic batch已消费的同步prompt轮数
        accum_has_update = False
        accum_effective_samples = 0
        accum_global_effective_samples = 0
        local_sample_records = []  # 每个 rank 本地缓存4条，step边界只 gather 一次
        # ── 监控计数器 ──
        mon = {"CE_targeted_no_hit": 0,
               "SKIP_overlong_insufficient": 0,
               "SKIP_no_signal": 0, "SKIP_no_improvement": 0,
               "SKIP_ddp_route_mismatch": 0,
               "SKIP_generation_timeout": 0,
               "PREFERENCE_only": 0, "GRPO": 0, "nogood_groups": 0,
               "targeted_no_hit_groups": 0,
               "sequence_preference_eligible": 0,
               "sequence_preference_applied": 0,
               "sequence_preference_identical_rejected": 0,
               "pairwise_preference_applied": 0,
               "greedy_invalid": 0, "KL_skip": 0, "anomaly_skip": 0,
               }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{GRPO_EPOCHS}") if IS_MAIN else dataloader
        for batch_idx, batch in enumerate(pbar):
            if (resume_state is not None and epoch == resume_epoch and
                    batch_idx < resume_next_batch_idx):
                del batch
                continue
            sample = batch[0]
            image       = sample["image"]
            image_path  = sample["image_path"]
            image_rel   = sample["image_rel"]
            prompt      = sample["prompt"]
            gt_response = sample["gt_response"]
            sample_task = classify_task(prompt)
            sample_max_tokens = max_tokens_for_bucket(
                sample_task, sample["cost_bucket"])

            # Step 1: 生成 G 组候选输出
            (candidate_texts, inputs, generated_ids, completion_mask,
             generation_runtime) = generate_candidates(
                 policy_model, processor, image, prompt, DEVICE,
                 num_candidates=G, max_new_tokens=sample_max_tokens)
            generation_timed_out = bool(generation_runtime["timed_out"])
            generation_diag = completion_generation_diagnostics(
                generated_ids, completion_mask, processor, sample_task,
                token_limit=sample_max_tokens)
            normalized_candidate_list = [
                re.sub(r"\s+", "", text) for text in candidate_texts]
            normalized_candidate_texts = set(normalized_candidate_list)
            candidate_unique_text_ratio = (
                len(normalized_candidate_texts) / max(len(candidate_texts), 1))
            normalized_sampled_texts = normalized_candidate_list[
                GREEDY_CANDIDATES:]
            sampled_unique_text_ratio = (
                len(set(normalized_sampled_texts)) /
                max(len(normalized_sampled_texts), 1))
            sampled_greedy_duplicate_rate = (
                sum(text == normalized_candidate_list[0]
                    for text in normalized_sampled_texts) /
                max(len(normalized_sampled_texts), 1))
            # eval模式不关闭梯度；保持rollout与policy-logprob的dropout状态完全一致。
            train_model.eval()
            policy_model.eval()
            torch.cuda.empty_cache()  # 释放 generate 的 KV cache

            # 先按评估口径解析全部候选。非空文本但无合法 bbox 也属于 invalid。
            reward_results, rewards_list = [], []
            for cand_idx, cand_text in enumerate(candidate_texts):
                r = compute_reward(cand_text, gt_response, image_path, prompt)
                length_penalty = soft_overlong_penalty(
                    sample_task, generation_diag["token_counts"][cand_idx],
                    token_limit=sample_max_tokens)
                r["base_total"] = float(r["total"])
                r["length_penalty"] = float(length_penalty)
                r["total"] = float(r["total"]) - length_penalty
                r["advantage_micro_raw"] = (
                    float(r["advantage_micro_raw"]) -
                    SOFT_OVERLONG_ADVANTAGE_SCALE * length_penalty)
                reward_results.append(r)
                rewards_list.append(r["total"])
            rewards_tensor = torch.tensor(
                rewards_list, device=DEVICE, dtype=torch.float32)
            advantage_micro_raw_tensor = torch.tensor(
                [r["advantage_micro_raw"] for r in reward_results],
                device=DEVICE, dtype=torch.float32)
            localization_tensor = torch.tensor(
                [r["localization"] for r in reward_results],
                device=DEVICE, dtype=torch.float32)
            greedy_reward = rewards_tensor[0].item()
            greedy_result = reward_results[0]
            sampled_indices = list(range(GREEDY_CANDIDATES, G))
            active_sampled_indices = [
                idx for idx in sampled_indices
                if not (MASK_TRUNCATED_COMPLETIONS and
                        bool(generation_diag["token_limit_reached"][idx]))]
            sampled_bbox_signatures = [
                reward_results[idx]["bbox_signature"]
                for idx in sampled_indices]
            sampled_bbox_unique_ratio = (
                len(set(sampled_bbox_signatures)) /
                max(len(sampled_bbox_signatures), 1))
            sampled_bbox_greedy_duplicate_rate = (
                sum(signature == greedy_result["bbox_signature"]
                    for signature in sampled_bbox_signatures) /
                max(len(sampled_bbox_signatures), 1))
            insufficient_active_sampled = (
                len(active_sampled_indices) <
                MIN_ACTIVE_SAMPLED_CANDIDATES)
            if active_sampled_indices:
                active_rewards = rewards_tensor[active_sampled_indices]
                r_mean = active_rewards.mean()
                r_std_raw = active_rewards.std(unbiased=False)
                active_micro_raw_std = advantage_micro_raw_tensor[
                    active_sampled_indices].std(unbiased=False)
                active_localization_std = localization_tensor[
                    active_sampled_indices].std(unbiased=False)
                best_idx = max(
                    active_sampled_indices,
                    key=lambda idx: float(rewards_tensor[idx].item()))
                metric_best_idx = metric_best_candidate_index(
                    reward_results, active_sampled_indices,
                    generation_diag["token_counts"])
                dense_best_idx = max(
                    active_sampled_indices,
                    key=lambda idx: (
                        float(reward_results[idx]["localization"]),
                        float(reward_results[idx]["micro_raw"]),
                        float(reward_results[idx]["total"]),
                        -idx))
            else:
                # 仅用于日志和CE回退；没有sampled rollout时绝不进入GRPO。
                r_mean = rewards_tensor[0]
                r_std_raw = torch.zeros_like(r_mean)
                active_micro_raw_std = torch.zeros_like(r_mean)
                active_localization_std = torch.zeros_like(r_mean)
                best_idx = 0
                metric_best_idx = 0
                dense_best_idx = 0
            micro_raw_zero_std = bool(
                len(active_sampled_indices) >=
                MIN_ACTIVE_SAMPLED_CANDIDATES and
                active_micro_raw_std.item() <= 1e-9)
            r_best = rewards_tensor[best_idx].item()
            best_result = reward_results[best_idx]
            metric_best_result = reward_results[metric_best_idx]
            dense_best_result = reward_results[dense_best_idx]
            any_hit_at_g = any(int(r["tp"]) > 0 for r in reward_results)
            perfect_at_g = any(
                int(r["tp"]) == int(r["gt_count"]) and
                int(r["fp"]) == 0 and int(r["fn"]) == 0
                for r in reward_results)
            sampled_any_hit = any(
                int(reward_results[idx]["tp"]) > 0
                for idx in sampled_indices)
            active_sampled_any_hit = any(
                int(reward_results[idx]["tp"]) > 0
                for idx in active_sampled_indices)
            sampled_perfect = any(
                int(reward_results[idx]["tp"]) ==
                int(reward_results[idx]["gt_count"]) and
                int(reward_results[idx]["fp"]) == 0 and
                int(reward_results[idx]["fn"]) == 0
                for idx in sampled_indices)
            sampled_rescue_at_g = (
                int(greedy_result["tp"]) == 0 and
                any(int(r["tp"]) > 0 for r in reward_results[1:]))
            sampled_better_than_greedy = sum(
                float(reward_results[idx]["micro_raw"]) >
                float(greedy_result["micro_raw"])
                for idx in sampled_indices)
            reward_metric_best_agreement = best_idx == metric_best_idx
            reward_best_metric_optimal = math.isclose(
                float(best_result["micro_raw"]),
                float(metric_best_result["micro_raw"]),
                rel_tol=0.0, abs_tol=1e-9)
            reward_best_not_worse_than_greedy = (
                float(best_result["micro_raw"]) >=
                float(greedy_result["micro_raw"]))
            sampled_metric_not_worse_than_greedy = (
                float(metric_best_result["micro_raw"]) >=
                float(greedy_result["micro_raw"]))
            metric_micro_raw_gap = (
                float(metric_best_result["micro_raw"]) -
                float(greedy_result["micro_raw"]))
            # v7 不再要求 sampled 先超过 greedy。greedy 是部署诊断而不是
            # behavior-policy baseline；只要 sampled 组自身存在可排序差异，就有
            # 标准 group-relative policy gradient。约一半旧
            # SKIP_no_improvement 因而重新进入 GRPO。
            sampled_group_has_signal = bool(
                len(active_sampled_indices) >=
                    MIN_ACTIVE_SAMPLED_CANDIDATES and
                (active_micro_raw_std.item() > 1.0e-9 or
                 r_std_raw.item() > 1.0e-9 or
                 active_localization_std.item() > 1.0e-9))
            learnable_sampled_improvement = sampled_group_has_signal
            num_valid = sum(
                reward_results[idx]["valid_count"] > 0
                for idx in active_sampled_indices)
            greedy_invalid = greedy_result["valid_count"] == 0
            # CGPC关闭SINGLE定向CE；保留条件计算仅用于现有诊断字段和路由审计。
            targeted_no_hit_ce = bool(
                TARGETED_NO_HIT_CE_ENABLED and
                sample_task in TARGETED_NO_HIT_CE_TASKS and
                int(greedy_result["tp"]) == 0 and
                not active_sampled_any_hit)
            sample_gt_count = greedy_result["gt_count"]
            sample_gt_weight = gt_scale_weight(sample_gt_count)
            if greedy_invalid:
                mon["greedy_invalid"] += 1

            lowvar_failure = not sampled_group_has_signal
            no_good_candidate = targeted_no_hit_ce
            if no_good_candidate:
                mon["nogood_groups"] += 1
            if targeted_no_hit_ce:
                mon["targeted_no_hit_groups"] += 1

            # ALL preference目标按外部micro边际量选择，防止把reward中recall等
            # 代理项的偏差直接固化到greedy策略。触顶走严格门；正常长度仅在
            # 高质量sampled显著优于greedy时进入，并在下方要求多候选支持。
            greedy_at_limit = bool(
                generation_diag["token_limit_reached"][0])
            greedy_has_eos = bool(generation_diag["eos_terminated"][0])
            overlong_gate_prompt = bool(
                sample_task == "ALL" and greedy_at_limit and
                not greedy_has_eos)
            preference_gate_prompt = sample_task == "ALL"
            preference_gate_rejections = Counter()
            qualified_preference_indices = []
            for idx in active_sampled_indices:
                rejection_reasons = (
                    all_preference_rejection_reasons(
                        sample_task, idx,
                        float(rewards_tensor[idx].item()), greedy_reward,
                        reward_results[idx], greedy_result,
                        greedy_at_limit, greedy_has_eos,
                        bool(generation_diag["token_limit_reached"][idx]),
                        bool(generation_diag["eos_terminated"][idx])))
                if rejection_reasons:
                    if preference_gate_prompt:
                        preference_gate_rejections.update(rejection_reasons)
                else:
                    qualified_preference_indices.append(idx)
            distill_idx = (
                metric_best_candidate_index(
                    reward_results, qualified_preference_indices,
                    generation_diag["token_counts"])
                if qualified_preference_indices else metric_best_idx)
            sequence_preference_kind = "none"
            sequence_preference_eligible = bool(
                qualified_preference_indices)
            # Route selection happens below, so reject an impossible identical
            # pair here rather than discovering it after GRPO was disabled.
            if sequence_preference_eligible:
                chosen_len = int(
                    completion_mask[distill_idx].bool().sum().item())
                greedy_len = int(completion_mask[0].bool().sum().item())
                pair_is_identical = bool(
                    chosen_len == greedy_len and
                    torch.equal(
                        generated_ids[distill_idx, :chosen_len],
                        generated_ids[0, :greedy_len]))
                if pair_is_identical:
                    sequence_preference_eligible = False
                    sequence_preference_kind = "identical_rejected"
                    mon["sequence_preference_identical_rejected"] += 1
            preference_bbox_support_count = 0
            preference_metric_support_count = 0
            preference_support_scale = 0.0
            if sequence_preference_eligible:
                chosen_signature = reward_results[distill_idx][
                    "bbox_signature"]
                chosen_metric = tuple(
                    int(reward_results[distill_idx][key])
                    for key in ("tp", "fp", "fn"))
                preference_bbox_support_count = sum(
                    reward_results[idx]["bbox_signature"] == chosen_signature
                    for idx in qualified_preference_indices)
                preference_metric_support_count = sum(
                    tuple(int(reward_results[idx][key])
                          for key in ("tp", "fp", "fn")) == chosen_metric
                    for idx in qualified_preference_indices)
                if (not overlong_gate_prompt and
                        max(preference_bbox_support_count,
                            preference_metric_support_count) <
                        ALL_QUALITY_PREFERENCE_MIN_SUPPORT):
                    sequence_preference_eligible = False
                    sequence_preference_kind = "support_rejected"
                    preference_gate_rejections["quality_support"] += 1
            if sequence_preference_eligible:
                sequence_preference_kind = (
                    "all_overlong" if overlong_gate_prompt
                    else "all_quality")
                mon["sequence_preference_eligible"] += 1
                preference_support_scale = (
                    1.0 if preference_bbox_support_count >= 2 else
                    0.75 if preference_metric_support_count >= 2 else 0.50)
            preference_first_base_weight = (
                ALL_OVERLONG_PREFERENCE_FIRST_WEIGHT
                if overlong_gate_prompt
                else ALL_QUALITY_PREFERENCE_FIRST_WEIGHT)
            preference_suffix_base_weight = (
                ALL_OVERLONG_PREFERENCE_SUFFIX_WEIGHT
                if overlong_gate_prompt
                else ALL_QUALITY_PREFERENCE_SUFFIX_WEIGHT)
            preference_pairwise_first_weight = (
                preference_first_base_weight *
                preference_support_scale
                if sequence_preference_eligible else 0.0)
            preference_pairwise_suffix_weight = (
                preference_suffix_base_weight *
                preference_support_scale
                if sequence_preference_eligible else 0.0)
            preference_chosen_has_positive_advantage = False
            preference_chosen_advantage = 0.0
            if sequence_preference_eligible and sampled_group_has_signal:
                preference_probe_advantages = centered_group_advantages(
                    rewards_tensor, advantage_micro_raw_tensor,
                    localization_tensor, active_sampled_indices,
                    sample_task, enable_localization=True)
                preference_chosen_advantage = float(
                    preference_probe_advantages[distill_idx].item())
                preference_chosen_has_positive_advantage = bool(
                    preference_chosen_advantage > 0.0)
                del preference_probe_advantages
            # chosen若不是sampled组内正优势，关闭该prompt普通GRPO，只保留
            # reference KL + first/suffix preference，避免同一轨迹正负冲突。
            preference_only_override = bool(
                sequence_preference_eligible and
                not preference_chosen_has_positive_advantage)

            # ── CGPC路径：SINGLE定向CE关闭；ALL可走严格安全偏好；其余正常GRPO ──
            sample_mode = "GRPO"
            sample_loss = 0.0
            sample_policy_loss = 0.0
            sample_kl = 0.0
            sample_token_nll = 0.0
            sample_has_update = False
            sample_kl_measured = False
            sample_kl_source = "none"
            pairwise_preference_applied = False
            sample_pairwise_chosen_kl = 0.0
            sample_pairwise_rejected_kl = 0.0
            sample_pairwise_divergence_token = -1
            sample_preference_first_loss = 0.0
            sample_preference_suffix_loss = 0.0
            sample_preference_first_coefficient = 0.0
            sample_preference_suffix_coefficient = 0.0
            structured_dapo_token_cap = 0
            sample_advantage_diag = {
                "normalized_adv_abs_mean": 0.0,
                "micro_raw_adv_abs_mean": 0.0,
                "localization_adv_abs_mean": 0.0,
                "total_adv_abs_mean": 0.0,
                "total_adv_abs_sum": 0.0,
                "total_adv_max_abs": 0.0,
                "micro_raw_clip_scale": 1.0,
                "micro_raw_clip_triggered": False,
                "task_advantage_gain": float(
                    TASK_ADVANTAGE_GAIN[sample_task]),
            }
            active_token_counts = [
                int(generation_diag["token_counts"][idx])
                for idx in active_sampled_indices]
            shortest_sync_idx = min(
                sampled_indices,
                key=lambda idx: int(
                    generation_diag["token_counts"][idx]))
            active_sampled_tokens = sum(active_token_counts)
            active_token_square_sum = sum(
                count * count for count in active_token_counts)
            dapo_effective_candidates = (
                active_sampled_tokens * active_sampled_tokens /
                active_token_square_sum
                if active_token_square_sum > 0 else 0.0)
            localization_advantage_enabled = True
            if (len(active_sampled_indices) >=
                    MIN_ACTIVE_SAMPLED_CANDIDATES and
                    learnable_sampled_improvement and
                    not preference_only_override):
                precomputed_advantages, sample_advantage_diag = (
                    centered_group_advantages(
                        rewards_tensor, advantage_micro_raw_tensor,
                        localization_tensor, active_sampled_indices,
                        sample_task,
                        enable_localization=
                            localization_advantage_enabled,
                        return_diagnostics=True))
            else:
                precomputed_advantages = torch.zeros_like(rewards_tensor)
            no_signal_group = bool(
                learnable_sampled_improvement and
                not preference_only_override and
                not sequence_preference_eligible and
                sample_advantage_diag["total_adv_abs_sum"] <
                    MIN_TOTAL_ADV_ABS)

            # DDP要求每个rank以完全相同的顺序进入collective。各rank处理不同
            # prompt，因此本地可能分别路由到SKIP、CE或GRPO；不能让一侧进入
            # G候选backward流程而另一侧直接走一次零梯度同步。先用固定的一次
            # one-hot collective对齐路径类型；只要路径类型不一致，两边都跳过
            # 当前样本对。候选级KL不再执行任何跨rank collective。
            # 动态有效批量会补足被跳过的prompt，不改变optimizer step的目标配额。
            if generation_timed_out:
                local_route_class = 0  # SKIP
            elif (insufficient_active_sampled and
                    not preference_only_override):
                local_route_class = 0  # SKIP
            elif targeted_no_hit_ce and not preference_only_override:
                local_route_class = 1  # CE
            elif no_signal_group and not preference_only_override:
                local_route_class = 0  # SKIP
            elif ((not learnable_sampled_improvement or
                   num_valid == 0 or lowvar_failure or
                   no_good_candidate) and
                  not preference_only_override):
                local_route_class = 0  # SKIP
            elif num_valid > 0:
                local_route_class = 2  # GRPO / preference
            else:
                local_route_class = 0  # defensive SKIP

            ddp_route_mismatch = False
            global_generation_timeout = generation_timed_out
            if is_ddp:
                route_votes = torch.zeros(
                    4, device=DEVICE, dtype=torch.int32)
                route_votes[local_route_class] = 1
                route_votes[3] = int(generation_timed_out)
                dist.all_reduce(route_votes, op=dist.ReduceOp.SUM)
                ddp_route_mismatch = bool(
                    int((route_votes[:3] > 0).sum().item()) != 1)
                global_generation_timeout = bool(route_votes[3].item() > 0)
                del route_votes

            if global_generation_timeout:
                zero_policy_sync_backward(
                    train_model, inputs, generated_ids, completion_mask,
                    shortest_sync_idx)
                sample_mode = "SKIP_generation_timeout"
                mon["SKIP_generation_timeout"] += 1
                del inputs, generated_ids, completion_mask
                torch.cuda.empty_cache()

            elif ddp_route_mismatch:
                zero_policy_sync_backward(
                    train_model, inputs, generated_ids, completion_mask,
                    shortest_sync_idx)
                sample_mode = "SKIP_ddp_route_mismatch"
                mon["SKIP_ddp_route_mismatch"] += 1
                del inputs, generated_ids, completion_mask
                torch.cuda.empty_cache()

            elif (insufficient_active_sampled and
                  not preference_only_override):
                # 有效候选不足通常是 sampled 触顶过滤造成的。用 GT CE 回退会把
                # 过长目标/循环再次写回策略；因此只做一次零梯度同步并等待后续
                # prompt 的有效 on-policy 信号。
                zero_policy_sync_backward(
                    train_model, inputs, generated_ids, completion_mask,
                    shortest_sync_idx)
                sample_mode = "SKIP_overlong_insufficient"
                mon["SKIP_overlong_insufficient"] += 1
                del inputs, generated_ids, completion_mask
                torch.cuda.empty_cache()

            elif targeted_no_hit_ce and not preference_only_override:
                # 仅SINGLE整组无TP时，以极弱GT suffix CE提供可达方向；其余
                # 空/低方差/无改进组保持SKIP，不让GRPO重新退化成广义SFT。
                ce_loss = compute_sft_ce_loss(
                    train_model, processor, image, prompt,
                    gt_response, DEVICE,
                    rejected_response=candidate_texts[0],
                    suffix_only=True)
                ce_token_nll = ce_loss.detach()
                ce_weight = TARGETED_NO_HIT_CE_WEIGHT[sample_task]
                sample_mode = "CE_targeted_no_hit"
                ce_objective = ce_weight * sample_gt_weight * ce_loss
                scaled_ce = ce_objective / GRADIENT_ACCUMULATION_STEPS
                ce_is_safe = (torch.isfinite(scaled_ce).item() and
                              abs(ce_objective.detach().item()) <=
                              ABNORMAL_LOSS)
                if ce_is_safe:
                    scaled_ce.backward()
                    sample_has_update = True
                    sample_token_nll = float(
                        ce_token_nll.detach().item())
                else:
                    (0.0 * ce_loss).backward()
                    mon["anomaly_skip"] += 1

                sample_loss = (
                    float(ce_objective.detach().item())
                    if ce_is_safe else 0.0)
                mon[sample_mode] += 1
                del inputs, generated_ids, completion_mask, ce_loss
                del ce_token_nll
                del ce_objective, scaled_ce
                torch.cuda.empty_cache()

            elif no_signal_group and not preference_only_override:
                zero_policy_sync_backward(
                    train_model, inputs, generated_ids, completion_mask,
                    shortest_sync_idx)
                sample_mode = "SKIP_no_signal"
                mon["SKIP_no_signal"] += 1
                del inputs, generated_ids, completion_mask
                torch.cuda.empty_cache()

            elif (not learnable_sampled_improvement or
                  num_valid == 0 or lowvar_failure or
                  no_good_candidate) and not (
                      preference_only_override):
                # sampled组无可排序差异；仅复用最短rollout维持DDP collective。
                zero_policy_sync_backward(
                    train_model, inputs, generated_ids, completion_mask,
                    shortest_sync_idx)
                sample_mode = "SKIP_no_improvement"
                mon["SKIP_no_improvement"] += 1
                del inputs, generated_ids, completion_mask
                torch.cuda.empty_cache()

            elif num_valid > 0:
                # ── 路径 C：正常GRPO，或ALL专用KL+preference ──
                # 不再过滤空候选：空文本至少有一个 EOS token，log_probs 可正常计算。
                # 空输出在total与主micro优势通道均为负，能够参与正常组内对比。

                ref_log_probs = get_ref_log_probs(
                    ref_model, inputs, generated_ids,
                    completion_mask=completion_mask)
                torch.cuda.empty_cache()

                group_grpo_enabled = not preference_only_override
                # 只对真实sampled、非截断rollout做策略梯度。ALL preference-only路由
                # 使用全零优势，只保留reference KL与严格偏好，不强化低质量整组。
                if group_grpo_enabled:
                    all_advantages = precomputed_advantages
                else:
                    all_advantages = torch.zeros_like(rewards_tensor)
                gt_serialized_token_count = len(
                    processor.tokenizer(
                        gt_response,
                        add_special_tokens=False).input_ids)
                structured_dapo_token_cap = (
                    min(
                        TASK_MAX_TOKENS[sample_task],
                        max(
                            int(math.ceil(
                                gt_serialized_token_count *
                                STRUCTURED_DAPO_GT_RATIO)) +
                            STRUCTURED_DAPO_EXTRA_TOKENS,
                            1))
                    if STRUCTURED_DAPO_LENGTH_CAP else None)
                candidate_dapo_weights = dapo_candidate_weights(
                    completion_mask, active_sampled_indices,
                    token_cap=structured_dapo_token_cap)
                active_sampled_set = set(active_sampled_indices)

                pairwise_first_stats = {
                    "loss": 0.0,
                    "coefficient": 0.0,
                    "relative_margin": 0.0,
                    "kl": 0.0,
                    "chosen_kl": 0.0,
                    "rejected_kl": 0.0,
                    "pair_safe": False,
                }
                pairwise_suffix_stats = dict(pairwise_first_stats)
                chosen_first_mask = torch.zeros_like(
                    completion_mask[distill_idx], dtype=torch.bool)
                rejected_first_mask = torch.zeros_like(
                    completion_mask[0], dtype=torch.bool)
                chosen_suffix_mask = torch.zeros_like(chosen_first_mask)
                rejected_suffix_mask = torch.zeros_like(rejected_first_mask)
                if sequence_preference_eligible:
                    (chosen_first_mask, rejected_first_mask,
                     chosen_suffix_mask, rejected_suffix_mask,
                     sample_pairwise_divergence_token) = (
                        preference_suffix_masks(
                            generated_ids, completion_mask,
                            distill_idx, 0))
                    if sample_pairwise_divergence_token < 0:
                        sequence_preference_eligible = False
                        sequence_preference_kind = "identical_rejected"
                        preference_only_override = False
                        group_grpo_enabled = True
                        if sampled_group_has_signal:
                            all_advantages = centered_group_advantages(
                                rewards_tensor,
                                advantage_micro_raw_tensor,
                                localization_tensor,
                                active_sampled_indices,
                                sample_task,
                                enable_localization=True)
                        preference_pairwise_first_weight = 0.0
                        preference_pairwise_suffix_weight = 0.0
                        mon["sequence_preference_eligible"] -= 1
                        mon[
                            "sequence_preference_identical_rejected"] += 1
                suffix_pair_available = bool(
                    chosen_suffix_mask.any().item() and
                    rejected_suffix_mask.any().item())
                if (sequence_preference_eligible and
                        (preference_pairwise_first_weight > 0.0 or
                         preference_pairwise_suffix_weight > 0.0)):
                    chosen_policy_snapshot = candidate_policy_log_probs(
                        policy_model, inputs, generated_ids, distill_idx)
                    rejected_policy_snapshot = candidate_policy_log_probs(
                        policy_model, inputs, generated_ids, 0)
                    if preference_pairwise_first_weight > 0.0:
                        pairwise_first_stats = (
                            reference_relative_pairwise_stats(
                                chosen_policy_snapshot, rejected_policy_snapshot,
                                ref_log_probs[distill_idx:distill_idx + 1],
                                ref_log_probs[0:1],
                                chosen_first_mask, rejected_first_mask,
                                preference_pairwise_first_weight))
                    if (preference_pairwise_suffix_weight > 0.0 and
                            suffix_pair_available):
                        pairwise_suffix_stats = (
                            reference_relative_pairwise_stats(
                                chosen_policy_snapshot, rejected_policy_snapshot,
                                ref_log_probs[distill_idx:distill_idx + 1],
                                ref_log_probs[0:1],
                                chosen_suffix_mask, rejected_suffix_mask,
                                preference_pairwise_suffix_weight))
                    relevant_pair_stats = [
                        stats for stats, weight in (
                            (pairwise_first_stats,
                             preference_pairwise_first_weight),
                            (pairwise_suffix_stats,
                             (preference_pairwise_suffix_weight
                              if suffix_pair_available else 0.0)))
                        if weight > 0.0]
                    pair_safe = bool(
                        relevant_pair_stats and
                        all(stats["pair_safe"]
                            for stats in relevant_pair_stats))
                    if not pair_safe:
                        for stats in relevant_pair_stats:
                            stats["coefficient"] = 0.0
                            stats["loss"] = 0.0
                        mon["KL_skip"] += 1
                    sample_pairwise_chosen_kl = max(
                        stats["chosen_kl"] for stats in relevant_pair_stats)
                    sample_pairwise_rejected_kl = (
                        max(stats["rejected_kl"]
                            for stats in relevant_pair_stats))
                    del chosen_policy_snapshot, rejected_policy_snapshot
                pairwise_stats = {
                    "loss": (
                        pairwise_first_stats["loss"] +
                        pairwise_suffix_stats["loss"]),
                    "coefficient": (
                        pairwise_first_stats["coefficient"] +
                        pairwise_suffix_stats["coefficient"]),
                    "relative_margin":
                        pairwise_first_stats["relative_margin"],
                    "suffix_relative_margin":
                        pairwise_suffix_stats["relative_margin"],
                    "kl": max(
                        pairwise_first_stats["kl"],
                        pairwise_suffix_stats["kl"]),
                    "chosen_kl": sample_pairwise_chosen_kl,
                    "rejected_kl": sample_pairwise_rejected_kl,
                    "pair_safe": bool(
                        pairwise_first_stats["coefficient"] > 0.0 or
                        pairwise_suffix_stats["coefficient"] > 0.0),
                }
                sample_preference_first_loss = float(
                    pairwise_first_stats["loss"])
                sample_preference_suffix_loss = float(
                    pairwise_suffix_stats["loss"])
                sample_preference_first_coefficient = float(
                    pairwise_first_stats["coefficient"])
                sample_preference_suffix_coefficient = float(
                    pairwise_suffix_stats["coefficient"])

                # Mini-batch GRPO loss + backward（遍历全部 G 个候选）
                MINI_BATCH = 1
                total_loss = 0.0; total_policy_loss = 0.0
                total_kl = 0.0; total_token_nll = 0.0
                pair_chosen_surrogate_applied = False
                pair_rejected_surrogate_applied = False

                for mb_start in range(0, G, MINI_BATCH):
                    mb_end = min(mb_start + MINI_BATCH, G)
                    mb_idx = list(range(mb_start, mb_end))
                    mb_inputs = {}
                    for k in ["input_ids", "attention_mask"]:
                        mb_inputs[k] = inputs[k][mb_idx] if inputs.get(k) is not None else None
                    mb_pv, mb_gw = _slice_vision_inputs(
                        inputs.get("pixel_values"), inputs.get("image_grid_thw"), mb_idx)
                    mb_inputs["pixel_values"] = mb_pv
                    mb_inputs["image_grid_thw"] = mb_gw
                    mb_gen_ids = generated_ids[mb_idx]
                    mb_ref_lp  = ref_log_probs[mb_idx]
                    mb_comp_mask = completion_mask[mb_idx]
                    mb_rewards = rewards_tensor[mb_idx]
                    mb_adv     = all_advantages[mb_idx]
                    candidate_is_active = mb_start in active_sampled_set
                    mb_dapo_weight = float(
                        candidate_dapo_weights[mb_start].item())

                    # 每个样本只在最后一个候选触发一次 DDP all-reduce；CE/GRPO
                    # 分支因而拥有相同的同步次数，不会因 rank 路径不同而错位。
                    sync_ctx = (train_model.no_sync()
                                if is_ddp and mb_end < G else nullcontext())
                    with sync_ctx:
                        mb_policy_lp = get_policy_log_probs(
                            train_model, mb_inputs, mb_gen_ids,
                            completion_mask=mb_comp_mask)
                        mlen = min(mb_policy_lp.shape[1], mb_ref_lp.shape[1],
                                   mb_comp_mask.shape[1])
                        if mlen > 0:
                            mb_policy_lp = mb_policy_lp[:, :mlen]
                            mb_ref_lp = mb_ref_lp[:, :mlen]
                            mb_comp_mask = mb_comp_mask[:, :mlen]

                        with torch.no_grad():
                            if mlen > 0:
                                _mask = mb_comp_mask.float()
                                _den = _mask.sum().clamp_min(1.0)
                                _d = (mb_ref_lp - mb_policy_lp).clamp(
                                    -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
                                _k3 = torch.exp(_d) - _d - 1.0
                                _kl_est = ((_k3 * _mask).sum() / _den).item()
                            else:
                                _kl_est = 0.0
                        # 路由类型已在进入本分支前完成固定的一次跨rank握手。
                        # 候选KL只做本地判定，避免每个prompt额外插入G次标量
                        # all-reduce；最后候选无目标时由全参数零梯度backward
                        # 完成与另一rank真实backward相同的DDP reducer同步。
                        candidate_kl_safe = bool(
                            math.isfinite(_kl_est) and
                            _kl_est <= KL_SKIP_THRESHOLD)
                        if candidate_is_active and not candidate_kl_safe:
                            mon["KL_skip"] += 1

                        mb_loss = None
                        mb_optimized_loss = 0.0 * mb_policy_lp.sum()
                        mb_has_chosen_pair = False
                        mb_has_rejected_pair = False
                        mb_has_objective = False
                        mb_metrics = {"loss": 0.0, "policy_loss": 0.0,
                                      "kl_div": 0.0, "token_nll": 0.0}
                        if (candidate_is_active and mlen > 0 and
                                candidate_kl_safe):
                            mb_loss, mb_metrics = compute_grpo_loss(
                                mb_policy_lp, mb_ref_lp, mb_rewards,
                                mb_comp_mask, beta=current_beta,
                                advantages=mb_adv)
                            mb_optimized_loss = mb_dapo_weight * mb_loss
                            mb_has_objective = True

                        # preference复用本prompt已有候选forward并合入同一loss；
                        # first直接改变argmax，suffix稳定JSON闭合/EOS，两者mask不重叠。
                        if (sequence_preference_eligible and mlen > 0 and
                                candidate_kl_safe):
                            if mb_start <= distill_idx < mb_end:
                                chosen_pair_surrogate = (
                                    0.0 * mb_policy_lp.sum())
                                if (pairwise_first_stats["coefficient"] >
                                        0.0):
                                    chosen_first_logp = masked_sequence_mean(
                                        mb_policy_lp, chosen_first_mask)
                                    chosen_pair_surrogate = (
                                        chosen_pair_surrogate -
                                        pairwise_first_stats["coefficient"] *
                                        chosen_first_logp)
                                if (pairwise_suffix_stats["coefficient"] >
                                        0.0):
                                    chosen_suffix_logp = masked_sequence_mean(
                                        mb_policy_lp, chosen_suffix_mask)
                                    chosen_pair_surrogate = (
                                        chosen_pair_surrogate -
                                        pairwise_suffix_stats["coefficient"] *
                                        chosen_suffix_logp)
                                if pairwise_stats["coefficient"] > 0.0:
                                    mb_optimized_loss = (
                                        mb_optimized_loss +
                                        chosen_pair_surrogate)
                                    mb_has_chosen_pair = True
                                    mb_has_objective = True
                            if (mb_start == 0 and
                                    pairwise_stats["coefficient"] > 0.0 and
                                    pairwise_stats["pair_safe"]):
                                rejected_pair_surrogate = (
                                    0.0 * mb_policy_lp.sum())
                                if (pairwise_first_stats["coefficient"] >
                                        0.0):
                                    rejected_first_logp = masked_sequence_mean(
                                        mb_policy_lp, rejected_first_mask)
                                    rejected_pair_surrogate = (
                                        rejected_pair_surrogate +
                                        pairwise_first_stats["coefficient"] *
                                        rejected_first_logp)
                                if (pairwise_suffix_stats["coefficient"] >
                                        0.0):
                                    rejected_suffix_logp = (
                                        masked_sequence_mean(
                                            mb_policy_lp,
                                            rejected_suffix_mask))
                                    rejected_pair_surrogate = (
                                        rejected_pair_surrogate +
                                        pairwise_suffix_stats["coefficient"] *
                                        rejected_suffix_logp)
                                mb_optimized_loss = (
                                    mb_optimized_loss +
                                    rejected_pair_surrogate)
                                mb_has_objective = True
                                mb_has_rejected_pair = True

                        if mb_has_objective:
                            loss_is_safe = (
                                torch.isfinite(mb_optimized_loss).item() and
                                abs(mb_optimized_loss.detach().item()) <=
                                ABNORMAL_LOSS)
                            if loss_is_safe:
                                (mb_optimized_loss /
                                 GRADIENT_ACCUMULATION_STEPS).backward()
                                sample_has_update = True
                                pair_chosen_surrogate_applied = (
                                    pair_chosen_surrogate_applied or
                                    mb_has_chosen_pair)
                                pair_rejected_surrogate_applied = (
                                    pair_rejected_surrogate_applied or
                                    mb_has_rejected_pair)
                            else:
                                mon["anomaly_skip"] += 1
                                mb_metrics = {
                                    "loss": 0.0, "policy_loss": 0.0,
                                    "kl_div": 0.0, "token_nll": 0.0}
                        if not mb_has_objective or not loss_is_safe:
                            # 前G-1个候选位于no_sync，过滤后可直接释放计算图。
                            # 最后候选必须完成一次reducer同步；但绝不能再对长
                            # completion执行(0 * log_probs).backward()，乘零仍会
                            # 遍历完整softmax/logit图并在ALL:140上触发OOM。
                            if is_ddp and mb_end == G:
                                if mb_loss is not None:
                                    del mb_loss
                                    mb_loss = None
                                if mb_optimized_loss is not None:
                                    del mb_optimized_loss
                                    mb_optimized_loss = None
                                del mb_policy_lp
                                mb_policy_lp = None
                                torch.cuda.empty_cache()
                                zero_parameter_sync_backward(train_model)

                    total_loss += mb_dapo_weight * mb_metrics["loss"]
                    total_policy_loss += (
                        mb_dapo_weight * mb_metrics["policy_loss"])
                    total_kl += mb_dapo_weight * mb_metrics["kl_div"]
                    total_token_nll += (
                        mb_dapo_weight * mb_metrics["token_nll"])
                    if mb_loss is not None:
                        del mb_loss
                    if mb_optimized_loss is not None:
                        del mb_optimized_loss
                    if mb_policy_lp is not None:
                        del mb_policy_lp
                    del mb_inputs, mb_pv, mb_gw

                pairwise_preference_applied = bool(
                    pair_chosen_surrogate_applied and
                    pair_rejected_surrogate_applied)
                if pairwise_preference_applied:
                    mon["pairwise_preference_applied"] += 1
                    mon["sequence_preference_applied"] += 1
                loss_metrics = {
                    # pairwise surrogate has the exact bounded-gradient
                    # but a different scalar value; report the true logistic
                    # loss once instead of summing its split linear surrogates.
                    "loss":        (
                        total_loss + pairwise_stats["loss"]),
                    "policy_loss": total_policy_loss,
                    "kl_div":      total_kl,
                    "token_nll":   total_token_nll,
                }
                sample_loss = loss_metrics["loss"]
                sample_policy_loss = loss_metrics["policy_loss"]
                sample_kl = max(
                    float(loss_metrics["kl_div"]),
                    (float(pairwise_stats["kl"])
                     if pairwise_preference_applied else 0.0))
                sample_kl_measured = bool(sample_has_update)
                sample_kl_source = (
                    "grpo+preference" if sequence_preference_eligible
                    else "grpo")
                sample_token_nll = loss_metrics["token_nll"]

                # 候选级异常已在 backward 前零梯度跳过；此处仅做聚合一致性检查。
                if (math.isnan(loss_metrics["loss"]) or math.isinf(loss_metrics["loss"]) or
                    abs(loss_metrics["loss"]) > ABNORMAL_LOSS):
                    if IS_MAIN:
                        print(f"[{datetime.now():%H:%M:%S}] ⚠ 聚合 loss 异常="
                              f"{loss_metrics['loss']:.2f}（候选已逐个防护）")

                if preference_only_override:
                    sample_mode = "PREFERENCE_only"
                    sample_kl_source = "all_preference_only"
                    mon["PREFERENCE_only"] += 1
                else:
                    sample_mode = "GRPO"
                    mon["GRPO"] += 1

                del inputs, generated_ids, completion_mask
                del ref_log_probs, all_advantages
                del chosen_suffix_mask, rejected_suffix_mask
                torch.cuda.empty_cache()

            # DAPO式动态有效批量：先在全rank统计本轮真实更新组。零优势轮仍完成
            # 必要的DDP零梯度同步，但不占用目标effective prompt配额。
            round_update = torch.tensor(
                [int(sample_has_update)], device=DEVICE, dtype=torch.int32)
            if is_ddp:
                dist.all_reduce(round_update, op=dist.ReduceOp.SUM)
            global_round_updates = int(round_update.item())
            accum_count += 1
            accum_global_effective_samples += global_round_updates
            accum_has_update = accum_has_update or sample_has_update
            accum_effective_samples += int(sample_has_update)
            if (sample_kl_measured and sample_has_update and
                    sample_kl_source in KL_CONTROLLER_SOURCES and
                    math.isfinite(sample_kl)):
                kl_window_local_sum += float(sample_kl)
                kl_window_local_count += 1

            # 每个微样本均记录；DDP 下汇集所有 rank，避免只看到 rank0 的最后一个样本。
            component_mean = {
                key: sum(float(r[key]) for r in reward_results) / max(len(reward_results), 1)
                for key in ("micro", "micro_raw", "advantage_micro_raw",
                            "iou", "recall", "precision", "f1",
                            "count", "label",
                            "center", "soft_iou", "localization",
                            "length_penalty", "under_penalty", "over_penalty",
                            "fn_ratio", "fp_ratio", "invalid_count")
            }
            logged_preference_idx = (
                int(distill_idx)
                if sequence_preference_eligible else -1)
            logged_preference_result = (
                reward_results[logged_preference_idx]
                if logged_preference_idx >= 0 else None)
            local_sample_records.append({
                "rank": GLOBAL_RANK,
                "accum_slot": int(accum_count),
                "idx": int(sample["idx"]),
                "image_rel": image_rel,
                "task": sample_task,
                "cost_bucket": sample["cost_bucket"],
                "generation_token_budget": int(sample_max_tokens),
                "generation_timed_out": bool(global_generation_timeout),
                "generation_local_timed_out": bool(generation_timed_out),
                "generation_elapsed_seconds": float(
                    generation_runtime["elapsed_seconds"]),
                "generation_timeout_seconds": float(
                    generation_runtime["timeout_seconds"]),
                "mode": sample_mode,
                "loss": float(sample_loss),
                "policy_loss": float(sample_policy_loss),
                "reward_mean": float(r_mean.item()),
                "reward_std": float(r_std_raw.item()),
                "reward_best": float(r_best),
                "reward_worst": float(
                    rewards_tensor[active_sampled_indices].min().item()
                    if active_sampled_indices else greedy_reward),
                "greedy_reward": float(rewards_tensor[0].item()),
                "greedy_micro": float(greedy_result["micro"]),
                "greedy_iou": float(greedy_result["iou"]),
                "greedy_soft_iou": float(greedy_result["soft_iou"]),
                "greedy_localization": float(greedy_result["localization"]),
                "greedy_length_penalty": float(
                    greedy_result["length_penalty"]),
                "greedy_recall": float(greedy_result["recall"]),
                "greedy_precision": float(greedy_result["precision"]),
                "greedy_f1": float(greedy_result["f1"]),
                "greedy_fp_ratio": float(greedy_result["fp_ratio"]),
                "greedy_fn_ratio": float(greedy_result["fn_ratio"]),
                "greedy_pred_count": float(greedy_result["pred_count"]),
                "greedy_tp": int(greedy_result["tp"]),
                "greedy_fp": int(greedy_result["fp"]),
                "greedy_fn": int(greedy_result["fn"]),
                "greedy_empty_text": bool(greedy_result["empty_text"]),
                "greedy_invalid_json": bool(greedy_result["invalid_json"]),
                "greedy_truncated": bool(greedy_result["truncated"]),
                "greedy_no_valid_box": bool(greedy_result["no_valid_box"]),
                "greedy_completion_tokens": int(
                    generation_diag["token_counts"][0]),
                "greedy_eos": bool(
                    generation_diag["eos_terminated"][0]),
                "greedy_token_limit": bool(
                    generation_diag["token_limit_reached"][0]),
                "best_idx": int(best_idx),
                "best_source": candidate_source(sample_task, best_idx),
                "best_micro": float(best_result["micro"]),
                "best_iou": float(best_result["iou"]),
                "best_recall": float(best_result["recall"]),
                "best_precision": float(best_result["precision"]),
                "best_f1": float(best_result["f1"]),
                "best_fp_ratio": float(best_result["fp_ratio"]),
                "best_fn_ratio": float(best_result["fn_ratio"]),
                "best_pred_count": float(best_result["pred_count"]),
                "best_tp": int(best_result["tp"]),
                "best_fp": int(best_result["fp"]),
                "best_fn": int(best_result["fn"]),
                "metric_best_idx": int(metric_best_idx),
                "metric_best_source": candidate_source(
                    sample_task, metric_best_idx),
                "metric_best_tp": int(metric_best_result["tp"]),
                "metric_best_fp": int(metric_best_result["fp"]),
                "metric_best_fn": int(metric_best_result["fn"]),
                "metric_best_f1": float(metric_best_result["f1"]),
                "metric_best_micro_raw": float(
                    metric_best_result["micro_raw"]),
                "metric_micro_raw_gap": float(metric_micro_raw_gap),
                "sampled_metric_not_worse_than_greedy": bool(
                    sampled_metric_not_worse_than_greedy),
                "dense_best_idx": int(dense_best_idx),
                "dense_best_localization": float(
                    dense_best_result["localization"]),
                "dense_greedy_gap": float(
                    dense_best_result["localization"] -
                    greedy_result["localization"]),
                "learnable_sampled_improvement": bool(
                    learnable_sampled_improvement),
                "active_sampled_candidates": int(
                    len(active_sampled_indices)),
                "active_sampled_tokens": int(active_sampled_tokens),
                "dapo_effective_candidates": float(
                    dapo_effective_candidates),
                "localization_advantage_enabled": bool(
                    localization_advantage_enabled),
                "no_signal_group": bool(no_signal_group),
                "masked_truncated_candidates": int(
                    len(sampled_indices) - len(active_sampled_indices)),
                "masked_overlong_candidates": int(
                    len(sampled_indices) - len(active_sampled_indices)),
                "any_hit_at_g": bool(any_hit_at_g),
                "perfect_at_g": bool(perfect_at_g),
                "sampled_any_hit": bool(sampled_any_hit),
                "active_sampled_any_hit": bool(active_sampled_any_hit),
                "targeted_no_hit_ce": bool(targeted_no_hit_ce),
                "sequence_preference_kind": sequence_preference_kind,
                "sequence_preference_eligible": bool(
                    sequence_preference_eligible),
                "sequence_preference_applied": bool(
                    pairwise_preference_applied),
                "preference_only": bool(preference_only_override),
                "preference_qualified_candidates": int(
                    len(qualified_preference_indices)),
                "preference_gate_rejections": dict(
                    preference_gate_rejections),
                "preference_chosen_idx": int(logged_preference_idx),
                "preference_chosen_advantage": float(
                    preference_chosen_advantage),
                "preference_chosen_reward": (
                    float(rewards_tensor[logged_preference_idx].item())
                    if logged_preference_idx >= 0 else None),
                "preference_chosen_f1": (
                    float(logged_preference_result["f1"])
                    if logged_preference_result is not None else None),
                "preference_chosen_precision": (
                    float(logged_preference_result["precision"])
                    if logged_preference_result is not None else None),
                "preference_chosen_recall": (
                    float(logged_preference_result["recall"])
                    if logged_preference_result is not None else None),
                "preference_chosen_tp": (
                    int(logged_preference_result["tp"])
                    if logged_preference_result is not None else None),
                "preference_chosen_fp": (
                    int(logged_preference_result["fp"])
                    if logged_preference_result is not None else None),
                "preference_chosen_fn": (
                    int(logged_preference_result["fn"])
                    if logged_preference_result is not None else None),
                "preference_chosen_tokens": (
                    int(generation_diag["token_counts"][
                        logged_preference_idx])
                    if logged_preference_idx >= 0 else None),
                "preference_chosen_eos": (
                    bool(generation_diag["eos_terminated"][
                        logged_preference_idx])
                    if logged_preference_idx >= 0 else None),
                "preference_bbox_support_count": int(
                    preference_bbox_support_count),
                "preference_metric_support_count": int(
                    preference_metric_support_count),
                "preference_support_scale": float(
                    preference_support_scale),
                "preference_chosen_positive_advantage": bool(
                    preference_chosen_has_positive_advantage),
                "preference_divergence_token": int(
                    sample_pairwise_divergence_token),
                "preference_first_loss": float(
                    sample_preference_first_loss),
                "preference_suffix_loss": float(
                    sample_preference_suffix_loss),
                "preference_first_coefficient": float(
                    sample_preference_first_coefficient),
                "preference_suffix_coefficient": float(
                    sample_preference_suffix_coefficient),
                "preference_chosen_kl": float(
                    sample_pairwise_chosen_kl),
                "preference_rejected_kl": float(
                    sample_pairwise_rejected_kl),
                "sampled_perfect": bool(sampled_perfect),
                "sampled_rescue_at_g": bool(sampled_rescue_at_g),
                "sampled_better_than_greedy": int(
                    sampled_better_than_greedy),
                "reward_metric_best_agreement": bool(
                    reward_metric_best_agreement),
                "reward_best_metric_optimal": bool(
                    reward_best_metric_optimal),
                "reward_best_not_worse_than_greedy": bool(
                    reward_best_not_worse_than_greedy),
                "candidate_unique_text_ratio": float(
                    candidate_unique_text_ratio),
                "sampled_unique_text_ratio": float(
                    sampled_unique_text_ratio),
                "sampled_greedy_duplicate_rate": float(
                    sampled_greedy_duplicate_rate),
                "sampled_bbox_unique_ratio": float(
                    sampled_bbox_unique_ratio),
                "sampled_bbox_greedy_duplicate_rate": float(
                    sampled_bbox_greedy_duplicate_rate),
                "active_micro_raw_std": float(
                    active_micro_raw_std.item()),
                "micro_raw_zero_std": bool(micro_raw_zero_std),
                "candidate_any_empty_text": any(
                    bool(r["empty_text"]) for r in reward_results),
                "candidate_any_invalid_json": any(
                    bool(r["invalid_json"]) for r in reward_results),
                "candidate_any_truncated": any(
                    bool(r["truncated"]) for r in reward_results),
                "candidate_any_no_valid_box": any(
                    bool(r["no_valid_box"]) for r in reward_results),
                "candidate_completion_tokens": list(
                    generation_diag["token_counts"]),
                "candidate_eos_terminated": list(
                    generation_diag["eos_terminated"]),
                "candidate_token_limit_reached": list(
                    generation_diag["token_limit_reached"]),
                "candidate_any_token_limit": any(
                    generation_diag["token_limit_reached"]),
                "max_candidate_completion_tokens": max(
                    generation_diag["token_counts"], default=0),
                "candidate_rewards": [float(v) for v in rewards_list],
                "candidate_length_penalties": [
                    float(r["length_penalty"]) for r in reward_results],
                "best_greedy_gap": float(r_best - greedy_reward),
                "num_valid": int(num_valid),
                "greedy_invalid": bool(greedy_invalid),
                "kl": float(sample_kl),
                "kl_measured": bool(sample_kl_measured),
                "kl_source": sample_kl_source,
                "token_nll": float(sample_token_nll),
                "gt_count": sum(float(r["gt_count"]) for r in reward_results) /
                            max(len(reward_results), 1),
                "pred_count": sum(float(r["pred_count"]) for r in reward_results) /
                              max(len(reward_results), 1),
                "micro": component_mean["micro"],
                "micro_raw": component_mean["micro_raw"],
                "advantage_micro_raw":
                    component_mean["advantage_micro_raw"],
                "iou": component_mean["iou"],
                "recall": component_mean["recall"],
                "precision": component_mean["precision"],
                "f1": component_mean["f1"],
                "count": component_mean["count"],
                "label": component_mean["label"],
                "center": component_mean["center"],
                "soft_iou": component_mean["soft_iou"],
                "localization": component_mean["localization"],
                "length_penalty": component_mean["length_penalty"],
                "under_penalty": component_mean["under_penalty"],
                "over_penalty": component_mean["over_penalty"],
                "fn_ratio": component_mean["fn_ratio"],
                "fp_ratio": component_mean["fp_ratio"],
                "invalid_count": component_mean["invalid_count"],
                **sample_advantage_diag,
                "has_update": bool(sample_has_update),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

            # ── 统一梯度累积检查（所有路径均会到达）──
            is_last_batch = batch_idx + 1 == len(dataloader)
            dynamic_step_now, dynamic_step_reason = (
                dynamic_batch_should_step(
                    accum_global_effective_samples,
                    accum_count, is_last_batch=is_last_batch))
            dynamic_batch_ready = dynamic_step_reason == "target"
            dynamic_batch_cap = dynamic_step_reason == "cap"
            if dynamic_step_now:
                update_stats = torch.tensor(
                    [int(accum_has_update), int(accum_effective_samples)],
                    device=DEVICE, dtype=torch.int32)
                if is_ddp:
                    dist.all_reduce(update_stats, op=dist.ReduceOp.SUM)
                contributing_ranks = int(update_stats[0].item())
                global_effective_samples = int(update_stats[1].item())
                if global_effective_samples != accum_global_effective_samples:
                    raise RuntimeError(
                        "dynamic effective sample accounting mismatch: "
                        f"{global_effective_samples} != "
                        f"{accum_global_effective_samples}")
                global_has_update = global_effective_samples > 0
                slot_samples = DYNAMIC_TARGET_EFFECTIVE_PROMPTS
                effective_sample_scale = 1.0
                if (global_has_update and
                        EFFECTIVE_SAMPLE_NORMALIZATION):
                    effective_sample_scale = min(
                        slot_samples / max(global_effective_samples, 1),
                        EFFECTIVE_SAMPLE_SCALE_MAX)
                effective_scale_at_cap = bool(
                    EFFECTIVE_SAMPLE_NORMALIZATION and global_has_update and
                    effective_sample_scale >=
                    EFFECTIVE_SAMPLE_SCALE_MAX - 1e-12)
                applied_lr_scale = (
                    effective_sample_scale *
                    current_kl_safety_lr_scale)
                lora_step_lr = applied_lr_scale * float(next(
                    group["lr"] for group in optimizer.param_groups
                    if group.get("group_name") == "lora"))
                projector_step_lr = applied_lr_scale * float(next(
                    (group["lr"] for group in optimizer.param_groups
                     if group.get("group_name") == "visual_merger"), 0.0))
                sample_group_grad = (
                    global_has_update and
                    ((global_step + 1) % GRAD_GROUP_MONITOR_INTERVAL == 0 or
                     (global_step + 1) % MONITOR_STEPS == 0))
                if global_has_update:
                    if sample_group_grad:
                        lora_grad_norm = parameter_group_grad_norm(
                            lora_params, DEVICE)
                        projector_grad_norm = parameter_group_grad_norm(
                            projector_params, DEVICE)
                    else:
                        lora_grad_norm = torch.tensor(0.0, device=DEVICE)
                        projector_grad_norm = torch.tensor(
                            0.0, device=DEVICE)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, MAX_GRAD_NORM)
                    optimizer_step_applied = bool(
                        torch.isfinite(grad_norm).item() and
                        float(grad_norm.item()) >
                        MIN_GRAD_NORM_FOR_OPTIMIZER_STEP)
                    if optimizer_step_applied:
                        # AdamW近似对统一梯度缩放不敏感（m/sqrt(v)会抵消）。
                        # 因此有效样本与KL保护必须缩放实际step LR，并在scheduler
                        # 推进前恢复基准LR，确保LambdaLR轨迹不被污染。
                        step_base_lrs = [
                            float(group["lr"])
                            for group in optimizer.param_groups]
                        try:
                            for group, base_lr in zip(
                                    optimizer.param_groups,
                                    step_base_lrs):
                                group["lr"] = (
                                    base_lr * applied_lr_scale)
                            optimizer.step()
                        finally:
                            for group, base_lr in zip(
                                    optimizer.param_groups,
                                    step_base_lrs):
                                group["lr"] = base_lr
                        scheduler.step()
                        optimizer_update_step += 1
                else:
                    optimizer_step_applied = False
                    grad_norm = torch.tensor(0.0, device=DEVICE)
                    lora_grad_norm = torch.tensor(0.0, device=DEVICE)
                    projector_grad_norm = torch.tensor(0.0, device=DEVICE)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                torch.cuda.empty_cache()

                for record in local_sample_records:
                    record["global_effective_samples"] = (
                        global_effective_samples)
                    record["dynamic_prompt_rounds"] = int(accum_count)
                    record["dynamic_target_effective_prompts"] = int(
                        DYNAMIC_TARGET_EFFECTIVE_PROMPTS)
                    record["dynamic_batch_hit_target"] = bool(
                        dynamic_batch_ready)
                    record["dynamic_batch_hit_cap"] = bool(
                        dynamic_batch_cap and not dynamic_batch_ready)
                    record["dynamic_batch_step_reason"] = (
                        dynamic_step_reason)
                accum_sample_records = gather_and_log_micro_samples(
                    local_sample_records, global_step)
                local_sample_records = []
                accum_count = 0
                accum_has_update = False
                accum_effective_samples = 0
                accum_global_effective_samples = 0

                if IS_MAIN:
                    grad_value = float(grad_norm.item())
                    lora_grad_value = float(lora_grad_norm.item())
                    projector_grad_value = float(projector_grad_norm.item())
                    lora_grad_rms_value = (
                        lora_grad_value /
                        math.sqrt(max(lora_parameter_count, 1))
                        if sample_group_grad else 0.0)
                    projector_grad_rms_value = (
                        projector_grad_value /
                        math.sqrt(max(projector_parameter_count, 1))
                        if sample_group_grad and projector_parameter_count > 0
                        else 0.0)
                    lora_relative_step_value = (
                        lora_step_lr * lora_grad_value /
                        max(lora_initial_norm, 1e-12)
                        if sample_group_grad else 0.0)
                    projector_relative_step_value = (
                        projector_step_lr * projector_grad_value /
                        max(projector_initial_norm, 1e-12)
                        if sample_group_grad and projector_parameter_count > 0
                        else 0.0)
                    monitor_grad_sum += grad_value
                    monitor_grad_count += 1
                    monitor_grad_max = max(monitor_grad_max, grad_value)
                    monitor_lora_grad_sum += lora_grad_value
                    monitor_lora_grad_max = max(
                        monitor_lora_grad_max, lora_grad_value)
                    monitor_projector_grad_sum += projector_grad_value
                    monitor_projector_grad_max = max(
                        monitor_projector_grad_max, projector_grad_value)
                    monitor_lora_grad_rms_sum += lora_grad_rms_value
                    monitor_lora_grad_rms_max = max(
                        monitor_lora_grad_rms_max, lora_grad_rms_value)
                    monitor_projector_grad_rms_sum += (
                        projector_grad_rms_value)
                    monitor_projector_grad_rms_max = max(
                        monitor_projector_grad_rms_max,
                        projector_grad_rms_value)
                    monitor_lora_relative_step_sum += (
                        lora_relative_step_value)
                    monitor_lora_relative_step_max = max(
                        monitor_lora_relative_step_max,
                        lora_relative_step_value)
                    monitor_projector_relative_step_sum += (
                        projector_relative_step_value)
                    monitor_projector_relative_step_max = max(
                        monitor_projector_relative_step_max,
                        projector_relative_step_value)
                    monitor_group_grad_count += int(sample_group_grad)
                    monitor_optimizer_updates += int(optimizer_step_applied)
                    monitor_zero_update_steps += int(
                        not optimizer_step_applied)
                    monitor_clip_steps += int(
                        math.isfinite(grad_value) and
                        grad_value > MAX_GRAD_NORM)
                    monitor_nonfinite_grad_steps += int(
                        not math.isfinite(grad_value))
                    monitor_contributing_rank_sum += contributing_ranks
                    monitor_contributing_rank_min = min(
                        monitor_contributing_rank_min, contributing_ranks)
                    monitor_contributing_rank_max = max(
                        monitor_contributing_rank_max, contributing_ranks)
                    monitor_effective_samples_sum += (
                        global_effective_samples)
                    monitor_effective_sample_scale_sum += (
                        effective_sample_scale)
                    monitor_effective_sample_scale_max = max(
                        monitor_effective_sample_scale_max,
                        effective_sample_scale)
                    monitor_effective_scale_cap_steps += int(
                        effective_scale_at_cap)
                    monitor_kl_safety_lr_scale_sum += float(
                        current_kl_safety_lr_scale)
                    n_records = max(len(accum_sample_records), 1)
                    task_dist = Counter(r["task"] for r in accum_sample_records)
                    mode_dist = Counter(r["mode"] for r in accum_sample_records)
                    task_str = "/".join(f"{t}:{c}" for t, c in task_dist.most_common())
                    mode_str = "/".join(f"{m}:{c}" for m, c in mode_dist.most_common())
                    mean_field = lambda key: (
                        sum(float(r[key]) for r in accum_sample_records) / n_records)
                    step_greedy_tp = sum(
                        int(r["greedy_tp"]) for r in accum_sample_records)
                    step_greedy_fp = sum(
                        int(r["greedy_fp"]) for r in accum_sample_records)
                    step_greedy_fn = sum(
                        int(r["greedy_fn"]) for r in accum_sample_records)
                    step_greedy_micro_f1 = micro_f1_from_counts(
                        step_greedy_tp, step_greedy_fp, step_greedy_fn)
                    reward_history.append(mean_field("reward_mean"))
                    loss_history.append(mean_field("loss"))
                    global_epoch_records.extend(accum_sample_records)
                    global_seen_task_counts.update(task_dist)
                    monitor_sample_records.extend(accum_sample_records)
                    raw_prompts_seen = len(global_epoch_records)
                    epoch_coverage = min(
                        raw_prompts_seen / max(epoch_prompt_exposures, 1),
                        1.0)
                    if (global_step <= 3 or
                            global_step % STEP_CONSOLE_INTERVAL == 0):
                        print(
                            f"[{datetime.now():%H:%M:%S}] "
                            f"slot={global_step:5d} opt={optimizer_update_step:4d} "
                            f"task=[{task_str}] mode=[{mode_str}] "
                            f"loss/policy={mean_field('loss'):.4f}/"
                            f"{mean_field('policy_loss'):.4f} "
                            f"R={mean_field('reward_mean'):.3f} "
                            f"GμF1={step_greedy_micro_f1:.3f} "
                            f"G TP/FP/FN={step_greedy_tp}/"
                            f"{step_greedy_fp}/{step_greedy_fn} "
                            f"activeS={mean_field('active_sampled_candidates'):.2f} "
                            f"pref={sum(int(r['sequence_preference_applied']) for r in accum_sample_records)}/"
                            f"{sum(int(r['sequence_preference_eligible']) for r in accum_sample_records)} "
                            f"coefF/S={mean_field('preference_first_coefficient'):.1e}/"
                            f"{mean_field('preference_suffix_coefficient'):.1e} "
                            f"update={mean_field('has_update'):.0%} "
                            f"KL={mean_field('kl'):.4f} "
                            f"grad={grad_norm.item():.3f} "
                            f"lr={optimizer.param_groups[0]['lr']:.2e} "
                            f"seen={raw_prompts_seen}/{epoch_prompt_exposures}"
                            f"({epoch_coverage:.1%}) "
                            f"effective={global_effective_samples}/{slot_samples} "
                            f"rounds={len(accum_sample_records)//WORLD_SIZE}")

                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({"step": global_step, "task": task_str})


                # 与checkpoint同频监控，便于把训练方向和外部F1逐点对齐。
                if global_step % MONITOR_STEPS == 0:
                    # 全部真实更新路径（GRPO/CE/distill/preference）的KL先求全局
                    # mean；rank0再从已汇集逐样本记录计算各任务p95。beta与下一窗口
                    # 下一窗口实际LR安全倍率随后广播，确保所有rank完全一致。
                    kl_window = torch.tensor(
                        [kl_window_local_sum, float(kl_window_local_count)],
                        device=DEVICE, dtype=torch.float64)
                    if is_ddp:
                        dist.all_reduce(kl_window, op=dist.ReduceOp.SUM)
                    global_kl_count = int(kl_window[1].item())
                    global_window_kl = (
                        float(kl_window[0].item()) / global_kl_count
                        if global_kl_count > 0 else None)
                    task_kl_p95_control = {}
                    if IS_MAIN:
                        for task_name in TASK_REPEAT:
                            values = sorted(
                                float(record["kl"])
                                for record in monitor_sample_records
                                if (record["task"] == task_name and
                                    record.get("kl_measured", False) and
                                    record.get("has_update", False) and
                                    record.get("kl_source") in
                                        KL_CONTROLLER_SOURCES and
                                    math.isfinite(float(record["kl"]))))
                            if len(values) >= KL_TASK_MIN_SAMPLES:
                                p95_index = higher_quantile_index(
                                    len(values), 0.95)
                                task_kl_p95_control[task_name] = {
                                    "count": len(values),
                                    "p95": float(values[p95_index]),
                                    "max": float(values[-1]),
                                }
                    kl_window_local_sum = 0.0
                    kl_window_local_count = 0

                    mon_keys = list(mon.keys())
                    mon_values = torch.tensor(
                        [mon[k] for k in mon_keys], device=DEVICE, dtype=torch.long)
                    if is_ddp:
                        dist.reduce(mon_values, dst=0, op=dist.ReduceOp.SUM)
                    memory_peak = torch.tensor([
                        float(torch.cuda.max_memory_allocated(DEVICE)),
                        float(torch.cuda.max_memory_reserved(DEVICE)),
                    ], device=DEVICE, dtype=torch.float64)
                    if is_ddp:
                        dist.reduce(
                            memory_peak, dst=0, op=dist.ReduceOp.MAX)
                    if IS_MAIN:
                        mon_global = {k: int(mon_values[i].item())
                                      for i, k in enumerate(mon_keys)}
                        live_mon_global = {
                            key: mon_global[key] for key in (
                                "GRPO", "SKIP_overlong_insufficient",
                                "SKIP_no_signal", "SKIP_no_improvement",
                                "SKIP_ddp_route_mismatch",
                                "SKIP_generation_timeout",
                                "PREFERENCE_only",
                                "sequence_preference_eligible",
                                "sequence_preference_applied",
                                "sequence_preference_identical_rejected",
                                "pairwise_preference_applied",
                                "KL_skip", "anomaly_skip",
                                "greedy_invalid")}
                        total_samples = (mon_global["CE_targeted_no_hit"] +
                                         mon_global[
                                             "SKIP_overlong_insufficient"] +
                                         mon_global["SKIP_no_signal"] +
                                         mon_global["SKIP_no_improvement"] +
                                         mon_global[
                                             "SKIP_ddp_route_mismatch"] +
                                         mon_global[
                                             "SKIP_generation_timeout"] +
                                         mon_global["PREFERENCE_only"] +
                                         mon_global["GRPO"])
                        skipped_samples = max(
                            total_samples - mon_global["GRPO"], 0)
                        current_drift_snapshot = (
                            snapshot_trainable_parameters(named_trainable))
                        cumulative_drift = compare_parameter_snapshots(
                            drift_run_start, current_drift_snapshot)
                        window_drift = compare_parameter_snapshots(
                            drift_window_start, current_drift_snapshot)
                        drift_window_start = current_drift_snapshot
                        mean_grad = (
                            monitor_grad_sum / monitor_grad_count
                            if monitor_grad_count > 0 else 0.0)
                        group_grad_den = max(monitor_group_grad_count, 1)
                        mean_lora_grad = (
                            monitor_lora_grad_sum / group_grad_den)
                        mean_projector_grad = (
                            monitor_projector_grad_sum / group_grad_den)
                        mean_lora_grad_rms = (
                            monitor_lora_grad_rms_sum / group_grad_den)
                        mean_projector_grad_rms = (
                            monitor_projector_grad_rms_sum /
                            group_grad_den)
                        mean_lora_relative_step = (
                            monitor_lora_relative_step_sum /
                            group_grad_den)
                        mean_projector_relative_step = (
                            monitor_projector_relative_step_sum /
                            group_grad_den)
                        update_rate = (
                            monitor_optimizer_updates /
                            max(monitor_grad_count, 1))
                        clip_rate = (
                            monitor_clip_steps /
                            max(monitor_optimizer_updates, 1))
                        mean_contributing_ranks = (
                            monitor_contributing_rank_sum /
                            max(monitor_grad_count, 1))
                        mean_effective_samples = (
                            monitor_effective_samples_sum /
                            max(monitor_grad_count, 1))
                        cumulative_by_kind = cumulative_drift["by_kind"]
                        lora_cumulative_relative = cumulative_by_kind.get(
                            "lora", {}).get("relative", 0.0)
                        kl_text = (
                            f"{global_window_kl:.6f}"
                            if global_window_kl is not None else "n/a")
                        task_summary = summarize_monitor_records(
                            monitor_sample_records)
                        cost_bucket_summary = summarize_cost_bucket_records(
                            monitor_sample_records)
                        all140_item = cost_bucket_summary.get(
                            ALL140_MONITOR_BUCKET)
                        pref_eligible = sum(
                            int(r["sequence_preference_eligible"])
                            for r in monitor_sample_records)
                        pref_applied = sum(
                            int(r["sequence_preference_applied"])
                            for r in monitor_sample_records)
                        pref_only = sum(
                            int(r["preference_only"])
                            for r in monitor_sample_records)
                        first_coef = sum(
                            float(r["preference_first_coefficient"])
                            for r in monitor_sample_records) / max(
                                pref_eligible, 1)
                        suffix_coef = sum(
                            float(r["preference_suffix_coefficient"])
                            for r in monitor_sample_records) / max(
                                pref_eligible, 1)
                        pref_support = sum(
                            float(r["preference_support_scale"])
                            for r in monitor_sample_records) / max(
                                pref_eligible, 1)
                        task_f1_text = ",".join(
                            f"{task}:{item['greedy_window_micro_f1']:.3f}"
                            for task, item in task_summary.items())
                        all_item = task_summary.get("ALL")
                        if all_item is not None:
                            all_counts = all_item["greedy_counts"]
                            all_recall = (
                                all_counts["tp"] /
                                max(all_counts["tp"] + all_counts["fn"], 1))
                            all_health_text = (
                                f"ALL(R/limit/eos)={all_recall:.3f}/"
                                f"{all_item['greedy_token_limit_rate']:.1%}/"
                                f"{all_item['greedy_eos_rate']:.1%}")
                        else:
                            all_health_text = "ALL(R/limit/eos)=-"
                        gate_rejections = Counter()
                        for record in monitor_sample_records:
                            gate_rejections.update(
                                record.get("preference_gate_rejections", {}))
                        gate_text = "/".join(
                            f"{reason}:{count}" for reason, count in
                            gate_rejections.most_common(3)) or "none"
                        print(
                            f"[MON {global_step}] GRPO={mon_global['GRPO']}/"
                            f"{total_samples} skip={skipped_samples} "
                            f"CEhit={mon_global['CE_targeted_no_hit']} "
                            f"pref(E/A/O)={pref_eligible}/{pref_applied}/{pref_only} "
                            f"support={pref_support:.2f} "
                            f"coef(F/S)={first_coef:.2e}/{suffix_coef:.2e} "
                            f"gate[{gate_text}] μF1[{task_f1_text}] "
                            f"{all_health_text} KL={kl_text} "
                            f"seen={raw_prompts_seen}/{epoch_prompt_exposures}"
                            f"({epoch_coverage:.1%};"
                            f"A/R/S={global_seen_task_counts.get('ALL', 0)}/"
                            f"{global_seen_task_counts.get('ROW', 0)}/"
                            f"{global_seen_task_counts.get('SINGLE', 0)}) "
                            f"grad={mean_grad:.3f}/{monitor_grad_max:.3f} "
                            f"clip={clip_rate:.1%} upd={update_rate:.1%} "
                            f"driftL/P={lora_cumulative_relative:.3%}/"
                            f"{cumulative_by_kind.get('projector', {}).get('relative', 0.0):.3%}")
                        if all140_item is not None:
                            print(
                                f"[ALL:140 {global_step}] n={all140_item['samples']} "
                                f"μF1(G/M)="
                                f"{all140_item['greedy_window_micro_f1']:.3f}/"
                                f"{all140_item['metric_best_window_micro_f1']:.3f} "
                                f"G(limit/eos)="
                                f"{all140_item['greedy_token_limit_rate']:.1%}/"
                                f"{all140_item['greedy_eos_rate']:.1%} "
                                f"S(limit/eos)="
                                f"{all140_item['sampled_candidate_token_limit_rate']:.1%}/"
                                f"{all140_item['sampled_candidate_eos_rate']:.1%} "
                                f"recover={all140_item['greedy_limit_recoverable_rate']:.1%} "
                                f"pref={pref_applied}/{pref_eligible}")
                        monitor_history.append({
                            "step": global_step,
                            "partial_window": False,
                            "optimizer_slots": monitor_grad_count,
                            "window_update_kl": global_window_kl,
                            "window_update_kl_samples": global_kl_count,
                            "task_kl_p95_control": task_kl_p95_control,
                            "grad_norm_mean": mean_grad,
                            "grad_norm_max": monitor_grad_max,
                            "lora_grad_norm_mean": mean_lora_grad,
                            "lora_grad_norm_max": monitor_lora_grad_max,
                            "projector_grad_norm_mean":
                                mean_projector_grad,
                            "projector_grad_norm_max":
                                monitor_projector_grad_max,
                            "group_grad_samples":
                                monitor_group_grad_count,
                            "lora_grad_rms_mean":
                                mean_lora_grad_rms,
                            "lora_grad_rms_max":
                                monitor_lora_grad_rms_max,
                            "projector_grad_rms_mean":
                                mean_projector_grad_rms,
                            "projector_grad_rms_max":
                                monitor_projector_grad_rms_max,
                            "lora_relative_step_mean":
                                mean_lora_relative_step,
                            "lora_relative_step_max":
                                monitor_lora_relative_step_max,
                            "projector_relative_step_mean":
                                mean_projector_relative_step,
                            "projector_relative_step_max":
                                monitor_projector_relative_step_max,
                            "exact_parameter_drift_reference": (
                                "resume_process_start"
                                if resume_state is not None
                                else "training_start"),
                            "exact_parameter_drift_cumulative":
                                cumulative_drift,
                            "exact_parameter_drift_window":
                                window_drift,
                            "optimizer_update_steps":
                                monitor_optimizer_updates,
                            "zero_update_steps":
                                monitor_zero_update_steps,
                            "optimizer_update_rate": update_rate,
                            "optimizer_update_step_total":
                                optimizer_update_step,
                            "clip_rate": clip_rate,
                            "nonfinite_grad_steps":
                                monitor_nonfinite_grad_steps,
                            "contributing_ranks_mean":
                                mean_contributing_ranks,
                            "contributing_ranks_min":
                                monitor_contributing_rank_min,
                            "contributing_ranks_max":
                                monitor_contributing_rank_max,
                            "effective_samples_mean":
                                mean_effective_samples,
                            "scheduler_last_epoch": scheduler.last_epoch,
                            "peak_cuda_allocated_gib":
                                float(memory_peak[0].item() / 2**30),
                            "peak_cuda_reserved_gib":
                                float(memory_peak[1].item() / 2**30),
                            "learning_rate": float(
                                optimizer.param_groups[0]["lr"]),
                            "parameter_group_lrs": {
                                group.get("group_name", str(i)):
                                    float(group["lr"])
                                for i, group in enumerate(
                                    optimizer.param_groups)},
                            "global_path_counts": live_mon_global,
                            "raw_prompt_exposures_seen": raw_prompts_seen,
                            "epoch_prompt_exposures": epoch_prompt_exposures,
                            "epoch_coverage": epoch_coverage,
                            "cumulative_task_exposures": dict(
                                global_seen_task_counts),
                            "preference_gate_rejections": dict(
                                gate_rejections),
                            "by_task": task_summary,
                            "by_cost_bucket": cost_bucket_summary,
                        })
                        beta_history.append({
                            "step": global_step,
                            "window_update_kl": global_window_kl,
                            "measured_update_samples": global_kl_count,
                            "task_kl_p95_control": task_kl_p95_control,
                        })
                        with open(MONITOR_JSON_FILE, "w", encoding="utf-8") as f:
                            json.dump({
                                "updated_at": datetime.now().isoformat(
                                    timespec="seconds"),
                                "latest_step": global_step,
                                "optimizer_update_step":
                                    optimizer_update_step,
                                "fixed_kl_history": beta_history,
                                "windows": monitor_history,
                            }, f, indent=2, ensure_ascii=False)
                        monitor_sample_records.clear()
                        monitor_grad_sum = 0.0
                        monitor_grad_count = 0
                        monitor_grad_max = 0.0
                        monitor_lora_grad_sum = 0.0
                        monitor_lora_grad_max = 0.0
                        monitor_projector_grad_sum = 0.0
                        monitor_projector_grad_max = 0.0
                        monitor_lora_grad_rms_sum = 0.0
                        monitor_lora_grad_rms_max = 0.0
                        monitor_projector_grad_rms_sum = 0.0
                        monitor_projector_grad_rms_max = 0.0
                        monitor_lora_relative_step_sum = 0.0
                        monitor_lora_relative_step_max = 0.0
                        monitor_projector_relative_step_sum = 0.0
                        monitor_projector_relative_step_max = 0.0
                        monitor_group_grad_count = 0
                        monitor_optimizer_updates = 0
                        monitor_zero_update_steps = 0
                        monitor_clip_steps = 0
                        monitor_nonfinite_grad_steps = 0
                        monitor_contributing_rank_sum = 0
                        monitor_contributing_rank_min = WORLD_SIZE
                        monitor_contributing_rank_max = 0
                        monitor_effective_samples_sum = 0
                        monitor_effective_sample_scale_sum = 0.0
                        monitor_effective_sample_scale_max = 0.0
                        monitor_effective_scale_cap_steps = 0
                        monitor_kl_safety_lr_scale_sum = 0.0
                    for k in mon:
                        mon[k] = 0
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                # 同一次GRPO训练每SAVE_STEPS步保存；所有rank在保存前后同步，避免rank0保存时
                # 其他 rank 已进入下一次 collective。
                if global_step % SAVE_STEPS == 0:
                    if is_ddp:
                        dist.barrier()
                    if IS_MAIN:
                        ckpt_path = os.path.join(
                            GRPO_OUTPUT_DIR, f"checkpoint-{global_step}")
                        policy_model.save_pretrained(ckpt_path)
                        print(f"  → Checkpoint 已保存: {ckpt_path}")
                    if is_ddp:
                        dist.barrier()

                if global_step % SAVE_STEPS == 0:
                    checkpoint_path = os.path.join(
                        GRPO_OUTPUT_DIR, f"checkpoint-{global_step}")
                    save_restart_state(
                        checkpoint_path, epoch, batch_idx + 1,
                        global_epoch_records)

        # 保存不足MONITOR_STEPS的最后一个窗口，避免完整epoch尾部诊断静默丢失。
        if global_step % MONITOR_STEPS != 0:
            kl_window = torch.tensor(
                [kl_window_local_sum, float(kl_window_local_count)],
                device=DEVICE, dtype=torch.float64)
            if is_ddp:
                dist.all_reduce(kl_window, op=dist.ReduceOp.SUM)
            residual_kl_count = int(kl_window[1].item())
            residual_kl = (
                float(kl_window[0].item()) / residual_kl_count
                if residual_kl_count > 0 else None)
            mon_keys = list(mon.keys())
            mon_values = torch.tensor(
                [mon[k] for k in mon_keys], device=DEVICE, dtype=torch.long)
            if is_ddp:
                dist.reduce(mon_values, dst=0, op=dist.ReduceOp.SUM)
            residual_memory_peak = torch.tensor([
                float(torch.cuda.max_memory_allocated(DEVICE)),
                float(torch.cuda.max_memory_reserved(DEVICE)),
            ], device=DEVICE, dtype=torch.float64)
            if is_ddp:
                dist.reduce(
                    residual_memory_peak, dst=0, op=dist.ReduceOp.MAX)
            if IS_MAIN and monitor_sample_records:
                mon_global = {
                    key: int(mon_values[i].item())
                    for i, key in enumerate(mon_keys)}
                live_mon_global = {
                    key: mon_global[key] for key in (
                        "GRPO", "SKIP_overlong_insufficient",
                        "SKIP_no_signal", "SKIP_no_improvement",
                        "SKIP_ddp_route_mismatch",
                        "SKIP_generation_timeout",
                        "KL_skip", "anomaly_skip", "greedy_invalid")}
                update_den = max(monitor_optimizer_updates, 1)
                group_grad_den = max(monitor_group_grad_count, 1)
                residual_grad_mean = (
                    monitor_grad_sum / max(monitor_grad_count, 1))
                residual_lora_grad_mean = (
                    monitor_lora_grad_sum / group_grad_den)
                residual_projector_grad_mean = (
                    monitor_projector_grad_sum / group_grad_den)
                residual_lora_grad_rms_mean = (
                    monitor_lora_grad_rms_sum / group_grad_den)
                residual_projector_grad_rms_mean = (
                    monitor_projector_grad_rms_sum / group_grad_den)
                residual_lora_relative_step_mean = (
                    monitor_lora_relative_step_sum / group_grad_den)
                residual_projector_relative_step_mean = (
                    monitor_projector_relative_step_sum /
                    group_grad_den)
                residual_update_rate = (
                    monitor_optimizer_updates /
                    max(monitor_grad_count, 1))
                residual_clip_rate = (
                    monitor_clip_steps / update_den)
                residual_rank_mean = (
                    monitor_contributing_rank_sum /
                    max(monitor_grad_count, 1))
                residual_effective_samples_mean = (
                    monitor_effective_samples_sum /
                    max(monitor_grad_count, 1))
                monitor_history.append({
                    "step": global_step,
                    "partial_window": True,
                    "optimizer_slots": monitor_grad_count,
                    "window_update_kl": residual_kl,
                    "window_update_kl_samples": residual_kl_count,
                    "grad_norm_mean": residual_grad_mean,
                    "grad_norm_max": monitor_grad_max,
                    "lora_grad_norm_mean": residual_lora_grad_mean,
                    "lora_grad_norm_max": monitor_lora_grad_max,
                    "projector_grad_norm_mean":
                        residual_projector_grad_mean,
                    "projector_grad_norm_max":
                        monitor_projector_grad_max,
                    "group_grad_samples": monitor_group_grad_count,
                    "lora_grad_rms_mean":
                        residual_lora_grad_rms_mean,
                    "lora_grad_rms_max":
                        monitor_lora_grad_rms_max,
                    "projector_grad_rms_mean":
                        residual_projector_grad_rms_mean,
                    "projector_grad_rms_max":
                        monitor_projector_grad_rms_max,
                    "lora_relative_step_mean":
                        residual_lora_relative_step_mean,
                    "lora_relative_step_max":
                        monitor_lora_relative_step_max,
                    "projector_relative_step_mean":
                        residual_projector_relative_step_mean,
                    "projector_relative_step_max":
                        monitor_projector_relative_step_max,
                    "optimizer_update_steps": monitor_optimizer_updates,
                    "zero_update_steps": monitor_zero_update_steps,
                    "optimizer_update_rate": residual_update_rate,
                    "optimizer_update_step_total": optimizer_update_step,
                    "clip_rate": residual_clip_rate,
                    "nonfinite_grad_steps":
                        monitor_nonfinite_grad_steps,
                    "contributing_ranks_mean": residual_rank_mean,
                    "contributing_ranks_min":
                        monitor_contributing_rank_min,
                    "contributing_ranks_max":
                        monitor_contributing_rank_max,
                    "effective_samples_mean":
                        residual_effective_samples_mean,
                    "scheduler_last_epoch": scheduler.last_epoch,
                    "peak_cuda_allocated_gib": float(
                        residual_memory_peak[0].item() / 2**30),
                    "peak_cuda_reserved_gib": float(
                        residual_memory_peak[1].item() / 2**30),
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]),
                    "parameter_group_lrs": {
                        group.get("group_name", str(i)): float(group["lr"])
                        for i, group in enumerate(optimizer.param_groups)},
                    "global_path_counts": live_mon_global,
                    "raw_prompt_exposures_seen": len(
                        global_epoch_records),
                    "epoch_prompt_exposures": epoch_prompt_exposures,
                    "epoch_coverage": min(
                        len(global_epoch_records) /
                        max(epoch_prompt_exposures, 1), 1.0),
                    "cumulative_task_exposures": dict(
                        global_seen_task_counts),
                    "by_task": summarize_monitor_records(
                        monitor_sample_records),
                    "by_cost_bucket": summarize_cost_bucket_records(
                        monitor_sample_records),
                })
                beta_history.append({
                    "step": global_step,
                    "partial_window": True,
                    "window_update_kl": residual_kl,
                    "measured_update_samples": residual_kl_count,
                })
                with open(MONITOR_JSON_FILE, "w", encoding="utf-8") as f:
                    json.dump({
                        "updated_at": datetime.now().isoformat(
                            timespec="seconds"),
                        "latest_step": global_step,
                        "optimizer_update_step": optimizer_update_step,
                        "fixed_kl_history": beta_history,
                        "windows": monitor_history,
                    }, f, indent=2, ensure_ascii=False)
                print(
                    f"  → 已保存尾窗口监控: slots={monitor_grad_count}, "
                    f"optimizer更新={monitor_optimizer_updates}, "
                    f"update_rate={residual_update_rate:.1%}")
                monitor_sample_records.clear()
            kl_window_local_sum = 0.0
            kl_window_local_count = 0
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # Epoch 汇总
        epoch_ckpt = os.path.join(
            GRPO_OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}-step-{global_step}")
        if IS_MAIN:
            if global_epoch_records:
                print(
                    f"\nEpoch {epoch+1}/{GRPO_EPOCHS} "
                    "阶段结束（完整epoch完成）: "
                    f"avg_reward="
                    f"{sum(r['reward_mean'] for r in global_epoch_records)/len(global_epoch_records):.4f}, "
                    f"avg_loss="
                    f"{sum(r['loss'] for r in global_epoch_records)/len(global_epoch_records):.4f}")
            policy_model.save_pretrained(epoch_ckpt)
            print(f"  → Epoch checkpoint 已保存: {epoch_ckpt}")
        if is_ddp:
            dist.barrier()
        save_restart_state(
            epoch_ckpt, epoch + 1, 0, global_epoch_records)

    # ================================================================
    # 最终保存（仅主进程）
    # ================================================================
    if IS_MAIN:
        print(f"\n[{datetime.now():%H:%M:%S}] 保存 GRPO 优化后的 LoRA 权重...")
        policy_model.save_pretrained(GRPO_LORA_SAVE)
        processor.save_pretrained(GRPO_LORA_SAVE)
        final_drift_snapshot = snapshot_trainable_parameters(named_trainable)
        final_exact_drift = compare_parameter_snapshots(
            drift_run_start, final_drift_snapshot)
        final_kind_drift = final_exact_drift["by_kind"]
        print(
            "  最终精确权重漂移 dW/Wref "
            f"LoRA={final_kind_drift.get('lora', {}).get('relative', 0.0):.3e}, "
            f"projector="
            f"{final_kind_drift.get('projector', {}).get('relative', 0.0):.3e}, "
            f"overall={final_exact_drift['relative']:.3e}")

        n_history = min(len(reward_history), 100)
        final_avg_reward = (sum(reward_history[-n_history:]) / n_history
                            if n_history > 0 else 0.0)

        stats = {
            "base_model": BASE_MODEL_PATH,
            "run_name": RUN_NAME,
            "policy_original_start_path": policy_lora_path,
            "policy_loaded_path": policy_lora_path,
            "resumed_from_state": resume_state_path,
            "reference_start_path": REFERENCE_LORA_PATH,
            "reference_is_original_sft_anchor": True,
            "reference_is_cgpc_start_anchor": True,
            "exact_parameter_drift_reference": (
                "resume_process_start"
                if resume_state is not None else "training_start"),
            "final_exact_parameter_drift": final_exact_drift,
            "training_script_sha256": file_sha256(__file__),
            "grpo_params": {"G": G,
                            "objective": POLICY_OBJECTIVE,
                            "token_denominator":
                                POLICY_TOKEN_DENOMINATOR,
                            "ppo_importance_ratio": False,
                            "ppo_clip_enabled": False,
                            "epsilon": EPSILON,
                            "beta_initial": BETA,
                            "beta_final": current_beta,
                            "gamma": GAMMA,
                            "lr": GRPO_LR,
                            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                            "gradient_accumulation_steps":
                                GRADIENT_ACCUMULATION_STEPS,
                            "world_size": WORLD_SIZE,
                            "global_prompt_batch":
                                (PER_DEVICE_BATCH_SIZE *
                                 GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE),
                            "task_max_tokens": TASK_MAX_TOKENS,
                            "train_bucket_max_tokens":
                                TRAIN_BUCKET_MAX_TOKENS,
                            "dynamic_effective_batching":
                                DYNAMIC_EFFECTIVE_BATCHING,
                            "dynamic_target_effective_prompts":
                                DYNAMIC_TARGET_EFFECTIVE_PROMPTS,
                            "dynamic_max_prompt_rounds":
                                DYNAMIC_MAX_PROMPT_ROUNDS,
                            "max_grad_norm": MAX_GRAD_NORM,
                            "min_grad_norm_for_optimizer_step":
                                MIN_GRAD_NORM_FOR_OPTIMIZER_STEP,
                            "projector_lr": PROJECTOR_LR,
                            "train_adapter_projector":
                                TRAIN_ADAPTER_PROJECTOR,
                            "weight_decay": WEIGHT_DECAY,
                            "warmup_ratio": WARMUP_RATIO,
                            "min_lr_ratio": MIN_LR_RATIO,
                            "lr_schedule":
                                "actual_optimizer_update_warmup_cosine",
                            "lr_schedule_advances_on_zero_update_slot":
                                False,
                            "temperature_low": SAMPLE_TEMPERATURE_LOW,
                            "temperature_high": SAMPLE_TEMPERATURE_HIGH,
                            "sample_top_p": SAMPLE_TOP_P,
                            "sample_top_k": SAMPLE_TOP_K,
                            "sample_behavior_policy": (
                                "raw current policy; T=1, top_p=1, top_k=0; "
                                "generation logits modifiers neutralized"),
                            "generation_neutral_overrides":
                                GENERATION_NEUTRAL_OVERRIDES,
                            "sample_generation_timeout_seconds":
                                SAMPLE_GENERATION_TIMEOUT_SECONDS,
                            "exploit_sampled_candidates": EXPLOIT_SAMPLED_CANDIDATES,
                            "epochs": GRPO_EPOCHS,
                            "planned_steps": planned_steps,
                            "schedule_steps": total_steps,
                            "steps_per_epoch": planned_steps // GRPO_EPOCHS,
                            "prompt_exposures_per_epoch":
                                epoch_prompt_exposures,
                            "full_epoch_training": True,
                            "save_steps": SAVE_STEPS,
                            "monitor_steps": MONITOR_STEPS,
                            "step_console_interval": STEP_CONSOLE_INTERVAL,
                            "grad_group_monitor_interval":
                                GRAD_GROUP_MONITOR_INTERVAL,
                            "lora_drift_target_min_relative":
                                LORA_DRIFT_TARGET_MIN_RELATIVE,
                            "lora_drift_target_max_relative":
                                LORA_DRIFT_TARGET_MAX_RELATIVE,
                            "optimizer_update_rate_warning":
                                OPTIMIZER_UPDATE_RATE_WARNING,
                            "optimizer_update_rate_critical":
                                OPTIMIZER_UPDATE_RATE_CRITICAL,
                            "greedy_candidates": GREEDY_CANDIDATES,
                            "greedy_excluded_from_grpo":
                                EXCLUDE_GREEDY_FROM_GRPO,
                            "advantage_normalization":
                                ADVANTAGE_NORMALIZATION,
                            "advantage_fixed_scale":
                                ADVANTAGE_FIXED_SCALE,
                            "micro_raw_advantage_scale":
                                MICRO_RAW_ADVANTAGE_SCALE,
                            "micro_raw_advantage_clip":
                                MICRO_RAW_ADVANTAGE_CLIP,
                            "soft_tp_quality_blend":
                                SOFT_TP_QUALITY_BLEND,
                            "localization_advantage_scale":
                                LOCALIZATION_ADVANTAGE_SCALE,
                            "task_advantage_gain": TASK_ADVANTAGE_GAIN,
                            "targeted_no_hit_ce": {
                                "enabled": TARGETED_NO_HIT_CE_ENABLED,
                                "tasks": sorted(TARGETED_NO_HIT_CE_TASKS),
                                "weights": TARGETED_NO_HIT_CE_WEIGHT,
                                "condition": (
                                    "greedy TP=0 and every active sampled "
                                    "candidate TP=0")},
                            "all_preference": {
                                "enabled":
                                    ALL_OVERLONG_PREFERENCE_ENABLED,
                                "task": "ALL",
                                "overlong_first_weight":
                                    ALL_OVERLONG_PREFERENCE_FIRST_WEIGHT,
                                "overlong_suffix_weight":
                                    ALL_OVERLONG_PREFERENCE_SUFFIX_WEIGHT,
                                "quality_first_weight":
                                    ALL_QUALITY_PREFERENCE_FIRST_WEIGHT,
                                "quality_suffix_weight":
                                    ALL_QUALITY_PREFERENCE_SUFFIX_WEIGHT,
                                "beta": PAIRWISE_PREFERENCE_BETA,
                                "overlong_min_reward":
                                    ALL_OVERLONG_PREFERENCE_MIN_REWARD,
                                "overlong_min_f1":
                                    ALL_OVERLONG_PREFERENCE_MIN_F1,
                                "quality_min_reward":
                                    ALL_QUALITY_PREFERENCE_MIN_REWARD,
                                "quality_min_reward_gap":
                                    ALL_QUALITY_PREFERENCE_MIN_REWARD_GAP,
                                "quality_min_f1":
                                    ALL_QUALITY_PREFERENCE_MIN_F1,
                                "quality_min_f1_gap":
                                    ALL_QUALITY_PREFERENCE_MIN_F1_GAP,
                                "quality_min_micro_raw_gap":
                                    ALL_QUALITY_PREFERENCE_MIN_MICRO_RAW_GAP,
                                "quality_min_support":
                                    ALL_QUALITY_PREFERENCE_MIN_SUPPORT,
                                "candidate_selection":
                                    "filter qualified first, then metric-best",
                                "support_scale":
                                    "quality requires >=2; overlong scales "
                                    "bbox>=2:1.0, metric>=2:0.75, single:0.5",
                                "first_suffix_masks_disjoint": True,
                                "negative_chosen_advantage_route":
                                    "preference-only+reference-KL"},
                            "min_total_adv_abs":
                                MIN_TOTAL_ADV_ABS,
                            "loss_reduction": LOSS_REDUCTION,
                            "mask_truncated_completions":
                                MASK_TRUNCATED_COMPLETIONS,
                            "min_active_sampled_candidates":
                                MIN_ACTIVE_SAMPLED_CANDIDATES,
                            "rl_disable_dropout": RL_DISABLE_DROPOUT,
                            "empty_reward": EMPTY_REWARD,
                            "kl_skip_threshold": KL_SKIP_THRESHOLD,
                            "nccl_timeout_minutes": NCCL_TIMEOUT_MINUTES},
            "reference_kl": {
                "beta": BETA,
                "adaptive": False,
                "candidate_anomaly_skip": KL_SKIP_THRESHOLD},
            "soft_overlong": {
                "cache_tokens": SOFT_OVERLONG_CACHE_TOKENS,
                "cache_ratio": SOFT_OVERLONG_CACHE_RATIO,
                "max_penalty": SOFT_OVERLONG_MAX_PENALTY,
                "advantage_scale": SOFT_OVERLONG_ADVANTAGE_SCALE,
                "enabled": any(
                    value > 0.0
                    for value in SOFT_OVERLONG_MAX_PENALTY.values()),
                "true_token_limit_candidates_masked_from_policy_gradient":
                    MASK_TRUNCATED_COMPLETIONS},
            "monitoring_schema": {
                "window_micro_f1": (
                    "reconstructed from exact summed TP/FP/FN, not "
                    "mean per-prompt F1"),
                "candidate_views": [
                    "greedy", "reward_best", "metric_best_micro_raw"],
                "oracle_rates": [
                    "any_hit_at_g", "perfect_at_g",
                    "sampled_rescue_at_g"],
                "reward_alignment": [
                    "reward_metric_best_agreement",
                    "reward_best_metric_optimal",
                    "reward_best_not_worse_than_greedy"],
                "overall_scope": (
                    "TRAIN_WEIGHTED_OVERALL uses TASK_REPEAT exposure"),
                "generation_failure_types": [
                    "empty_text", "invalid_json", "truncated",
                    "no_valid_box", "token_limit", "eos_terminated"],
                "completion_length_and_text_diversity": True,
                "low_overhead_optimizer_diagnostics": [
                    "actual_optimizer_updates", "zero_update_slots",
                    "contributing_ranks", "lora_grad_norm",
                    "projector_grad_norm", "grad_rms",
                    "lr*grad_norm/initial_param_norm relative_step",
                    "clip_rate",
                    "scheduler_last_epoch",
                    "peak_cuda_allocated/reserved"],
                "advantage_diagnostics": [
                    "hard+soft-TP/localization/total advantage magnitudes",
                    "quality-aware TP radial clip scale",
                    "active sampled tokens",
                    "token-level effective candidate count"],
                "dynamic_batch_diagnostics": [
                    "global effective prompts",
                    "prompt rounds to optimizer boundary",
                    "target/cap/epoch-tail boundary reason"],
                "structural_exploration_diagnostics": [
                    "sampled quantized bbox-set uniqueness",
                    "sampled bbox duplicate-to-greedy rate",
                    "active sampled quality-aware TP std/zero-std rate"],
                "all_preference_diagnostics": [
                    "qualified/applied/preference-only counts",
                    "first/suffix bounded gradient coefficients",
                    "bbox/metric support and support scale",
                    "chosen/rejected KL and divergence token"],
                "all140_diagnostics": {
                    "scope": ALL140_MONITOR_BUCKET,
                    "quality": [
                        "greedy/metric-best micro-F1", "greedy TP/FP/FN",
                        "sampled rescue rate"],
                    "length": [
                        "greedy limit/EOS/tokens/budget",
                        "sampled candidate limit/EOS/tokens",
                        "active sampled candidates",
                        "sampled soft-overlong penalty"],
                    "optimization_router": (
                        "greedy_limit_recoverable_rate separates a greedy "
                        "transfer gap from a policy-wide length failure")},
                "partial_final_window_persisted": True},
            "optimizer_progress": {
                "accumulation_slots": global_step,
                "actual_optimizer_updates": optimizer_update_step,
                "zero_update_slots": global_step - optimizer_update_step,
                "scheduler_last_epoch": scheduler.last_epoch,
                "scheduler_basis": "actual_optimizer_updates",
                "checkpoint_step_semantics": (
                    "global_step counts gradient-accumulation slots; "
                    "actual_optimizer_updates counts optimizer.step calls")},
            "parameter_group_monitor": {
                "sample_interval_slots": GRAD_GROUP_MONITOR_INTERVAL,
                "lora_parameter_count": lora_parameter_count,
                "projector_parameter_count": projector_parameter_count,
                "lora_initial_l2_norm": lora_initial_norm,
                "projector_initial_l2_norm": projector_initial_norm,
                "grad_rms_formula": "grad_l2_norm/sqrt(parameter_count)",
                "relative_step_formula": (
                    "group_lr*grad_l2_norm/initial_parameter_l2_norm; "
                    "diagnostic approximation before Adam preconditioning")},
            "reward_weights": TASK_REWARD_WEIGHTS,
            "micro_f1_objective": {
                "target": MICRO_F1_TARGET,
                "utility_floor": MICRO_UTILITY_FLOOR,
                "formula": "(2*TP-target*Pred)/(GT*(2-target))",
                "raw_formula": "2*TP-target*Pred",
                "training_soft_tp_formula":
                    "clip((matched_IoU-0.5)/(1-0.5),0,1)",
                "training_advantage_raw_formula":
                    (f"2*sum({1.0-SOFT_TP_QUALITY_BLEND:.2f}*hard_TP+"
                     f"{SOFT_TP_QUALITY_BLEND:.2f}*soft_TP)-target*Pred"),
                "matching": "external_csv_gt_order_greedy_iou_gt_0.5",
                "invalid_predictions_count_as_fp": True,
                "iou_reward": "sum(IoU of thresholded TP)/GT"},
            "undergen_penalty": UNDERGEN_PENALTY,
            "overgen_penalty": OVERGEN_PENALTY,
            "overgen_ratio_cap": OVERGEN_RATIO_CAP,
            "task_repeat": TASK_REPEAT,
            "sampled_task_counts": dict(dataset.sampled_task_counts),
            "cost_balanced_sampler": {
                "enabled": True,
                "per_rank_samples": len(sampler),
                "bucket_counts": {
                    key: len(indices) for key, indices in sampler.buckets.items()}},
            "image_augmentation": "none",
            "iou_threshold": IOU_THRESHOLD,
            "total_steps": global_step,
            "actual_optimizer_updates": optimizer_update_step,
            "final_avg_reward": final_avg_reward,
            "micro_samples_jsonl": MICRO_LOG_FILE,
            "monitor_json": MONITOR_JSON_FILE,
            "monitor_history": monitor_history,
            "reward_history": reward_history, "loss_history": loss_history}
        stats_path = os.path.join(GRPO_OUTPUT_DIR, "grpo_training_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n{'=' * 60}")
        print("GRPO 训练完成!")
        print(f"  LoRA 权重:  {GRPO_LORA_SAVE}")
        print(f"  训练统计:   {stats_path}")
        print(f"  日志:       {LOG_FILE}")
        print(f"  异常样本JSONL: {MICRO_LOG_FILE}")
        print(f"  窗口监控JSON: {MONITOR_JSON_FILE}")
        print(f"  完成时间:   {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'=' * 60}")

        close_micro_log()
        release_training_lock()
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

    if is_ddp:
        dist.destroy_process_group()


# ========================= 主入口 =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 优化 LoRA 权重 — 棉花包语义定位")
    parser.add_argument(
        "--resume_state", type=str, default=None,
        help="从CGPC checkpoint目录内trainer_state.pt精确恢复；reference仍为原始SFT")
    args = parser.parse_args()

    # 唯一fresh训练链路：从原始SFT LoRA启动；只允许恢复同一CGPC run。
    train(
        policy_lora_path=POLICY_LORA_PATH,
        resume_state_path=args.resume_state)
