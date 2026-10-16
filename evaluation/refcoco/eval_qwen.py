"""Evaluate Qwen3-VL Base, LoRA, or CGPC checkpoints on dense RefCOCO."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import os as _release_os
import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IOU_THRESHOLD = 0.5
TASK_MAX_TOKENS = {"ALL": 6144, "LEFT": 1024, "RIGHT": 1024, "MIDDLE": 1024}
CATEGORY_KEYS = ["ALL", "LEFT", "RIGHT", "MIDDLE"]
CATEGORY_NAMES = {"ALL": "检测所有对象", "LEFT": "检测最左侧对象",
                  "RIGHT": "检测最右侧对象", "MIDDLE": "检测最中间对象"}
METRIC_NAMES = ["CountAcc", "HitMacro", "Recall", "Precision", "F1", "mIoU", "OverGen"]

def _env(name, default):
    value = os.environ.get(name, "").strip()
    return value or default

class Tee:
    """同时输出到终端和日志文件 + 实时 flush"""

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

def _extract_json_text(text):
    """从模型输出中截取一段可被 json.loads 解析的完整 JSON。

    兼容 markdown 代码围栏、前后说明文字、顶层对象与顶层数组两种结构。
    """
    if not text:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    first_idx, first_ch = len(s), None
    for ch in ("{", "["):
        idx = s.find(ch)
        if idx != -1 and idx < first_idx:
            first_idx, first_ch = idx, ch
    if first_ch is None:
        return None
    stack, in_str, esc = [], False, False
    for i in range(first_idx, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                return s[first_idx:i]
            opener = stack.pop()
            if (opener == "{" and c != "}") or (opener == "[" and c != "]"):
                return None
            if not stack:
                return s[first_idx:i + 1]
    return None

def _coerce_box_list(value):
    """把 bbox_2d 归一化为二维列表 [[x1,y1,x2,y2], ...]。"""
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], list):
        return [[b[0], b[1], b[2], b[3]] for b in value
                if isinstance(b, list) and len(b) >= 4]
    if len(value) >= 4:
        return [[value[0], value[1], value[2], value[3]]]
    return []

def _coerce_label_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is not None:
        return [str(value)]
    return []

def parse_bbox_json(text):
    return protocol.parse_grounding_answer(text)

def scale_boxes_to_image(boxes, image_size):
    return protocol.scale_boxes_preserving_count(boxes, image_size)

def sort_boxes(boxes, labels=None):
    return protocol.sort_boxes_with_labels(boxes, labels)

def classify_prompt(prompt_text):
    """所有 -> ALL，最左侧 -> LEFT，最右侧 -> RIGHT，最中间 -> MIDDLE。"""
    p = prompt_text.replace("<image>\n", "")
    if "所有" in p:
        return "ALL"
    if "最左侧" in p:
        return "LEFT"
    if "最右侧" in p:
        return "RIGHT"
    if "最中间" in p:
        return "MIDDLE"
    return "ALL"

def extract_target_category(prompt_text):
    p = prompt_text.replace("<image>\n", "")
    match = re.search(
        r'labels["”]?\s*[:：]\s*["“]?([^"”}\s,]+(?:\s+[^"”}\s,]+)*)', p)
    return match.group(1).strip().lower() if match else "unknown"

def match_boxes(gt_boxes, pred_boxes):
    matches = protocol.ordered_box_matches(gt_boxes, pred_boxes, IOU_THRESHOLD)
    tp = len(matches)
    return (tp, len(pred_boxes)-tp, len(gt_boxes)-tp,
            sum(m[0] for m in matches)/tp if tp else 0.0,
            tp/len(gt_boxes) if gt_boxes else 0.0)

class MetricsAccumulator:
    def __init__(self):
        self.N = 0
        self.total_hit_macro = 0.0
        self.total_count_acc = 0
        self.total_over_gen = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_iou_sum = 0.0

    def add_sample(self, gt_count, pred_count, tp, fp, fn, mean_iou, hit):
        self.N += 1
        self.total_hit_macro += hit
        self.total_count_acc += 1 if pred_count == gt_count else 0
        self.total_over_gen += 1 if pred_count > gt_count else 0
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_iou_sum += mean_iou

    def compute(self):
        n = max(self.N, 1)
        return OrderedDict([
            ("样本数", self.N),
            ("CountAcc", round(self.total_count_acc / n, 4)),
            ("HitMacro", round(self.total_hit_macro / n, 4)),
            ("Recall", round(self.total_tp / (self.total_tp + self.total_fn + 1e-6), 4)),
            ("Precision", round(self.total_tp / (self.total_tp + self.total_fp + 1e-6), 4)),
            ("F1", round(2 * self.total_tp /
                         (2 * self.total_tp + self.total_fp + self.total_fn + 1e-6), 4)),
            ("mIoU", round(self.total_iou_sum / n, 4)),
            ("OverGen", round(self.total_over_gen / n, 4)),
        ])

def evaluate_records(records):
    """按指令类型 + 总体聚合指标。"""
    acc = {key: MetricsAccumulator() for key in CATEGORY_KEYS}
    acc["OVERALL"] = MetricsAccumulator()
    for record in records:
        for target in (record["category"], "OVERALL"):
            acc[target].add_sample(
                record["gt_count"], record["pred_count"], record["tp"],
                record["fp"], record["fn"], record["miou"], record["hit"])
    return {key: value.compute() for key, value in acc.items()}

def print_stage_table(stage_title, metrics):
    print("\n" + "=" * 100)
    print(f"【{stage_title}】指标")
    print("=" * 100)
    header = (f"{'类别':<12s} {'样本':>6s}" +
              "".join(f" {name:>10s}" for name in METRIC_NAMES))
    print(header)
    print("-" * len(header))
    for key in ["OVERALL"] + CATEGORY_KEYS:
        values = metrics[key]
        name = "总体" if key == "OVERALL" else CATEGORY_NAMES[key]
        print(f"{name:<12s} {values['样本数']:>6d}" +
              "".join(f" {values[m]:>10.4f}" for m in METRIC_NAMES))
    print("=" * 100)

class QwenVLModel:
    def __init__(self, base_path, lora_path):
        print(f"[{datetime.now():%H:%M:%S}] 加载 processor: {base_path}")
        self.processor = AutoProcessor.from_pretrained(
            base_path, trust_remote_code=True)
        # decoder-only 模型批量生成必须 left-padding
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = \
                self.processor.tokenizer.eos_token_id

        print(f"[{datetime.now():%H:%M:%S}] 加载基础模型权重: {base_path}")
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_path, attn_implementation="flash_attention_2", **load_kwargs)
            print("  FlashAttention2 enabled.")
        except Exception as exc:  # noqa: BLE001 - 退回 sdpa 属预期分支
            print(f"  FlashAttention2 unavailable: {exc}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_path, **load_kwargs)

        if lora_path:
            print(f"加载适配器: {lora_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            self.model = base_model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[{datetime.now():%H:%M:%S}] 模型就绪, device={self.device}")

    def generate(self, image_path, prompt, category):
        image = Image.open(image_path).convert("RGB")
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=TASK_MAX_TOKENS.get(category, 6144),
                do_sample=False)
        outputs = outputs[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

def build_record(rank, source_idx, sample, image_path, image_size,
                 pred_text, stage_key):
    prompt = sample["conversations"][0]["value"]
    category = classify_prompt(prompt)
    object_category = extract_target_category(prompt)

    gt_boxes, gt_labels = parse_bbox_json(sample["conversations"][1]["value"])
    pred_boxes, pred_labels = parse_bbox_json(pred_text)
    gt_boxes = scale_boxes_to_image(gt_boxes, image_size)
    pred_boxes = scale_boxes_to_image(pred_boxes, image_size)
    gt_boxes, gt_labels = sort_boxes(gt_boxes, gt_labels)
    pred_boxes, pred_labels = sort_boxes(pred_boxes, pred_labels)

    tp, fp, fn, mean_iou, hit = match_boxes(gt_boxes, pred_boxes)
    return {
        "stage": stage_key,
        "rank": rank,
        "source_idx": source_idx,
        "image_rel": sample["images"][0],
        "image_path": image_path,
        "category": category,
        "object_category": object_category,
        "prompt": prompt,
        "raw_output": pred_text,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "pred_boxes": pred_boxes,
        "pred_labels": pred_labels,
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "miou": round(mean_iou, 4),
        "hit": round(hit, 4),
    }

BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/Qwen3-VL-2B-Instruct')
ADAPTER_PATH = _release_os.environ.get('CGPC_ADAPTER_PATH', './checkpoints/cgpc')
DATA_JSON = _release_os.environ.get('CGPC_EVAL_JSON', './data/valid.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')
OUTPUT_DIR = _env("CGPC_EVAL_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "eval_refcoco_qwen"))


def matching_details(gt_boxes, pred_boxes):
    matches = [{'gt_index': gi, 'pred_index': pi, 'iou': score, 'is_tp': True}
               for score, gi, pi in protocol.ordered_box_matches(gt_boxes, pred_boxes, IOU_THRESHOLD)]
    gt_used = {m['gt_index'] for m in matches}
    pred_used = {m['pred_index'] for m in matches}
    return {'assignments': matches, 'matches': matches,
            'unmatched_gt_indices': [i for i in range(len(gt_boxes)) if i not in gt_used],
            'unmatched_pred_indices': [i for i in range(len(pred_boxes)) if i not in pred_used]}


def enrich_record(record, sample, image_size):
    """Add export fields without changing original metric fields or calculations."""
    gt_text = sample['conversations'][1]['value']
    gt_raw, _ = parse_bbox_json(gt_text)
    pred_raw, _ = parse_bbox_json(record['raw_output'])
    details = matching_details(record['gt_boxes'], record['pred_boxes'])
    assert len(details['matches']) == record['tp']
    tp, fp, fn = record['tp'], record['fp'], record['fn']
    pair_key = hashlib.sha256(json.dumps([record['image_rel'], record['prompt']],
        ensure_ascii=False).encode('utf-8')).hexdigest()
    record.update({
        'sample_id': record['source_idx'], 'pair_key': pair_key,
        'image_name': os.path.basename(record['image_rel']),
        'image_width': image_size[0], 'image_height': image_size[1],
        'ground_truth_text': gt_text,
        'gt_boxes_1000_raw': gt_raw, 'pred_boxes_1000_raw': pred_raw,
        'coordinate_format': 'xyxy pixels; indices refer to sorted gt_boxes/pred_boxes',
        'matching': details,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'count_acc': int(record['gt_count'] == record['pred_count']),
        'over_gen': int(record['pred_count'] > record['gt_count']),
    })
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('base', 'lora', 'cgpc'), default='cgpc')
    parser.add_argument('--base-model-path', default=BASE_MODEL_PATH)
    parser.add_argument('--adapter-path', default=ADAPTER_PATH,
                        help='Required for LoRA and CGPC; ignored for Base.')
    parser.add_argument('--data-json', default=DATA_JSON)
    parser.add_argument('--image-root', default=IMAGE_ROOT)
    parser.add_argument('--output-dir', default=None,
                        help='Exact new run directory; default OUTPUT_DIR/timestamp')
    return parser.parse_args()


def main():
    args = parse_args()
    stage = args.stage
    adapter_path = None if stage == 'base' else args.adapter_path
    if stage != 'base' and not adapter_path:
        raise ValueError('LoRA and CGPC require a nonempty adapter path')
    run_dir = args.output_dir or os.path.join(OUTPUT_DIR,
        datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    os.makedirs(run_dir, exist_ok=True)
    predictions_path = os.path.join(run_dir, f'predictions_{stage}.txt')
    if os.path.exists(predictions_path):
        raise FileExistsError(f'{predictions_path} exists; use a new output directory.')
    log_path = os.path.join(run_dir, 'eval.log')
    metadata_path = os.path.join(run_dir, 'run_metadata.json')
    tee = Tee(log_path)
    sys.stdout = tee
    model = None
    model_title = os.path.basename(os.path.normpath(args.base_model_path))
    metadata = {'status': 'running', 'model': model_title, 'stage': stage,
        'base_model_path': args.base_model_path, 'adapter_path': adapter_path,
        'data_json': args.data_json, 'image_root': args.image_root,
        'expected_records': None, 'exported_records': 0,
        'record_selection': 'all source records, no image deduplication',
        'task_max_new_tokens': TASK_MAX_TOKENS, 'do_sample': False,
        'matching': 'spatial GT-order greedy, one-to-one, strict IoU > 0.5',
        'metric_scale': '0 to 1', 'started_at': datetime.now().isoformat()}
    def save_metadata():
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    save_metadata()
    try:
        with open(args.data_json, 'rb') as handle:
            data_bytes = handle.read()
        dataset = json.loads(data_bytes.decode('utf-8-sig'))
        metadata['dataset_sha256'] = hashlib.sha256(data_bytes).hexdigest()
        metadata['expected_records'] = len(dataset)
        save_metadata()
        print(f'{model_title}: {stage}, all {len(dataset)} RefCOCO records')
        print(f'Adapter: {adapter_path}')
        print(f'Predictions: {predictions_path}')
        global torch, PeftModel, AutoModelForImageTextToText, AutoProcessor
        import torch
        if adapter_path:
            from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = QwenVLModel(args.base_model_path, adapter_path)
        records = []
        started = time.perf_counter()
        with open(predictions_path, 'w', encoding='utf-8') as predictions, open(
                os.path.join(run_dir, f'result_{stage}.csv'), 'w', newline='',
                encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['sample_id', 'image', 'prompt', 'task', 'target_category',
                'gt_count', 'pred_count', 'TP', 'FP', 'FN', 'Precision', 'Recall',
                'F1', 'mIoU', 'HitMacro', 'CountAcc', 'OverGen'])
            for source_idx, sample in enumerate(dataset):
                image_path = os.path.join(args.image_root, sample['images'][0])
                with Image.open(image_path) as image:
                    image_size = image.size
                prompt = sample['conversations'][0]['value']
                category = classify_prompt(prompt)
                pred_text = model.generate(image_path, prompt, category)
                record = build_record(source_idx + 1, source_idx, sample, image_path,
                                      image_size, pred_text, stage)
                enrich_record(record, sample, image_size)
                predictions.write(json.dumps(record, ensure_ascii=False) + '\n')
                predictions.flush()
                writer.writerow([record['sample_id'], record['image_rel'], prompt,
                    category, record['object_category'], record['gt_count'],
                    record['pred_count'], record['tp'], record['fp'], record['fn'],
                    record['precision'], record['recall'], record['f1'], record['miou'],
                    record['hit'], record['count_acc'], record['over_gen']])
                csv_file.flush()
                # Keep only aggregation fields in memory, not full long responses.
                records.append({k: record[k] for k in ('category', 'gt_count', 'pred_count',
                    'tp', 'fp', 'fn', 'miou', 'hit')})
                metadata['exported_records'] = source_idx + 1
                elapsed = time.perf_counter() - started
                print(f'[{source_idx + 1}/{len(dataset)}] {category} '
                      f'GT={record["gt_count"]} Pred={record["pred_count"]} '
                      f'F1={record["f1"]:.4f} {elapsed / (source_idx + 1):.1f}s/record')
        assert len(records) == len(dataset)
        metrics = evaluate_records(records)
        print_stage_table(f'{stage.upper()} RefCOCO', metrics)
        with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump({stage: {'adapter': adapter_path, 'metrics': metrics}},
                      handle, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, f'metrics_{stage}.csv'), 'w', newline='',
                  encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['category', 'samples'] + METRIC_NAMES)
            for category in ['OVERALL'] + CATEGORY_KEYS:
                values = metrics[category]
                writer.writerow([category, values['样本数']] + [values[k] for k in METRIC_NAMES])
        metadata['status'] = 'complete'
        print(f'Complete: {len(records)} records exported to {predictions_path}')
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        print(f'Evaluation interrupted: {metadata["error"]}; completed records retained.')
        raise
    finally:
        metadata['finished_at'] = datetime.now().isoformat()
        try:
            save_metadata()
            if model is not None:
                model.close()
        finally:
            sys.stdout = tee.terminal
            tee.log.close()


if __name__ == '__main__':
    main()
