"""Evaluate Qwen3-VL Base, LoRA, or CGPC checkpoints on cotton-bale data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_tools import grounding_protocol as protocol
import os as _release_os
import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from PIL import Image

BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/Qwen3-VL-2B-Instruct')
ADAPTER_PATH = _release_os.environ.get('CGPC_ADAPTER_PATH', './checkpoints/cgpc')
DATA_JSON = _release_os.environ.get('CGPC_EVAL_JSON', './data/valid.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')
CN_NUM = r"[一二三四五六七八九十两]+"
TASK_MAX_TOKENS = {'SINGLE': 1024, 'ROW': 3072, 'ALL': 10240}
CATEGORY_NAMES = {'ALL': '检测所有棉包', 'ROW': '检测指定排', 'SINGLE': '检测指定排+指定个'}

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

def parse_bbox_json(text):
    return protocol.parse_grounding_answer(text)

def scale_boxes_to_image(boxes, image_path):
    with Image.open(image_path) as image:
        return protocol.scale_boxes_preserving_count(boxes, image.size)

def sort_boxes(boxes, labels=None):
    return protocol.sort_boxes_with_labels(boxes, labels)

def classify_prompt(prompt_text):
    """返回: 'ALL' | 'ROW' | 'SINGLE'"""
    p = prompt_text.replace("<image>\n", "")
    if "所有棉包" in p or "全部棉包" in p:
        return "ALL"
    has_row_num = bool(re.search(r"第" + CN_NUM + r"排", p))
    has_pos_num = bool(re.search(r"第" + CN_NUM + r"个", p))
    if has_row_num and has_pos_num:
        return "SINGLE"
    if has_row_num or "排" in p:
        return "ROW"
    return "ROW"

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

    def add_sample(self, gt_boxes, pred_boxes):
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)

        matched = set()
        tp = 0
        iou_list = []
        for g in gt_boxes:
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(pred_boxes):
                if j in matched:
                    continue
                cur_iou = iou(g, p)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_j = j
            if best_iou > 0.5:
                tp += 1
                matched.add(best_j)
                iou_list.append(best_iou)

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
        n = max(self.N, 1)
        return OrderedDict([
            ("样本数", self.N),
            ("CountAcc", round(self.total_count_acc / n, 4)),
            ("HitMacro", round(self.total_hit_macro / n, 4)),
            ("Recall", round(self.total_tp / (self.total_tp + self.total_fn + 1e-6), 4)),
            ("Precision", round(self.total_tp / (self.total_tp + self.total_fp + 1e-6), 4)),
            ("F1", round(2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn + 1e-6), 4)),
            ("mIoU", round(self.total_iou_sum / n, 4)),
            ("OverGen", round(self.total_over_gen / n, 4)),
        ])

class QwenVLModel:
    def __init__(self, base_path, adapter_path):
        print(f"[{datetime.now():%H:%M:%S}] 加载基础模型: {base_path}")
        self.processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        # decoder-only 模型批量生成必须用 left-padding，right-padding 会导致
        # 模型从 PAD 位置继续生成 → 立即 EOS / 空文本 / 非法 JSON
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_path,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            print("FlashAttention2 enabled.")
        except Exception as e:
            print(f"FlashAttention2 unavailable: {e}")
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_path,
                dtype=torch.bfloat16,
                device_map="auto"
            )
        print(f"[{datetime.now():%H:%M:%S}] 加载适配器: {adapter_path}")
        self.model = (PeftModel.from_pretrained(base_model, adapter_path)
                      if adapter_path else base_model)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[{datetime.now():%H:%M:%S}] 模型就绪, device={self.device}")

    def generate_batch(self, image_paths, prompts, categories):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        texts = []
        max_tokens_list = []
        for prompt, cat in zip(prompts, categories):
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}
                ]}],
                add_generation_prompt=True, tokenize=False,
            )
            texts.append(text)
            max_tokens_list.append(TASK_MAX_TOKENS.get(cat, 10240))

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max(max_tokens_list), do_sample=False)
        outputs = outputs[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(outputs, skip_special_tokens=True)

def match_details(gt_boxes, pred_boxes):
    """Reference matching: GT-order greedy, one-to-one, strict IoU > 0.5."""
    matched = set()
    matches = []
    for gidx, gt in enumerate(gt_boxes):
        best_iou, best_j = 0.0, -1
        for j, pred in enumerate(pred_boxes):
            if j in matched:
                continue
            value = iou(gt, pred)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_iou > 0.5:
            matched.add(best_j)
            matches.append({'gt_index': gidx, 'pred_index': best_j, 'iou': best_iou})
    matched_gt = {m['gt_index'] for m in matches}
    return {
        'matches': matches,
        'unmatched_gt_indices': [j for j in range(len(gt_boxes)) if j not in matched_gt],
        'unmatched_pred_indices': [j for j in range(len(pred_boxes)) if j not in matched],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('base', 'lora', 'cgpc'), default='cgpc')
    parser.add_argument('--base-model-path', default=BASE_MODEL_PATH)
    parser.add_argument('--adapter-path', default=ADAPTER_PATH,
                        help='Required for LoRA and CGPC; ignored for Base.')
    parser.add_argument('--data-json', default=DATA_JSON)
    parser.add_argument('--image-root', default=IMAGE_ROOT)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    stage = args.stage
    adapter = None if stage == 'base' else args.adapter_path
    if stage != 'base' and not adapter:
        parser.error('LoRA and CGPC require a nonempty adapter path.')
    # Keep runs separate. TXT contains predictions only; metadata is a separate JSON.
    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / ('eval_qwen_' + stage + '_export') /
        datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.base_model_path).name.lower().replace('-', '_')
    txt_path = out / (model_tag + '_' + stage + '_predictions.txt')
    if txt_path.exists():
        raise FileExistsError(f'{txt_path} already exists; choose another --output-dir.')
    data_bytes = Path(args.data_json).read_bytes()
    dataset = json.loads(data_bytes.decode('utf-8-sig'))
    indexed = list(enumerate(dataset))
    # Seed changes only traversal order relative to the reference's unseeded shuffle.
    random.Random(42).shuffle(indexed)
    metadata = {
        'stage': stage, 'base_model': args.base_model_path, 'adapter': adapter,
        'data_json': args.data_json, 'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
        'image_root': args.image_root, 'expected_records': len(dataset),
        'exported_records': 0, 'status': 'running', 'batch_size': 1,
        'do_sample': False, 'task_max_new_tokens': TASK_MAX_TOKENS,
        'matching': 'GT-order greedy, one-to-one, strict IoU > 0.5; labels ignored for matching',
        'coordinate_format': 'xyxy, image pixels, sorted by y1 then x1',
        'metrics_scale': '0 to 1',
        'miou': 'mean over samples of mean IoU of matched TPs; zero for no TP',
        'parsing': 'reference parser, scaling, and label-dependent sorting retained unchanged',
        'started_at': datetime.now().isoformat(),
    }
    def save_metadata():
        (out / 'run_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    save_metadata()
    print(f'Stage: {stage}; adapter: {adapter}; dataset: {args.data_json}', flush=True)
    print(f'Output: {out}', flush=True)
    acc = {cat: MetricsAccumulator() for cat in ['ALL', 'ROW', 'SINGLE', 'OVERALL']}
    try:
        global torch, AutoProcessor, AutoModelForImageTextToText, PeftModel
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if adapter:
            from peft import PeftModel
        model = QwenVLModel(args.base_model_path, adapter)
        with txt_path.open('w', encoding='utf-8') as export, (out / 'per_sample.csv').open(
                'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['sample_id', 'image', 'instruction', 'task', 'GT', 'Pred',
                             'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'mIoU', 'CountAcc'])
            for ordinal, (sample_id, sample) in enumerate(indexed, 1):
                img_rel = sample['images'][0]
                image_path = os.path.join(args.image_root, img_rel)
                prompt = sample['conversations'][0]['value']
                gt_text = sample['conversations'][1]['value']
                cat = classify_prompt(prompt)
                pred_text = model.generate_batch([image_path], [prompt], [cat])[0]
                gt_raw, gt_labels_raw = parse_bbox_json(gt_text)
                pred_raw, pred_labels_raw = parse_bbox_json(pred_text)
                gt_scaled = scale_boxes_to_image(gt_raw, image_path)
                pred_scaled = scale_boxes_to_image(pred_raw, image_path)
                gt_boxes, gt_labels = sort_boxes(gt_scaled, gt_labels_raw)
                pred_boxes, pred_labels = sort_boxes(pred_scaled, pred_labels_raw)
                tp, fp, fn, avg_iou, hit = acc[cat].add_sample(gt_boxes, pred_boxes)
                acc['OVERALL'].add_sample(gt_boxes, pred_boxes)
                matching = match_details(gt_boxes, pred_boxes)
                assert len(matching['matches']) == tp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                with Image.open(image_path) as image:
                    width, height = image.size
                pair_key = hashlib.sha256(json.dumps([img_rel, prompt],
                    ensure_ascii=False).encode('utf-8')).hexdigest()
                record = {
                    'sample_id': sample_id, 'pair_key': pair_key, 'stage': stage,
                    'image_name': os.path.basename(img_rel), 'image_relative_path': img_rel,
                    'image_path': image_path, 'instruction': prompt, 'task': cat,
                    'image_width': width, 'image_height': height,
                    'prediction_text': pred_text, 'ground_truth_text': gt_text,
                    'pred_boxes_1000_raw': pred_raw, 'gt_boxes_1000_raw': gt_raw,
                    'pred_boxes_px_before_sort': pred_scaled,
                    'pred_boxes_px': pred_boxes, 'gt_boxes_px': gt_boxes,
                    'pred_labels_raw': pred_labels_raw, 'gt_labels_raw': gt_labels_raw,
                    'pred_labels': pred_labels, 'gt_labels': gt_labels,
                    'matching': matching,
                    'metrics': {'gt_count': len(gt_boxes), 'pred_count': len(pred_boxes),
                        'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall,
                        'F1': f1, 'mIoU': avg_iou, 'HitMacro': hit,
                        'CountAcc': int(len(pred_boxes) == len(gt_boxes)),
                        'OverGen': int(len(pred_boxes) > len(gt_boxes))},
                }
                export.write(json.dumps(record, ensure_ascii=False) + '\n')
                export.flush()
                writer.writerow([sample_id, img_rel, prompt, cat, len(gt_boxes), len(pred_boxes),
                                 tp, fp, fn, precision, recall, f1, avg_iou,
                                 int(len(pred_boxes) == len(gt_boxes))])
                csv_file.flush()
                metadata['exported_records'] = ordinal
                print(f'[{ordinal}/{len(dataset)}] id={sample_id} {cat} '
                      f'GT={len(gt_boxes)} Pred={len(pred_boxes)} TP={tp} F1={f1:.4f}', flush=True)
        assert metadata['exported_records'] == len(dataset)
        results = OrderedDict((('总体' if cat == 'OVERALL' else CATEGORY_NAMES[cat]),
                               acc[cat].compute()) for cat in ['OVERALL', 'ALL', 'ROW', 'SINGLE'])
        (out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
        metadata['status'] = 'complete'
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        print(f'All predictions: {txt_path}', flush=True)
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        # Keep completed records, but never report an incomplete run as full evaluation.
        raise
    finally:
        metadata['finished_at'] = datetime.now().isoformat()
        save_metadata()


if __name__ == '__main__':
    main()
