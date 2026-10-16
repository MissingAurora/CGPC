"""CGPC release: visualization/gradient_attribution.py. See the README in this directory for configuration."""
import os as _release_os

from pathlib import Path

# ======================== 用户配置 ========================
TEST_JSON = _release_os.environ.get('CGPC_EVAL_JSON', './data/valid.json')
IMAGE_ROOT = _release_os.environ.get('CGPC_IMAGE_ROOT', './data')
OUTPUT_DIR = _release_os.environ.get('CGPC_OUTPUT_DIR', './outputs/attribution')

BASE_MODEL_PATH = _release_os.environ.get('CGPC_BASE_MODEL', './models/Qwen3-VL-2B-Instruct')
LORA_PATH = _release_os.environ.get('CGPC_SFT_PATH', './checkpoints/lora')
CGPC_PATH = _release_os.environ.get('CGPC_ADAPTER_PATH', './checkpoints/cgpc')
DEVICE = "cuda:0"
DTYPE = "bfloat16"

# 归因信号：bbox_digits —— 只回传 bbox_2d 数组里的数字 token（推荐）；answer —— 整个回答。
LOSS_TOKENS = "bbox_digits"

# 绘图参数（与 gradcam_comparison_unfiltered.png 保持一致）
COLORMAP = "jet"            # 蓝→青→绿→黄→红
OVERLAY_ALPHA = 0.5         # 三模型相同透明度
NORM_MODE = "per-model"     # 各图除以自身百分位上限
CLIP_PERCENTILE = 98.0
SMOOTH_SIGMA = 1.0          # 34x60 网格高斯平滑
PRED_COLOR = "#ff0000"      # 红色：模型预测框
GT_COLOR = "#00a000"        # 绿色：真实标注框
BOX_LINEWIDTH = 1.2         # 统一线宽 pt
TITLE_COLOR = "#000000"
TEXT_COLOR = "#000000"
MUTED_COLOR = "#444444"
BORDER_COLOR = "#000000"
DRAW_BBOX = True            # 模型面板画红色预测框；输入面板不画
DRAW_GT = True              # 四个面板均画绿色 GT 框
DPI = 600
STAGES = ["base", "lora", "cgpc"]
SAMPLE_STRIDE = 3          # 测试集每隔 3 张取 1 张（索引 0,3,6,...）；None 或 1 表示全部
IOU_THRESHOLD = 0.5        # 预测框与 GT 框匹配的 IoU 阈值（用于 TP/FP/FN 汇总）
# ============================== 实现部分 ==============================

import argparse
import gc
import json
import re
import warnings

import numpy as np
from PIL import Image

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ordinal(n):
    if n % 10 == 1 and n % 100 != 11:
        return f"{n}st"
    if n % 10 == 2 and n % 100 != 12:
        return f"{n}nd"
    if n % 10 == 3 and n % 100 != 13:
        return f"{n}rd"
    return f"{n}th"


def make_display_instruction(instruction):
    """把中文定位指令压成一行英文标题，用于对比图 suptitle。"""
    if "所有" in instruction:
        return "Locate all bales in the image"
    m = re.search(r"第([一二三四五六七八九十]+)排第([一二三四五六七八九十]+)个", instruction)
    if m:
        row = CN_NUM.get(m.group(1), m.group(1))
        idx = CN_NUM.get(m.group(2), m.group(2))
        return f"Locate the {ordinal(idx)} bale in row {row}"
    m = re.search(r"第([一二三四五六七八九十]+)排", instruction)
    if m:
        row = CN_NUM.get(m.group(1), m.group(1))
        if "从下往上" in instruction:
            return f"Locate row {row} from the bottom"
        return f"Locate row {row} from the top"
    if "最上面两排" in instruction:
        return "Locate the top two rows"
    return "Gradient-based attribution comparison"


def parse_bboxes(text):
    """从回答文本里解析 bbox_2d 数组中的所有 [x1,y1,x2,y2]，返回 0-1000 归一化框列表。"""
    m = re.search(r'"bbox_2d"\s*:\s*\[', text)
    if m is None:
        return []
    start = m.end() - 1  # 指向 bbox_2d 的外层 '['
    depth, end = 0, None
    for pos in range(start, len(text)):
        if text[pos] == "[":
            depth += 1
        elif text[pos] == "]":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        return []
    boxes = []
    for a, b, c, d in re.findall(
            r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
            r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text[start:end]):
        a, b, c, d = float(a), float(b), float(c), float(d)
        if 0 <= a <= 1000 and 0 <= b <= 1000 and c > a and d > b:
            boxes.append([a, b, c, d])
    return boxes


def parse_gt(text):
    """从 GT 回答里解析 bbox_2d 与 labels（优先 json，失败回退正则）。"""
    try:
        obj = json.loads(text)
        raw = obj.get("bbox_2d", [])
        labels = obj.get("labels", [])
        boxes = []
        for b in raw:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = (float(v) for v in b)
                if 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        return boxes, labels
    except Exception:
        return parse_bboxes(text), []


def build_index():
    """读取整个测试集，产出 {index, name, image_path, instruction, display_instruction,
    gt_boxes, gt_labels} 列表。"""
    data = json.load(open(TEST_JSON, encoding="utf-8"))
    index = []
    for i, e in enumerate(data):
        if SAMPLE_STRIDE and SAMPLE_STRIDE > 1 and i % SAMPLE_STRIDE != 0:
            continue
        human = gpt = None
        for c in e.get("conversations", []):
            if c.get("from") == "human":
                human = c.get("value", "")
            elif c.get("from") == "gpt":
                gpt = c.get("value", "")
        instruction = human.replace("<image>", "").strip() if human else ""
        gt_boxes, gt_labels = parse_gt(gpt or "")
        img_rel = e.get("images", [""])[0]
        stem = Path(img_rel).stem
        index.append({
            "index": i,
            "name": f"{i:05d}_{stem}",
            "image_path": str(Path(IMAGE_ROOT) / img_rel),
            "image_rel": img_rel,
            "instruction": instruction,
            "display_instruction": make_display_instruction(instruction),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        })
    return index


def select_range(items, start, end, limit):
    items = items[start:end]
    if limit is not None:
        items = items[:limit]
    return items


def task_type(instruction):
    """把指令归类为稳定的任务类型键（用于汇总分组）。"""
    if "所有" in instruction:
        return "all_bales"
    m = re.search(r"第([一二三四五六七八九十]+)排第([一二三四五六七八九十]+)个", instruction)
    if m:
        row = CN_NUM.get(m.group(1), m.group(1))
        col = CN_NUM.get(m.group(2), m.group(2))
        return f"bale_r{row}_c{col}"
    m = re.search(r"第([一二三四五六七八九十]+)排", instruction)
    if m:
        row = CN_NUM.get(m.group(1), m.group(1))
        return f"row_bottom_{row}" if "从下往上" in instruction else f"row_top_{row}"
    if "最上面两排" in instruction:
        return "top_two_rows"
    return "other"


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_thresh=IOU_THRESHOLD):
    """贪心匹配预测框与 GT 框，返回 (tp, fp, fn)。"""
    cand = []
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou = compute_iou(p, g)
            if iou >= iou_thresh:
                cand.append((iou, i, j))
    cand.sort(key=lambda t: t[0], reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for _, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ------------------------ 归因计算（与单图脚本一致） ------------------------

def bbox_number_token_mask(tokenizer, ids):
    """返回 answer token 中属于 bbox_2d 数字部分的布尔掩码（与 token 对齐）。"""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    match = re.search(r'"bbox_2d"\s*:\s*\[', text)
    if match is None:
        return None
    start, depth, end = match.end() - 1, 0, None
    for pos in range(start, len(text)):
        if text[pos] == "[":
            depth += 1
        elif text[pos] == "]":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        return None
    spans = [(start + m.start(), start + m.end())
             for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text[start:end])]
    if not spans:
        return None
    offsets, previous = [], ""
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        offsets.append((len(previous), len(prefix)))
        previous = prefix
    mask = [False] * len(ids)
    for i, (a, b) in enumerate(offsets):
        if b > a and any(a < e and b > s for s, e in spans):
            mask[i] = True
    return mask if any(mask) else None


def gradient_saliency(model, pixel_values, image_grid_thw, full_ids, prompt_len, target_mask):
    """对 target_mask 标记的 answer token 求 NLL，反向传播得到视觉 token saliency。"""
    import torch
    captured = {}

    def hook(module, args, kwargs, output):
        emb = output[0]
        emb.retain_grad()
        captured["emb"] = emb
        return output

    handle = model.model.visual.register_forward_hook(hook, with_kwargs=True)
    n = int(target_mask.sum().item())
    positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(target_mask),
                             device=full_ids.device)[target_mask]
    targets = full_ids[0, positions + 1]
    try:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=full_ids, pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw, use_cache=False)
        logits = out.logits[0]
        logp = torch.log_softmax(logits[positions].float(), dim=-1)
        loss = -logp[torch.arange(n, device=logits.device), targets].sum()
        loss.backward()
        emb = captured["emb"].detach()
        grad = captured["emb"].grad
        sal = (grad * emb).sum(dim=-1).abs().float().cpu().numpy()
    finally:
        handle.remove()
    return sal, n


def generate_answer(model, inputs, budget=256):
    import torch
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
                             use_cache=True, cache_implementation="dynamic")
    prompt_len = inputs.input_ids.shape[1]
    ids = gen[0, prompt_len:].cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    while ids and ids[-1] in eos:
        ids.pop()
    return ids, gen


def process_one(model, processor, image, instruction, out_dir, stage):
    """对一张图跑一个 stage，保存 response/saliency/metadata，返回 detail。"""
    import torch
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                       {"type": "text", "text": instruction}]}],
        add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    prompt_len = int(inputs.input_ids.shape[1])
    ids = inputs.input_ids[0]
    image_token_id = model.config.image_token_id
    positions = torch.where(ids == image_token_id)[0].cpu().tolist()
    grid = inputs.image_grid_thw.cpu().tolist()
    if len(grid) != 1 or grid[0][0] != 1:
        raise ValueError("本脚本只支持单张静态图像。")
    _, height, width = grid[0]
    merge = int(model.config.vision_config.spatial_merge_size)
    shape = (height // merge, width // merge)
    if len(positions) != shape[0] * shape[1]:
        raise RuntimeError("视觉 token 数与空间网格不一致。")

    answer_ids, gen = generate_answer(model, inputs)
    answer_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
    (out_dir / f"{stage}_response.txt").write_text(answer_text, encoding="utf-8")
    if not answer_ids:
        raise RuntimeError(f"[{stage}] 模型未生成回答。")

    if LOSS_TOKENS == "bbox_digits":
        mask = bbox_number_token_mask(processor.tokenizer, answer_ids)
        if mask is None:
            warnings.warn(f"[{stage}] 回答里没有 bbox_2d，回退到整个回答的似然。")
            mask = [True] * len(answer_ids)
    elif LOSS_TOKENS == "answer":
        mask = [True] * len(answer_ids)
    else:
        raise ValueError("LOSS_TOKENS 只能为 bbox_digits 或 answer。")
    target_mask = torch.tensor(mask, device=DEVICE)

    full_ids = torch.cat([inputs.input_ids,
                          torch.tensor([answer_ids], device=DEVICE)], dim=1)
    sal, n_tok = gradient_saliency(model, inputs.pixel_values, inputs.image_grid_thw,
                                   full_ids, prompt_len, target_mask)
    salmap = sal.reshape(shape)
    np.savez_compressed(out_dir / f"{stage}_saliency.npz", saliency=salmap,
                        raw_saliency=sal, answer_ids=np.array(answer_ids),
                        image_positions=np.array(positions), image_grid_thw=np.array(grid))
    detail = {"stage": stage, "spatial_grid": list(shape),
              "visual_token_count": len(positions), "prompt_token_count": prompt_len,
              "output_token_count": len(answer_ids), "loss_tokens": LOSS_TOKENS,
              "target_token_count": n_tok, "response": answer_text,
              "bboxes": parse_bboxes(answer_text),
              "saliency_max": float(salmap.max()), "saliency_mean": float(salmap.mean())}
    write_json(out_dir / f"{stage}_metadata.json", detail)
    return detail


# ------------------------ 绘图（unfiltered，与参考图同款） ------------------------

def prepare_maps(maps):
    from scipy.ndimage import gaussian_filter
    smoothed = [gaussian_filter(a, sigma=SMOOTH_SIGMA) if SMOOTH_SIGMA > 0
                else np.array(a, copy=True) for a in maps]
    scales = [float(np.percentile(s, CLIP_PERCENTILE)) for s in smoothed]
    normalized = [np.clip(s / max(scale, 1e-12), 0, 1)
                  for s, scale in zip(smoothed, scales)]
    return normalized, scales


def render_sample(rec, sample_dir, comparison_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    image = Image.open(sample_dir / "input_image.png").convert("RGB")
    maps = [np.load(sample_dir / f"{s}_saliency.npz")["saliency"] for s in STAGES]
    if len({a.shape for a in maps}) != 1:
        raise RuntimeError(f"{rec['name']}: 三组视觉网格不同，不能直接比较。")
    if any(not np.isfinite(a).all() for a in maps):
        raise ValueError(f"{rec['name']}: saliency 数据应为有限值。")

    smoothed, scales = prepare_maps(maps)
    gt_boxes = rec["gt_boxes"]
    pred_boxes = {}
    for s in STAGES:
        resp = (sample_dir / f"{s}_response.txt").read_text(encoding="utf-8")
        pred_boxes[s] = parse_bboxes(resp)

    cmap = plt.get_cmap(COLORMAP)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
                         "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    width, height = image.size
    figure_width = 183 / 25.4
    figure_height = figure_width / 4 * height / width + 1.00
    fig, axes = plt.subplots(1, 4, figsize=(figure_width, figure_height))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.34, top=0.86, wspace=0.025)
    titles = ["(a) Input image", "(b) Base", "(c) LoRA", "(d) LoRA + CGPC"]
    image_array = np.asarray(image).astype(np.float32) / 255
    norm = Normalize(vmin=0, vmax=1, clip=True)

    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.imshow(image, interpolation="nearest")
        if i:
            enlarged = np.asarray(Image.fromarray(smoothed[i - 1].astype(np.float32)).resize(
                image.size, getattr(Image, "Resampling", Image).BILINEAR))
            overlay = ((1 - OVERLAY_ALPHA) * image_array
                       + OVERLAY_ALPHA * cmap(norm(enlarged))[..., :3])
            ax.imshow(overlay)
            Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255)).save(
                sample_dir / f"{STAGES[i - 1]}_overlay.png")
        if DRAW_GT:
            for x1, y1, x2, y2 in gt_boxes:
                ax.add_patch(Rectangle((x1 / 1000 * width, y1 / 1000 * height),
                    (x2 - x1) / 1000 * width, (y2 - y1) / 1000 * height, fill=False,
                    edgecolor=GT_COLOR, linewidth=BOX_LINEWIDTH, zorder=4))
        if i > 0 and DRAW_BBOX:
            for x1, y1, x2, y2 in pred_boxes[STAGES[i - 1]]:
                ax.add_patch(Rectangle((x1 / 1000 * width, y1 / 1000 * height),
                    (x2 - x1) / 1000 * width, (y2 - y1) / 1000 * height, fill=False,
                    edgecolor=PRED_COLOR, linewidth=BOX_LINEWIDTH, zorder=5))
        ax.set_title(title, fontsize=8, fontweight="bold", pad=5, color=TITLE_COLOR)
        ax.axis("off")

    fig.suptitle(rec["display_instruction"], fontsize=9, y=0.98)
    handles = []
    if DRAW_BBOX:
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=PRED_COLOR,
                                 linewidth=BOX_LINEWIDTH, label="Prediction"))
    if DRAW_GT:
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=GT_COLOR,
                                 linewidth=BOX_LINEWIDTH, label="Ground truth"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .255),
               ncol=2, frameon=False, fontsize=7)
    bar_ax = fig.add_axes([0.38, 0.19, 0.34, 0.030])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax, orientation="horizontal")
    bar.set_ticks([0, 0.5, 1])
    bar.set_ticklabels(["low", "", "high"])
    bar.ax.tick_params(labelsize=7, length=2, pad=1, colors=TEXT_COLOR)
    bar.outline.set_edgecolor(BORDER_COLOR)
    bar.outline.set_linewidth(0.8)
    bar.set_label("Gradient-based attribution", fontsize=7, labelpad=1)
    scale_note = "normalized within each model" if NORM_MODE == "per-model" else "shared scale across models"
    filter_note = "No GT neighborhood weighting"
    fig.text(.5, .018, f"{filter_note}; {scale_note}", ha="center", fontsize=7, color=MUTED_COLOR)
    fig.savefig(comparison_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    write_json(sample_dir / "render_metadata.json", {
        "colormap": COLORMAP, "alpha": OVERLAY_ALPHA, "norm_mode": NORM_MODE,
        "clip_percentile": CLIP_PERCENTILE, "smooth_sigma": SMOOTH_SIGMA,
        "resize": "bilinear", "loss_tokens": LOSS_TOKENS, "input_noise": False,
        "normalization_scales": scales, "gt_boxes": gt_boxes,
        "bbox_filter": False, "prediction_color_hex": PRED_COLOR,
        "gt_color_hex": GT_COLOR, "box_linewidth_pt": BOX_LINEWIDTH,
        "display_instruction": rec["display_instruction"]})


def write_run_metadata(rec, sample_dir):
    stages = []
    for s in STAGES:
        meta = json.loads((sample_dir / f"{s}_metadata.json").read_text(encoding="utf-8"))
        stages.append(meta)
    write_json(sample_dir / "run_metadata.json", {
        "sample_index": rec["index"], "name": rec["name"],
        "image_path": rec["image_path"], "instruction": rec["instruction"],
        "display_instruction": rec["display_instruction"],
        "gt_boxes": rec["gt_boxes"], "gt_labels": rec["gt_labels"],
        "loss_tokens": LOSS_TOKENS, "attribution_version": 2,
        "input_noise": False, "dtype": DTYPE,
        "method": "gradient * embedding saliency of model's own greedy bbox answer",
        "stages": stages})


def write_summary(root, samples_dir):
    """扫描已计算的样本，输出每张图的数据汇总（gt / tp / fp / fn / 任务类型）。"""
    import csv
    index = {r["name"]: r for r in json.loads((root / "samples_index.json").read_text(encoding="utf-8"))}
    rows = []
    for d in sorted(samples_dir.iterdir()):
        if not d.is_dir():
            continue
        rec = index.get(d.name)
        if rec is None or not (d / "base_metadata.json").exists():
            continue
        row = {"index": rec["index"], "name": d.name, "image": rec["image_rel"],
               "task_type": task_type(rec["instruction"]),
               "task": rec["display_instruction"], "gt_count": len(rec["gt_boxes"])}
        for s in STAGES:
            meta_path = d / f"{s}_metadata.json"
            if not meta_path.exists():
                for k in ("pred", "tp", "fp", "fn", "prec", "rec"):
                    row[f"{s}_{k}"] = None
                continue
            preds = json.loads(meta_path.read_text(encoding="utf-8")).get("bboxes", [])
            tp, fp, fn = match_boxes(preds, rec["gt_boxes"])
            row[f"{s}_pred"] = len(preds)
            row[f"{s}_tp"] = tp
            row[f"{s}_fp"] = fp
            row[f"{s}_fn"] = fn
            row[f"{s}_prec"] = round(tp / (tp + fp), 4) if tp + fp else 0.0
            row[f"{s}_rec"] = round(tp / (tp + fn), 4) if tp + fn else 0.0
        rows.append(row)

    cols = ["index", "name", "image", "task_type", "task", "gt_count"]
    for s in STAGES:
        cols += [f"{s}_pred", f"{s}_tp", f"{s}_fp", f"{s}_fn", f"{s}_prec", f"{s}_rec"]
    with open(root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    agg = {}
    for r in rows:
        a = agg.setdefault(r["task_type"], {"samples": 0, "gt_total": 0})
        a["samples"] += 1
        a["gt_total"] += r["gt_count"]
        for s in STAGES:
            if r[f"{s}_tp"] is None:
                continue
            d = a.setdefault(s, {"pred": 0, "tp": 0, "fp": 0, "fn": 0})
            d["pred"] += r[f"{s}_pred"]
            d["tp"] += r[f"{s}_tp"]
            d["fp"] += r[f"{s}_fp"]
            d["fn"] += r[f"{s}_fn"]
    for a in agg.values():
        for s in STAGES:
            if s not in a:
                continue
            d = a[s]
            d["precision"] = round(d["tp"] / (d["tp"] + d["fp"]), 4) if d["tp"] + d["fp"] else 0.0
            d["recall"] = round(d["tp"] / (d["tp"] + d["fn"]), 4) if d["tp"] + d["fn"] else 0.0
    write_json(root / "summary_by_task.json", {"iou_threshold": IOU_THRESHOLD, "by_task": agg})
    print(f"汇总: {root / 'summary.csv'}（{len(rows)} 行）, {root / 'summary_by_task.json'}")


# ------------------------ 主流程 ------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="只用已保存的归因数据重新绘图")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="跳过已存在 saliency 的样本（默认开启，用于断点续跑）")
    args = parser.parse_args()

    root = Path(OUTPUT_DIR)
    comparisons_dir = root / "comparisons"
    samples_dir = root / "samples"
    root.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    index = build_index()
    write_json(root / "samples_index.json", index)
    selected = select_range(index, args.start, args.end, args.limit)
    print(f"测试集共 {len(index)} 个样本，本次处理 {len(selected)} 个。")

    # 先把每张输入图落到各自样本文件夹
    for rec in selected:
        sample_dir = samples_dir / rec["name"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        if not (sample_dir / "input_image.png").exists():
            Image.open(rec["image_path"]).convert("RGB").save(sample_dir / "input_image.png")

    if not args.render_only:
        import torch
        import transformers
        import peft
        from transformers import AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用。")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        torch.manual_seed(42)

        for stage, adapter in [("base", None), ("lora", LORA_PATH), ("cgpc", CGPC_PATH)]:
            from transformers import AutoModelForImageTextToText
            from peft import PeftModel
            print(f"\n[{stage}] loading {adapter or BASE_MODEL_PATH}", flush=True)
            model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL_PATH, dtype=getattr(torch, DTYPE), device_map={"": DEVICE},
                attn_implementation="sdpa")
            if adapter:
                model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
            model.eval()
            done = skipped = 0
            for rec in selected:
                sample_dir = samples_dir / rec["name"]
                if args.skip_existing and (sample_dir / f"{stage}_saliency.npz").exists():
                    skipped += 1
                    continue
                image = Image.open(sample_dir / "input_image.png").convert("RGB")
                detail = process_one(model, processor, image, rec["instruction"],
                                     sample_dir, stage)
                print(f"[{stage}] {rec['name']}: {detail['output_token_count']} tokens, "
                      f"{detail['target_token_count']} target, "
                      f"bboxes={len(detail['bboxes'])}, saliency max={detail['saliency_max']:.4g}",
                      flush=True)
                done += 1
            print(f"[{stage}] done={done} skipped={skipped}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 渲染：每个样本一张对比图 + 三张单模型热力图
    for rec in selected:
        sample_dir = samples_dir / rec["name"]
        if not all((sample_dir / f"{s}_saliency.npz").exists() for s in STAGES):
            print(f"[skip render] {rec['name']}: saliency 数据不完整", flush=True)
            continue
        try:
            render_sample(rec, sample_dir, comparisons_dir / f"{rec['name']}.png")
            write_run_metadata(rec, sample_dir)
            print(f"[render] {rec['name']} -> comparisons/{rec['name']}.png", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render error] {rec['name']}: {e}", flush=True)

    # 末尾汇总：扫描所有已计算样本，输出每张图的 gt/tp/fp/fn/任务类型
    write_summary(root, samples_dir)


if __name__ == "__main__":
    main()
