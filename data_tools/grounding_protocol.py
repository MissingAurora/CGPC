"""Task classification and box evaluation used by the CGPC release."""

import json
import math
import re


def classify_grounding_task(prompt):
    """Map cotton and RefCOCO instructions to ALL, ROW, or SINGLE."""
    text = prompt.replace('<image>', '').strip()
    lower = text.lower()
    if any(word in text for word in ('所有', '全部')) or re.search(r'\ball\b', lower):
        return 'ALL'
    if any(word in text for word in ('最左', '最右', '最中间', '正中间')) or any(
            word in lower for word in ('leftmost', 'rightmost', 'middlemost', 'central instance')):
        return 'SINGLE'
    number = r'(?:[零一二三四五六七八九十百两]+|\d+)'
    if re.search(r'第\s*' + number + r'\s*(?:个|块|包)', text):
        return 'SINGLE'
    if '排' in text or re.search(r'\brows?\b', lower):
        return 'ROW'
    return 'SINGLE'


def parse_grounding_answer(text):
    """Parse a complete JSON answer; retain malformed box entries for FP counting."""
    record = parse_grounding_record(text)
    return record['boxes'], record['labels']


def parse_grounding_record(text):
    """Return parsed boxes and response-structure metadata."""
    text = '' if text is None else str(text)
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    empty = not text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and 'bbox_2d' in obj:
            objects = [obj]
        elif isinstance(obj, list) and obj and all(isinstance(x, dict) and 'bbox_2d' in x for x in obj):
            objects = obj
        else:
            continue
        boxes, labels = [], []
        for item in objects:
            raw = item['bbox_2d']
            if isinstance(raw, list) and len(raw) == 4 and all(not isinstance(x, (list, dict)) for x in raw):
                raw = [raw]
            elif not isinstance(raw, list):
                raw = [raw]
            names = item.get('labels', item.get('label', []))
            names = names if isinstance(names, list) else [names]
            boxes.extend(raw)
            labels.extend((names + [''] * len(raw))[:len(raw)])
        return dict(boxes=boxes, labels=labels, empty_text=empty,
                    invalid_json=False, parseable_json=True, parse_mode='strict_json')
    return dict(boxes=[], labels=[], empty_text=empty,
                invalid_json=not empty, parseable_json=False, parse_mode='unparsed')


def scale_normalized_boxes(boxes, size):
    """Return legal pixel boxes, their source indices, and the invalid count."""
    width, height = size
    valid, indices, invalid = [], [], 0
    for index, box in enumerate(boxes):
        try:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError('four coordinates required')
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in box):
                raise ValueError('numeric coordinates required')
            x1, y1, x2, y2 = box
            if not all(math.isfinite(x) for x in box) or not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError('invalid normalized box')
            valid.append([int(x1 * width / 1000), int(y1 * height / 1000),
                          int(x2 * width / 1000), int(y2 * height / 1000)])
            indices.append(index)
        except (ValueError, TypeError, OverflowError):
            invalid += 1
    return valid, indices, invalid


def scale_boxes_preserving_count(boxes, size):
    """Use zero-area placeholders to retain invalid entries in prediction counts."""
    valid, indices, _ = scale_normalized_boxes(boxes, size)
    result = [[0, 0, 0, 0] for _ in boxes]
    for index, box in zip(indices, valid):
        result[index] = box
    return result


def box_iou(first, second):
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area1 = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area2 = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return overlap / (area1 + area2 - overlap + 1e-6)


def ordered_box_matches(gt_boxes, pred_boxes, threshold=0.5):
    """Spatial GT-order greedy matching with strict IoU > threshold."""
    used, matches = set(), []
    gt_order = sorted(range(len(gt_boxes)), key=lambda i: (gt_boxes[i][1], gt_boxes[i][0], i))
    pred_order = sorted(range(len(pred_boxes)), key=lambda i: (pred_boxes[i][1], pred_boxes[i][0], i))
    for gi in gt_order:
        best, pi = 0.0, -1
        for candidate in pred_order:
            if candidate in used:
                continue
            score = box_iou(gt_boxes[gi], pred_boxes[candidate])
            if score > best:
                best, pi = score, candidate
        if best > threshold:
            used.add(pi)
            matches.append((best, gi, pi))
    return matches


def sort_boxes_with_labels(boxes, labels=None):
    names = list(labels or []) + [''] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0], i))
    return [boxes[i] for i in order], [names[i] for i in order]
