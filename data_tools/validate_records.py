"""Validate user-owned grounding records without changing annotations."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path


def load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, list):
        raise ValueError('Expected a JSON array of records')
    return value


def normalized_image(record):
    return record['images'][0].replace('\\', '/')


def validate(records, image_root=None):
    pairs = Counter()
    images = set()
    errors = []
    for idx, record in enumerate(records):
        try:
            assert isinstance(record, dict), 'record must be an object'
            paths = record['images']
            assert isinstance(paths, list) and len(paths) == 1, 'exactly one image required'
            assert isinstance(paths[0], str) and paths[0], 'image path must be nonempty text'
            turns = record['conversations']
            assert len(turns) == 2, 'expected one human and one assistant turn'
            assert turns[0]['from'] == 'human' and turns[1]['from'] == 'gpt', 'incorrect roles'
            prompt = turns[0]['value']
            assert isinstance(prompt, str) and '<image>' in prompt, 'missing image token'
            answer = turns[1]['value']
            if '</think>' in answer:
                answer = answer.split('</think>', 1)[1].strip()
            obj = json.loads(answer)
            boxes, labels = obj['bbox_2d'], obj['labels']
            assert isinstance(boxes, list) and isinstance(labels, list), 'boxes/labels must be lists'
            assert len(boxes) == len(labels), 'box-label count mismatch'
            for box in boxes:
                assert isinstance(box, list) and len(box) == 4, 'box must have four coordinates'
                assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box), 'nonfinite or nonnumeric coordinate'
                assert 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000, 'invalid normalized box'
            image = normalized_image(record)
            if image_root:
                assert (Path(image_root) / image).is_file(), 'image does not exist'
            pairs[(image, prompt)] += 1
            images.add(image)
        except (AssertionError, KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append({'record': idx, 'error': str(exc)})
    return {'records': len(records), 'unique_image_paths': len(images),
            'duplicate_image_instruction_pairs': sum(n - 1 for n in pairs.values()),
            'invalid_records': len(errors), 'errors': errors}, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--image-root')
    parser.add_argument('--compare')
    parser.add_argument('--register-name')
    parser.add_argument('--registry')
    args = parser.parse_args()
    if bool(args.register_name) != bool(args.registry):
        parser.error('--register-name and --registry must be supplied together')
    report, images = validate(load(args.input), args.image_root)
    if args.compare:
        other_report, other_images = validate(load(args.compare))
        report['comparison_invalid_records'] = other_report['invalid_records']
        report['exact_image_path_overlap'] = len(images & other_images)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report['invalid_records']:
        raise SystemExit(1)
    if args.registry:
        registry = Path(args.registry).resolve()
        existing = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else {}
        entry = {
            'file_name': os.path.relpath(Path(args.input).resolve(), registry.parent).replace('\\', '/'),
            'formatting': 'sharegpt',
            'columns': {'messages': 'conversations', 'images': 'images'},
            'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt'},
        }
        if args.register_name in existing and existing[args.register_name] != entry:
            raise ValueError('Registry name already has a different definition; choose a new name')
        existing[args.register_name] = entry
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
