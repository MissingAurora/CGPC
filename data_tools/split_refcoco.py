"""Stratify a preselected COCO image subset by dominant object category."""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        parser.error('--validation-ratio must be between 0 and 1')
    from sklearn.model_selection import train_test_split

    coco = json.loads(args.instances.read_text(encoding='utf-8-sig'))
    names = {Path(image['file_name']).stem: image['id'] for image in coco['images']}
    counts = defaultdict(Counter)
    for ann in coco['annotations']:
        counts[ann['image_id']][ann['category_id']] += 1
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    groups = defaultdict(list)
    missing = []
    for path in sorted(args.image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        image_id = names.get(path.stem)
        if image_id is None:
            image_id = names.get(path.stem.rsplit('_', 1)[0])
        if image_id is None:
            missing.append(path.name)
        else:
            groups[image_id].append(path.name)
    if missing:
        raise ValueError(f'{len(missing)} images lack a COCO mapping; first examples: {missing[:5]}')
    if len(groups) < 2:
        raise ValueError('At least two unique source images are required')
    image_ids = sorted(groups)
    dominant = {}
    for image_id in image_ids:
        category_counts = counts[image_id]
        category = max(category_counts, key=lambda key: (category_counts[key], -key)) if category_counts else None
        dominant[image_id] = categories[category] if category is not None else 'none'
    category_sizes = Counter(dominant.values())
    labels = [dominant[key] if category_sizes[dominant[key]] >= 2 else '__other__' for key in image_ids]
    try:
        train_ids, valid_ids = train_test_split(
            image_ids, test_size=args.validation_ratio, random_state=args.seed, stratify=labels)
    except ValueError as exc:
        raise ValueError('The subset is too small for the requested stratified split. '
                         'Adjust the subset or validation ratio. No unstratified fallback was applied.') from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError('Use an empty output directory to avoid mixing different splits')
    summary = {'seed': args.seed, 'validation_ratio': args.validation_ratio,
               'grouping': 'original COCO image ID', 'input_images': str(args.image_dir.resolve()),
               'instances': str(args.instances.resolve()), 'splits': {}}
    for name, ids in [('train', train_ids), ('valid', valid_ids)]:
        ids = sorted(ids)
        rows = [{'image_id': key, 'file_name': filename, 'dominant_category': dominant[key]}
                for key in ids for filename in groups[key]]
        (args.output_dir / f'{name}.json').write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary['splits'][name] = {'unique_images': len(ids), 'files': len(rows),
                                  'categories': dict(Counter(dominant[key] for key in ids))}
        if args.copy_images:
            destination = args.output_dir / name
            destination.mkdir()
            for row in rows:
                shutil.copy2(args.image_dir / row['file_name'], destination / row['file_name'])
    summary['overlapping_image_ids'] = len(set(train_ids) & set(valid_ids))
    (args.output_dir / 'split_report.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
