"""Render aligned ALL/ROW/SINGLE examples from three cotton prediction exports."""
import argparse
import json
from pathlib import Path
import textwrap


def read_records(path):
    records = {}
    with open(path, encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            idx = record['sample_id']
            if idx in records:
                raise ValueError(f'Duplicate sample_id {idx}')
            records[idx] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('base', 'lora', 'cgpc'):
        parser.add_argument('--' + stage, required=True)
    parser.add_argument('--sample-ids', type=int, nargs=3, required=True, metavar=('ALL_ID', 'ROW_ID', 'SINGLE_ID'))
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--line-width', type=float, default=1.5)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from PIL import Image
    stages = [read_records(getattr(args, name)) for name in ('base', 'lora', 'cgpc')]
    tasks = ('ALL', 'ROW', 'SINGLE')
    for col, idx in enumerate(args.sample_ids):
        recs = [stage[idx] for stage in stages]
        identity = [(r['image_relative_path'], r['instruction']) for r in recs]
        if len(set(identity)) != 1:
            raise ValueError(f'Stage records do not describe the same example: {idx}')
        if any(r['task'] != tasks[col] for r in recs):
            raise ValueError(f'Expected {tasks[col]} in column {col + 1}')
        if any(r['gt_boxes_px'] != recs[0]['gt_boxes_px'] for r in recs):
            raise ValueError(f'GT differs across stages: {idx}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), facecolor='white')
    for row, (stage, label) in enumerate(zip(stages, ('Base', 'LoRA', 'CGPC'))):
        for col, idx in enumerate(args.sample_ids):
            rec = stage[idx]
            ax = axes[row, col]
            image_path = Path(args.image_root) / rec['image_relative_path'].replace('\\', '/')
            with Image.open(image_path) as image:
                ax.imshow(image.convert('RGB'))
            for field, color in (('gt_boxes_px', '#00a000'), ('pred_boxes_px', '#ff0000')):
                for x1, y1, x2, y2 in rec[field]:
                    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=args.line_width))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#c2ccd3')
                spine.set_linewidth(1)
            if row == 0:
                ax.set_title(tasks[col], fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(label, fontweight='bold', fontsize=12)
            if row == 2:
                instruction = rec['instruction'].replace('<image>', '').strip().splitlines()[0]
                ax.set_xlabel('\n'.join(textwrap.wrap(instruction, 48)), fontsize=9)
    fig.legend(handles=[Patch(facecolor='none', edgecolor='#00a000', label='Ground truth'), Patch(facecolor='none', edgecolor='#ff0000', label='Prediction')], loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
