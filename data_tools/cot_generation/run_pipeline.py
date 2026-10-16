"""Run teacher generation, validation, and CoT export on user-owned data."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', default='qwen3-vl-flash')
    parser.add_argument('--base-url', default=os.getenv('TEACHER_API_URL'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exclude-images', type=Path,
                        help='JSON array of image paths excluded following manual review')
    args = parser.parse_args()
    root = args.root.resolve()
    code = Path(__file__).resolve().parent
    if not args.validate_only and not args.base_url:
        parser.error('Set TEACHER_API_URL or pass --base-url')
    if not args.validate_only and not (os.getenv('TEACHER_API_KEY') or os.getenv('DASHSCOPE_API_KEY')):
        parser.error('Set TEACHER_API_KEY or DASHSCOPE_API_KEY')

    def run(script, *arguments):
        subprocess.run([sys.executable, str(code / script), *map(str, arguments)], cwd=root, check=True)

    run('build_manifest.py', '--root', root)
    for split in ('train', 'valid'):
        prefix = f'teacher_work/manifest/{split}'
        report = f'teacher_work/reports/{split}'
        raw = f'teacher_work/raw/{split}'
        accepted = f'teacher_work/accepted/{split}'
        run('normalize_answers.py', '--input', prefix + '.jsonl', '--output', prefix + '_ordered.jsonl',
            '--report', report + '_order.json')
        run('filter_annotations.py', '--input', prefix + '_ordered.jsonl', '--accepted', prefix + '_ready.jsonl',
            '--review', f'teacher_work/review/{split}_annotations.jsonl', '--report', report + '_annotations.json')
        if not args.validate_only:
            extras = ['--resume'] if args.resume else []
            run('generate_teacher.py', '--input', prefix + '_ready.jsonl', '--output-dir', raw,
                '--root', root, '--mode', 'openai-compatible', '--model', args.model,
                '--base-url', args.base_url, '--workers', args.workers, '--max-retries', args.max_retries,
                '--max-output-tokens', args.max_output_tokens, *extras)
        run('sanitize_teacher.py', '--manifest', prefix + '_ready.jsonl', '--raw-dir', raw,
            '--output-dir', raw + '_clean', '--report', report + '_clean.json')
        run('validate_structure.py', '--manifest', prefix + '_ready.jsonl', '--raw-dir', raw + '_clean',
            '--accepted', accepted + '_hard.jsonl', '--rejected', f'teacher_work/rejected/{split}.jsonl',
            '--report', report + '_structure.json')
        run('validate_reasoning.py', '--input', accepted + '_hard.jsonl', '--accepted', accepted + '.jsonl',
            '--retry', f'teacher_work/retry/{split}.jsonl', '--report', report + '_reasoning.json')
        extra = ['--exclude-images', args.exclude_images.resolve()] if args.exclude_images else []
        run('merge_dataset.py', '--source', f'new_{split}.json', '--split', split,
            '--accepted', accepted + '.jsonl', '--require-accepted', '--output', f'new_{split}_cot_final.json',
            '--report', report + '_merge.json', *extra)
    run('export_cot.py', '--root', root)
    run('audit_export.py', '--root', root)


if __name__ == '__main__':
    main()
