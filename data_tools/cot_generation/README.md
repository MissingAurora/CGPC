# Cotton CoT generation

This workflow uses Qwen3-VL-Flash to generate task understanding, scene description, and object-selection reasoning. The teacher receives the image, instruction, and annotations. Postprocessing fixes boxes, labels, and counts from the annotations and constructs the per-object confirmations. The final exporter retains scene-specific reasoning and derives spatial ordering from the annotated geometry.

## Inputs and setup

Install `requirements.txt` in this directory. Provide a data root with `new_train.json`, `new_valid.json`, and the images referenced by those records. Records use `images: [relative_path]` and two `conversations` turns (`human`, `gpt`). Original answers contain `bbox_2d` and `labels`, with normalized coordinates in 0–1000. The source annotations use `front` and `side` labels.

```bash
export TEACHER_API_KEY=your_api_key
export TEACHER_API_URL=your_openai_compatible_base_url
python data_tools/cot_generation/run_pipeline.py --root /data/cotton --resume
```

The command sends images and annotations to the configured teacher API. Credentials are read from environment variables; `DASHSCOPE_API_KEY` is also supported. Use `--validate-only` to reuse existing raw responses without making API calls. Use a separate data/work directory when changing the teacher, prompt, or source records: resume identifies previously generated records by sample ID.

## Stages

| Script | Operation |
|---|---|
| `build_manifest.py` | Read records and check source structure, image availability, and annotations |
| `normalize_answers.py` | Standardize annotated object order |
| `filter_annotations.py` | Separate annotation anomalies for review |
| `generate_teacher.py` | Generate structured teacher text using `prompts/teacher_system.txt` |
| `sanitize_teacher.py` | Clean text and populate authoritative boxes, labels, and counts |
| `validate_structure.py` | Check response structure and annotation consistency |
| `validate_reasoning.py` | Flag detectable semantic conflicts and unsuitable descriptions |
| `sample_review.py` | Select records for human inspection |
| `merge_dataset.py` | Merge accepted records with the original instructions |
| `export_cot.py` | Export `<think>` reasoning followed by the localization JSON |
| `audit_export.py` | Verify exported answers, reasoning structure, and image files |
| `sample_pilot.py`, `audit_dataset.py` | Optional pilot sampling and intermediate dataset checks |

`cot_common.py` provides shared parsing, task classification, and serialization. All scripts expose `--help`. The generation prompt is `prompts/teacher_system.txt`, with runtime identifier `grounded`.

## Review and outputs

Intermediate manifests, raw responses, reports, accepted/rejected records, and retry lists are written under `teacher_work/` inside the data root. Review the flagged records and a sample of accepted records against their images. Supply `--exclude-images` with a JSON list of excluded relative paths when rebuilding the export. Regenerate and revalidate records on the semantic retry lists before merging them.

Final outputs are `qwen_data/cotton_cot_train.json` and `qwen_data/cotton_cot_valid.json`, matching the repository's CoT training dataset names. Both use the same `images` list schema as other repository records. Registration commands are in the parent README. The export report gives the number of accepted records in each split.

Use `--mode mock` for offline pipeline checks. The normal runner uses API mode, and the standard merge accepts API-generated records. To inspect a few API requests first, use `sample_pilot.py` and invoke the generator with `--mode openai-compatible --limit 3`.
