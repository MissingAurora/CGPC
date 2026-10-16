# Data preparation

| Entry | Purpose |
|---|---|
| `grounding_protocol.py` | Shared task mapping, JSON parsing, box validation, and spatial GT-order matching |
| `validate_records.py` | Check image-conversation records, normalized boxes, image overlap, and LLaMA-Factory registration |
| `cot_generation/` | Generate teacher reasoning, validate it, and export the auxiliary cotton CoT dataset |
| `generate_refcoco_records.py` | Convert existing COCO image splits into ALL and SINGLE conversation records |

All datasets, images, annotations, generated responses, and API credentials remain outside this repository.

## Dense RefCOCO subset

Prepare density-filtered images in existing `train/` and `valid/` directories, then run from the repository root:

```bash
python data_tools/generate_refcoco_records.py --instances /data/instances_train2014.json --split-dir /data/refcoco_images --output-dir /data/refcoco_records --seed 42
```

For each retained image, the generator selects the most frequent category (smallest category ID for ties), applies the original category exclusion list, and creates two independent conversations: ALL objects and one randomly chosen LEFT, RIGHT, or MIDDLE instruction. SINGLE selects the leftmost box edge, rightmost box edge, or box center nearest the image's horizontal center. Coordinates are normalized to integers in `[0,1000]`; ALL boxes are ordered by left edge and then top edge. Chinese instructions and English category labels follow the original generator.

Outputs are `refcoco_train.json` and `refcoco_valid.json`, matching the Qwen LoRA dataset names. Each record contains `conversations`, `images`, and `imagesize`. Image paths are relative to `/data/refcoco_images`; use this as the image root during training and evaluation. For an existing `val/` directory, add `--valid-subdir val`; the output filename remains `refcoco_valid.json`. The generator preserves the supplied split membership and uses Python's standard library.

Register the outputs with LLaMA-Factory:

```bash
python data_tools/validate_records.py --input /data/refcoco_records/refcoco_train.json --image-root /data/refcoco_images --register-name refcoco_train --registry /path/to/LLaMA-Factory/data/dataset_info.json
python data_tools/validate_records.py --input /data/refcoco_records/refcoco_valid.json --image-root /data/refcoco_images --register-name refcoco_valid --registry /path/to/LLaMA-Factory/data/dataset_info.json
```

## CoT and training registration

See [cot_generation/README.md](cot_generation/README.md) for the generation workflow.

Register the exported files under the names used by `training/lora/qwen2b_cot.yaml`:

```bash
python data_tools/validate_records.py --input /data/cotton/qwen_data/cotton_cot_train.json --image-root /data/cotton --register-name cotton_cot_train --registry /path/to/LLaMA-Factory/data/dataset_info.json
python data_tools/validate_records.py --input /data/cotton/qwen_data/cotton_cot_valid.json --image-root /data/cotton --register-name cotton_cot_valid --registry /path/to/LLaMA-Factory/data/dataset_info.json
```

Set LLaMA-Factory's image/media root to `/data/cotton` for these relative image paths, and set `dataset_dir` to the directory containing the registry.
