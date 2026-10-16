# Data preparation

| Entry | Purpose |
|---|---|
| `grounding_protocol.py` | Shared task mapping, JSON parsing, box validation, and spatial GT-order matching |
| `validate_records.py` | Check image-conversation records, normalized boxes, image overlap, and LLaMA-Factory registration |
| `cot_generation/` | Generate teacher reasoning, validate it, and export the auxiliary cotton CoT dataset |
| `split_refcoco.py` | Split an already selected COCO image subset by dominant object category |

All datasets, images, annotations, generated responses, and API credentials remain outside this repository.

## Dense RefCOCO subset

Install `data_tools/requirements.txt`, then run from the repository root:

```bash
python data_tools/split_refcoco.py --instances /data/instances_train2014.json --image-dir /data/selected_dense_images --output-dir /data/refcoco_split --copy-images
```

The splitter takes a density-filtered image directory and uses dominant-category stratification with a validation fraction of 0.2 and seed 42. It groups filename variants by their original COCO image ID to keep each source image in one split. Referring expressions are prepared separately.

Outputs are `train.json`, `valid.json`, and `split_report.json`. Optional `--copy-images` creates corresponding image directories. The JSON files list image membership. Use their image IDs to assign referring records to each split, then export image-conversation training records. Rare dominant categories are pooled before stratification; each resulting stratum needs enough images for both splits.

## CoT and training registration

See [cot_generation/README.md](cot_generation/README.md) for the generation workflow.

Register the exported files under the names used by `training/lora/qwen2b_cot.yaml`:

```bash
python data_tools/validate_records.py --input /data/cotton/qwen_data/cotton_cot_train.json --image-root /data/cotton --register-name cotton_cot_train --registry /path/to/LLaMA-Factory/data/dataset_info.json
python data_tools/validate_records.py --input /data/cotton/qwen_data/cotton_cot_valid.json --image-root /data/cotton --register-name cotton_cot_valid --registry /path/to/LLaMA-Factory/data/dataset_info.json
```

Set LLaMA-Factory's image/media root to `/data/cotton` for these relative image paths, and set `dataset_dir` to the directory containing the registry.
