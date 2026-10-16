# LoRA supervised fine-tuning

## Qwen3-VL: LLaMA-Factory

| File | Model | Data |
|---|---|---|
| `qwen2b_cotton.yaml` | Qwen3-VL-2B | cotton-bale direct coordinates |
| `qwen8b_cotton.yaml` | Qwen3-VL-8B | cotton-bale direct coordinates |
| `qwen2b_refcoco.yaml` | Qwen3-VL-2B | dense RefCOCO |
| `qwen8b_refcoco.yaml` | Qwen3-VL-8B | dense RefCOCO |
| `qwen2b_cot.yaml` | Qwen3-VL-2B | auxiliary cotton-bale CoT comparison |

Register the dataset names in LLaMA-Factory, update `model_name_or_path`, `dataset_dir`, and `output_dir`, then run `llamafactory-cli train <config>`.

All recipes read an existing validation dataset through `eval_dataset`, run periodic loss evaluation, and select the checkpoint with the lowest `eval_loss`. Checkpoint saving and evaluation use the same step interval.

| Data | Training registration | Validation registration |
|---|---|---|
| Cotton | `cotton_train` | `cotton_valid` |
| Dense RefCOCO | `refcoco_train` | `refcoco_valid` |
| Cotton CoT | `cotton_cot_train` | `cotton_cot_valid` |

For the cotton split, register the two existing JSON files:

```bash
python data_tools/validate_records.py --input /data/cotton/train_split.json --image-root /data/cotton --register-name cotton_train --registry /path/to/LLaMA-Factory/data/dataset_info.json
python data_tools/validate_records.py --input /data/cotton/valid_split.json --image-root /data/cotton --register-name cotton_valid --registry /path/to/LLaMA-Factory/data/dataset_info.json
llamafactory-cli train training/lora/qwen2b_cotton.yaml
```

Set `dataset_dir` to the registry directory and the image/media root to the location containing the referenced images. The cotton split has 7,652 training and 1,277 validation records; 3,836 test records are used for final evaluation.

## InternVL3: XTuner

| File | Data |
|---|---|
| `internvl8b_cotton.py` | cotton-bale direct coordinates |
| `internvl8b_refcoco.py` | dense RefCOCO |

These are XTuner configuration files. Update model/data paths and launch them with the XTuner command appropriate to the installed version. Both configurations train the language LoRA and visual projector while keeping the pretrained backbones frozen.

Set `CGPC_TRAIN_JSON`, `CGPC_VALID_JSON`, and `CGPC_IMAGE_ROOT` for the existing splits. The validation loader uses one record per device and deterministic ordering. `validation.py` integrates teacher-forced loss with MMEngine's validation loop; `CheckpointHook` saves the lowest `grounding/loss` checkpoint. The loss is averaged over validation records, with distributed sampler padding removed during metric collection.

Run from the repository root so the custom validation module is importable:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CGPC_TRAIN_JSON=/data/cotton/train_split.json
export CGPC_VALID_JSON=/data/cotton/valid_split.json
export CGPC_IMAGE_ROOT=/data/cotton
NPROC_PER_NODE=4 xtuner train training/lora/internvl8b_cotton.py
```

For RefCOCO, use `internvl8b_refcoco.py` and point the two JSON variables to `refcoco_train.json` and `refcoco_valid.json`. Initialize CGPC from the selected LoRA checkpoint, including its visual projector.
