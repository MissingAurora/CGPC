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

## InternVL3: XTuner

| File | Data |
|---|---|
| `internvl8b_cotton.py` | cotton-bale direct coordinates |
| `internvl8b_refcoco.py` | dense RefCOCO |

These are XTuner configuration files. Update model/data paths and launch them with the XTuner command appropriate to the installed version. Both configurations train the language LoRA and visual projector while keeping the pretrained backbones frozen.
