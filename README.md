# CGPC
Code for **CGPC: Composite Grounding and Preference Consolidation for Multilevel Referring Detection in Dense Scenes**. The repository contains LoRA supervised fine-tuning, CGPC training, component ablations, evaluation, and visualization. Datasets, images, model weights, prediction dumps, and training logs are not distributed.

## Structure
| Directory | Contents |
|---|---|
| `training/lora/` | Cotton, dense RefCOCO, and auxiliary CoT LoRA recipes |
| `training/grpo/` | CGPC training for Qwen3-VL-2B, Qwen3-VL-8B, and InternVL3-8B |
| `training/ablations/` | GRPO-base, CGA-only, and GPC-only controls |
| `evaluation/cotton/` | Cotton-bale evaluation and complete prediction export |
| `evaluation/refcoco/` | Dense RefCOCO evaluation and complete prediction export |
| `visualization/` | Qualitative grids and gradient attribution |
| `data_tools/` | Shared grounding protocol, record validation, teacher CoT generation, and dense RefCOCO instruction generation |

Each directory has its own README with entry points and required arguments.
`data_tools/grounding_protocol.py` provides the common task mapping, response parsing, coordinate validation, and spatial GT-order matching used by the training and evaluation entry points.

## Training toolchains
- **Qwen3-VL LoRA** uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The YAML recipes under `training/lora/` run through LLaMA-Factory.
- **InternVL3 LoRA** uses [XTuner](https://github.com/InternLM/xtuner). The Python files under `training/lora/` are XTuner configurations.
- **CGPC and its ablations** use the custom distributed PyTorch scripts under `training/grpo/` and `training/ablations/`.

The training recipes use one CGPC epoch. The LoRA recipes use rank 16, alpha 32, dropout 0.05, and 20 epochs so that the compared model families and datasets follow the same published setup. No geometric augmentation is applied.

## Paths
Configure paths through environment variables:
```bash
export CGPC_BASE_MODEL=/path/to/base/model
export CGPC_SFT_PATH=/path/to/lora/checkpoint
export CGPC_ADAPTER_PATH=/path/to/cgpc/checkpoint
export CGPC_TRAIN_JSON=/path/to/train.json
export CGPC_VALID_JSON=/path/to/validation.json
export CGPC_EVAL_JSON=/path/to/test.json
export CGPC_IMAGE_ROOT=/path/to/images
export CGPC_OUTPUT_DIR=/path/to/output
```

For dense RefCOCO, use `data_tools/generate_refcoco_records.py` with existing image splits to export `refcoco_train.json` and `refcoco_valid.json`. See [data preparation](data_tools/README.md) for commands and dataset registration.

LoRA uses validation loss for checkpoint selection; CGPC uses training-set performance during refinement. XTuner reads `CGPC_VALID_JSON`, while LLaMA-Factory reads the validation dataset registered under each recipe's `eval_dataset`. See [LoRA configuration](training/lora/README.md) for both interfaces.

## Minimal workflow

```bash
# Qwen LoRA through LLaMA-Factory
llamafactory-cli train training/lora/qwen2b_cotton.yaml
# CGPC from the selected LoRA checkpoint
torchrun --standalone --nproc_per_node=4 training/grpo/train_qwen2b.py
# Qwen evaluation; replace paths/model size as needed
python evaluation/cotton/eval_qwen.py \
  --stage cgpc \
  --base-model-path /path/to/Qwen3-VL-2B-Instruct \
  --adapter-path /path/to/cgpc/checkpoint \
  --data-json /path/to/eval.json \
  --image-root /path/to/images \
  --output-dir ./outputs/eval_qwen2b_cgpc
```
