# CGPC ablations

| File | Composite Grounding Advantage | Gate-based Preference Consolidation |
|---|:---:|:---:|
| `train_grpo_base.py` | no | no |
| `train_cga_only.py` | yes | no |
| `train_gpc_only.py` | no | yes |

The full method is `training/grpo/train_qwen2b.py`. Run all four variants from the same Qwen3-VL-2B LoRA checkpoint, dataset, seed, one-epoch budget, and sampling configuration. Use separate output directories.
