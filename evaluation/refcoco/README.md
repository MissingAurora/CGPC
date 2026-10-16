# Dense RefCOCO evaluation

- `eval_qwen.py`: shared Qwen3-VL-2B/8B evaluator for Base, LoRA, and CGPC (`--stage`).
- `eval_internvl_base.py`: InternVL3-8B Base evaluator.
- `eval_internvl_lora.py`: InternVL3-8B XTuner LoRA evaluator.
- `eval_internvl_cgpc.py`: InternVL3-8B CGPC evaluator loading the policy adapter and trained `visual_projector.pt`.

The evaluator records the ALL, LEFT, RIGHT, and MIDDLE task subsets and exports every response. Choose the model size through `--base-model-path` and provide the corresponding dataset-specific checkpoint.
