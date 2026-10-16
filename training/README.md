# Training

Training follows two stages: LoRA supervised fine-tuning and one CGPC epoch. The CGPC policy and frozen reference both start from the selected LoRA checkpoint.

- `lora/`: LLaMA-Factory YAML recipes for Qwen and XTuner configurations for InternVL.
- `grpo/`: complete CGPC training.
- `ablations/`: GRPO-base, CGA-only, and GPC-only controls using the same Qwen3-VL-2B initialization and training budget.

Set a different `CGPC_OUTPUT_DIR` for every independent run.
