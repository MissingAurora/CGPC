# Cotton-bale evaluation

- `eval_qwen.py`: shared Qwen3-VL-2B/8B evaluator for Base, LoRA, and CGPC (`--stage`).
- `eval_internvl_base.py`: InternVL3-8B Base evaluator.
- `eval_internvl_lora.py`: InternVL3-8B XTuner LoRA evaluator.
- `eval_internvl_cgpc.py`: InternVL3-8B CGPC evaluator loading the policy adapter and trained `visual_projector.pt`.
- `eval_cot.py`: Qwen3-VL-2B auxiliary CoT evaluation.

Select Base with `--stage base`. For LoRA and CGPC, also provide the corresponding checkpoint through `--adapter-path`. Set a separate output directory for each run.
