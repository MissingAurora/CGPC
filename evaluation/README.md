# Evaluation

Evaluation is organized by dataset. Qwen uses one shared evaluator per dataset; select the model size with `--base-model-path` and the stage with `--stage base|lora|cgpc`. The common output includes raw responses, parsed boxes, per-sample metrics, aggregate metrics, and run metadata.

InternVL uses separate loading code because Base, XTuner LoRA `.pth`, and CGPC PEFT checkpoints have different state formats. Its entry points are documented in the dataset subdirectories.

Metrics use spatially ordered, per-ground-truth greedy matching with strict IoU > 0.5 and report Precision, Recall, F1, mIoU, CountAcc, and OverGen. Invalid parsed prediction boxes contribute to false positives and prediction counts. Box counting is independent of label availability. These rules are shared by Base, LoRA, and CGPC evaluations.
