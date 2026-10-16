# CGPC training

| File | Model |
|---|---|
| `train_qwen2b.py` | Qwen3-VL-2B |
| `train_qwen8b.py` | Qwen3-VL-8B |
| `train_internvl8b.py` | InternVL3-8B |

All scripts start from `CGPC_SFT_PATH`, keep a frozen reference anchored to that LoRA checkpoint, and train for one epoch. Qwen uses eight responses per prompt; InternVL uses six responses and candidate mini-batches of two. Training budgets are SINGLE 256, ROW 1024, and density-dependent ALL up to `CGPC_ALL_MAX_TOKENS` (default 10240). No geometric augmentation is used.

The three trainers share Composite Grounding Advantage and Gate-Based Preference Consolidation. CGA combines centered set utility and continuous localization credit. GPC selects qualified sampled improvements over the greedy response on ALL tasks and applies reference-relative preference losses at the first differing token and the subsequent suffix. Its group-level policy gate switches the ordinary policy-gradient term; the reference KL term remains active for eligible sampled responses.

For dense RefCOCO, use the same trainers with the RefCOCO LoRA checkpoint, records, and image root. Instructions requesting all instances map to ALL; LEFT, RIGHT, and MIDDLE instance requests map to SINGLE.

LoRA parameters and the visual projector are trainable. InternVL checkpoints contain the policy adapter and `visual_projector.pt`; the frozen reference retains the original SFT adapter and projector. Both InternVL CGPC evaluators restore this saved projector.

Example:

```bash
CGPC_BASE_MODEL=/path/to/model \
CGPC_SFT_PATH=/path/to/lora \
CGPC_TRAIN_JSON=/path/to/train.json \
CGPC_IMAGE_ROOT=/path/to/images \
CGPC_OUTPUT_DIR=./outputs/qwen2b_cgpc \
torchrun --standalone --nproc_per_node=4 training/grpo/train_qwen2b.py
```
