"""Teacher-forced validation loss for XTuner InternVL LoRA training."""

import math
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import LOOPS, METRICS
from mmengine.runner import ValLoop


@METRICS.register_module()
class GroundingValidationLoss(BaseMetric):
    """Mean per-record loss; distributed collection removes sampler padding."""

    default_prefix = 'grounding'

    def process(self, data_batch, data_samples):
        self.results.extend(float(sample['loss']) for sample in data_samples)

    def compute_metrics(self, results):
        if not results or not all(math.isfinite(value) for value in results):
            raise ValueError('Validation requires finite losses for a nonempty dataset')
        return {'loss': sum(results) / len(results)}


@LOOPS.register_module()
class GroundingLossValLoop(ValLoop):
    """Use InternVL's teacher-forced loss output in the standard validation loop."""

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        if self.dataloader.batch_size != 1:
            raise ValueError('GroundingLossValLoop requires validation batch_size=1')
        self.runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)
        # XTuner InternVL_V1_5.forward returns {'loss': ...} for val_step as well.
        # val_step retains the runner's preprocessing and DDP/DeepSpeed wrapper.
        losses = self.runner.model.val_step(data_batch)
        if not isinstance(losses, dict) or 'loss' not in losses:
            raise TypeError('Expected InternVL teacher-forced loss from val_step')
        loss = losses['loss'].detach().float().mean().item()
        outputs = [{'loss': loss}]
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook('after_val_iter', batch_idx=idx,
                              data_batch=data_batch, outputs=outputs)
