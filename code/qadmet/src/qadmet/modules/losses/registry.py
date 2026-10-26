"""Loss registry: two types - per-task functional and nn.Module (parent loss)."""

from __future__ import annotations

from qadmet.core.registry import Registry

# per-task: (preds, targets, mask) → (B, n_tasks) per-task loss tensor
PER_TASK_LOSSES: Registry = Registry("per_task_losses")

# Parent loss: nn.Module — holds learnable parameters/buffers such as log_vars / task_weights.
# At build time, kwargs (n_tasks, task_types, task_weights, ...) are passed.
LOSSES: Registry = Registry("losses")
