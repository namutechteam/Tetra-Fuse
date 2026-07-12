"""Loss registry: per-task functional 과 nn.Module(상위 loss) 두 종류."""

from __future__ import annotations

from qadmet.core.registry import Registry

# per-task: (preds, targets, mask) → (B, n_tasks) per-task loss tensor
PER_TASK_LOSSES: Registry = Registry("per_task_losses")

# 상위 loss: nn.Module — log_vars / task_weights 등 학습 가능 파라미터/버퍼 보유.
# build 시 (n_tasks, task_types, task_weights, ...) kwargs 전달.
LOSSES: Registry = Registry("losses")
