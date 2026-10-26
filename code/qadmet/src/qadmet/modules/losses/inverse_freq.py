"""Inverse-frequency task weighting.

Numerically identical to old ``pipeline_mtl.py:compute_inverse_freq_weights`` (line ~429).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor


def compute_inverse_freq_weights(
    samples_per_task: Mapping[str, int],
    task_names: list[str],
) -> Tensor:
    """Compute task weights as the inverse of data size → mean normalization.

    Args:
        samples_per_task: task_name → sample count.
        task_names: list of task names defining the weight order.

    Returns:
        (n_tasks,) float32 tensor. Normalized so the mean is 1.

    Raises:
        KeyError: a key among task_names that is not in samples_per_task.
    """
    counts = np.array([samples_per_task[t] for t in task_names], dtype=np.float32)
    if (counts <= 0).any():
        raise ValueError(f"tasks with sample count <= 0: {[t for t, c in zip(task_names, counts) if c <= 0]}")
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)
