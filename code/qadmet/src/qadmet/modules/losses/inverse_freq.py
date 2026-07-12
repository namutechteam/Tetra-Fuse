"""Inverse-frequency task weighting.

옛 ``pipeline_mtl.py:compute_inverse_freq_weights`` (line ~429) 와 수치 동일.
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
    """데이터 크기의 역수로 task 가중치 계산 → mean 정규화.

    Args:
        samples_per_task: task_name → sample 수.
        task_names: 가중치 순서를 정의하는 task name 리스트.

    Returns:
        (n_tasks,) float32 텐서. 평균이 1 이 되도록 정규화.

    Raises:
        KeyError: task_names 중 samples_per_task 에 없는 키.
    """
    counts = np.array([samples_per_task[t] for t in task_names], dtype=np.float32)
    if (counts <= 0).any():
        raise ValueError(f"sample 수가 0 이하인 task: {[t for t, c in zip(task_names, counts) if c <= 0]}")
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)
