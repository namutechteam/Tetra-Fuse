"""Loss 모듈: per-task functional + task weighting (Kendall) + inverse-freq.

Why:
    옛 ``pipeline_mtl.py`` 의 ``_grouped_loss`` / ``_mixed_loss`` 안에 섞여 있던
    BCE/MSE 계산, Kendall log_vars regularizer, inverse-freq task weight 를
    각각 pure-tensor 모듈로 분해해 단위 테스트·재조합 가능하게 한다.

Registry:
    - ``PER_TASK_LOSSES`` : (preds, targets, mask) → (B, n_tasks) functional.
      LossSpec.classification / LossSpec.regression 의 키로 lookup.
    - ``LOSSES`` : ``nn.Module`` factory. Kendall log_vars 등을 보유.
      LossSpec.type 의 키로 lookup. (build 시 ``n_tasks``, ``task_types``, ``task_weights`` 전달)

import-time 등록:
    본 ``__init__`` 가 functional / mixed_task / single_task / inverse_freq 모듈을
    import 하여 데코레이터를 발동시킨다.
"""

from __future__ import annotations

from .registry import LOSSES, PER_TASK_LOSSES  # noqa: F401
from . import functional  # noqa: F401  — 등록 트리거
from . import mixed_task  # noqa: F401
from . import single_task  # noqa: F401
from . import gradnorm  # noqa: F401
from .mixed_task import MixedTaskUncertaintyLoss, MixedTaskUncertaintyPCGradLoss
from .single_task import SingleTaskLoss
from .gradnorm import MixedTaskGradNormLoss
from .inverse_freq import compute_inverse_freq_weights
from .pcgrad import project_conflicting_gradients, cosine_conflict_rate

__all__ = [
    "LOSSES",
    "PER_TASK_LOSSES",
    "MixedTaskUncertaintyLoss",
    "MixedTaskUncertaintyPCGradLoss",
    "MixedTaskGradNormLoss",
    "SingleTaskLoss",
    "compute_inverse_freq_weights",
    "project_conflicting_gradients",
    "cosine_conflict_rate",
]
