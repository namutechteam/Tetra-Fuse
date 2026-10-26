"""Loss module: per-task functional + task weighting (Kendall) + inverse-freq.

Why:
    Mixed into the old ``pipeline_mtl.py``'s ``_grouped_loss`` / ``_mixed_loss``
    BCE/MSE computation, Kendall log_vars regularizer, inverse-freq task weight
    decomposed each into pure-tensor modules to enable unit testing and recombination.

Registry:
    - ``PER_TASK_LOSSES`` : (preds, targets, mask) → (B, n_tasks) functional.
      Lookup by key of LossSpec.classification / LossSpec.regression.
    - ``LOSSES`` : ``nn.Module`` factory. Holds Kendall log_vars, etc.
      Lookup by key of LossSpec.type. (Passes ``n_tasks``, ``task_types``, ``task_weights`` at build time)

import-time registration:
    This ``__init__`` imports the functional / mixed_task / single_task / inverse_freq modules
    to trigger the decorators.
"""

from __future__ import annotations

from .registry import LOSSES, PER_TASK_LOSSES  # noqa: F401
from . import functional  # noqa: F401  — registration trigger
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
