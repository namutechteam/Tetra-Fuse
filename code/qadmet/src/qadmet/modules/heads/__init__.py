"""Head module: Z (B, input_dim) → (B, n_tasks).

Moved as-is from legacy ``pipeline_mtl.py:GroupedHead`` + added LinearHead.
Keys of the HEADS registry:
    - ``linear``         : simple nn.Linear(input_dim, n_tasks).
    - ``grouped``        : task affinity 7-group routing, 2D Z (B, D) input.
    - ``grouped_routed`` : per-group fused Z (B, n_groups, D) input — paired with GR-MF.
    - ``bottleneck``     : Linear→ReLU→Dropout→Linear→ReLU→Linear (MLP head for STL).
"""

from __future__ import annotations

from .registry import HEADS  # noqa: F401
from . import linear  # noqa: F401  — registration trigger
from . import grouped  # noqa: F401
from . import grouped_routed  # noqa: F401
from . import bottleneck  # noqa: F401
from .linear import LinearHead
from .grouped import GroupedHead
from .grouped_routed import GroupedRoutedHead
from .bottleneck import BottleneckHead

__all__ = [
    "HEADS",
    "LinearHead",
    "GroupedHead",
    "GroupedRoutedHead",
    "BottleneckHead",
]
