"""Head 모듈: Z (B, input_dim) → (B, n_tasks).

옛 ``pipeline_mtl.py:GroupedHead`` 그대로 이동 + LinearHead 추가.
HEADS registry 의 키:
    - ``linear``         : 단순 nn.Linear(input_dim, n_tasks).
    - ``grouped``        : task affinity 7-group routing, 2D Z (B, D) 입력.
    - ``grouped_routed`` : per-group fused Z (B, n_groups, D) 입력 — GR-MF 와 페어링.
    - ``bottleneck``     : Linear→ReLU→Dropout→Linear→ReLU→Linear (STL 용 MLP head).
"""

from __future__ import annotations

from .registry import HEADS  # noqa: F401
from . import linear  # noqa: F401  — 등록 트리거
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
