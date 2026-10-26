"""Fusion module: nn module that combines K modality embeddings into one.

Common forward signature:
    forward(embeddings: list[Tensor (B, D_i)]) -> tuple[Tensor (B, output_dim), Tensor (B, K)]
        - Second return is alpha (gate weight). Maintain format even for uniform or meaningless fusion.

FUSIONS registry keys:
    - ``concat``      : simple concat → linear projection.
    - ``attention``   : old ``AttentionFusion`` (softmax gate).
    - ``gated``       : sigmoid gate — slot to mitigate alpha-collapse.
    - ``robust_attention`` : tau annealing + modality dropout + residual.
    - ``uncertainty_attention`` : robust + learned modality precision (linked with MixedTaskUncertaintyLoss).
    - ``film``        : affine modulation of primary modality by auxiliary modalities
      (CheMeleon main + MoLFormer/UniMol2/GraphMVP auxiliary pattern).
    - ``group_routed`` : per-group static gate + optional cross-modal self-attn
      enrichment. Removes RGR dynamics, brings the per-task routing insight of stacking
      GR-MF moved to the fusion stage. (B, n_groups, D) output → grouped_routed head.
    - ``multi_film`` : FiLM-modulates each of K modalities as primary using the rest
      K branches → cross-attention late fusion. symmetric FiLM (weak symmetry).
    - ``cross_attention`` : asymmetric primary(query) + aux(K/V) cross-attention injection
      (TransformerDecoder, select primary via primary_idx). Redesigned after discarding multi_film symmetry.
"""

from __future__ import annotations

from .registry import FUSIONS  # noqa: F401
from . import concat  # noqa: F401
from . import attention  # noqa: F401
from . import gated  # noqa: F401
from . import robust_attention  # noqa: F401
from . import uncertainty_attention  # noqa: F401
from . import film  # noqa: F401
from . import group_routed  # noqa: F401
from . import multi_film  # noqa: F401
from . import cross_attention  # noqa: F401
from .concat import ConcatFusion
from .attention import AttentionFusion
from .gated import GatedFusion
from .robust_attention import RobustAttentionFusion
from .uncertainty_attention import UncertaintyAttentionFusion
from .film import FilmFusion
from .group_routed import GroupRoutedFusion
from .multi_film import MultiFilmFusion
from .cross_attention import CrossAttentionFusion

__all__ = [
    "FUSIONS",
    "ConcatFusion",
    "AttentionFusion",
    "GatedFusion",
    "RobustAttentionFusion",
    "UncertaintyAttentionFusion",
    "FilmFusion",
    "GroupRoutedFusion",
    "MultiFilmFusion",
    "CrossAttentionFusion",
]
