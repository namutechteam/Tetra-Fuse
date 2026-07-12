"""Fusion 모듈: K 개 modality embedding 을 하나로 합치는 nn 모듈.

공통 forward 시그니처:
    forward(embeddings: list[Tensor (B, D_i)]) -> tuple[Tensor (B, output_dim), Tensor (B, K)]
        - 두 번째 반환은 alpha (gate weight). uniform 또는 의미 없는 fusion 도 형식 유지.

FUSIONS registry 키:
    - ``concat``      : 단순 concat → linear projection.
    - ``attention``   : 옛 ``AttentionFusion`` (softmax gate).
    - ``gated``       : sigmoid 게이트 — alpha-collapse 완화 슬롯.
    - ``robust_attention`` : tau annealing + modality dropout + residual.
    - ``uncertainty_attention`` : robust + learned modality precision (MixedTaskUncertaintyLoss 연동).
    - ``film``        : primary modality 를 보조 modality 들로 affine modulation
      (CheMeleon main + MoLFormer/UniMol2/GraphMVP 보조 패턴).
    - ``group_routed`` : per-group static gate + optional cross-modal self-attn
      enrichment. RGR 다이내믹 제거, stacking 의 per-task routing 인사이트를
      fusion 단으로 옮긴 GR-MF. (B, n_groups, D) 출력 → grouped_routed head.
    - ``multi_film`` : K 개 modality 를 각각 primary 로 두고 나머지로 FiLM 변조한
      branch K 개 → cross-attention late fusion. symmetric FiLM (약한 대칭).
    - ``cross_attention`` : asymmetric primary(query) + aux(K/V) cross-attention 주입
      (TransformerDecoder, primary_idx 로 primary 선택). multi_film 대칭 폐기 후 재설계.
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
