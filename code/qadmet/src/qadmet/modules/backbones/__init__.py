"""nn backbone 패키지 — BACKBONES registry 등록 트리거.

각 backbone 의 무거운 의존(chemprop / transformers+peft / unimol_v2 vendored)은
``__init__`` 시점이 아니라 클래스 ``__init__`` 안에서 lazy import 된다. 따라서
본 패키지 import 만으로는 chemprop/transformers/peft 가 로드되지 않는다.
"""

from __future__ import annotations

from .registry import BACKBONES  # noqa: F401
from . import dmpnn  # noqa: F401  — DMPNNBackbone 등록
from . import chemeleon  # noqa: F401  — CheMeleonBackbone 등록
from . import molformer_lora  # noqa: F401  — MoLFormerLoRABackbone 등록
from . import unimol2_lora  # noqa: F401  — UniMol2LoRABackbone 등록
from . import graphmvp  # noqa: F401  — GraphMVPBackbone 등록
from . import mlp_embedding  # noqa: F401  — MLPEmbeddingBackbone 등록
from .dmpnn import DMPNNBackbone
from .chemeleon import CheMeleonBackbone
from .molformer_lora import MoLFormerLoRABackbone
from .unimol2_lora import UniMol2LoRABackbone
from .graphmvp import GraphMVPBackbone
from .mlp_embedding import MLPEmbeddingBackbone

__all__ = [
    "BACKBONES",
    "DMPNNBackbone",
    "CheMeleonBackbone",
    "MoLFormerLoRABackbone",
    "UniMol2LoRABackbone",
    "GraphMVPBackbone",
    "MLPEmbeddingBackbone",
]
