"""nn backbone package — BACKBONES registry registration trigger.

The heavy dependencies of each backbone (chemprop / transformers+peft / unimol_v2 vendored)
are lazily imported not at ``__init__`` time but inside the class ``__init__``. Therefore
importing this package alone does not load chemprop/transformers/peft.
"""

from __future__ import annotations

from .registry import BACKBONES  # noqa: F401
from . import dmpnn  # noqa: F401  — DMPNNBackbone registration
from . import chemeleon  # noqa: F401  — CheMeleonBackbone registration
from . import molformer_lora  # noqa: F401  — MoLFormerLoRABackbone registration
from . import unimol2_lora  # noqa: F401  — UniMol2LoRABackbone registration
from . import graphmvp  # noqa: F401  — GraphMVPBackbone registration
from . import mlp_embedding  # noqa: F401  — MLPEmbeddingBackbone registration
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
