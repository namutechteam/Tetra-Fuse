"""MLPEmbeddingBackbone — Precomputed embedding vector, as-is (or Linear projection) pass-through.

frozen-backbone ablation-only passthrough backbone. consumes-obtained single modality's
Precomputed embedding ``(B, input_dim)`` is received as input, output_dim is absent, or input_dim
equals, use Identity, otherwise Linear(input_dim, output_dim) is used for projection.

Note:
    - ``output_dim`` attribute, which lit_module reads, must be set (chemeleon.py convention).
    - train.py injects ``d_xd``, so it is accepted but ignored (embedding-only passthrough).
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


@BACKBONES.register("mlp_embedding")
class MLPEmbeddingBackbone(nn.Module):
    """Precomputed embedding (B, input_dim) → Identity or Linear projection.

    Args:
        input_dim: Input embedding dimension.
        output_dim: Output dimension. None or input_dim, if equal, use Identity.
        d_xd: train.py injection-purpose — ignored (embedding-only passthrough).

    Attributes:
        output_dim: Output dimension (lit_module reads it).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        d_xd: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if output_dim is None or output_dim == input_dim:
            self.net: nn.Module = nn.Identity()
            self.output_dim = input_dim
        else:
            self.net = nn.Linear(input_dim, output_dim)
            self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
