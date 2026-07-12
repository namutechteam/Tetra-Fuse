"""MLPEmbeddingBackbone — 사전계산 embedding 벡터를 그대로(또는 Linear 투영) 통과.

frozen-backbone ablation 용 passthrough backbone. consumes 로 받은 한 modality 의
사전계산 embedding ``(B, input_dim)`` 을 입력으로 받아, output_dim 이 없거나 input_dim
과 같으면 Identity, 다르면 Linear(input_dim, output_dim) 으로 투영한다.

Note:
    - ``output_dim`` attribute 는 lit_module 이 읽으므로 반드시 설정 (chemeleon.py 규약).
    - train.py 가 ``d_xd`` 를 주입하므로 받아 무시한다 (embedding-only passthrough).
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


@BACKBONES.register("mlp_embedding")
class MLPEmbeddingBackbone(nn.Module):
    """사전계산 embedding (B, input_dim) → Identity 또는 Linear 투영.

    Args:
        input_dim: 입력 embedding 차원.
        output_dim: 출력 차원. None 또는 input_dim 과 같으면 Identity.
        d_xd: train.py 주입용 — 무시 (embedding-only passthrough).

    Attributes:
        output_dim: 출력 차원 (lit_module 이 읽음).
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
