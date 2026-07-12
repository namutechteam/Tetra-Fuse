"""AdapterGINBlock — LoRA-style bottleneck mini-GIN adapter on node embeddings.

설계 (AdapterGNN; Li et al., AAAI'24 + LoRA 초기화 컨벤션):
    h_in (N, d)
      → LayerNorm
      → Linear down (d → r)
      → GINConv (r-dim message passing, optional edge_attr)
      → Linear up (r → d)  [zero-init: 초기 residual = identity]
      → Dropout
      → + h_in (residual)

GraphMVP / CheMeleon backbone 의 frozen layer 사이(또는 위)에 삽입되는
trainable PEFT 모듈. ``up`` 가중치는 0 으로 초기화되어 학습 시작 시점에
frozen backbone 의 출력 분포가 그대로 보존된다 (LoRA convention).

Lazy import:
    ``torch_geometric`` / ``ogb`` 는 ``__init__`` 안에서만 import (PyG inspector 가
    message() 시그니처를 보고 propagate kwargs 를 결정하므로 conv 클래스를
    edge_attr 사용/미사용 두 종류로 분리).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _make_atom_only_conv(r: int) -> nn.Module:
    """Vanilla GIN conv at bottleneck dim r (no edge_attr).

    PyG ``MessagePassing.propagate`` 는 ``message`` 시그니처 기반으로 kwargs 를
    인스펙트하므로, edge_attr 사용/미사용 클래스를 분리해야 한다.
    """
    from torch_geometric.nn import MessagePassing

    class _AtomOnlyConv(MessagePassing):
        def __init__(self) -> None:
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(r, 2 * r),
                nn.ReLU(),
                nn.Linear(2 * r, r),
            )
            self.eps = nn.Parameter(torch.zeros(1))

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))

        def message(self, x_j: Tensor) -> Tensor:
            return x_j

    return _AtomOnlyConv()


def _make_edge_attr_conv(r: int) -> nn.Module:
    """GIN(E)-like conv at bottleneck dim r with OGB BondEncoder."""
    from ogb.graphproppred.mol_encoder import BondEncoder
    from torch_geometric.nn import MessagePassing

    class _EdgeAttrConv(MessagePassing):
        def __init__(self) -> None:
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(r, 2 * r),
                nn.ReLU(),
                nn.Linear(2 * r, r),
            )
            self.eps = nn.Parameter(torch.zeros(1))
            self.bond_encoder = BondEncoder(emb_dim=r)

        def forward(
            self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
        ) -> Tensor:
            edge_emb = self.bond_encoder(edge_attr)
            return self.mlp(
                (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_emb)
            )

        def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
            return F.relu(x_j + edge_attr)

    return _EdgeAttrConv()


class AdapterGINBlock(nn.Module):
    """LoRA-style bottleneck mini-GIN adapter on node embeddings.

    Args:
        d: input/output node embedding dim (frozen backbone hidden dim).
        r: bottleneck dim (보통 d 보다 훨씬 작음).
        with_edge_attr: GraphMVP path 처럼 edge_attr 가 있으면 True (BondEncoder 사용).
        dropout: residual branch 의 dropout 확률.

    Attributes:
        d, r, with_edge_attr.
    """

    def __init__(
        self,
        d: int,
        r: int,
        with_edge_attr: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.r = r
        self.with_edge_attr = with_edge_attr

        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, r)
        self.gin = (
            _make_edge_attr_conv(r) if with_edge_attr else _make_atom_only_conv(r)
        )
        self.up = nn.Linear(r, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA-style zero init for the up-projection so initial residual = identity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None
    ) -> Tensor:
        z = self.norm(h)
        z = self.down(z)
        if self.with_edge_attr:
            if edge_attr is None:
                raise ValueError(
                    "AdapterGINBlock(with_edge_attr=True) 는 edge_attr 인자가 필요."
                )
            z = self.gin(z, edge_index, edge_attr)
        else:
            z = self.gin(z, edge_index)
        z = self.up(z)
        z = self.drop(z)
        return h + z
