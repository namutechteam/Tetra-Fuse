"""GraphMVP GIN encoder — paper port (state_dict 호환 보장).

GraphMVP/src_regression/models_complete_feature/molecule_gnn_model.py:62-125
의 ``GNNComplete`` 와 ``GINConv`` 를 그대로 옮긴 형태. 모듈 이름/구조를 변경하면
사전학습 가중치 (``.pth`` 의 ``"model"`` key state_dict) 가 strict load 되지 않으므로
**모듈 키는 paper 코드와 동일** 하게 유지한다 (``atom_encoder``, ``gnns[i].mlp``,
``gnns[i].eps``, ``gnns[i].bond_encoder``, ``batch_norms[i]``).

Lazy import:
    ``ogb`` / ``torch_geometric`` 은 본 모듈 top-level 에서 import — backbone 측이
    이미 lazy import 후 호출하는 흐름이라 추가 격리 불필요.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing


class GINConv(MessagePassing):
    """GraphMVP src_regression GINConv 동등 (paper:12-30)."""

    def __init__(self, emb_dim: int, aggr: str = "add") -> None:
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_embedding = self.bond_encoder(edge_attr)
        return self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


class GraphMVPGIN(nn.Module):
    """GraphMVP src_regression ``GNNComplete`` 동등 (paper:62-125).

    이름만 ``GNNComplete -> GraphMVPGIN`` 으로 변경 — sub-module 키는 동일 유지.
    forward 는 명시 인자 (x, edge_index, edge_attr) 만 받음 (PyG Data 인터페이스는
    backbone 래퍼 측에서 풀어 호출).

    Args:
        num_layer: GIN 레이어 수 (paper 기본 5).
        emb_dim: hidden / output 차원 (paper 기본 300).
        JK: jumping-knowledge 모드 — 'last' / 'concat' / 'max' / 'sum'.
        drop_ratio: 각 레이어 출력 dropout 확률 (frozen 사용 시 0 권장).
    """

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layer < 2:
            raise ValueError("num_layer must be >= 2")
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.drop_ratio = drop_ratio

        self.atom_encoder = AtomEncoder(emb_dim)
        self.gnns = nn.ModuleList(
            [GINConv(emb_dim, aggr="add") for _ in range(num_layer)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.atom_encoder(x)
        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        if self.JK == "max":
            return torch.stack(h_list, dim=0).max(dim=0).values
        if self.JK == "sum":
            return torch.stack(h_list, dim=0).sum(dim=0)
        raise ValueError(f"unknown JK: {self.JK!r}")
