"""GraphMVP frozen pooled embedding builder (300-dim, pre-projection).

frozen-backbone ablation 용 — LoRA/adapter 를 전혀 학습하지 않고 fusion+head 만
학습할 때 매 epoch GIN 재계산을 피하기 위해 frozen graph embedding 을 미리 parquet 로
저장한다.

frozen-only forward (adapter 없음):
    atom_encoder(x) 뒤로 각 layer i:
        edge_emb = gnn.bond_encoder(edge_attr)
        mp_out   = (1 + gnn.eps) * x_l + gnn.propagate(edge_index, x=x_l, edge_attr=edge_emb)
        h_l      = batch_norms[i](gnn.mlp(mp_out))
        x_{l+1}  = relu(h_l)  (i < num_layer-1)  /  h_l  (last layer)
    graph_h = backbone._pool(h, batch.batch)   # (B, 300)

이 흐름은 GraphMVPBackbone.forward 의 ``adapter_kind=="paper"`` 분기에서 adapter
기여(a1/a2)를 0 으로 둔 것과 동일하며, frozen GIN (GraphMVPGIN.forward) 의
``drop_ratio=0`` 결과와도 일치한다.

Note:
    - SMILES → PyG Batch 는 학습 경로와 동일하게 ``smiles_batch_to_pyg`` 사용.
    - present_mask=False (RDKit parse 실패 / atom 0) 분자는 출력 dict 에서 제외한다.
    - 무거운 의존성(ogb / torch_geometric)은 method 안에서만 import.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

logger = logging.getLogger(__name__)

EMBED_DIM = 300
DEFAULT_CKPT = "data/weights/graphmvp/graphmvp_hybrid.pth"


@FEATURE_BUILDERS.register("graphmvp_frozen")
class GraphMVPFrozenBuilder(PrecomputeBase):
    """GraphMVP frozen graph embedding (300-dim, pre-projection/pre-adapter)."""

    name = "graphmvp_frozen"
    column_prefix = "gm_"
    output_dim = EMBED_DIM
    default_batch_size = 4096
    default_checkpoint_every = 4096

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CKPT,
        gpu_batch_size: int = 256,
        device: str = "cuda:0",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.gpu_batch_size = gpu_batch_size
        self.device = device
        self._backbone = None

    def _ensure_loaded(self) -> None:
        if self._backbone is not None:
            return
        import torch  # noqa: F401

        from qadmet.modules.backbones.graphmvp import GraphMVPBackbone

        logger.info("building GraphMVP frozen backbone from %s", self.checkpoint_path)
        # adapter_kind="lora_gin" 은 최소 비용 — adapter 는 생성되지만 frozen GIN
        # submodule 만 직접 호출하므로 사용되지 않는다.
        backbone = GraphMVPBackbone(
            checkpoint_path=self.checkpoint_path,
            num_layer=5,
            emb_dim=EMBED_DIM,
            JK="last",
            pool="mean",
            adapter_kind="lora_gin",
        ).eval()
        self._backbone = backbone.to(self.device)

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as F

        from qadmet.modules.backbones._graphmvp_features import smiles_batch_to_pyg

        self._ensure_loaded()
        backbone = self._backbone
        gin = backbone.gin
        out: dict[str, np.ndarray] = {}
        smiles = list(smiles)

        for start in range(0, len(smiles), self.gpu_batch_size):
            chunk = smiles[start : start + self.gpu_batch_size]
            pyg_batch, present_mask = smiles_batch_to_pyg(chunk)
            pyg_batch = pyg_batch.to(self.device)

            x, edge_index, edge_attr = (
                pyg_batch.x,
                pyg_batch.edge_index,
                pyg_batch.edge_attr,
            )
            with torch.no_grad():
                h = gin.atom_encoder(x)
                for i in range(backbone.num_layer):
                    xl = h
                    gnn = gin.gnns[i]
                    edge_emb = gnn.bond_encoder(edge_attr)
                    mp_out = (1 + gnn.eps) * xl + gnn.propagate(
                        edge_index, x=xl, edge_attr=edge_emb
                    )
                    hl = gin.batch_norms[i](gnn.mlp(mp_out))
                    h = F.relu(hl) if i < backbone.num_layer - 1 else hl
                graph_h = backbone._pool(h, pyg_batch.batch)  # (B, 300)
            arr = graph_h.detach().cpu().numpy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            present = present_mask.cpu().numpy()
            for j, s in enumerate(chunk):
                if not present[j]:
                    continue  # invalid SMILES — 출력 dict 에서 제외.
                out[s] = arr[j]
        return out
