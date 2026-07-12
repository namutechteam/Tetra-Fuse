"""GraphMVPBackbone — 사전학습 GraphMVP GIN(frozen) + AdapterGNN(trainable).

설계:
    - GraphMVP_hybrid (Eq.8) regression variant 가중치 (OGB 9-d atom / 3-d bond feat 으로
      학습된 GIN) 를 frozen 으로 로드.
    - frozen GIN 은 항상 eval 모드 — ``train(mode)`` 호출 시에도 BN running stat 보존.
    - Pool → trainable ``self.proj`` (Linear + ReLU [+ Dropout]) 로 d_h 차원 출력.

Adapter 모드 (``adapter_kind``):
    - ``"paper"`` (default) — AdapterGNN (Li et al., AAAI'24) 충실 구현.
        * 각 frozen GIN layer 마다 pre-adapter A1 + post-adapter A2 한 쌍을 **parallel**
          로 wiring (논문 Eq. 8 그대로):
            h_l = BN(MLP(MP(x_l))) + s1·A1(x_l) + s2·A2(MP(x_l))
            x_{l+1} = Dropout(ReLU(h_l))
        * 각 adapter A: down → ReLU → up → BN (Eq. 7), 외부에서 s 로 스케일.
        * s1, s2 는 각 adapter 의 ``gamma`` (learnable scalar, init=0). 학습 시작 시점에는
          어댑터 기여 0 → forward = frozen-only baseline 과 완전 동등.
        * MP(x_l) 추출을 위해 frozen ``GINConv`` 의 ``mlp/eps/bond_encoder`` 만 인라인 호출.
    - ``"lora_gin"`` (legacy) — 기존 LoRA-style mini-GIN bottleneck (``AdapterGINBlock``).
        * 각 frozen GIN layer 출력 뒤에 1개씩 (LayerNorm + down + GIN + up + dropout).
        * up.weight 0-init 으로 zero-residual.

Lazy import:
    ``ogb`` / ``torch_geometric`` 은 ``__init__`` 안에서만 import (무거운 의존 격리).

State_dict 호환:
    GraphMVP paper 의 가중치 파일은
    ``{'model': GIN_state_dict, 'model_3D': ..., 'AE_2D_3D_model': ..., 'AE_3D_2D_model': ...}``
    구조. 본 backbone 은 ``'model'`` key 만 추출해 GIN 에 ``strict=False`` 로 로드.

Backward-compat:
    - 이전 버전의 ``adaptor_dropout`` kwarg 는 ``adapter_dropout`` 의 alias 로 수용 (deprecation).
    - ``adapter_kind`` default 가 "paper" 로 바뀜 — 기존 학습 가중치 호환 안 됨, 재학습 필요.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from ._adapter_gin import AdapterGINBlock
from ._adapter_gnn import AdapterGNNBlock
from .registry import BACKBONES

logger = logging.getLogger(__name__)


@BACKBONES.register("graphmvp")
class GraphMVPBackbone(nn.Module):
    """GraphMVP 사전학습 GIN(frozen) + AdapterGNN/AdapterGIN(trainable) 래퍼.

    Args:
        d_h: ``self.proj`` 출력 차원 (= ``output_dim``).
        checkpoint_path: GraphMVP ``.pth`` 가중치 경로. None 이면 random-init smoke.
        num_layer: GIN 레이어 수 (paper 기본 5).
        emb_dim: GIN hidden / output 차원 (paper 기본 300).
        JK: jumping-knowledge 모드 — 현재 ``"last"`` 만 지원.
        adapter_kind: ``"paper"`` (default, AdapterGNN paper-faithful) 또는
            ``"lora_gin"`` (legacy ``AdapterGINBlock``).
        adapter_bottleneck: adapter 의 bottleneck dim ``r``.
        adapter_dropout: adapter residual branch dropout.
        use_edge_attr: ``adapter_kind="lora_gin"`` 에서만 의미 — adapter 내부 mini-GIN 이
            OGB BondEncoder 로 edge_attr 사용. ``"paper"`` 에서는 무시 (MLP-only adapter).
        proj_dropout: ``self.proj`` 의 dropout (없으면 0).
        pool: graph-level pooling — ``"mean"`` / ``"sum"`` / ``"max"``.
        adaptor_dropout: **Deprecated** — ``adapter_dropout`` 의 alias.

    Attributes:
        output_dim: ``d_h``.
        adapter_kind: 사용한 adapter 모드.
    """

    def __init__(
        self,
        d_h: int = 300,
        checkpoint_path: str | None = None,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        adapter_kind: Literal["paper", "lora_gin"] = "paper",
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.0,
        use_edge_attr: bool = True,
        proj_dropout: float = 0.0,
        pool: str = "mean",
        adaptor_dropout: float | None = None,
    ) -> None:
        super().__init__()
        import torch
        from torch_geometric.nn import (
            global_add_pool,
            global_max_pool,
            global_mean_pool,
        )

        from ._graphmvp_gin import GraphMVPGIN

        if JK != "last":
            raise ValueError(
                f"GraphMVPBackbone (Adapter variant) 는 JK='last' 만 지원 (받은 값: {JK!r}). "
                "concat/max/sum 모드는 adapter 삽입 위치가 모호 — out-of-scope."
            )

        if adapter_kind not in ("paper", "lora_gin"):
            raise ValueError(
                f"adapter_kind 는 'paper' 또는 'lora_gin' (받은 값: {adapter_kind!r})."
            )

        if adaptor_dropout is not None:
            warnings.warn(
                "GraphMVPBackbone: `adaptor_dropout` 은 deprecated — `adapter_dropout` 을 사용. "
                "이번엔 alias 로 매핑.",
                DeprecationWarning,
                stacklevel=2,
            )
            adapter_dropout = adaptor_dropout

        gin = GraphMVPGIN(num_layer=num_layer, emb_dim=emb_dim, JK=JK, drop_ratio=0.0)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            gin_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            # OGB AtomEncoder categorical 임베딩 dim 이 paper(2021) vs 현 OGB 사이에
            # 다를 수 있어 (e.g. chirality 4→5), shape 일치하는 키만 load.
            own_state = gin.state_dict()
            filtered = {
                k: v for k, v in gin_state.items()
                if k in own_state and v.shape == own_state[k].shape
            }
            shape_skipped = [
                (k, tuple(v.shape), tuple(own_state[k].shape))
                for k, v in gin_state.items()
                if k in own_state and v.shape != own_state[k].shape
            ]
            missing, unexpected = gin.load_state_dict(filtered, strict=False)
            if shape_skipped:
                logger.warning(
                    "GraphMVP GIN 가중치 shape mismatch %d keys skipped (random init): %s",
                    len(shape_skipped), shape_skipped[:4],
                )
            if missing:
                logger.warning("GraphMVP GIN 가중치 missing keys: %s", missing[:8])
            if unexpected:
                logger.warning("GraphMVP GIN 가중치 unexpected keys: %s", unexpected[:8])
        else:
            logger.info("GraphMVP checkpoint_path=None — random-init smoke 모드")

        self.gin = gin
        self.gin.requires_grad_(False)
        self.gin.eval()

        if pool == "mean":
            self._pool = global_mean_pool
        elif pool == "sum":
            self._pool = global_add_pool
        elif pool == "max":
            self._pool = global_max_pool
        else:
            raise ValueError(f"unknown pool: {pool!r}")

        self.adapter_kind = adapter_kind
        self.use_edge_attr = use_edge_attr

        if adapter_kind == "paper":
            # ---- paper-faithful: 각 frozen GIN layer 마다 pre/post adapter 쌍 ----
            self.pre_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(d=emb_dim, r=adapter_bottleneck, dropout=adapter_dropout)
                    for _ in range(num_layer)
                ]
            )
            self.post_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(d=emb_dim, r=adapter_bottleneck, dropout=adapter_dropout)
                    for _ in range(num_layer)
                ]
            )
            # legacy attr (ModuleList 이름 일치 깨도 외부 의존성 없음 — 신규 학습)
            self.adapter_blocks = nn.ModuleList()
        else:
            # ---- legacy lora_gin: 각 frozen GIN layer 출력 뒤에 1개 adapter ----
            self.adapter_blocks = nn.ModuleList(
                [
                    AdapterGINBlock(
                        d=emb_dim,
                        r=adapter_bottleneck,
                        with_edge_attr=use_edge_attr,
                        dropout=adapter_dropout,
                    )
                    for _ in range(num_layer)
                ]
            )

        # ---- trainable post-pool projection ----
        proj_layers: list[nn.Module] = [nn.Linear(emb_dim, d_h), nn.ReLU()]
        if proj_dropout > 0:
            proj_layers.append(nn.Dropout(proj_dropout))
        self.proj = nn.Sequential(*proj_layers)

        self.d_h = d_h
        self.output_dim = d_h
        self.num_layer = num_layer

    def train(self, mode: bool = True):
        """frozen GIN 은 항상 eval 유지 (Lightning train() 전파 차단).

        Args:
            mode: trainable 서브모듈 (adapters / proj) 만 train/eval 토글.

        Returns:
            self.
        """
        super().train(mode)
        self.gin.eval()
        return self

    def forward(self, batch) -> Tensor:
        """PyG Batch → adapter-augmented graph embedding.

        Args:
            batch: ``torch_geometric.data.Batch`` (``x``, ``edge_index``, ``edge_attr``,
                ``batch`` index).

        Returns:
            ``(B, d_h)`` projection 출력.
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # frozen atom encoder
        h = self.gin.atom_encoder(x)

        if self.adapter_kind == "paper":
            # paper Eq. (8) parallel wiring:
            #   h_l = BN(MLP(MP(x_l))) + A1(x_l) + A2(MP(x_l))
            #   x_{l+1} = ReLU(h_l) for inner layers; raw h_l for the last layer
            # (drop_ratio=0 으로 init 했으므로 Dropout 생략)
            for i in range(self.num_layer):
                xl = h
                gnn = self.gin.gnns[i]
                # MP(x_l): GINConv aggregation step (mlp 적용 전)
                edge_emb = gnn.bond_encoder(edge_attr)
                mp_out = (1 + gnn.eps) * xl + gnn.propagate(
                    edge_index, x=xl, edge_attr=edge_emb
                )
                # BN(MLP(MP(x_l))): frozen MLP + outer BN
                frozen_path = self.gin.batch_norms[i](gnn.mlp(mp_out))
                # parallel branches (each adapter returns s · A(·))
                a1 = self.pre_adapters[i](xl)
                a2 = self.post_adapters[i](mp_out)
                hl = frozen_path + a1 + a2
                # x_{l+1} = ReLU(h_l) for inner layers (last layer raw)
                h = F.relu(hl) if i < self.num_layer - 1 else hl
        else:
            # legacy: frozen GIN layer 출력 뒤에 1개 adapter
            for i in range(self.num_layer):
                h = self.gin.gnns[i](h, edge_index, edge_attr)
                h = self.gin.batch_norms[i](h)
                if i < self.num_layer - 1:
                    h = F.relu(h)
                h = self.adapter_blocks[i](h, edge_index, edge_attr)

        graph_h = self._pool(h, batch.batch)
        return self.proj(graph_h)
