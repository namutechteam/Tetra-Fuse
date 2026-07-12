"""CheMeleonBackbone — 사전학습 CheMeleon GNN(frozen) + AdapterGNN/AdapterGIN(trainable).

설계:
    - CheMeleon BondMessagePassing + MeanAggregation 가중치는 항상 frozen.
    - chemprop ``BondMessagePassing`` 는 내부에서 ``depth`` 회 message passing 을
      closed-loop 으로 수행. 각 loop iteration 을 한 "MP layer" 로 간주.

Adapter 모드 (``adapter_kind``):
    - ``"paper"`` (default) — AdapterGNN (Li et al., AAAI'24) 충실 구현.
        * chemprop forward 를 인라인하고, 각 depth iteration 마다 paper Eq. (8)
          parallel wiring 으로 두 어댑터를 합산 (bond-level, d=2048):

            H_l = update(message(H_{l-1}, bmg), H_0)
                  + s1 · A1(H_{l-1})
                  + s2 · A2(message(H_{l-1}, bmg))

          여기서 ``update(M, H_0) = tau(H_0 + W_h(M)) + dropout`` (chemprop frozen 경로)
          가 paper 의 ``BN(MLP(MP(x_l)))`` 자리. chemprop 자체에는 layer 내부 BN 이
          없으므로 paper 의 outer BN 은 adapter 내부 BN 이 대신함.
        * 각 adapter A: down → ReLU → up → BN (Eq. 7), 외부 scalar s 로 스케일.
        * s1, s2 는 각 adapter 의 ``gamma`` (learnable, init=0). 학습 시작 시점에는
          forward = frozen-only baseline 과 완전 동등.
        * depth=3 (chemprop default) → 2 pre + 2 post = 4 adapter blocks.
        * BondMP 가 atom-level d_h 중간 상태를 노출하지 않으므로 adapter 도 bond-level.
    - ``"lora_gin"`` (legacy) — outer-only AdapterGINBlock 스택.
        * frozen MP 전체 출력 (N_atoms, 2048) 위에 ``n_adapter_layers`` 개 atom-only GIN.

이후 frozen ``agg`` (MeanAggregation) → graph-level → trainable ``self.proj``
(Linear + ReLU [+ Dropout]) → ``(B, d_h)``.

Lazy import:
    chemprop 은 ``__init__`` 안에서만 import (무거운 패키지).

Backward-compat:
    - 이전 버전의 ``adaptor_dropout`` kwarg 는 ``adapter_dropout`` 의 alias (deprecation).
    - ``adapter_kind`` default 가 "paper" 로 바뀜 — 기존 학습 가중치 호환 안 됨, 재학습 필요.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from torch import Tensor, nn

from ._adapter_gin import AdapterGINBlock
from ._adapter_gnn import AdapterGNNBlock
from .registry import BACKBONES


@BACKBONES.register("chemeleon")
class CheMeleonBackbone(nn.Module):
    """CheMeleon 사전학습 GNN(frozen) + AdapterGNN/AdapterGIN(trainable) 래퍼.

    Args:
        d_h: ``self.proj`` 출력 차원 (= ``output_dim``).
        checkpoint_path: CheMeleon 가중치 파일 경로 (None 이면 random-init smoke test).
        d_xd: 추가 descriptor concat 차원 (기본 0 — descriptor 미사용).
        adapter_kind: ``"paper"`` (default, AdapterGNN paper-faithful) 또는
            ``"lora_gin"`` (legacy outer-only ``AdapterGINBlock``).
        adapter_bottleneck: adapter bottleneck dim ``r``.
        adapter_dropout: adapter residual branch dropout.
        n_adapter_layers: ``"lora_gin"`` 에서만 의미 — outer AdapterGINBlock 개수.
            ``"paper"`` 모드에서는 ``depth-1`` 개 (chemprop depth) 가 자동 결정.
        proj_dropout: ``self.proj`` 의 dropout (없으면 0).
        adaptor_dropout: **Deprecated** — ``adapter_dropout`` 의 alias.

    Attributes:
        CHEMELEON_HIDDEN: BondMessagePassing 출력 차원 (항상 2048).
        output_dim: d_h.
        adapter_kind: 사용한 adapter 모드.

    Note:
        - frozen mp/agg 는 ``requires_grad=False`` + ``eval()`` 고정.
        - ``train(mode)`` 호출해도 frozen 모듈은 eval 유지.
    """

    CHEMELEON_HIDDEN = 2048  # BondMessagePassing output dim

    def __init__(
        self,
        d_h: int = 300,
        checkpoint_path: str | None = None,
        d_xd: int = 0,
        adapter_kind: Literal["paper", "lora_gin"] = "paper",
        adapter_bottleneck: int = 256,
        adapter_dropout: float = 0.0,
        n_adapter_layers: int = 2,
        proj_dropout: float = 0.0,
        adaptor_dropout: float | None = None,
    ) -> None:
        super().__init__()
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MeanAggregation
        from chemprop.nn.predictors import RegressionFFN

        if adapter_kind not in ("paper", "lora_gin"):
            raise ValueError(
                f"adapter_kind 는 'paper' 또는 'lora_gin' (받은 값: {adapter_kind!r})."
            )

        if adaptor_dropout is not None:
            warnings.warn(
                "CheMeleonBackbone: `adaptor_dropout` 은 deprecated — `adapter_dropout` 을 사용. "
                "이번엔 alias 로 매핑.",
                DeprecationWarning,
                stacklevel=2,
            )
            adapter_dropout = adaptor_dropout

        if adapter_kind == "lora_gin" and n_adapter_layers < 1:
            raise ValueError(
                f"n_adapter_layers must be >= 1 for adapter_kind='lora_gin' "
                f"(받은 값: {n_adapter_layers})."
            )

        # 체크포인트에서 로드 또는 random init
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, weights_only=True)
            mp = BondMessagePassing(**ckpt["hyper_parameters"])
            mp.load_state_dict(ckpt["state_dict"])
        else:
            # smoke test: random-init (notebook hyper_parameters 와 동일 d_h=2048)
            mp = BondMessagePassing(d_h=self.CHEMELEON_HIDDEN)

        agg = MeanAggregation()

        # MPNN 생성자 호환용 dummy predictor (실제 사용 안 함 — message_passing/agg 만 호출)
        dummy_predictor = RegressionFFN(
            n_tasks=1,
            input_dim=self.CHEMELEON_HIDDEN + d_xd,
            hidden_dim=4,
            n_layers=1,
        )
        self._mpnn = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=dummy_predictor,
            batch_norm=False,
        )

        # ---- frozen mp/agg/bn/X_d_transform ----
        self._mpnn.message_passing.requires_grad_(False)
        self._mpnn.agg.requires_grad_(False)
        self._mpnn.message_passing.eval()
        self._mpnn.agg.eval()
        # bn 은 batch_norm=False 로 Identity 일 가능성이 크지만, 안전하게 frozen 처리
        self._mpnn.bn.requires_grad_(False)
        self._mpnn.bn.eval()
        if hasattr(self._mpnn, "X_d_transform"):
            self._mpnn.X_d_transform.requires_grad_(False)
            self._mpnn.X_d_transform.eval()

        self.adapter_kind = adapter_kind

        if adapter_kind == "paper":
            # ---- paper-faithful: 각 chemprop MP iteration 마다 pre/post 쌍 (bond-level) ----
            depth = self._mpnn.message_passing.depth
            if depth < 2:
                raise ValueError(
                    f"adapter_kind='paper' 는 chemprop depth>=2 필요 (현재: {depth}). "
                    "depth=1 이면 message-passing iteration 이 없어 adapter 삽입 자리 없음."
                )
            n_pairs = depth - 1  # chemprop loop runs (depth-1) iterations
            self.pre_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_pairs)
                ]
            )
            self.post_adapters = nn.ModuleList(
                [
                    AdapterGNNBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_pairs)
                ]
            )
            self.n_adapter_pairs = n_pairs
            # legacy attr 비움 (외부 의존성 없음 — 신규 학습)
            self.adapter_blocks = nn.ModuleList()
        else:
            # ---- legacy lora_gin: outer-only AdapterGINBlock 스택 ----
            self.adapter_blocks = nn.ModuleList(
                [
                    AdapterGINBlock(
                        d=self.CHEMELEON_HIDDEN,
                        r=adapter_bottleneck,
                        with_edge_attr=False,
                        dropout=adapter_dropout,
                    )
                    for _ in range(n_adapter_layers)
                ]
            )
            self.n_adapter_layers = n_adapter_layers

        # ---- trainable post-pool projection ----
        proj_layers: list[nn.Module] = [
            nn.Linear(self.CHEMELEON_HIDDEN + d_xd, d_h),
            nn.ReLU(),
        ]
        if proj_dropout > 0:
            proj_layers.append(nn.Dropout(proj_dropout))
        self.proj = nn.Sequential(*proj_layers)

        self.d_h = d_h
        self.d_xd = d_xd
        self.output_dim = d_h

    def train(self, mode: bool = True):
        """frozen 서브모듈은 항상 eval 유지 (Lightning train() 전파 차단).

        Args:
            mode: trainable 서브모듈 (adapters / proj) 만 train/eval 토글.

        Returns:
            self
        """
        super().train(mode)
        # frozen 모듈 eval 강제
        self._mpnn.message_passing.eval()
        self._mpnn.agg.eval()
        self._mpnn.bn.eval()
        if hasattr(self._mpnn, "X_d_transform"):
            self._mpnn.X_d_transform.eval()
        return self

    def _frozen_mp_forward_with_paper_adapters(
        self, bmg, V_d: Tensor | None = None
    ) -> Tensor:
        """chemprop ``BondMessagePassing.forward`` 인라인 + paper Eq.(8) parallel adapter.

        chemprop 2.2.3 의 ``_MessagePassingBase.forward`` 구조 재현 + bond-level
        AdapterGNNBlock 두 갈래 (A1: H_in, A2: M=message(H_in, bmg)) 를 frozen update
        결과에 element-wise 합산. chemprop internals (``W_i/W_h/W_o``, ``message``,
        ``update``, ``tau``, ``depth``, ``undirected``, ``initialize``, ``finalize``,
        ``graph_transform``, ``V_d_transform``) 는 그대로 호출.

        Args:
            bmg: chemprop ``BatchMolGraph``.
            V_d: atom feature 보강 (optional).

        Returns:
            atom-level ``(N_atoms, d_h)`` (finalize 출력).
        """
        mp = self._mpnn.message_passing
        bmg = mp.graph_transform(bmg)
        H_0 = mp.initialize(bmg)  # bond-level (V_bonds, d_h)
        H = mp.tau(H_0)

        for idx in range(mp.depth - 1):
            if mp.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2
            # frozen branch: MP(x_l)=M, then update — = BN(MLP(MP)) 자리
            M = mp.message(H, bmg)
            H_frozen = mp.update(M, H_0)
            # parallel adapter branches (s · A(·), no residual inside)
            a1 = self.pre_adapters[idx](H)
            a2 = self.post_adapters[idx](M)
            H = H_frozen + a1 + a2

        # atom-level scatter (chemprop base.py:208-211 그대로)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(
            len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device
        ).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        return mp.finalize(M, bmg.V, V_d)

    def forward(
        self, bmg, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """CheMeleon mp + adapter → agg → proj.

        Args:
            bmg: batch molecule graph (chemprop BatchMolGraph).
            V_d: atom feature 보강 (optional).
            X_d: descriptor concat 차원 (optional, d_xd > 0 일 때만 사용).

        Returns:
            h: (B, d_h) projection 출력.
        """
        if self.adapter_kind == "paper":
            # bond-level pre/post adapter 가 인라인된 MP forward
            H_v = self._frozen_mp_forward_with_paper_adapters(bmg, V_d)
        else:
            # frozen mp → atom-level (N, 2048)
            H_v = self._mpnn.message_passing(bmg, V_d)
            # trainable atom-only adapter GINs (residual, LoRA-zero-init)
            edge_index = bmg.edge_index
            for adapter in self.adapter_blocks:
                H_v = adapter(H_v, edge_index, edge_attr=None)

        # frozen agg → (B, 2048)
        graph_h = self._mpnn.agg(H_v, bmg.batch)
        graph_h = self._mpnn.bn(graph_h)

        # X_d concat (chemprop fingerprint 와 동일 시퀀스)
        if X_d is not None:
            graph_h = torch.cat((graph_h, self._mpnn.X_d_transform(X_d)), dim=1)

        return self.proj(graph_h)
