"""DMPNNBackbone — chemprop ``BondMessagePassing + NormAggregation`` 래퍼.

옛 ``pipeline_mtl.build_mtl_mpnn`` (line 1218-1365) 의 mp/agg 부분만 분리한 형태이다.
predictor 는 외부 head 가 담당하므로 dummy ``RegressionFFN`` 을 넣어 chemprop ``MPNN``
인스턴스를 만들고, 학습/추론 시에는 ``fingerprint(bmg, V_d, X_d)`` 만 호출한다.

Why dummy predictor:
    chemprop ``MPNN`` 생성자가 predictor 를 필수로 받기 때문이다. 새 설계에서는
    head/loss 를 외부에서 조립하므로 predictor 는 사용하지 않는다.

Lazy import:
    chemprop 은 import-time 에 약 0.5 ~ 1 초 걸리는 무거운 패키지라 ``__init__``
    안에서만 import 한다 (등록 데코레이터 동작 시점에는 import 되지 않음).
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


@BACKBONES.register("dmpnn")
class DMPNNBackbone(nn.Module):
    """D-MPNN 그래프 인코더 (post-aggregation fingerprint 출력).

    Args:
        d_h: hidden dim. message passing 출력 차원.
        depth: message passing 단계 수.
        dropout: 노드/엣지 dropout.
        d_v: 원자 feature 차원 (chemprop 기본 72).
        d_e: 결합 feature 차원 (chemprop 기본 14).
        agg_norm: ``NormAggregation`` 정규화 상수 (기본 100.0).
        d_xd: 추가 descriptor concat 차원 (forward 의 X_d 가 None 이 아닐 때
            출력에 합쳐지는 차원). output_dim 계산에만 사용된다.

    Note:
        forward 의 ``X_d`` 인자는 chemprop ``MPNN.fingerprint`` 가 직접 처리한다.
        다중 모달리티를 fusion 에서 합치는 새 설계에서는 보통 ``X_d=None`` 으로
        호출하고 descriptor 는 fusion 의 별도 branch 로 흘려보낸다.
    """

    def __init__(
        self,
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        d_v: int = 72,
        d_e: int = 14,
        agg_norm: float = 100.0,
        d_xd: int = 0,
    ) -> None:
        super().__init__()
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, NormAggregation
        from chemprop.nn.predictors import RegressionFFN

        mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=d_h, depth=depth, dropout=dropout)
        agg = NormAggregation(norm=agg_norm)
        dummy_predictor = RegressionFFN(
            n_tasks=1, input_dim=1, hidden_dim=4, n_layers=1,
        )
        self._mpnn = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=dummy_predictor,
            batch_norm=False,
        )
        self.d_h = d_h
        self.d_xd = d_xd
        self.output_dim = d_h + d_xd

    def forward(self, bmg, V_d: Tensor | None = None, X_d: Tensor | None = None) -> Tensor:
        return self._mpnn.fingerprint(bmg, V_d, X_d)
