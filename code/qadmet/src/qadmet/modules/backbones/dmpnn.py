"""DMPNNBackbone — chemprop ``BondMessagePassing + NormAggregation`` wrapper.

Form that separates only the mp/agg part of the old ``pipeline_mtl.build_mtl_mpnn`` (line 1218-1365).
Since predictor is handled by an external head, insert a dummy ``RegressionFFN`` to build the chemprop ``MPNN``
instance, and at train/inference time only ``fingerprint(bmg, V_d, X_d)`` is called.

Why dummy predictor:
    This is because the chemprop ``MPNN`` constructor requires predictor as mandatory. In the new design,
    head/loss is assembled externally, so predictor is not used.

Lazy import:
    chemprop is a heavy package that takes about 0.5 ~ 1 second at import-time, so it is imported only inside ``__init__``
    (not imported at the time the registration decorator runs).
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


@BACKBONES.register("dmpnn")
class DMPNNBackbone(nn.Module):
    """D-MPNN graph encoder (post-aggregation fingerprint output).

    Args:
        d_h: hidden dim. message passing output dimension.
        depth: number of message passing steps.
        dropout: node/edge dropout.
        d_v: atom feature dimension (chemprop default 72).
        d_e: bond feature dimension (chemprop default 14).
        agg_norm: ``NormAggregation`` normalization constant (default 100.0).
        d_xd: additional descriptor concat dimension (when forward's X_d is not None
            dimension concatenated to output). Used only for output_dim calculation.

    Note:
        forward's ``X_d`` argument is handled directly by chemprop ``MPNN.fingerprint``.
        In the new design where multiple modalities are combined in fusion, usually with ``X_d=None``
        call, and descriptor is passed through fusion's separate branch.
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
