"""CheMeleon frozen pooled embedding builder (2048-dim, pre-projection).

frozen-backbone ablation 용 — LoRA/adapter 를 전혀 학습하지 않고 fusion+head 만
학습할 때 매 epoch GNN 재계산을 피하기 위해 frozen pooled embedding 을 미리 parquet 로
저장한다.

frozen pooled embedding 정의:
    ``bn(agg(message_passing(bmg), bmg.batch))`` — frozen MP → frozen MeanAggregation →
    frozen BN. adapter / proj 는 거치지 않는다 (사전학습 GNN 만).

Note:
    - chemprop BatchMolGraph 생성은 학습 경로(loaders.py ``_build_chemprop_dataset``)와
      동일하게 ``make_datapoints`` → ``make_dataset`` → ``collate_batch`` 로 만든다.
    - 사전학습 GNN 은 항상 frozen — adapter_kind/n_adapter_layers 는 (생성은 되지만)
      직접 submodule 을 호출하므로 사용되지 않는다.
    - 무거운 의존성(chemprop)은 method 안에서만 import.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

logger = logging.getLogger(__name__)

EMBED_DIM = 2048
DEFAULT_CKPT = "data/weights/chemeleon_mp.pt"


@FEATURE_BUILDERS.register("chemeleon_frozen")
class CheMeleonFrozenBuilder(PrecomputeBase):
    """CheMeleon frozen pooled embedding (2048-dim, pre-projection/pre-adapter)."""

    name = "chemeleon_frozen"
    column_prefix = "cm_"
    output_dim = EMBED_DIM
    default_batch_size = 4096
    default_checkpoint_every = 4096

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CKPT,
        gpu_batch_size: int = 256,
        device: str = "cuda:0",
        n_workers: int = 8,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.gpu_batch_size = gpu_batch_size
        self.device = device
        self.n_workers = n_workers
        self._backbone = None

    def _ensure_loaded(self) -> None:
        if self._backbone is not None:
            return
        import torch  # noqa: F401

        from qadmet.modules.backbones.chemeleon import CheMeleonBackbone

        logger.info("building CheMeleon frozen backbone from %s", self.checkpoint_path)
        # adapter_kind="lora_gin" + n_adapter_layers=1 은 최소 비용 — adapter 는 생성되지만
        # frozen submodule(message_passing/agg/bn)만 직접 호출하므로 사용되지 않는다.
        backbone = CheMeleonBackbone(
            checkpoint_path=self.checkpoint_path,
            adapter_kind="lora_gin",
            n_adapter_layers=1,
        ).eval()
        self._backbone = backbone.to(self.device)

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        import torch
        from chemprop.data.collate import collate_batch

        from qadmet.datasets.loaders import _build_chemprop_dataset

        self._ensure_loaded()
        backbone = self._backbone
        mpnn = backbone._mpnn
        out: dict[str, np.ndarray] = {}
        smiles = list(smiles)

        for start in range(0, len(smiles), self.gpu_batch_size):
            chunk = smiles[start : start + self.gpu_batch_size]
            # 학습 경로와 동일한 chemprop dataset 구성 (dummy targets, no descriptors).
            dset = _build_chemprop_dataset(
                chunk,
                np.zeros((len(chunk), 1), dtype=np.float32),
                None,
                n_workers=self.n_workers,
            )
            items = [dset[i] for i in range(len(dset))]
            tb = collate_batch(items)
            bmg = tb.bmg
            bmg.to(self.device)  # in-place — None 반환.
            with torch.no_grad():
                # frozen message_passing → frozen agg → frozen bn (NO adapter, NO proj).
                pooled = mpnn.bn(mpnn.agg(mpnn.message_passing(bmg), bmg.batch))
            arr = pooled.detach().cpu().numpy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for s, vec in zip(chunk, arr):
                out[s] = vec
        return out
