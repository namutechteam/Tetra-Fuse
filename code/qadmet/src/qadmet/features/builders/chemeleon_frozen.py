"""CheMeleon frozen pooled embedding builder (2048-dim, pre-projection).

frozen-backbone ablation — LoRA/adapter is never trained, only fusion+head
To avoid recomputing the GNN every epoch during training, frozen pooled embeddings are precomputed into parquet
and saved.

frozen pooled embedding definition:
    ``bn(agg(message_passing(bmg), bmg.batch))`` — frozen MP → frozen MeanAggregation →
    frozen BN. Does not pass through adapter / proj (pretrained GNN only).

Note:
    - chemprop BatchMolGraph generation follows the training path (loaders.py ``_build_chemprop_dataset``) and
      is built identically via ``make_datapoints`` → ``make_dataset`` → ``collate_batch``.
    - The pretrained GNN is always frozen — adapter_kind/n_adapter_layers is (created but)
      not used, since submodules are called directly.
    - Heavy dependencies (chemprop) are imported only inside the method.
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
        # adapter_kind="lora_gin" + n_adapter_layers=1 is minimal cost — adapter is created but
        # not used, since only the frozen submodule(message_passing/agg/bn) is called directly.
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
            # chemprop dataset construction identical to the training path (dummy targets, no descriptors).
            dset = _build_chemprop_dataset(
                chunk,
                np.zeros((len(chunk), 1), dtype=np.float32),
                None,
                n_workers=self.n_workers,
            )
            items = [dset[i] for i in range(len(dset))]
            tb = collate_batch(items)
            bmg = tb.bmg
            bmg.to(self.device)  # in-place — returns None.
            with torch.no_grad():
                # frozen message_passing → frozen agg → frozen bn (NO adapter, NO proj).
                pooled = mpnn.bn(mpnn.agg(mpnn.message_passing(bmg), bmg.batch))
            arr = pooled.detach().cpu().numpy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for s, vec in zip(chunk, arr):
                out[s] = vec
        return out
