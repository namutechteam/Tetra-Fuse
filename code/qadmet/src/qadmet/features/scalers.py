"""Per-block StandardScaler + clipping.

Legacy ``fit_xd_scalers`` / ``apply_xd_scalers`` (scripts-old/pipeline_mtl.py:334-365) of
Generalization. When RDKit/xTB/MoLFormer/UniMol2 are concatenated into one X_d matrix, each block
receives index ranges and performs fit·transform·clip on a per-block basis.

Why:
    The reason block-level scaling is needed is (a) each modality's distribution differs greatly, and
    (b) mask channels are 0/1, so they are not scaling targets. The legacy code had slices
    hardcoded, but here they are received as a Spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BlockSpec:
    """Defines one slice of X_d.

    Attributes:
        name: block name (e.g., "rdkit", "xtb", "molformer").
        start: start index (inclusive).
        end: end index (exclusive).
        clip: ±σ clip value for training stability. If None, no clipping is applied.
        observed_only: fit using only observed rows (e.g., when missing rows are imputed as 0, as in xTB).
    """

    name: str
    start: int
    end: int
    clip: float | None = 10.0
    observed_only: bool = False
    mask_block: tuple[int, int] | None = None  # mask slice when observed_only=True.


class BlockScaler:
    """Per-block StandardScaler container. Separates fit / transform into two stages."""

    def __init__(self, blocks: list[BlockSpec]) -> None:
        self.blocks = blocks
        self._scalers: dict[str, StandardScaler] = {}

    def fit(self, train_xd: np.ndarray) -> "BlockScaler":
        for b in self.blocks:
            block = train_xd[:, b.start : b.end]
            if b.observed_only:
                if b.mask_block is None:
                    raise ValueError(f"BlockSpec {b.name!r}: mask_block is required when observed_only=True")
                ms, me = b.mask_block
                mask = train_xd[:, ms:me]
                is_obs = (mask == 0).all(axis=1)
                if is_obs.sum() < 2:
                    # fallback: fit on the entire set (same as legacy code).
                    s = StandardScaler().fit(block)
                else:
                    s = StandardScaler().fit(block[is_obs])
            else:
                s = StandardScaler().fit(block)
            self._scalers[b.name] = s
        return self

    def transform(self, xd: np.ndarray) -> np.ndarray:
        out = xd.astype(np.float64).copy()
        for b in self.blocks:
            s = self._scalers[b.name]
            out[:, b.start : b.end] = s.transform(xd[:, b.start : b.end])
            if b.clip is not None:
                out[:, b.start : b.end] = np.clip(out[:, b.start : b.end], -b.clip, b.clip)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)

    def fit_transform(self, train_xd: np.ndarray) -> np.ndarray:
        return self.fit(train_xd).transform(train_xd)

    def state(self) -> Mapping[str, dict]:
        """Serialization dict for bundling with checkpoint."""
        return {
            name: {"mean": s.mean_.tolist(), "scale": s.scale_.tolist(), "var": s.var_.tolist()}
            for name, s in self._scalers.items()
        }


def fit_block_scalers(train_xd: np.ndarray, blocks: list[BlockSpec]) -> BlockScaler:
    """Convenience function: create a BlockScaler in one line."""
    return BlockScaler(blocks).fit(train_xd)
