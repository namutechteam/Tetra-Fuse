"""SharedPrecomputeBase: Consolidates ~80% of boilerplate from the 6 legacy precompute scripts.

Each modality builder ``compute_batch(smiles: list[str]) -> dict[str, np.ndarray]`` only
needs to implement this. resume / parquet IO / multiprocessing are handled by this base.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from qadmet.features.stores import ParquetStore

logger = logging.getLogger(__name__)


class PrecomputeBase(ABC):
    """Feature precomputation base.

    Things the subclass must define:
    - ``column_prefix``: output column prefix (e.g. "rd_", "mf_", "um_").
    - ``output_dim``: feature dimension per SMILES.
    - ``compute_batch(smiles)``: SMILES batch → ``{smi: np.ndarray(output_dim,)}``.

    Optional overrides for subclass:
    - ``columns``: list of column names. Defaults to ``f"{prefix}{i:03d}"``.
    - ``default_batch_size``: default outer batch (unit of compute_batch calls).
    - ``default_checkpoint_every``: interval for intermediate parquet saves.
    - ``run_parallel``: whether to use multiprocessing (default False — subclass implements its own).
    """

    name: str = ""
    column_prefix: str = ""
    output_dim: int = 0
    default_batch_size: int = 256
    default_checkpoint_every: int = 5000

    @property
    def columns(self) -> list[str]:
        return [f"{self.column_prefix}{i:03d}" for i in range(self.output_dim)]

    @abstractmethod
    def compute_batch(self, smiles: list[str]) -> dict[str, np.ndarray]:
        """SMILES batch → {smi: vector}. Failed SMILES are either excluded from the dict or filled with zeros."""

    # ------------------------------------------------------------------ run
    def run(
        self,
        smiles: Iterable[str],
        output_path: str | Path,
        *,
        resume: bool = True,
        batch_size: int | None = None,
        checkpoint_every: int | None = None,
    ) -> ParquetStore:
        """Compute features for all SMILES + save to parquet.

        Args:
            smiles: input SMILES (duplicates allowed, sorted/deduped internally).
            output_path: save location.
            resume: if True, SMILES already in the existing parquet are skipped without recomputation.
            batch_size: number of SMILES passed to ``compute_batch`` at once.
            checkpoint_every: save intermediate parquet each time this many accumulate.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if checkpoint_every is None:
            checkpoint_every = self.default_checkpoint_every
        store = ParquetStore(output_path)
        smiles = sorted(set(smiles))
        if resume and store.exists():
            done = store.existing_smiles()
            before = len(smiles)
            smiles = [s for s in smiles if s not in done]
            logger.info("resume: %s already done, %s remaining", before - len(smiles), len(smiles))
        if not smiles:
            logger.info("nothing to compute — exit.")
            return store

        logger.info("computing %s for %s SMILES (batch_size=%s)", self.name, len(smiles), batch_size)

        accumulated: dict[str, np.ndarray] = {}
        with tqdm(total=len(smiles), desc=self.name) as pbar:
            for start in range(0, len(smiles), batch_size):
                chunk = smiles[start : start + batch_size]
                results = self.compute_batch(chunk)
                accumulated.update(results)
                pbar.update(len(chunk))
                if len(accumulated) >= checkpoint_every:
                    self._flush(store, accumulated)
                    accumulated.clear()
        if accumulated:
            self._flush(store, accumulated)
        return store

    # --------------------------------------------------------------- internal
    def _flush(self, store: ParquetStore, results: dict[str, np.ndarray]) -> None:
        """Upsert accumulated results into the store (idempotent)."""
        cols = self.columns
        smis = list(results.keys())
        mat = np.stack([results[s] for s in smis], axis=0)
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, "smiles", smis)
        store.upsert(df)
