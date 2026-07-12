"""SharedPrecomputeBase: 옛 6 종 precompute 스크립트의 ~80% 보일러플레이트 통합.

각 modality builder 는 ``compute_batch(smiles: list[str]) -> dict[str, np.ndarray]`` 만
구현하면 된다. resume / parquet IO / multiprocessing 은 본 베이스가 처리.
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
    """feature 사전 계산 베이스.

    Subclass 가 정의해야 할 것:
    - ``column_prefix``: 출력 컬럼 prefix (예: "rd_", "mf_", "um_").
    - ``output_dim``: 한 SMILES 당 feature 차원.
    - ``compute_batch(smiles)``: SMILES 배치 → ``{smi: np.ndarray(output_dim,)}``.

    Subclass 가 옵션으로 override:
    - ``columns``: 컬럼 이름 리스트. 기본은 ``f"{prefix}{i:03d}"``.
    - ``default_batch_size``: 기본 외부 batch (compute_batch 호출 단위).
    - ``default_checkpoint_every``: parquet 중간 저장 주기.
    - ``run_parallel``: 멀티프로세싱 여부 (기본 False — subclass 가 자체 구현).
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
        """SMILES 배치 → {smi: vector}. 실패한 SMILES 는 dict 에서 제외하거나 zeros 로 채움."""

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
        """SMILES 전체에 대해 feature 계산 + parquet 저장.

        Args:
            smiles: 입력 SMILES (중복 허용, 내부에서 정렬·dedupe).
            output_path: 저장 위치.
            resume: True 면 기존 parquet 의 SMILES 는 재계산 없이 건너뜀.
            batch_size: ``compute_batch`` 한 번에 넘길 SMILES 수.
            checkpoint_every: 이 만큼 누적될 때마다 parquet 중간 저장.
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
        """누적된 결과를 store 에 upsert (멱등)."""
        cols = self.columns
        smis = list(results.keys())
        mat = np.stack([results[s] for s in smis], axis=0)
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, "smiles", smis)
        store.upsert(df)
