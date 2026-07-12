"""feature 캐시 추상화.

ParquetStore: SMILES indexed parquet 파일. 옛 6 종 precompute 스크립트가 매번
재구현하던 read/append/lookup/resume 패턴을 한 클래스로 압축한다.

규약:
- 첫 컬럼은 항상 ``smiles`` (lookup 키).
- 나머지 컬럼은 builder 가 자유롭게 정의 (예: rd_000..rd_199).
- ``upsert(rows)`` 는 같은 SMILES 가 이미 있으면 덮어쓴다.
- ``lookup(smiles_list, columns)`` 는 누락 SMILES 행을 NaN 으로 채운 (N, len(columns)) array 를 반환.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class ParquetStore:
    """SMILES indexed parquet feature 캐시."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ----------------------------------------------------------------- 읽기
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(f"ParquetStore 파일 없음: {self.path}")
        df = pd.read_parquet(self.path)
        if "smiles" not in df.columns:
            raise ValueError(f"{self.path}: 'smiles' 컬럼이 필요합니다.")
        return df

    def load_or_empty(self, columns: Sequence[str]) -> pd.DataFrame:
        if self.exists():
            return self.load()
        return pd.DataFrame(columns=["smiles", *columns])

    def existing_smiles(self) -> set[str]:
        if not self.exists():
            return set()
        return set(self.load()["smiles"].tolist())

    # ----------------------------------------------------------------- 쓰기
    def upsert(self, df: pd.DataFrame) -> None:
        """SMILES 충돌 시 새 값으로 덮어쓰기. 컬럼 일관성 검사."""
        if "smiles" not in df.columns:
            raise ValueError("upsert: df 에 'smiles' 컬럼 필요")
        if self.exists():
            existing = self.load()
            keep = existing[~existing["smiles"].isin(df["smiles"])]
            merged = pd.concat([keep, df], ignore_index=True)
        else:
            merged = df.copy()
            self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)

    # ----------------------------------------------------------------- 조회
    def lookup(
        self, smiles_list: Sequence[str], columns: Sequence[str]
    ) -> np.ndarray:
        """SMILES 리스트 순서대로 (N, D) 배열. 누락 행은 NaN."""
        df = self.load() if self.exists() else pd.DataFrame(columns=["smiles", *columns])
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.path}: 누락된 컬럼 {missing[:5]} ...")
        lookup_df = df.set_index("smiles")[list(columns)]
        out = np.full((len(smiles_list), len(columns)), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            if smi in lookup_df.index:
                row = lookup_df.loc[smi]
                if isinstance(row, pd.DataFrame):  # 중복 SMILES 보호.
                    row = row.iloc[0]
                out[i] = row.to_numpy(dtype=np.float32)
        return out
