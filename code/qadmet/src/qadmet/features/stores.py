"""feature cache abstraction.

ParquetStore: SMILES indexed parquet file. The old 6 precompute scripts each
reimplemented the read/append/lookup/resume pattern; condensed here into a single class.

Conventions:
- The first column is always ``smiles`` (lookup key).
- The remaining columns are freely defined by the builder (e.g. rd_000..rd_199).
- ``upsert(rows)`` overwrites if the same SMILES already exists.
- ``lookup(smiles_list, columns)`` returns an (N, len(columns)) array with missing SMILES rows filled with NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class ParquetStore:
    """SMILES indexed parquet feature cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ----------------------------------------------------------------- Read
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(f"ParquetStore file not found: {self.path}")
        df = pd.read_parquet(self.path)
        if "smiles" not in df.columns:
            raise ValueError(f"{self.path}: 'smiles' column is required.")
        return df

    def load_or_empty(self, columns: Sequence[str]) -> pd.DataFrame:
        if self.exists():
            return self.load()
        return pd.DataFrame(columns=["smiles", *columns])

    def existing_smiles(self) -> set[str]:
        if not self.exists():
            return set()
        return set(self.load()["smiles"].tolist())

    # ----------------------------------------------------------------- Write
    def upsert(self, df: pd.DataFrame) -> None:
        """Overwrite with new value on SMILES collision. Checks column consistency."""
        if "smiles" not in df.columns:
            raise ValueError("upsert: df requires 'smiles' column")
        if self.exists():
            existing = self.load()
            keep = existing[~existing["smiles"].isin(df["smiles"])]
            merged = pd.concat([keep, df], ignore_index=True)
        else:
            merged = df.copy()
            self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)

    # ----------------------------------------------------------------- Lookup
    def lookup(
        self, smiles_list: Sequence[str], columns: Sequence[str]
    ) -> np.ndarray:
        """(N, D) array in order of the SMILES list. Missing rows are NaN."""
        df = self.load() if self.exists() else pd.DataFrame(columns=["smiles", *columns])
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.path}: missing columns {missing[:5]} ...")
        lookup_df = df.set_index("smiles")[list(columns)]
        out = np.full((len(smiles_list), len(columns)), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            if smi in lookup_df.index:
                row = lookup_df.loc[smi]
                if isinstance(row, pd.DataFrame):  # Guard against duplicate SMILES.
                    row = row.iloc[0]
                out[i] = row.to_numpy(dtype=np.float32)
        return out
