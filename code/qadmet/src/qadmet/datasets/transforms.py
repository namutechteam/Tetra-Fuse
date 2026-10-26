"""target column transform (log1p etc.).

Generalizes the part where the old ``LOG_TRANSFORM_TASKS`` set was hardcoded, based on Task metadata
(Task.log_transform=True).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def apply_log1p(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Returns a copy with log1p applied in-place to the given column."""
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = np.log1p(out[c])
    return out


def invert_log1p(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)
