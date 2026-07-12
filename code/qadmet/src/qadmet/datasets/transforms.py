"""target column transform (log1p 등).

옛 ``LOG_TRANSFORM_TASKS`` 셋이 hardcoded 였던 부분을 Task 메타데이터 기반으로
일반화한다 (Task.log_transform=True).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def apply_log1p(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """주어진 컬럼에 log1p 를 in-place 적용한 사본을 반환."""
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = np.log1p(out[c])
    return out


def invert_log1p(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)
