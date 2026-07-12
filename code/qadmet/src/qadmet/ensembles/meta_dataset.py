"""``MetaDataset`` — 여러 ``PredictionSet`` 을 (seed, task, smiles) 키로 결합한 wide table.

``MetaLearner`` 가 받는 표준 입력 형태. 옛 ``run_stacking_v*.py`` 가 base 별 dict 를
직접 zip 해서 사용하던 패턴을, 명시적인 wide DataFrame + (seed, task) 그룹핑으로 통합.

스키마 (wide):
    [seed, task, smiles, y_true, y_pred__<base1>, y_pred__<base2>, ...]

    - join key: (seed, task, smiles, __row_in_group). 같은 SMILES 가 같은
      (seed, task, split) 안에 여러 번 등장하는 TDC 데이터셋 (PPBR_AZ 같이 같은
      분자의 여러 측정값) 을 row 순서 보존으로 매칭하기 위해 ``cumcount`` 기반
      sub-key 를 inner-join 에 포함한다. base 들의 row 순서가 일치한다는
      가정 (같은 fold loader → 같은 입력 순서) 위에서 동작.
    - y_true 는 첫 base 의 값을 사용. 다른 base 와 불일치 시 ValueError (분류 0/1
      값이 base 마다 같아야 정상).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterator

import numpy as np
import pandas as pd

from qadmet.ensembles.prediction_set import PredictionSet


def _y_pred_col(base_name: str) -> str:
    return f"y_pred__{base_name}"


@dataclass
class MetaDataset:
    """다중 base 의 wide-join 결과.

    Attributes:
        bases: 컬럼 순서를 결정하는 base 이름 튜플 (입력 순서 보존).
        df: wide DataFrame (위 스키마).
        split: "test" | "val" — 어떤 split 으로 결합했는지 기록.
    """

    bases: tuple[str, ...]
    df: pd.DataFrame
    split: str

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_predictionsets(
        cls,
        predictionsets: list[PredictionSet] | tuple[PredictionSet, ...],
        *,
        split: str = "test",
        check_y_true: bool = True,
    ) -> "MetaDataset":
        """K base 의 PredictionSet 을 inner-join 으로 결합한다.

        Args:
            predictionsets: 결합할 base 들. 이름이 모두 unique 해야 한다.
            split: 사용할 split ("test" 또는 "val").
            check_y_true: True 면 base 간 y_true 일치 여부를 검증 (분류 0/1 등).

        Returns:
            MetaDataset.

        Raises:
            ValueError: base 이름 중복 / 빈 입력 / y_true 불일치 / 모든 base 가 비어
                있어 join 결과가 0 행.
        """
        if not predictionsets:
            raise ValueError("predictionsets 가 비어있습니다.")
        names: list[str] = []
        per_base: list[pd.DataFrame] = []
        for ps in predictionsets:
            if ps.name in names:
                raise ValueError(f"base 이름 중복: {ps.name}")
            names.append(ps.name)
            sub = ps.select(split=split)
            if sub.empty:
                raise ValueError(
                    f"PredictionSet {ps.name!r} 에 split={split!r} 데이터가 없습니다."
                )
            # 같은 SMILES 의 multiplicity (PPBR_AZ 등) 를 row 순서로 매칭하기 위한 sub-key.
            sub = sub.assign(
                __row_in_group=sub.groupby(["seed", "task", "smiles"]).cumcount()
            )
            per_base.append(
                sub.rename(columns={"y_pred": _y_pred_col(ps.name)})[
                    ["seed", "task", "smiles", "__row_in_group", "y_true", _y_pred_col(ps.name)]
                ]
            )

        # 첫 base 를 기준으로 차례차례 inner-join.
        keys = ["seed", "task", "smiles", "__row_in_group"]
        first = per_base[0]
        merged = first.copy()
        for other_name, other in zip(names[1:], per_base[1:]):
            other_renamed = other.rename(columns={"y_true": f"y_true__{other_name}"})
            merged = merged.merge(other_renamed, on=keys, how="inner")
            if check_y_true:
                col = f"y_true__{other_name}"
                # NaN 케이스도 일치로 본다 (regression 에서 missing target 이 정상).
                eq = (
                    merged["y_true"].eq(merged[col])
                    | (merged["y_true"].isna() & merged[col].isna())
                )
                if not eq.all():
                    bad = (~eq).sum()
                    raise ValueError(
                        f"base {other_name!r} 의 y_true 가 기준과 {bad} 행 불일치."
                    )
                merged = merged.drop(columns=[col])

        if merged.empty:
            raise ValueError(
                "모든 base 의 (seed, task, smiles) inner-join 결과가 비어있습니다."
            )

        # 컬럼 순서 정규화.
        cols = ["seed", "task", "smiles", "y_true", *[_y_pred_col(n) for n in names]]
        merged = merged[cols].reset_index(drop=True)
        return cls(bases=tuple(names), df=merged, split=split)

    # ----------------------------------------------------------------- accessors
    @property
    def y_pred_columns(self) -> list[str]:
        return [_y_pred_col(b) for b in self.bases]

    def predictions_matrix(self, sub_df: pd.DataFrame | None = None) -> np.ndarray:
        """선택된 행의 base 예측 행렬 ``(N, K)`` 을 반환.

        Args:
            sub_df: 부분집합. None 이면 전체 ``self.df``.
        """
        df = self.df if sub_df is None else sub_df
        return df[self.y_pred_columns].to_numpy(dtype=np.float64)

    def y_true(self, sub_df: pd.DataFrame | None = None) -> np.ndarray:
        df = self.df if sub_df is None else sub_df
        return df["y_true"].to_numpy(dtype=np.float64)

    def iter_tasks(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """``(task, task_df)`` 단위로 그룹 yield (task 정렬 순)."""
        for task in sorted(self.df["task"].unique()):
            yield task, self.df[self.df["task"] == task].reset_index(drop=True)

    def iter_seeds(self) -> Iterator[tuple[int, pd.DataFrame]]:
        for seed in sorted(self.df["seed"].unique()):
            yield int(seed), self.df[self.df["seed"] == seed].reset_index(drop=True)
