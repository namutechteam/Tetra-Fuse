"""``MetaDataset`` — wide table combining multiple ``PredictionSet`` by (seed, task, smiles) key.

``MetaLearner`` 's standard input format. The old ``run_stacking_v*.py`` pattern of zipping per-base dicts
directly is unified into an explicit wide DataFrame + (seed, task) grouping.

Schema (wide):
    [seed, task, smiles, y_true, y_pred__<base1>, y_pred__<base2>, ...]

    - join key: (seed, task, smiles, __row_in_group). Same SMILES appearing multiple times in the same
      (seed, task, split) within TDC datasets (e.g. PPBR_AZ, where the same
      molecule has multiple measurements) is matched by preserving row order, via a ``cumcount``-based
      sub-key included in the inner-join. This operates on the assumption that bases' row order matches
      (same fold loader → same input order).
    - y_true uses the value from the first base. Raises ValueError on mismatch with other bases (classification 0/1
      values should be identical across bases under normal conditions).
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
    """Result of a wide-join across multiple bases.

    Attributes:
        bases: tuple of base names determining column order (preserves input order).
        df: wide DataFrame (schema above).
        split: "test" | "val" — records which split was used for combining.
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
        """Combines the PredictionSet of K bases via inner-join.

        Args:
            predictionsets: bases to combine. All names must be unique.
            split: split to use ("test" or "val").
            check_y_true: if True, validates y_true consistency across bases (e.g. classification 0/1).

        Returns:
            MetaDataset.

        Raises:
            ValueError: duplicate base names / empty input / y_true mismatch / all bases empty
                resulting in a join with 0 rows.
        """
        if not predictionsets:
            raise ValueError("predictionsets is empty.")
        names: list[str] = []
        per_base: list[pd.DataFrame] = []
        for ps in predictionsets:
            if ps.name in names:
                raise ValueError(f"Duplicate base name: {ps.name}")
            names.append(ps.name)
            sub = ps.select(split=split)
            if sub.empty:
                raise ValueError(
                    f"PredictionSet {ps.name!r} has no data for split={split!r}."
                )
            # Sub-key for matching multiplicity of the same SMILES (e.g. PPBR_AZ) by row order.
            sub = sub.assign(
                __row_in_group=sub.groupby(["seed", "task", "smiles"]).cumcount()
            )
            per_base.append(
                sub.rename(columns={"y_pred": _y_pred_col(ps.name)})[
                    ["seed", "task", "smiles", "__row_in_group", "y_true", _y_pred_col(ps.name)]
                ]
            )

        # Inner-join sequentially, starting from the first base as the reference.
        keys = ["seed", "task", "smiles", "__row_in_group"]
        first = per_base[0]
        merged = first.copy()
        for other_name, other in zip(names[1:], per_base[1:]):
            other_renamed = other.rename(columns={"y_true": f"y_true__{other_name}"})
            merged = merged.merge(other_renamed, on=keys, how="inner")
            if check_y_true:
                col = f"y_true__{other_name}"
                # NaN cases are also treated as matching (missing targets are normal in regression).
                eq = (
                    merged["y_true"].eq(merged[col])
                    | (merged["y_true"].isna() & merged[col].isna())
                )
                if not eq.all():
                    bad = (~eq).sum()
                    raise ValueError(
                        f"y_true of base {other_name!r} mismatches the reference in {bad} rows."
                    )
                merged = merged.drop(columns=[col])

        if merged.empty:
            raise ValueError(
                "Inner-join result of (seed, task, smiles) across all bases is empty."
            )

        # Normalize column order.
        cols = ["seed", "task", "smiles", "y_true", *[_y_pred_col(n) for n in names]]
        merged = merged[cols].reset_index(drop=True)
        return cls(bases=tuple(names), df=merged, split=split)

    # ----------------------------------------------------------------- accessors
    @property
    def y_pred_columns(self) -> list[str]:
        return [_y_pred_col(b) for b in self.bases]

    def predictions_matrix(self, sub_df: pd.DataFrame | None = None) -> np.ndarray:
        """Returns the base prediction matrix ``(N, K)`` for the selected rows.

        Args:
            sub_df: subset. If None, uses the entire ``self.df``.
        """
        df = self.df if sub_df is None else sub_df
        return df[self.y_pred_columns].to_numpy(dtype=np.float64)

    def y_true(self, sub_df: pd.DataFrame | None = None) -> np.ndarray:
        df = self.df if sub_df is None else sub_df
        return df["y_true"].to_numpy(dtype=np.float64)

    def iter_tasks(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yields groups in units of ``(task, task_df)`` (sorted by task)."""
        for task in sorted(self.df["task"].unique()):
            yield task, self.df[self.df["task"] == task].reset_index(drop=True)

    def iter_seeds(self) -> Iterator[tuple[int, pd.DataFrame]]:
        for seed in sorted(self.df["seed"].unique()):
            yield int(seed), self.df[self.df["seed"] == seed].reset_index(drop=True)
