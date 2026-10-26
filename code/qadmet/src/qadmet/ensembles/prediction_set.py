"""``PredictionSet`` — prediction bundle per (seed, task, split) for a single base model.

``predictions.parquet`` (long format) created by the ``train`` runner, through a single entry point,
is reinterpreted. The old ``run_stacking_v1.py``'s ``load_phase1_1_preds`` /
``load_phase2_stl_preds`` (JSON per-(task,seed)) are consolidated into this class.

Schema:
    [seed, task, split, smiles, y_true, y_pred]

    - seed: int
    - task: str (TDC ADMET task name, e.g., "Caco2_Wang")
    - split: str ("val" | "test")
    - smiles: str
    - y_true: float (0/1 for classification)
    - y_pred: float (raw — for classification tasks, logit or prob, depending on each base model's definition)

Lookup methods:
    1. ``PredictionSet.from_artifact(name, store)`` — automatically resolves the most recent train artifact.
    2. ``PredictionSet.from_artifact(name, store, run_id=...)`` — a specific run.
    3. ``PredictionSet.from_parquet(path, name)`` — loads directly from an arbitrary path (for wrapping legacy results).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from qadmet.core.artifacts import Artifact, ArtifactStore

REQUIRED_COLUMNS: tuple[str, ...] = (
    "seed", "task", "split", "smiles", "y_true", "y_pred",
)
VALID_SPLITS: frozenset[str] = frozenset({"val", "test"})


@dataclass
class PredictionSet:
    """Long-format predictions + lineage metadata for a single base model.

    Attributes:
        name: base model identifier (usually the train recipe name. ex: "mtl_dmpnn_grouped").
        df: predictions long DataFrame. Guaranteed to contain the ``REQUIRED_COLUMNS`` above.
        artifact: original ``Artifact`` (None when loaded directly from parquet).
        source_path: actual loaded parquet path (for debugging).
    """

    name: str
    df: pd.DataFrame
    artifact: Artifact | None = None
    source_path: Path | None = None

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_artifact(
        cls,
        name: str,
        store: ArtifactStore,
        *,
        run_id: str | None = None,
        kind: str = "train",
    ) -> "PredictionSet":
        """Automatically loads ``predictions.parquet`` from a train artifact.

        Args:
            name: artifact name (the recipe's ``name`` field).
            store: ArtifactStore instance.
            run_id: specifies a particular run. If None, uses the result of ``store.latest``.
            kind: artifact kind. Default ``"train"`` — also compatible with other kinds such as ``predict``.

        Returns:
            PredictionSet.

        Raises:
            FileNotFoundError: artifact does not exist or predictions.parquet is missing.
        """
        if run_id is not None:
            artifact = store.load(kind, name, run_id)
        else:
            artifact = store.latest(kind, name)
            if artifact is None:
                raise FileNotFoundError(
                    f"artifact not found: kind={kind} name={name}. Run training first."
                )
        pq_path = artifact.data_dir / "predictions.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"predictions.parquet not found: {pq_path}")
        df = _load_and_validate(pq_path)
        return cls(name=name, df=df, artifact=artifact, source_path=pq_path)

    @classmethod
    def from_parquet(cls, path: Path | str, *, name: str) -> "PredictionSet":
        """Loads directly from an arbitrary parquet (for testing/legacy result conversion)."""
        p = Path(path)
        df = _load_and_validate(p)
        return cls(name=name, df=df, artifact=None, source_path=p)

    # ----------------------------------------------------------------- accessors
    @property
    def seeds(self) -> list[int]:
        return sorted(int(s) for s in self.df["seed"].unique())

    @property
    def tasks(self) -> list[str]:
        return sorted(str(t) for t in self.df["task"].unique())

    @property
    def splits(self) -> list[str]:
        return sorted(str(s) for s in self.df["split"].unique())

    def select(
        self,
        *,
        split: str | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Returns a filtered long DataFrame (shallow copy).

        Why:
            Since stacking strategies usually combine by split="test" / per-task units,
            a single entry point is needed to apply the same filter consistently across multiple bases.
        """
        df = self.df
        if split is not None:
            if split not in VALID_SPLITS:
                raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}: {split!r}")
            df = df[df["split"] == split]
        if task is not None:
            df = df[df["task"] == task]
        if seed is not None:
            df = df[df["seed"] == int(seed)]
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------- helpers

def _load_and_validate(path: Path) -> pd.DataFrame:
    """Validates schema and normalizes dtypes after loading parquet."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"PredictionSet schema violation: missing columns {missing} in {path}. "
            f"required: {list(REQUIRED_COLUMNS)}"
        )
    # Validate split values.
    bad = set(df["split"].unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"PredictionSet split value violation: {bad} (allowed: {sorted(VALID_SPLITS)})")
    # Normalize dtypes (to handle cases where float32 gets mixed in during parquet conversion).
    df = df.assign(
        seed=df["seed"].astype("int64"),
        task=df["task"].astype("string"),
        split=df["split"].astype("string"),
        smiles=df["smiles"].astype("string"),
        y_true=df["y_true"].astype("float64"),
        y_pred=df["y_pred"].astype("float64"),
    )
    return df


def stack_predictions(
    predictions: Iterable[PredictionSet],
) -> dict[str, PredictionSet]:
    """Dict of PredictionSet by name. Raises ValueError on name collision."""
    out: dict[str, PredictionSet] = {}
    for ps in predictions:
        if ps.name in out:
            raise ValueError(f"PredictionSet name collision: {ps.name}")
        out[ps.name] = ps
    return out
