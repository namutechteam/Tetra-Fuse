"""``PredictionSet`` — 한 base 모델의 (seed, task, split) 별 예측 묶음.

``train`` runner 가 만든 ``predictions.parquet`` (long 포맷) 를 단일 진입점으로
재해석한다. 옛 ``run_stacking_v1.py`` 의 ``load_phase1_1_preds`` /
``load_phase2_stl_preds`` (JSON per-(task,seed)) 가 이 클래스로 통합된다.

스키마:
    [seed, task, split, smiles, y_true, y_pred]

    - seed: int
    - task: str (TDC ADMET task 이름, 예: "Caco2_Wang")
    - split: str ("val" | "test")
    - smiles: str
    - y_true: float (분류면 0/1)
    - y_pred: float (raw — 분류 task 는 logit 또는 prob, base 모델별 정의 따름)

조회 방법:
    1. ``PredictionSet.from_artifact(name, store)`` — 가장 최근 train artifact 자동 resolve.
    2. ``PredictionSet.from_artifact(name, store, run_id=...)`` — 특정 run.
    3. ``PredictionSet.from_parquet(path, name)`` — 임의 path 직접 로드 (옛 결과 wrap 용).
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
    """한 base 모델의 long-format 예측 + lineage 메타데이터.

    Attributes:
        name: base 모델 식별자 (보통 train recipe 이름. ex: "mtl_dmpnn_grouped").
        df: predictions long DataFrame. 위 ``REQUIRED_COLUMNS`` 보유 보장.
        artifact: 원본 ``Artifact`` (직접 parquet 로드 시 None).
        source_path: 실제 로드된 parquet 경로 (디버그용).
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
        """train artifact 로부터 ``predictions.parquet`` 자동 로드.

        Args:
            name: artifact 이름 (recipe ``name`` 필드).
            store: ArtifactStore 인스턴스.
            run_id: 특정 run 지정. None 이면 ``store.latest`` 의 결과.
            kind: artifact 종류. 기본 ``"train"`` — ``predict`` 등 다른 kind 도 호환.

        Returns:
            PredictionSet.

        Raises:
            FileNotFoundError: artifact 가 없거나 predictions.parquet 미존재.
        """
        if run_id is not None:
            artifact = store.load(kind, name, run_id)
        else:
            artifact = store.latest(kind, name)
            if artifact is None:
                raise FileNotFoundError(
                    f"artifact 미발견: kind={kind} name={name}. 학습을 먼저 돌리세요."
                )
        pq_path = artifact.data_dir / "predictions.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"predictions.parquet 없음: {pq_path}")
        df = _load_and_validate(pq_path)
        return cls(name=name, df=df, artifact=artifact, source_path=pq_path)

    @classmethod
    def from_parquet(cls, path: Path | str, *, name: str) -> "PredictionSet":
        """임의 parquet 직접 로드 (테스트·옛 결과 변환용)."""
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
        """필터링된 long DataFrame 반환 (얕은 사본).

        Why:
            stacking strategy 가 보통 split="test" / per-task 단위로 결합하므로,
            같은 필터를 여러 base 에 일관되게 적용할 단일 진입점이 필요.
        """
        df = self.df
        if split is not None:
            if split not in VALID_SPLITS:
                raise ValueError(f"split 은 {sorted(VALID_SPLITS)} 중 하나: {split!r}")
            df = df[df["split"] == split]
        if task is not None:
            df = df[df["task"] == task]
        if seed is not None:
            df = df[df["seed"] == int(seed)]
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------- helpers

def _load_and_validate(path: Path) -> pd.DataFrame:
    """parquet 로드 후 schema 검증 + dtype 정규화."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"PredictionSet schema 위반: {path} 에 누락 컬럼 {missing}. "
            f"필수: {list(REQUIRED_COLUMNS)}"
        )
    # split 값 검증.
    bad = set(df["split"].unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"PredictionSet split 값 위반: {bad} (허용: {sorted(VALID_SPLITS)})")
    # dtype 정규화 (parquet 변환 시 float32 가 섞이는 케이스 대비).
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
    """이름별 PredictionSet 사전. 같은 이름 충돌 시 ValueError."""
    out: dict[str, PredictionSet] = {}
    for ps in predictions:
        if ps.name in out:
            raise ValueError(f"PredictionSet 이름 충돌: {ps.name}")
        out[ps.name] = ps
    return out
