"""Polaris ADME-Fang 6-task fold 로더 (독립 외부 MTL 블록).

data/polaris_adme_fang/<stem>_{train_val,test}.csv → SMILES(Drug) outer-merge로
6-task wide 프레임 생성. 미측정 화합물 target=NaN → presence-mask가 loss에서 무시.

반환 타입은 TdcAdmetFold 동일 → runners/train.py 무변경으로 작동.

Val split:
    train_val.csv를 seed별 random 분할(val_frac=0.125). PXR loader와 동일 규약.
Test:
    Polaris가 test target을 숨김 → CSV는 [Drug]만. test_dfs에는 Y=NaN placeholder를
    넣어 predictions.parquet에 y_pred가 남게 하고, 실제 평가는 Polaris 제출로 수행.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from qadmet.datasets.tdc_admet import TdcAdmetFold, _merge
from qadmet.tasks import TASK_SETS
from qadmet.tasks.polaris_fang import FANG_CSV_STEM

logger = logging.getLogger(__name__)

FANG_DIR = Path("data/polaris_adme_fang")
FANG_VAL_FRAC = 0.125


def _load_fang_task(task_name: str, seed: int, val_frac: float = FANG_VAL_FRAC):
    """data/polaris_adme_fang/<stem>_{train_val,test}.csv → (train, val, test).

    train/val은 seed별 random 분할. test는 target 숨김 → Y=NaN.
    Returns: (train_df, val_df, test_df) — 각 columns=[Drug, Y].
    """
    stem = FANG_CSV_STEM[task_name]
    tv = pd.read_csv(FANG_DIR / f"{stem}_train_val.csv")[["Drug", "Y"]]
    te_raw = pd.read_csv(FANG_DIR / f"{stem}_test.csv")
    te = pd.DataFrame({"Drug": te_raw["Drug"].values, "Y": np.nan})

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tv))
    n_val = max(1, int(round(len(tv) * val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr = tv.iloc[train_idx].reset_index(drop=True)
    va = tv.iloc[val_idx].reset_index(drop=True)
    return tr, va, te.reset_index(drop=True)


def load_polaris_fang_fold(task_set_name: str, seed: int, *, group=None) -> TdcAdmetFold:
    """Polaris ADME-Fang 6-task 한 fold 로드.

    Args:
        task_set_name: "polaris_fang_6" 또는 "polaris_fang_6_pertask".
        seed: train/val random 분할 seed.
        group: 미사용(TDC admet_group 불필요) — 시그니처 호환용.
    """
    ts = TASK_SETS.get(task_set_name)

    train_dfs: list[pd.DataFrame] = []
    val_dfs: list[pd.DataFrame] = []
    test_dfs: dict[str, pd.DataFrame] = {}
    samples_per_task: dict[str, int] = {}

    for task_name in ts.tasks:
        tr, va, te = _load_fang_task(task_name, seed)
        train_dfs.append(tr.rename(columns={"Y": task_name}))
        val_dfs.append(va.rename(columns={"Y": task_name}))
        test_dfs[task_name] = te
        samples_per_task[task_name] = len(tr)

    train_merged = _merge(train_dfs)
    val_merged = _merge(val_dfs)

    logger.info(
        "Polaris-Fang fold: task_set=%s seed=%s train=%d val=%d tasks=%d samples=%s",
        task_set_name, seed, len(train_merged), len(val_merged),
        len(ts.tasks), samples_per_task,
    )
    return TdcAdmetFold(
        train_df=train_merged,
        val_df=val_merged,
        test_dfs=test_dfs,
        samples_per_task=samples_per_task,
    )


def collect_polaris_fang_smiles(task_set_name: str = "polaris_fang_6", *, seeds=(0,)) -> list[str]:
    """precompute 용 — 6-task 전체 SMILES 합집합 (train_val + test)."""
    ts = TASK_SETS.get(task_set_name)
    smis: set[str] = set()
    for task_name in ts.tasks:
        stem = FANG_CSV_STEM[task_name]
        tv = pd.read_csv(FANG_DIR / f"{stem}_train_val.csv")
        te = pd.read_csv(FANG_DIR / f"{stem}_test.csv")
        smis.update(tv["Drug"].tolist())
        smis.update(te["Drug"].tolist())
    return sorted(smis)
