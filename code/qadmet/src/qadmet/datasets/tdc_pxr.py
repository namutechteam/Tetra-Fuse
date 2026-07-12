"""TDC22 + PXR3 = 25-task 통합 fold 로더 (plan_details.md §2.4).

TDC 22-task 는 기존 ``tdc.benchmark_group`` 경로로, PXR 3-task 는 ``data/pxr/<task>/``
CSV(build_pxr_csv.py 산출) 에서 로드한 뒤 SMILES(Drug) outer-merge 로 25-task wide
프레임을 만든다. 각 task 미측정 화합물은 target=NaN → presence-mask 가 loss 에서 무시.

반환 타입은 ``TdcAdmetFold`` 동일 → runners/train.py 무변경으로 작동.

PXR val split:
    train_val.csv 를 seed 별 random 분할(val_frac=0.125). TDC 의 seed별 default split 과
    동일하게 seed 마다 다른 val 을 쓴다. PXR test 는 고정(build_pxr_csv 의 scaffold holdout
    또는 PHASE1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qadmet.datasets.tdc_admet import TdcAdmetFold, _merge
from qadmet.tasks import TASK_SETS

logger = logging.getLogger(__name__)

PXR_DIR = Path("data/pxr")
PXR_VAL_FRAC = 0.125


def _load_pxr_task(task: str, seed: int, val_frac: float = PXR_VAL_FRAC):
    """data/pxr/<task>/ → (train, val, test). train/val 은 seed별 random 분할.

    Returns:
        (train_df, val_df, test_df) — 각각 columns=[Drug, Y].
    """
    base = PXR_DIR / task
    tv = pd.read_csv(base / "train_val.csv")[["Drug", "Y"]]
    te = pd.read_csv(base / "test.csv")[["Drug", "Y"]]

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tv))
    n_val = max(1, int(round(len(tv) * val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr = tv.iloc[train_idx].reset_index(drop=True)
    va = tv.iloc[val_idx].reset_index(drop=True)
    return tr, va, te.reset_index(drop=True)


def load_tdc_pxr_fold(task_set_name: str, seed: int, *, group=None) -> TdcAdmetFold:
    """25-task(TDC22 + PXR3) 한 fold 로드.

    Args:
        task_set_name: "tdc_pxr_25" (causal6+PXR) 또는 "tdc_pxr_25_pertask" (STL).
            둘 다 task list 는 동일 → 데이터는 같고 grouping 만 다름.
        seed: TDC split seed + PXR val 분할 seed.
        group: TDC admet_group 주입(테스트용). None 이면 새로 생성.
    """
    from qadmet.tasks.pxr import PXR_3

    ts = TASK_SETS.get(task_set_name)
    pxr_set = set(PXR_3)
    tdc_tasks = [t for t in ts.tasks if t not in pxr_set]
    pxr_tasks = [t for t in ts.tasks if t in pxr_set]

    train_dfs: list[pd.DataFrame] = []
    val_dfs: list[pd.DataFrame] = []
    test_dfs: dict[str, pd.DataFrame] = {}
    samples_per_task: dict[str, int] = {}

    # --- TDC 22 ---
    if group is None:
        from tdc.benchmark_group import admet_group  # type: ignore

        group = admet_group(path="data/")
    for task_name in tdc_tasks:
        bench = group.get(task_name)
        test_dfs[task_name] = bench["test"]
        tr, va = group.get_train_valid_split(
            benchmark=task_name, split_type="default", seed=seed,
        )
        train_dfs.append(tr[["Drug", "Y"]].rename(columns={"Y": task_name}))
        val_dfs.append(va[["Drug", "Y"]].rename(columns={"Y": task_name}))
        samples_per_task[task_name] = len(tr)

    # --- PXR 3 (data/pxr CSV) ---
    for task_name in pxr_tasks:
        tr, va, te = _load_pxr_task(task_name, seed)
        train_dfs.append(tr.rename(columns={"Y": task_name}))
        val_dfs.append(va.rename(columns={"Y": task_name}))
        test_dfs[task_name] = te
        samples_per_task[task_name] = len(tr)

    # --- submit 모드: PXR_pEC50 test 에 BLINDED+structure(챌린지 제출 대상) 추가 ---
    # QADMET_PXR_SUBMIT=1 이면 학습된 ckpt 로 미라벨 셋을 추론하기 위해 test 에 append.
    # y_true=NaN 이라 metric 은 NaN(무해, 저장 후 계산)이고 predictions.parquet 에는 y_pred 가 남는다.
    if os.environ.get("QADMET_PXR_SUBMIT") and "PXR_pEC50" in test_dfs:
        extra = []
        for fn in ("TEST_BLINDED.csv", "structure_TEST_BLINDED.csv"):
            p = PXR_DIR / "_holdout" / fn
            if p.exists():
                sm = pd.read_csv(p)["Drug"]
                extra.append(pd.DataFrame({"Drug": sm.values, "Y": np.nan}))
        # HTChem residual-anchor 예측용 (Step B) — QADMET_PXR_HTCHEM=1 일 때만 append.
        if os.environ.get("QADMET_PXR_HTCHEM"):
            p = PXR_DIR / "_clean" / "htchem.csv"
            if p.exists():
                sm = pd.read_csv(p)["Drug"]
                extra.append(pd.DataFrame({"Drug": sm.values, "Y": np.nan}))
        if extra:
            test_dfs["PXR_pEC50"] = pd.concat(
                [test_dfs["PXR_pEC50"]] + extra, ignore_index=True
            ).drop_duplicates("Drug", keep="first").reset_index(drop=True)
            logger.info("PXR_SUBMIT: PXR_pEC50 test → %d rows (PHASE1+BLINDED+structure)",
                        len(test_dfs["PXR_pEC50"]))

    train_merged = _merge(train_dfs)
    val_merged = _merge(val_dfs)

    logger.info(
        "TDC+PXR fold: task_set=%s seed=%s train=%d val=%d (TDC %d + PXR %d tasks) "
        "samples_pxr=%s",
        task_set_name, seed, len(train_merged), len(val_merged),
        len(tdc_tasks), len(pxr_tasks),
        {t: samples_per_task[t] for t in pxr_tasks},
    )
    return TdcAdmetFold(
        train_df=train_merged,
        val_df=val_merged,
        test_dfs=test_dfs,
        samples_per_task=samples_per_task,
    )


def collect_tdc_pxr_smiles(task_set_name: str = "tdc_pxr_25", *, seeds=(0,)) -> list[str]:
    """precompute 용 — 25-task 전체 SMILES 합집합 (TDC + PXR train/val/test)."""
    from qadmet.datasets.tdc_admet import collect_smiles
    from qadmet.tasks.pxr import PXR_3

    ts = TASK_SETS.get(task_set_name)
    tdc_tasks = [t for t in ts.tasks if t not in set(PXR_3)]

    smis: set[str] = set()
    # TDC: 기존 헬퍼 재사용 (22-task TaskSet 이 아니라 직접 모으기 위해 placeholder)
    from tdc.benchmark_group import admet_group  # type: ignore

    group = admet_group(path="data/")
    for task_name in tdc_tasks:
        bench = group.get(task_name)
        smis.update(bench["test"]["Drug"].tolist())
        for seed in seeds:
            tr, va = group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed,
            )
            smis.update(tr["Drug"].tolist())
            smis.update(va["Drug"].tolist())
    # PXR: train_val + test + holdout
    for task in PXR_3:
        for fn in ("train_val.csv", "test.csv"):
            smis.update(pd.read_csv(PXR_DIR / task / fn)["Drug"].tolist())
    for fn in ("TEST_BLINDED.csv", "structure_TEST_BLINDED.csv"):
        p = PXR_DIR / "_holdout" / fn
        if p.exists():
            smis.update(pd.read_csv(p)["Drug"].tolist())
    return sorted(smis)
