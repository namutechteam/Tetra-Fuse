"""TDC ADMET group 로더.

옛 ``merge_tdc_datasets`` (scripts-old/pipeline_mtl.py:371-400) 와 동일한 outer-merge
로직을 제공하되, chemprop 객체화는 별도 단계로 분리한다 (학습 코드의 책임).

본 모듈의 출력:
    TdcAdmetFold(train_df, val_df, test_dfs, samples_per_task)
    - train_df / val_df: SMILES outer-merged DataFrame, target 컬럼은 task 이름과 동일.
    - test_dfs: {task_name: DataFrame} (각 task 별로 그대로 보존).
    - samples_per_task: {task_name: int} 학습 셋 row 수 (inverse-frequency 가중치 산출용).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from typing import Sequence

import pandas as pd

from qadmet.datasets.registry import DATASETS
from qadmet.tasks import TASK_SETS

logger = logging.getLogger(__name__)


@dataclass
class TdcAdmetFold:
    """TDC scaffold split 한 fold 의 데이터 묶음."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_dfs: dict[str, pd.DataFrame]
    samples_per_task: dict[str, int]


def _merge(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """SMILES (Drug) outer merge — 한 task 만 있으면 그대로 반환."""
    if len(dfs) == 1:
        return dfs[0]
    return reduce(lambda l, r: pd.merge(l, r, how="outer", on="Drug"), dfs)


def load_fold(task_set_name: str, seed: int, *, group=None) -> TdcAdmetFold:
    """TDC admet_group 한 seed fold 를 로드한다.

    Args:
        task_set_name: ``TASK_SETS`` 에 등록된 이름 (예: "tdc_admet_22").
        seed: TDC ``get_train_valid_split(seed=...)`` 에 그대로 전달.
        group: 외부 주입용 (테스트 등). None 이면 ``tdc.benchmark_group.admet_group``
               를 새로 생성.
    """
    # TDC22 + PXR3 = 25-task 통합 분기 (plan_details.md §2.4) — pxr_ 보다 먼저 매칭.
    if task_set_name.startswith("tdc_pxr_25"):
        from qadmet.datasets.tdc_pxr import load_tdc_pxr_fold
        return load_tdc_pxr_fold(task_set_name, seed, group=group)

    if task_set_name.startswith("polaris_fang"):
        from qadmet.datasets.polaris_fang import load_polaris_fang_fold
        return load_polaris_fang_fold(task_set_name, seed, group=group)

    # PXR challenge 분기 — HuggingFace 로부터 직접 로드 (TDC 와 별개)
    if task_set_name.startswith("pxr_"):
        from qadmet.datasets.pxr_activity import load_pxr_fold
        return load_pxr_fold(task_set_name, seed)

    ts = TASK_SETS.get(task_set_name)
    if group is None:
        from tdc.benchmark_group import admet_group  # type: ignore

        group = admet_group(path="data/")

    train_dfs: list[pd.DataFrame] = []
    val_dfs: list[pd.DataFrame] = []
    test_dfs: dict[str, pd.DataFrame] = {}
    samples_per_task: dict[str, int] = {}

    for task_name in ts.tasks:
        bench = group.get(task_name)
        test_dfs[task_name] = bench["test"]
        train_df, val_df = group.get_train_valid_split(
            benchmark=task_name, split_type="default", seed=seed,
        )
        train_dfs.append(train_df[["Drug", "Y"]].rename(columns={"Y": task_name}))
        val_dfs.append(val_df[["Drug", "Y"]].rename(columns={"Y": task_name}))
        samples_per_task[task_name] = len(train_df)

    train_merged = _merge(train_dfs)
    val_merged = _merge(val_dfs)

    logger.info(
        "TDC fold loaded: task_set=%s seed=%s train=%s val=%s tasks=%s",
        task_set_name, seed, len(train_merged), len(val_merged), len(ts.tasks),
    )
    return TdcAdmetFold(
        train_df=train_merged,
        val_df=val_merged,
        test_dfs=test_dfs,
        samples_per_task=samples_per_task,
    )


def collect_smiles(task_set_name: str, *, seeds: Sequence[int] = (0,), group=None) -> list[str]:
    """task_set 의 모든 fold 에서 등장하는 SMILES 합집합.

    Why:
        precompute 스크립트가 학습 전에 TDC SMILES 를 모두 모아 feature 를 만든다.
        매번 fold 를 로드하지 않고 캐시 가능한 함수로 분리.
    """
    ts = TASK_SETS.get(task_set_name)
    if group is None:
        from tdc.benchmark_group import admet_group  # type: ignore

        group = admet_group(path="data/")

    smis: set[str] = set()
    for task_name in ts.tasks:
        bench = group.get(task_name)
        smis.update(bench["test"]["Drug"].tolist())
        for seed in seeds:
            tr, va = group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed,
            )
            smis.update(tr["Drug"].tolist())
            smis.update(va["Drug"].tolist())
    return sorted(smis)


# ---------------------------------------------------------------------------- registry 등록

@DATASETS.register("tdc_admet_22")
class TdcAdmet22:
    """DatasetSpec(name="tdc_admet_22") 매핑용 placeholder.

    실제 로딩은 ``load_fold`` 가 담당한다. 본 클래스는 registry lookup 시
    DatasetSpec 의 의도(어느 task_set 을 사용할지)를 문서화하는 역할만.
    """

    task_set = "tdc_admet_22"

    def load(self, seed: int) -> TdcAdmetFold:
        return load_fold(self.task_set, seed)


@DATASETS.register("tdc_admet_5_sanity")
class TdcAdmet5Sanity:
    """4-modality overfit 진단용 5-task sanity (TaskSet ``tdc_admet_5_sanity``)."""

    task_set = "tdc_admet_5_sanity"

    def load(self, seed: int) -> TdcAdmetFold:
        return load_fold(self.task_set, seed)
