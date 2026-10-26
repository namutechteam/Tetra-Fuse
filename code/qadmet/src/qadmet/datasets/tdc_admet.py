"""TDC ADMET group loader.

Same outer-merge as the old ``merge_tdc_datasets`` (scripts-old/pipeline_mtl.py:371-400)
logic, but separates chemprop objectification into a separate step (the responsibility of the training code).

Output of this module:
    TdcAdmetFold(train_df, val_df, test_dfs, samples_per_task)
    - train_df / val_df: SMILES outer-merged DataFrame, target column is the same as the task name.
    - test_dfs: {task_name: DataFrame} (preserved as-is per task).
    - samples_per_task: {task_name: int} number of training-set rows (for computing inverse-frequency weights).
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
    """Data bundle for one fold of a TDC scaffold split."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_dfs: dict[str, pd.DataFrame]
    samples_per_task: dict[str, int]


def _merge(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """SMILES (Drug) outer merge — returns as-is if only one task is present."""
    if len(dfs) == 1:
        return dfs[0]
    return reduce(lambda l, r: pd.merge(l, r, how="outer", on="Drug"), dfs)


def load_fold(task_set_name: str, seed: int, *, group=None) -> TdcAdmetFold:
    """Loads one seed fold of the TDC admet_group.

    Args:
        task_set_name: name registered in ``TASK_SETS`` (e.g. "tdc_admet_22").
        seed: passed through as-is to TDC ``get_train_valid_split(seed=...)``.
        group: for external injection (e.g. testing). If None, ``tdc.benchmark_group.admet_group``
               is newly created.
    """
    # TDC22 + PXR3 = 25-task unified branch (plan_details.md §2.4) — matched before pxr_.
    if task_set_name.startswith("tdc_pxr_25"):
        from qadmet.datasets.tdc_pxr import load_tdc_pxr_fold
        return load_tdc_pxr_fold(task_set_name, seed, group=group)

    if task_set_name.startswith("polaris_fang"):
        from qadmet.datasets.polaris_fang import load_polaris_fang_fold
        return load_polaris_fang_fold(task_set_name, seed, group=group)

    # PXR challenge branch — loaded directly from HuggingFace (separate from TDC)
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
    """Union of SMILES appearing across all folds of task_set.

    Why:
        The precompute script gathers all TDC SMILES before training to build features.
        Separated into a cacheable function instead of loading the fold each time.
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


# ---------------------------------------------------------------------------- registry registration

@DATASETS.register("tdc_admet_22")
class TdcAdmet22:
    """Placeholder for DatasetSpec(name="tdc_admet_22") mapping.

    Actual loading is handled by ``load_fold``. This class, during registry lookup,
    only documents the intent of DatasetSpec (which task_set to use).
    """

    task_set = "tdc_admet_22"

    def load(self, seed: int) -> TdcAdmetFold:
        return load_fold(self.task_set, seed)


@DATASETS.register("tdc_admet_5_sanity")
class TdcAdmet5Sanity:
    """5-task sanity for diagnosing 4-modality overfit (TaskSet ``tdc_admet_5_sanity``)."""

    task_set = "tdc_admet_5_sanity"

    def load(self, seed: int) -> TdcAdmetFold:
        return load_fold(self.task_set, seed)
