"""TDC22 + PXR3 = 25-task unified fold loader (plan_details.md §2.4).

TDC 22-task uses the existing ``tdc.benchmark_group`` path, PXR 3-task uses ``data/pxr/<task>/``
loaded from CSV (output of build_pxr_csv.py), then outer-merged on SMILES(Drug) to build a 25-task wide
frame. For each task, unmeasured compounds have target=NaN → presence-mask ignores them in the loss.

Return type is the same ``TdcAdmetFold`` → works with runners/train.py unchanged.

PXR val split:
    train_val.csv split randomly per seed (val_frac=0.125). Same as TDC's per-seed default split
    uses a different val per seed in the same way. PXR test is fixed (build_pxr_csv's scaffold holdout
    or PHASE1).
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
    """data/pxr/<task>/ → (train, val, test). train/val split randomly per seed.

    Returns:
        (train_df, val_df, test_df) — each with columns=[Drug, Y].
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
    """Load one fold of 25-task (TDC22 + PXR3).

    Args:
        task_set_name: "tdc_pxr_25" (causal6+PXR) or "tdc_pxr_25_pertask" (STL).
            Both have the same task list → data is identical, only grouping differs.
        seed: TDC split seed + PXR val split seed.
        group: injected TDC admet_group (for testing). If None, created anew.
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

    # --- submit mode: append BLINDED+structure (challenge submission target) to PXR_pEC50 test ---
    # If QADMET_PXR_SUBMIT=1, append the unlabeled set to test for inference using the trained ckpt.
    # Since y_true=NaN, metric is NaN (harmless, computed after saving), and y_pred remains in predictions.parquet.
    if os.environ.get("QADMET_PXR_SUBMIT") and "PXR_pEC50" in test_dfs:
        extra = []
        for fn in ("TEST_BLINDED.csv", "structure_TEST_BLINDED.csv"):
            p = PXR_DIR / "_holdout" / fn
            if p.exists():
                sm = pd.read_csv(p)["Drug"]
                extra.append(pd.DataFrame({"Drug": sm.values, "Y": np.nan}))
        # For HTChem residual-anchor prediction (Step B) — append only when QADMET_PXR_HTCHEM=1.
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
    """For precompute — union of all SMILES across 25 tasks (TDC + PXR train/val/test)."""
    from qadmet.datasets.tdc_admet import collect_smiles
    from qadmet.tasks.pxr import PXR_3

    ts = TASK_SETS.get(task_set_name)
    tdc_tasks = [t for t in ts.tasks if t not in set(PXR_3)]

    smis: set[str] = set()
    # TDC: reuse existing helper (placeholder for collecting directly instead of 22-task TaskSet)
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
