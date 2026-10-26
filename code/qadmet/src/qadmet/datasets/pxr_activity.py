"""OpenADMET PXR challenge dataset adapter.

Core transformations:
  1. Load 3 HF dataset configs (default train/test, counter_assay, single_concentration).
  2. **Batch-centering**: Extract pEC50 mean per OCNT Batch suffix (001~005), then
     Shift pEC50 of all rows to batch-004 reference frame. The blind test set
     is effectively batch-004 distribution, based on SMILES-classifier analysis (≈98.5%).
  3. **Selectivity = PXR pEC50 − null pEC50** (OCNT_ID join).
     Since batches are 99.8% synchronized, same-plate differences are automatically batch-free.
  4. Scaffold 5-fold + Active-cluster 5-fold split.

The training-side interface is identical to ``TdcAdmetFold`` — runners/train.py works as-is.

Split semantics:
  - scaffold_kfold: Murcko scaffold 5-fold (general OOD evaluation).
  - active_cluster_kfold: pEC50≥6 65 compounds + ECFP4 Tanimoto>0.4 analog cluster
    5-fold split. Mimics challenge test (analog of potent hits) scenario.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qadmet.datasets.registry import DATASETS
from qadmet.datasets.tdc_admet import TdcAdmetFold
from qadmet.tasks import TASK_SETS

logger = logging.getLogger(__name__)

# HF dataset constants
HF_REPO = "openadmet/pxr-challenge-train-test"
REF_BATCH = "004"  # SMILES-fingerprint classifier → test ≈ 98.5% batch-004

# 4 target column names → HF original column mapping (rename after load)
EMAX_COL = "Emax_estimate (log2FC vs. baseline)"


def _batch_suffix(s: object) -> str | None:
    """Extract trailing -NNN from OCNT Batch string."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = re.search(r"-(\d{3})$", str(s))
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, pd.DataFrame]:
    """Load all 3 configs from HF (LRU caching)."""
    from datasets import load_dataset

    train   = load_dataset(HF_REPO, "default")["train"].to_pandas()
    test    = load_dataset(HF_REPO, "default")["test"].to_pandas()
    counter = load_dataset(HF_REPO, "counter_assay")["train"].to_pandas()
    return {"train": train, "test": test, "counter": counter}


def _build_pxr_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build PXR 4-task train/test DataFrame.

    Returns:
        (train_df, test_df) — both have SMILES (key=Drug) + 4 target columns.
        train_df has 4,139 rows, test_df has 513 rows (all NaN target).
    """
    raw = _load_raw()
    train_src = raw["train"].copy()
    test_src  = raw["test"].copy()
    counter   = raw["counter"]

    # --- Extract batch_group ---
    train_src["batch_group"] = train_src["OCNT Batch"].map(_batch_suffix)

    # --- target = raw pEC50 (centering removed 2026-05-26) ---
    # Previously: shifted to batch-004 reference frame. Diagnosis showed R² −0.29, RAE +0.23 degradation.
    # (see diagnose_r2_bak.ipynb) The assumption that Test is analog of batch-004 actives, and
    # the side effect of ignoring that actual activity-rate differences between batches are real signal.
    batch_means = train_src.groupby("batch_group")["pEC50"].mean()
    ref_mean = float(batch_means[REF_BATCH])  # kept only for logging purposes
    train_src["PXR_pEC50"] = train_src["pEC50"].astype(float)

    train_src["PXR_Emax"] = train_src[EMAX_COL]

    # --- selectivity via OCNT_ID merge ---
    counter_lookup = (
        counter[["OCNT_ID", "pEC50"]]
        .rename(columns={"pEC50": "PXR_null_pEC50"})
    )
    train_src = train_src.merge(counter_lookup, on="OCNT_ID", how="left")
    train_src["PXR_selectivity"] = (
        train_src["pEC50"] - train_src["PXR_null_pEC50"]
    )

    cols = ["SMILES", "Molecule Name", "batch_group",
            "PXR_pEC50", "PXR_Emax", "PXR_selectivity", "PXR_null_pEC50",
            "pEC50_std.error (-log10(molarity))"]
    train_df = train_src[cols].copy()
    train_df = train_df.rename(columns={
        "SMILES": "Drug",
        "pEC50_std.error (-log10(molarity))": "pEC50_std_error",
    })

    # test: target column is all NaN (blinded). No batch_group.
    test_df = pd.DataFrame({
        "Drug": test_src["SMILES"].values,
        "Molecule Name": test_src["Molecule Name"].values,
        "batch_group": "004",  # assumption: test ≈ batch-004
        "PXR_pEC50": np.nan,
        "PXR_Emax": np.nan,
        "PXR_selectivity": np.nan,
        "PXR_null_pEC50": np.nan,
        "pEC50_std_error": np.nan,
    })

    logger.info(
        "PXR data prep: train=%d (sel known=%d), test=%d. batch_means=%s, ref_mean=%.3f",
        len(train_df), train_df["PXR_selectivity"].notna().sum(),
        len(test_df),
        batch_means.round(3).to_dict(), ref_mean,
    )
    return train_df, test_df


# ============================================================================
# Split implementations
# ============================================================================

def _murcko_scaffold(smi: str) -> str:
    """Murcko scaffold SMILES (chirality not considered)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def _scaffold_kfold(df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Murcko-scaffold k-fold. Molecules with the same scaffold are grouped into the same fold.

    Returns:
        (N,) ndarray of fold idx in [0, n_splits).
    """
    rng = np.random.RandomState(seed)
    scaffolds = df["Drug"].map(_murcko_scaffold)

    # Round-robin starting from the largest scaffold-size group (TDC-like)
    grouped = (
        pd.DataFrame({"idx": df.index, "scaf": scaffolds})
        .groupby("scaf", sort=False)["idx"].apply(list)
    )
    order = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    fold_sizes = [0] * n_splits
    fold_of = np.zeros(len(df), dtype=int)
    # Fill in order of largest scaffold, prioritizing assignment to the smallest fold (greedy balancing)
    for _scaf, idxs in order:
        target_fold = int(np.argmin(fold_sizes))
        for i in idxs:
            fold_of[df.index.get_loc(i)] = target_fold
        fold_sizes[target_fold] += len(idxs)

    # Slight shuffle effect per seed via rng fold permutation (deterministic)
    perm = rng.permutation(n_splits)
    return perm[fold_of]


def _ecfp4(smi: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _active_cluster_kfold(
    df: pd.DataFrame, n_splits: int, seed: int,
    *, active_threshold: float = 6.0, sim_threshold: float = 0.4,
) -> np.ndarray:
    """Active-cluster k-fold.

    Procedure:
      1. Extract 'active' compounds with pEC50 ≥ active_threshold.
      2. Include all train compounds with ECFP4 Tanimoto ≥ sim_threshold in the active cluster.
      3. Evenly split the active cluster into 5-fold.
      4. Remaining train compounds become common train for all folds (marked as -1).

    Returns:
        (N,) fold idx in [0, n_splits) for active cluster members,
        -1 for "always-train" non-cluster members.
    """
    from rdkit import DataStructs

    rng = np.random.RandomState(seed)
    # Identify active compounds (based on raw pEC50 — note this is not the centered training target:
    # since the active definition is absolute activity, raw pEC50 is appropriate).
    # However, since we only have PXR_pEC50 (centered), add batch_means back
    # to restore the original pEC50.
    train_src = _load_raw()["train"]
    train_src = train_src.set_index("Molecule Name")
    raw_pec50 = train_src.loc[df["Molecule Name"], "pEC50"].values

    is_active = raw_pec50 >= active_threshold
    active_idx = np.where(is_active)[0]

    # Compute FP
    fps = [_ecfp4(s) for s in df["Drug"]]
    valid = np.array([f is not None for f in fps])

    # Include molecules with Tanimoto ≥ sim_threshold relative to actives in the cluster
    cluster_mask = np.zeros(len(df), dtype=bool)
    cluster_mask[active_idx] = True
    for ai in active_idx:
        if fps[ai] is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(
            fps[ai], [f if f is not None else fps[ai] for f in fps]
        ))
        sims[~valid] = 0.0
        cluster_mask |= sims >= sim_threshold

    cluster_members = np.where(cluster_mask)[0]
    logger.info(
        "active_cluster: %d actives → %d cluster members (incl analogs) / %d total",
        len(active_idx), len(cluster_members), len(df),
    )

    # Assign fold
    fold_of = np.full(len(df), -1, dtype=int)
    perm = rng.permutation(cluster_members)
    fold_assignments = np.arange(len(perm)) % n_splits
    fold_assignments = rng.permutation(fold_assignments)  # shuffle within
    fold_of[perm] = fold_assignments

    return fold_of


# ============================================================================
# Fold loader
# ============================================================================

def load_pxr_fold(task_set_name: str, seed: int) -> TdcAdmetFold:
    """Load PXR fold — behavior per task_set_name:

      - pxr_4task         : scaffold 5-fold, 4 task
      - pxr_4task_active  : active-cluster 5-fold, 4 task
      - pxr_pec50_only    : scaffold 5-fold, primary task only
      - pxr_pec50_only_active : active-cluster 5-fold, primary task only

    If seed ∈ [1..5], fold idx = (seed-1).
    seed=0 is equivalent to fold 0 (convention compatibility).
    """
    ts = TASK_SETS.get(task_set_name)
    n_splits = 5
    fold_idx = max(0, seed - 1) % n_splits if seed >= 1 else 0

    train_df, test_df = _build_pxr_dataframes()

    # Select split
    if "active" in task_set_name:
        fold_of = _active_cluster_kfold(train_df, n_splits=n_splits, seed=0)
        # active-cluster: -1 is always train, rows with fold==fold_idx are val
        val_mask = (fold_of == fold_idx)
        train_mask = ~val_mask  # -1 + all other folds included in train
    else:
        fold_of = _scaffold_kfold(train_df, n_splits=n_splits, seed=0)
        val_mask = (fold_of == fold_idx)
        train_mask = ~val_mask

    target_cols = list(ts.tasks)
    keep_cols = ["Drug"] + target_cols

    train_split = train_df.loc[train_mask, keep_cols].reset_index(drop=True)
    val_split   = train_df.loc[val_mask,   keep_cols].reset_index(drop=True)

    # test_dfs: per-task. Since it's a blind set, target is all NaN.
    test_dfs: dict[str, pd.DataFrame] = {}
    for t in target_cols:
        td = pd.DataFrame({
            "Drug": test_df["Drug"].values,
            "Y": np.nan,
        })
        test_dfs[t] = td

    samples_per_task = {t: int(train_split[t].notna().sum()) for t in target_cols}

    logger.info(
        "PXR fold: %s seed=%d fold=%d train=%d val=%d test=%d samples_per_task=%s",
        task_set_name, seed, fold_idx, len(train_split), len(val_split),
        len(test_df), samples_per_task,
    )

    return TdcAdmetFold(
        train_df=train_split,
        val_df=val_split,
        test_dfs=test_dfs,
        samples_per_task=samples_per_task,
    )


def collect_pxr_smiles(task_set_name: str = "pxr_4task") -> list[str]:
    """Union of PXR SMILES for precompute to use (train + test)."""
    train_df, test_df = _build_pxr_dataframes()
    return sorted(set(train_df["Drug"].tolist() + test_df["Drug"].tolist()))


# ---------------------------------------------------------------------------- registry registration
# Why class-based: same pattern as tdc_admet. The actual dispatch is in load_fold (tdc_admet.py)'s
# pxr_ prefix branching.

@DATASETS.register("pxr_4task")
class Pxr4Task:
    """4-task PXR MTL (scaffold split)."""
    task_set = "pxr_4task"
    def load(self, seed: int) -> TdcAdmetFold:
        return load_pxr_fold(self.task_set, seed)


@DATASETS.register("pxr_4task_active")
class Pxr4TaskActive:
    """4-task PXR MTL (active-cluster split)."""
    task_set = "pxr_4task_active"
    def load(self, seed: int) -> TdcAdmetFold:
        return load_pxr_fold(self.task_set, seed)


@DATASETS.register("pxr_pec50_only")
class PxrPec50Only:
    """STL pEC50 only (scaffold split)."""
    task_set = "pxr_pec50_only"
    def load(self, seed: int) -> TdcAdmetFold:
        return load_pxr_fold(self.task_set, seed)


@DATASETS.register("pxr_pec50_only_active")
class PxrPec50OnlyActive:
    """STL pEC50 only (active-cluster split)."""
    task_set = "pxr_pec50_only_active"
    def load(self, seed: int) -> TdcAdmetFold:
        return load_pxr_fold(self.task_set, seed)
