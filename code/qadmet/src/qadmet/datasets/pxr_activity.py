"""OpenADMET PXR challenge dataset adapter.

핵심 변환:
  1. HF dataset 3종 로드 (default train/test, counter_assay, single_concentration).
  2. **Batch-centering**: OCNT Batch suffix (001~005) 별 pEC50 평균 추출 후
     모든 row 의 pEC50 를 batch-004 reference frame 으로 shift. blind test 셋이
     사실상 batch-004 분포라는 SMILES-classifier 분석 (≈98.5%) 기반.
  3. **Selectivity = PXR pEC50 − null pEC50** (OCNT_ID join).
     배치가 99.8% 동기되어 있으므로 same-plate 차분은 자동으로 배치-free.
  4. Scaffold 5-fold + Active-cluster 5-fold 분할.

학습 측면 인터페이스는 ``TdcAdmetFold`` 와 동일 — runners/train.py 가 그대로 작동.

Split 의미:
  - scaffold_kfold: Murcko scaffold 5-fold (일반 OOD 평가).
  - active_cluster_kfold: pEC50≥6 65화합물 + ECFP4 Tanimoto>0.4 analog 클러스터를
    5-fold 분할. 챌린지 test (analog of potent hits) 시나리오 모사.
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

# 4 target column 명 → HF 원본 컬럼 매핑 (load 후 rename)
EMAX_COL = "Emax_estimate (log2FC vs. baseline)"


def _batch_suffix(s: object) -> str | None:
    """OCNT Batch 문자열에서 trailing -NNN 추출."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = re.search(r"-(\d{3})$", str(s))
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, pd.DataFrame]:
    """HF에서 3 config 모두 로드 (LRU 캐싱)."""
    from datasets import load_dataset

    train   = load_dataset(HF_REPO, "default")["train"].to_pandas()
    test    = load_dataset(HF_REPO, "default")["test"].to_pandas()
    counter = load_dataset(HF_REPO, "counter_assay")["train"].to_pandas()
    return {"train": train, "test": test, "counter": counter}


def _build_pxr_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """PXR 4-task 학습용/테스트용 DataFrame 빌드.

    Returns:
        (train_df, test_df) — 둘 다 SMILES (key=Drug) + 4 target 컬럼.
        train_df 는 4,139 rows, test_df 는 513 rows (모두 NaN target).
    """
    raw = _load_raw()
    train_src = raw["train"].copy()
    test_src  = raw["test"].copy()
    counter   = raw["counter"]

    # --- batch_group 추출 ---
    train_src["batch_group"] = train_src["OCNT Batch"].map(_batch_suffix)

    # --- target = raw pEC50 (centering 제거 2026-05-26) ---
    # 이전: batch-004 reference frame 으로 shift. 진단 결과 R² −0.29, RAE +0.23 손해 발생.
    # (diagnose_r2_bak.ipynb 참조) Test 가 analog of batch-004 actives 라는 가정과
    # 실제 batch 간 활성률 차이가 실제 신호임을 무시하는 부작용.
    batch_means = train_src.groupby("batch_group")["pEC50"].mean()
    ref_mean = float(batch_means[REF_BATCH])  # 로그 용도로만 유지
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

    # test: target 컬럼은 모두 NaN (blinded). batch_group 없음.
    test_df = pd.DataFrame({
        "Drug": test_src["SMILES"].values,
        "Molecule Name": test_src["Molecule Name"].values,
        "batch_group": "004",  # 가정: test ≈ batch-004
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
    """Murcko scaffold SMILES (chirality 미고려)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def _scaffold_kfold(df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Murcko-scaffold k-fold. 같은 scaffold 의 분자는 동일 fold 에 모인다.

    Returns:
        (N,) ndarray of fold idx in [0, n_splits).
    """
    rng = np.random.RandomState(seed)
    scaffolds = df["Drug"].map(_murcko_scaffold)

    # scaffold size 가 큰 그룹부터 round-robin (TDC-like)
    grouped = (
        pd.DataFrame({"idx": df.index, "scaf": scaffolds})
        .groupby("scaf", sort=False)["idx"].apply(list)
    )
    order = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    fold_sizes = [0] * n_splits
    fold_of = np.zeros(len(df), dtype=int)
    # 큰 scaffold 순으로 채우되 가장 적은 fold 에 우선 배정 (그리디 균등)
    for _scaf, idxs in order:
        target_fold = int(np.argmin(fold_sizes))
        for i in idxs:
            fold_of[df.index.get_loc(i)] = target_fold
        fold_sizes[target_fold] += len(idxs)

    # seed 별 약간의 shuffle 효과는 rng 로 fold 순열 (deterministic)
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

    절차:
      1. pEC50 ≥ active_threshold 인 'active' 화합물 추출.
      2. ECFP4 Tanimoto ≥ sim_threshold 인 모든 train 화합물을 active cluster 에 포함.
      3. active cluster 를 5-fold 로 균등 분할.
      4. 나머지 train 화합물은 모든 fold 의 공통 train (= -1 표시) 으로.

    Returns:
        (N,) fold idx in [0, n_splits) for active cluster members,
        -1 for "always-train" non-cluster members.
    """
    from rdkit import DataStructs

    rng = np.random.RandomState(seed)
    # 활성 화합물 식별 (raw pEC50 기준 — 학습 target 인 centered 가 아님에 주의:
    # active 정의는 절대 활성도이므로 raw pEC50 가 적절).
    # 하지만 우리는 PXR_pEC50 (centered) 만 갖고 있으므로, batch_means 를 다시 더해
    # 원본 pEC50 를 복원한다.
    train_src = _load_raw()["train"]
    train_src = train_src.set_index("Molecule Name")
    raw_pec50 = train_src.loc[df["Molecule Name"], "pEC50"].values

    is_active = raw_pec50 >= active_threshold
    active_idx = np.where(is_active)[0]

    # FP 계산
    fps = [_ecfp4(s) for s in df["Drug"]]
    valid = np.array([f is not None for f in fps])

    # active 들에 대해 Tanimoto ≥ sim_threshold 인 분자들을 cluster 에 포함
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

    # fold 할당
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
    """PXR fold 로드 — task_set_name 별 동작:

      - pxr_4task         : scaffold 5-fold, 4 task
      - pxr_4task_active  : active-cluster 5-fold, 4 task
      - pxr_pec50_only    : scaffold 5-fold, primary task 만
      - pxr_pec50_only_active : active-cluster 5-fold, primary task 만

    seed ∈ [1..5] 면 fold idx = (seed-1).
    seed=0 은 fold 0 와 같음 (관습 호환).
    """
    ts = TASK_SETS.get(task_set_name)
    n_splits = 5
    fold_idx = max(0, seed - 1) % n_splits if seed >= 1 else 0

    train_df, test_df = _build_pxr_dataframes()

    # split 선택
    if "active" in task_set_name:
        fold_of = _active_cluster_kfold(train_df, n_splits=n_splits, seed=0)
        # active-cluster: -1 은 항상 train, fold==fold_idx 인 row 가 val
        val_mask = (fold_of == fold_idx)
        train_mask = ~val_mask  # -1 + 다른 fold 모두 train 에 포함
    else:
        fold_of = _scaffold_kfold(train_df, n_splits=n_splits, seed=0)
        val_mask = (fold_of == fold_idx)
        train_mask = ~val_mask

    target_cols = list(ts.tasks)
    keep_cols = ["Drug"] + target_cols

    train_split = train_df.loc[train_mask, keep_cols].reset_index(drop=True)
    val_split   = train_df.loc[val_mask,   keep_cols].reset_index(drop=True)

    # test_dfs: per-task. 블라인드 셋이므로 target 은 모두 NaN.
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
    """precompute 가 사용할 PXR SMILES 합집합 (train + test)."""
    train_df, test_df = _build_pxr_dataframes()
    return sorted(set(train_df["Drug"].tolist() + test_df["Drug"].tolist()))


# ---------------------------------------------------------------------------- registry 등록
# Why class-based: tdc_admet 와 동일 패턴. 실제 dispatch 는 load_fold (tdc_admet.py) 의
# pxr_ 접두사 분기로 이뤄짐.

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
