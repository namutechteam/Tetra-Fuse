"""학습용 DataLoader 빌더 + collate 어댑터.

본 모듈은 ``TdcAdmetFold`` 와 ``FeatureSpec`` 을 받아
``MultitaskLitModule`` 의 batch dict 계약을 만족하는 PyTorch DataLoader 를
만든다.

핵심 흐름:
    1. ``build_modality_block`` — features.descriptors + features.embeddings 순서로
       precomputed parquet 들을 SMILES 별로 stack 한 ``X_d_combined (N, ΣD)`` 와
       모달리티별 slice 인덱스 맵을 만든다.
    2. ``X_d_combined`` 을 chemprop ``MoleculeDatapoint`` 의 ``X_d`` 자리에 통째로
       전달 → chemprop dataloader 가 batch 마다 ``TrainingBatch.X_d (B, ΣD)`` 로
       반환.
    3. ``make_collate_fn`` 이 chemprop collate_batch 위에 한 단계 더 얹어
       descriptor slice 는 ``batch["X_d"]`` 로, embedding slice 는
       ``batch["modalities"]`` dict 로 분리 → ``MultitaskLitModule`` 이 받는 dict
       를 yield.

Why 이 구조 (Step B 디자인, 2026-05-01):
    descriptor (rdkit2d / xtb / gfn_missing_vector) 는 옛 chemprop X_d concat
    경로를 그대로 살린다 (DMPNN ``MPNN.fingerprint`` 가 fp 뒤에 X_d 를 concat → 출력
    dim = d_h + d_xd). embedding (molformer / unimol2 frozen CLS, 또는 multi-backbone
    인 LoRA path 의 token modality) 만 fusion 모듈로 보낸다. 이로써:
        - descriptors 만 쓰는 recipe (mtl_dmpnn_rdkit / rdkit_xtb / +missing) 는
          fusion 미사용 → 단일 backbone 학습이 옛 phase1.3a/b 와 동일 경로를 탄다.
        - embeddings 가 추가된 recipe (mtl_with_molformer / mtl_dmpnn_molformer_lora)
          는 fusion 입력 = ``[H_dmpnn(d_h+d_xd), *embeddings]`` 형태.
    chemprop dataloader 의 shuffle/batching 매커니즘은 그대로 활용한다.

Lazy import:
    ``chemprop`` / ``torch`` 는 함수 안에서만 import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from qadmet.core.logging import get_logger
from qadmet.core.specs import FeatureSpec
from qadmet.datasets.tdc_admet import TdcAdmetFold
from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.features.stores import ParquetStore

logger = get_logger("qadmet.datasets.loaders")

LOG_TRANSFORM_TASKS: frozenset[str] = frozenset(
    {"VDss_Lombardo", "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"}
)
"""regression target 에 ``log1p`` 를 씌우는 task 집합 (옛 ``pipeline_mtl.LOG_TRANSFORM_TASKS``)."""

_XTB_KEY = "xtb"
_XTB_MASK_KEY = "gfn_missing_vector"

#: 사전계산 parquet/lmdb 가 없고 SMILES 로부터 collate 단계에서 즉석 생성되는 modality.
#: graphmvp_data: collate 가 SMILES → ``torch_geometric.data.Batch`` 변환 → graphmvp
#: backbone 이 consumes. on-the-fly 변환 비용이 작아 LMDB precompute 불필요.
SYNTHETIC_MODALITY_KEYS: frozenset[str] = frozenset({"graphmvp_data"})


def expand_descriptors(descriptors: Sequence[str]) -> list[str]:
    """xtb 가 포함된 경우 gfn_missing_vector 를 바로 뒤에 자동 삽입.

    recipe 에 gfn_missing_vector 를 직접 쓰지 않아도 xtb 와 항상 같이 흐른다.
    이미 포함되어 있으면 no-op.
    """
    result = list(descriptors)
    if _XTB_KEY in result and _XTB_MASK_KEY not in result:
        idx = result.index(_XTB_KEY)
        result.insert(idx + 1, _XTB_MASK_KEY)
    return result


# ---------------------------------------------------------------------------- modality block


@dataclass
class ModalityBlock:
    """한 modality 의 (N, dim) 텐서 + 메타.

    Attributes:
        name: modality 키 (FEATURE_BUILDERS 에 등록된 이름).
        dim: 채널 수.
        array: (N, dim) float32. SMILES 누락 행은 0 (zeros) 으로 채움.
    """

    name: str
    dim: int
    array: np.ndarray


def _builder_columns(name: str) -> tuple[list[str], int]:
    """builder 클래스에서 columns + dim 을 가져온다 (인스턴스화 1회만)."""
    cls = FEATURE_BUILDERS.get(name)
    builder = cls()
    return list(builder.columns), int(builder.output_dim)


def build_modality_block(
    smiles: Sequence[str],
    name: str,
    parquet_path: Path,
) -> ModalityBlock:
    """단일 modality 의 (N, dim) 텐서를 SMILES 순서대로 만든다.

    Args:
        smiles: 행 순서를 결정하는 SMILES 리스트 (정렬 안 됨, fold 의 원본 순서 유지).
        name: FEATURE_BUILDERS 키.
        parquet_path: 해당 modality 의 사전 계산 parquet 위치 (``ParquetStore``).

    Returns:
        ``ModalityBlock(name, dim, array)``. 누락 SMILES 는 zeros 채움.
    """
    cols, dim = _builder_columns(name)
    store = ParquetStore(parquet_path)
    arr = store.lookup(list(smiles), cols)  # (N, dim), 누락 NaN.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return ModalityBlock(name=name, dim=dim, array=arr)


# ---------------------------------------------------------------------------- modality scaler


@dataclass
class ModalityScalerState:
    """train fold 에서 학습된 modality 별 scaler 상태.

    NaN-safe StandardScaler — train_arr 에서 mean/std 만 계산해 transform 시 적용.
    ``molformer_tokens`` 처럼 정수 토큰 modality 는 ``skip_keys`` 로 우회한다.
    """

    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    clip: float
    skip_keys: tuple[str, ...]

    def transform(self, name: str, arr: np.ndarray) -> np.ndarray:
        if name in self.skip_keys:
            return arr.astype(np.float32)
        mean = self.means[name]
        std = self.stds[name]
        out = (arr.astype(np.float64) - mean) / np.where(std > 1e-12, std, 1.0)
        out = np.clip(out, -self.clip, self.clip)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ``molformer_tokens`` 는 token_id (정수) + attention mask (0/1) — scaling 대상 아님.
# ``gfn_missing_vector`` 는 binary 0/1 mask — standardize 시 의미 훼손.
DEFAULT_SCALE_SKIP: tuple[str, ...] = ("molformer_tokens", "gfn_missing_vector")


def fit_modality_scalers(
    blocks: Mapping[str, ModalityBlock],
    *,
    clip: float = 10.0,
    skip_keys: tuple[str, ...] = DEFAULT_SCALE_SKIP,
) -> ModalityScalerState:
    """train modality blocks 에서 mean/std 산출. clip ±10 (옛 코드 관습).

    xTB 블록: ``gfn_missing_vector`` 가 함께 있으면 모든 채널이 관측된 행만으로
    fit (QW-MTL observed-only strategy). 0-imputed 결측치가 분포를 왜곡하는 것을 방지.
    관측 행 < 2 이면 전체 행으로 fallback (옛 ``fit_xd_scalers`` 와 동일).
    """
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for name, block in blocks.items():
        if name in skip_keys:
            continue
        a = block.array.astype(np.float64)
        if name == _XTB_KEY and _XTB_MASK_KEY in blocks:
            mask_arr = blocks[_XTB_MASK_KEY].array  # (N, d_xtb) binary
            is_obs = (mask_arr == 0).all(axis=1)
            if is_obs.sum() >= 2:
                a = a[is_obs]
            else:
                logger.warning(
                    "xTB observed samples < 2 in train fold; falling back to all-row fit"
                )
        means[name] = a.mean(axis=0)
        stds[name] = a.std(axis=0)
    return ModalityScalerState(means=means, stds=stds, clip=clip, skip_keys=skip_keys)


# ---------------------------------------------------------------------------- X_d stack


def stack_modality_blocks(
    blocks: list[ModalityBlock],
) -> tuple[np.ndarray | None, dict[str, slice], dict[str, int]]:
    """modality block 리스트 → (X_d_combined, slices, dims).

    Returns:
        - X_d_combined: ``(N, ΣD)`` float32. blocks 가 비어있으면 ``None``.
        - slices: ``{name: slice(start, end)}``.
        - dims: ``{name: D}``.
    """
    if not blocks:
        return None, {}, {}
    arrays: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    dims: dict[str, int] = {}
    cursor = 0
    for b in blocks:
        slices[b.name] = slice(cursor, cursor + b.dim)
        dims[b.name] = b.dim
        cursor += b.dim
        arrays.append(b.array)
    return np.concatenate(arrays, axis=1).astype(np.float32), slices, dims


# ---------------------------------------------------------------------------- collate adapter


# 프로세스 + path 별 LMDB env 캐시 — 같은 process 내에서 같은 LMDB 를 두 번 open 하면
# ``lmdb.Error: environment is already open in this process`` 가 발생하므로 공유한다.
# DataLoader worker fork 후엔 ``os.getpid()`` 가 달라져 새로 open (정상 동작).
_LMDB_ENV_CACHE: dict[tuple[int, str], Any] = {}


def _get_shared_lmdb_txn(path: str):
    """프로세스 단위로 LMDB env/txn 캐시. 같은 path 첫 호출 시 open, 이후 재사용."""
    import os
    import lmdb  # type: ignore

    key = (os.getpid(), path)
    if key not in _LMDB_ENV_CACHE:
        env = lmdb.open(
            path, subdir=False, lock=False, readonly=True, max_readers=1024,
        )
        _LMDB_ENV_CACHE[key] = env
    return _LMDB_ENV_CACHE[key].begin()


class _PairedChempropLmdbDataset:
    """chemprop dataset + LMDB feat 동시 인덱싱 (MTL UniMol2 LoRA 경로용).

    base chemprop dataset 의 ``__getitem__`` 결과와 LMDB 의 (smiles → feat dict)
    lookup 결과를 튜플로 묶어 반환. (process, path) 별 env 캐시 공유.

    None feat (LMDB miss) 는 그대로 반환 — collate 가 placeholder 채우고 presence
    mask 로 fusion 단계에서 해당 modality 차단.
    """

    def __init__(self, chemprop_dset, smiles_list: Sequence[str], lmdb_path: Path) -> None:
        self.chemprop = chemprop_dset
        self._smis = list(smiles_list)
        self._lmdb_path = str(lmdb_path)
        self._txn = None

    def __len__(self) -> int:
        return len(self.chemprop)

    def __getitem__(self, idx: int):
        import pickle

        cp_item = self.chemprop[idx]
        if self._txn is None:
            self._txn = _get_shared_lmdb_txn(self._lmdb_path)
        smi = self._smis[idx]
        v = self._txn.get(smi.encode("utf-8"))
        feat = pickle.loads(v) if v is not None else None
        return (cp_item, feat)


def make_collate_fn_with_lmdb(
    descriptor_slices: Mapping[str, slice],
    embedding_slices: Mapping[str, slice],
    lmdb_modality: str,
):
    """LMDB feat 를 inject 하는 collate (UniMol2 LoRA MTL 경로).

    base ``make_collate_fn`` 과 동일 책임 + LMDB 에서 받은 feat dict 들을
    vendored ``collate_unimolv2`` 로 합쳐 ``batch["modalities"][lmdb_modality]`` 에
    dict 로 주입. ``lit_module`` 의 unimol2_lora input adapter 가 ``__consumed__``
    슬롯에서 같은 dict 를 읽는다.

    Args:
        descriptor_slices / embedding_slices: ``make_collate_fn`` 과 동일.
        lmdb_modality: LMDB modality 이름 (예: ``"unimol2_lmdb"``). consumes 되어
            fusion 입력에서 빠지므로 차원은 의미 없음.

    Note:
        feat None 인 항목 (LMDB miss / featurize 실패) 은 batch 에서 제거. 전부
        제거되면 ``None`` 반환 → Lightning skip.
    """
    import torch

    from chemprop.data.collate import collate_batch

    def _collate(items):
        # items: list of (chemprop_item, feat_dict_or_None)
        # missing handling: 누락 sample 도 batch 에 유지 (placeholder feat 으로 forward 통과)
        # 후 fusion 단계에서 ``unimol2_present`` mask 로 해당 modality alpha 무효화.
        present_flags = [f is not None for (_cp, f) in items]
        if not any(present_flags):
            return None  # 전부 missing → batch 자체 skip
        placeholder = next(f for (_cp, f) in items if f is not None)
        feats_filled = [f if f is not None else placeholder for (_cp, f) in items]
        cp_items = [cp for (cp, _f) in items]

        tb = collate_batch(cp_items)
        modalities: dict[str, Any] = {}
        x_d_for_backbone = None
        if tb.X_d is not None:
            x_d = tb.X_d
            if descriptor_slices:
                desc_parts = [x_d[:, sl].float() for sl in descriptor_slices.values()]
                x_d_for_backbone = (
                    desc_parts[0] if len(desc_parts) == 1 else torch.cat(desc_parts, dim=1)
                )
            for name, sl in embedding_slices.items():
                modalities[name] = x_d[:, sl].float()

        from qadmet.modules.backbones._vendor import unimol_v2 as uv2

        unimol2_batch = uv2.collate_unimolv2(feats_filled)
        modalities[lmdb_modality] = unimol2_batch  # dict — consumes 경로로만 흐름.

        present_mask = torch.tensor(present_flags, dtype=torch.bool)  # (B,)
        modality_present: dict[str, Any] = {lmdb_modality: present_mask}

        return {
            "bmg": tb.bmg,
            "V_d": tb.V_d,
            "X_d": x_d_for_backbone,
            "modalities": modalities,
            "modality_present": modality_present,
            "targets": tb.Y,
            "sample_weights": None,
        }

    return _collate


class _ChempropSmilesDataset:
    """chemprop dataset + SMILES 동시 인덱싱 (synthetic modality 경로용).

    items: ``(chemprop_item, smiles)``. collate 가 SMILES 로부터 PyG Batch 등
    synthetic modality payload 를 즉석 생성한다.
    """

    def __init__(self, chemprop_dset, smiles_list: Sequence[str]) -> None:
        self.chemprop = chemprop_dset
        self._smis = list(smiles_list)

    def __len__(self) -> int:
        return len(self.chemprop)

    def __getitem__(self, idx: int):
        return (self.chemprop[idx], self._smis[idx])


class _PairedChempropLmdbSmilesDataset(_PairedChempropLmdbDataset):
    """LMDB + SMILES 동시 인덱싱 (LMDB modality + synthetic modality 동시 사용)."""

    def __getitem__(self, idx: int):
        cp_item, feat = super().__getitem__(idx)
        return (cp_item, feat, self._smis[idx])


def make_collate_fn_with_graphmvp(
    descriptor_slices: Mapping[str, slice],
    embedding_slices: Mapping[str, slice],
    lmdb_modality: str | None,
    graphmvp_key: str = "graphmvp_data",
):
    """graphmvp synthetic modality + (옵션) LMDB modality 동시 지원 collate.

    SMILES → PyG Batch 변환을 collate 단계에서 수행 (precompute 불필요). LMDB 경로와
    조합되면 unimol2 feat dict 까지 함께 합쳐 ``modalities`` dict 의 다른 키로 노출.

    Args:
        descriptor_slices / embedding_slices: ``make_collate_fn`` 과 동일.
        lmdb_modality: LMDB modality 이름 (예: "unimol2_lmdb"). None 이면 LMDB 경로 우회.
        graphmvp_key: synthetic modality 키 — backbone consumes 와 일치해야 함.

    Items 형태 (loader 가 dataset 을 어떻게 wrap 하느냐에 따라):
        - LMDB 없음:    ``(chemprop_item, smiles)`` — ``_ChempropSmilesDataset``.
        - LMDB 있음:    ``(chemprop_item, lmdb_feat_or_None, smiles)`` —
          ``_PairedChempropLmdbSmilesDataset``.

    Note:
        graphmvp invalid SMILES 는 1-atom placeholder + presence False 로 처리되어
        fusion 단계에서 alpha 차단. LMDB miss 는 placeholder feat + presence False.
    """
    import torch

    from chemprop.data.collate import collate_batch

    from qadmet.modules.backbones._graphmvp_features import smiles_batch_to_pyg

    def _collate(items):
        if lmdb_modality is None:
            cp_items = [it[0] for it in items]
            smis = [it[1] for it in items]
            lmdb_present_flags: list[bool] | None = None
            feats_filled = None
        else:
            cp_items = [it[0] for it in items]
            feats = [it[1] for it in items]
            smis = [it[2] for it in items]
            lmdb_present_flags = [f is not None for f in feats]
            if not any(lmdb_present_flags):
                # 모든 LMDB miss 인 batch → 학습 의미 없음. skip.
                return None
            placeholder = next(f for f in feats if f is not None)
            feats_filled = [f if f is not None else placeholder for f in feats]

        tb = collate_batch(cp_items)
        modalities: dict[str, Any] = {}
        x_d_for_backbone = None
        if tb.X_d is not None:
            x_d = tb.X_d
            if descriptor_slices:
                desc_parts = [x_d[:, sl].float() for sl in descriptor_slices.values()]
                x_d_for_backbone = (
                    desc_parts[0] if len(desc_parts) == 1 else torch.cat(desc_parts, dim=1)
                )
            for name, sl in embedding_slices.items():
                modalities[name] = x_d[:, sl].float()

        modality_present: dict[str, Any] = {}

        if lmdb_modality is not None and feats_filled is not None:
            from qadmet.modules.backbones._vendor import unimol_v2 as uv2

            unimol2_batch = uv2.collate_unimolv2(feats_filled)
            modalities[lmdb_modality] = unimol2_batch  # dict — consumes 경로로만.
            modality_present[lmdb_modality] = torch.tensor(
                lmdb_present_flags, dtype=torch.bool
            )

        # ---- graphmvp synthetic modality ----
        graphmvp_batch, graphmvp_present = smiles_batch_to_pyg(smis)
        modalities[graphmvp_key] = graphmvp_batch  # PyG Batch — consumes 경로로만.
        modality_present[graphmvp_key] = graphmvp_present

        return {
            "bmg": tb.bmg,
            "V_d": tb.V_d,
            "X_d": x_d_for_backbone,
            "modalities": modalities,
            "modality_present": modality_present,
            "targets": tb.Y,
            "sample_weights": None,
        }

    return _collate


def make_collate_fn(
    descriptor_slices: Mapping[str, slice],
    embedding_slices: Mapping[str, slice],
):
    """chemprop ``collate_batch`` → ``MultitaskLitModule`` batch dict.

    descriptor 와 embedding modality 둘 다 chemprop ``X_d`` 단일 채널에 stack 된
    상태로 들어오므로, 본 collate 가 두 그룹으로 split 한다.

    Args:
        descriptor_slices: ``{name: slice(start, end)}`` X_d 안에서 descriptor 의
            범위. 비어있으면 ``batch["X_d"] = None`` 으로 DMPNN backbone 이
            descriptor 없이 동작.
        embedding_slices: ``{name: slice(start, end)}`` X_d 안에서 embedding 의
            범위. ``batch["modalities"]`` dict 의 key 가 된다 (fusion 입력).

    Returns:
        ``collate_fn`` callable. PyTorch DataLoader 의 ``collate_fn`` 인자에 사용.

    Note:
        descriptor 는 stack 시 embedding 보다 먼저 배치되므로 (loaders 에서 그 순서로
        쌓음), 보통은 contiguous slice 다. 그러나 본 함수는 slice 가 contiguous 인지
        가정하지 않고 명시적으로 ``torch.cat`` 한다 — 옛 비-순차 stacking 에도 안전.
    """
    import torch

    from chemprop.data.collate import collate_batch

    def _collate(items):
        tb = collate_batch(items)
        modalities: dict[str, Any] = {}
        x_d_for_backbone = None
        if tb.X_d is not None:
            x_d = tb.X_d
            if descriptor_slices:
                desc_parts = [x_d[:, sl].float() for sl in descriptor_slices.values()]
                x_d_for_backbone = (
                    desc_parts[0] if len(desc_parts) == 1 else torch.cat(desc_parts, dim=1)
                )
            for name, sl in embedding_slices.items():
                # slice 결과를 float tensor 로 보장 (token modality 도 float 으로 통과 — backbone adapter 가 .long() 등으로 캐스트).
                modalities[name] = x_d[:, sl].float()
        return {
            "bmg": tb.bmg,
            "V_d": tb.V_d,
            "X_d": x_d_for_backbone,
            "modalities": modalities,
            "targets": tb.Y,
            "sample_weights": None,  # batch 단위 weight 는 LitModule.task_weights buffer 가 담당.
        }

    return _collate


# ---------------------------------------------------------------------------- 통합 빌더


@dataclass
class FoldLoaders:
    """한 seed fold 에 대한 dataloader 묶음.

    Attributes:
        train_loader / val_loader: PyTorch DataLoader (collate 가 dict 반환).
            ``val_loader`` 는 학습 중 EarlyStopping 모니터용 (outer-merged val).
        val_loaders: ``{task_name: (DataLoader, smiles_list, y_true)}``.
            task 별 val 예측용 — OOF stacking (Step 4.2 InverseVariance) 의 입력.
            ``test_loaders`` 와 동일 패턴: 한 task 의 finite val 행만 골라 single-task
            loader 로 빌드 + 같은 modality scaler 적용.
        test_loaders: ``{task_name: (DataLoader, smiles_list, y_true)}``.
            test 는 task 별로 별도 dataloader (옛 train_mtl_mpnn 패턴).
        modality_dims: ``{name: D}`` 모든 modality (descriptors + embeddings) 의 차원.
            runner 가 descriptors 합 (= DMPNN d_xd) 과 embeddings 합 (= fusion 입력)
            을 분리 산출하는 데 사용한다.
        modality_keys_order: descriptors 순서 + embeddings 순서.
        descriptor_dim_total: descriptor 들 차원 합 (= DMPNN backbone 의 ``d_xd``).
            descriptors 가 비어있으면 0.
        scaler_state: train fit 된 ModalityScalerState (val/test loader 도 같은 상태로 transform).
        output_scaler: regression normalize 용 sklearn StandardScaler (옛 train_dset.normalize_targets 결과). None 일 수 있음.
        log_tasks: log1p 적용된 regression task 이름 set (예측 후 expm1 역변환에 필요).
    """

    train_loader: Any
    val_loader: Any
    val_loaders: dict[str, tuple[Any, list[str], np.ndarray]]
    test_loaders: dict[str, tuple[Any, list[str], np.ndarray]]
    modality_dims: dict[str, int]
    modality_keys_order: list[str]
    descriptor_dim_total: int
    scaler_state: ModalityScalerState | None
    output_scaler: Any | None
    log_tasks: frozenset[str]


def _build_chemprop_dataset(
    smiles: Sequence[str],
    targets: np.ndarray,
    x_d: np.ndarray | None,
    *,
    n_workers: int,
):
    """chemprop ``make_datapoints`` → ``make_dataset`` 호출 wrap."""
    from chemprop.cli.utils.parsing import make_datapoints, make_dataset

    dps, _ = make_datapoints(
        smiss=[list(smiles)], rxnss=None, Y=targets,
        weights=None, lt_mask=None, gt_mask=None, X_d=x_d,
        V_fss=None, E_fss=None, V_dss=None,
        molecule_featurizers=None, keep_h=False, add_h=False,
        ignore_stereo=False, reorder_atoms=False,
        use_cuikmolmaker_featurization=False, n_workers=n_workers,
    )
    dset = make_dataset(dps[0], n_workers=n_workers)
    dset.cache = True
    return dset


def _build_torch_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int | None,
    pin_memory: bool,
    collate_fn,
):
    """순수 PyTorch DataLoader — chemprop build_dataloader 우회 (collate_fn override 위해)."""
    import torch
    from torch.utils.data import DataLoader

    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
    )


def prepare_fold_loaders(
    fold: TdcAdmetFold,
    feature_spec: FeatureSpec,
    task_names: Sequence[str],
    task_types: Sequence[str],
    *,
    parquet_root: Path,
    parquet_paths: Mapping[str, Path] | None = None,
    lmdb_paths: Mapping[str, Path] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 0,
    apply_log_transform: bool = True,
) -> FoldLoaders:
    """한 fold 의 train/val/test DataLoader 일괄 생성.

    Args:
        fold: ``load_fold`` 결과.
        feature_spec: ``FeatureSpec`` (descriptors + embeddings 순서).
        task_names: target 컬럼 순서.
        task_types: 길이 ``len(task_names)``, 각 task 의 ``"regression"`` / ``"classification"``.
        parquet_root: precompute 산출물 파일들의 부모 디렉토리.
            각 modality 는 ``{parquet_root}/{name}.parquet`` 가정.
        parquet_paths: modality 별 명시 path map. 누락 키는 ``parquet_root/{name}.parquet``
            컨벤션으로 fallback. 옛 데이터(``data/rdkit_2d.parquet`` 등 비-컨벤션 이름)를
            바로 쓸 때 사용.
        batch_size / num_workers / seed: chemprop dataloader 옵션.
        apply_log_transform: regression task 중 ``LOG_TRANSFORM_TASKS`` 에 포함된 컬럼에
            ``log1p`` 적용 (옛 코드 ``pipeline_mtl.py:468-473``).

    Returns:
        ``FoldLoaders`` — train/val/test loader + 메타.
    """
    import torch

    train_df = fold.train_df.copy()
    val_df = fold.val_df.copy()
    log_tasks: set[str] = set()
    if apply_log_transform:
        for t in task_names:
            if t in LOG_TRANSFORM_TASKS and t in train_df.columns:
                idx = task_names.index(t)
                if task_types[idx] == "regression":
                    train_df[t] = np.log1p(train_df[t])
                    val_df[t] = np.log1p(val_df[t])
                    log_tasks.add(t)
                    logger.info("log1p applied: %s", t)

    expanded_descriptors = expand_descriptors(feature_spec.descriptors)
    modality_keys_order = expanded_descriptors + list(feature_spec.embeddings)
    pp_map: dict[str, Path] = dict(parquet_paths or {})
    lmdb_map: dict[str, Path] = dict(lmdb_paths or {})

    # LMDB modality 는 parquet block stack 에 들어가지 않음 — 별도 collate 로 inject.
    # synthetic modality (graphmvp_data 등) 는 SMILES 로부터 collate 단계에서 즉석 생성.
    parquet_modality_keys = [
        k for k in modality_keys_order
        if k not in lmdb_map and k not in SYNTHETIC_MODALITY_KEYS
    ]
    lmdb_modality_keys = [k for k in modality_keys_order if k in lmdb_map]
    synthetic_modality_keys = [
        k for k in modality_keys_order if k in SYNTHETIC_MODALITY_KEYS
    ]
    if len(lmdb_modality_keys) > 1:
        raise NotImplementedError(
            f"여러 LMDB modality 동시 사용 미지원 (현재 {lmdb_modality_keys}). "
            f"한 번에 하나만 가능."
        )
    if len(synthetic_modality_keys) > 1:
        raise NotImplementedError(
            f"여러 synthetic modality 동시 사용 미지원 (현재 {synthetic_modality_keys})."
        )

    def _resolve(name: str) -> Path:
        return pp_map.get(name, parquet_root / f"{name}.parquet")

    # ---- modality blocks (train/val/test 각각, 같은 scaler 적용) ----
    train_blocks_dict: dict[str, ModalityBlock] = {}
    val_blocks_dict: dict[str, ModalityBlock] = {}
    train_smiles = train_df["Drug"].tolist()
    val_smiles = val_df["Drug"].tolist()

    for name in parquet_modality_keys:
        parquet = _resolve(name)
        if not parquet.exists():
            raise FileNotFoundError(
                f"modality {name!r} 의 사전 계산 parquet 누락: {parquet}. "
                f"`qadmet precompute --recipe recipes/precompute/{name}.yaml` 으로 먼저 빌드."
            )
        train_blocks_dict[name] = build_modality_block(train_smiles, name, parquet)
        val_blocks_dict[name] = build_modality_block(val_smiles, name, parquet)

    # LMDB modality 존재 여부 검증.
    for name in lmdb_modality_keys:
        if not lmdb_map[name].exists():
            raise FileNotFoundError(
                f"modality {name!r} 의 LMDB 누락: {lmdb_map[name]}.\n"
                f"  옛 precompute LMDB (atomic feature dict) 가 필요합니다."
            )

    scaler_state: ModalityScalerState | None = None
    if parquet_modality_keys:
        scaler_state = fit_modality_scalers(train_blocks_dict)
        for name in parquet_modality_keys:
            train_blocks_dict[name] = ModalityBlock(
                name=name,
                dim=train_blocks_dict[name].dim,
                array=scaler_state.transform(name, train_blocks_dict[name].array),
            )
            val_blocks_dict[name] = ModalityBlock(
                name=name,
                dim=val_blocks_dict[name].dim,
                array=scaler_state.transform(name, val_blocks_dict[name].array),
            )

    train_blocks_list = [train_blocks_dict[k] for k in parquet_modality_keys]
    val_blocks_list = [val_blocks_dict[k] for k in parquet_modality_keys]

    if train_blocks_list:
        train_xd, slices, modality_dims = stack_modality_blocks(train_blocks_list)
        val_xd, _, _ = stack_modality_blocks(val_blocks_list)
    else:
        train_xd, val_xd = None, None
        slices, modality_dims = {}, {}

    descriptor_keys = expanded_descriptors
    embedding_keys = list(feature_spec.embeddings)
    descriptor_slices = {k: slices[k] for k in descriptor_keys if k in slices}
    embedding_slices = {k: slices[k] for k in embedding_keys if k in slices}
    descriptor_dim_total = sum(modality_dims.get(k, 0) for k in descriptor_keys)
    # LMDB modality 도 modality_dims 에 sentinel 등록 (lit_module 검증 통과용 — fusion 입력 미사용).
    for name in lmdb_modality_keys:
        modality_dims.setdefault(name, 1)
    for name in synthetic_modality_keys:
        modality_dims.setdefault(name, 1)

    # ---- targets ----
    train_y = train_df[list(task_names)].to_numpy(np.float32)
    val_y = val_df[list(task_names)].to_numpy(np.float32)

    # ---- chemprop dataset ----
    train_dset = _build_chemprop_dataset(train_smiles, train_y, train_xd, n_workers=num_workers)
    val_dset = _build_chemprop_dataset(val_smiles, val_y, val_xd, n_workers=num_workers)

    # ---- regression target normalize (옛 train_dset.normalize_targets) ----
    output_scaler = None
    has_regression = any(t == "regression" for t in task_types)
    if has_regression:
        from sklearn.preprocessing import StandardScaler

        reg_idx = [i for i, t in enumerate(task_types) if t == "regression"]
        scaler = StandardScaler().fit(train_dset._Y[:, reg_idx])
        train_y_full = train_dset._Y.copy()
        val_y_full = val_dset._Y.copy()
        train_y_full[:, reg_idx] = scaler.transform(train_dset._Y[:, reg_idx])
        val_y_full[:, reg_idx] = scaler.transform(val_dset._Y[:, reg_idx])
        train_dset.Y = train_y_full
        val_dset.Y = val_y_full
        output_scaler = (scaler, tuple(reg_idx))

    pin_memory = torch.cuda.is_available()
    lmdb_modality = lmdb_modality_keys[0] if lmdb_modality_keys else None
    synthetic_modality = synthetic_modality_keys[0] if synthetic_modality_keys else None
    if synthetic_modality is not None:
        # graphmvp_data — collate 가 SMILES → PyG Batch. LMDB 와 조합 가능.
        collate_fn = make_collate_fn_with_graphmvp(
            descriptor_slices, embedding_slices, lmdb_modality, synthetic_modality,
        )
        if lmdb_modality is not None:
            train_dset_eff = _PairedChempropLmdbSmilesDataset(
                train_dset, train_smiles, lmdb_map[lmdb_modality],
            )
            val_dset_eff = _PairedChempropLmdbSmilesDataset(
                val_dset, val_smiles, lmdb_map[lmdb_modality],
            )
        else:
            train_dset_eff = _ChempropSmilesDataset(train_dset, train_smiles)
            val_dset_eff = _ChempropSmilesDataset(val_dset, val_smiles)
    elif lmdb_modality is not None:
        collate_fn = make_collate_fn_with_lmdb(
            descriptor_slices, embedding_slices, lmdb_modality,
        )
        train_dset_eff = _PairedChempropLmdbDataset(train_dset, train_smiles, lmdb_map[lmdb_modality])
        val_dset_eff = _PairedChempropLmdbDataset(val_dset, val_smiles, lmdb_map[lmdb_modality])
    else:
        collate_fn = make_collate_fn(descriptor_slices, embedding_slices)
        train_dset_eff = train_dset
        val_dset_eff = val_dset

    train_loader = _build_torch_loader(
        train_dset_eff, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, seed=seed, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    val_loader = _build_torch_loader(
        val_dset_eff, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, seed=None, pin_memory=pin_memory, collate_fn=collate_fn,
    )

    def _build_per_task_loader(
        smis: list[str],
        y_true_arr: np.ndarray,
    ):
        """task 별 val/test 단일-task loader 빌더.

        Why:
            chemprop dataloader 는 SMILES 를 보존하지 않는다. OOF stacking 에서
            (seed, task, smiles) 키로 base 예측들을 결합하려면 입력 순서 = SMILES 리스트
            를 직접 보존한 채 loader 를 만들어야 한다.

        같은 modality scaler / collate / chemprop 빌더를 재사용한다.
        """
        dummy_y = np.full((len(smis), len(task_names)), np.nan, dtype=np.float32)
        per_blocks: list[ModalityBlock] = []
        for name in parquet_modality_keys:
            parquet = _resolve(name)
            block = build_modality_block(smis, name, parquet)
            if scaler_state is not None:
                block = ModalityBlock(
                    name=name, dim=block.dim,
                    array=scaler_state.transform(name, block.array),
                )
            per_blocks.append(block)
        if per_blocks:
            per_xd, _, _ = stack_modality_blocks(per_blocks)
        else:
            per_xd = None
        per_dset = _build_chemprop_dataset(smis, dummy_y, per_xd, n_workers=num_workers)
        if synthetic_modality is not None:
            if lmdb_modality is not None:
                per_dset = _PairedChempropLmdbSmilesDataset(
                    per_dset, smis, lmdb_map[lmdb_modality],
                )
            else:
                per_dset = _ChempropSmilesDataset(per_dset, smis)
        elif lmdb_modality is not None:
            per_dset = _PairedChempropLmdbDataset(per_dset, smis, lmdb_map[lmdb_modality])
        per_loader = _build_torch_loader(
            per_dset, batch_size=batch_size, num_workers=num_workers,
            shuffle=False, seed=None, pin_memory=pin_memory, collate_fn=collate_fn,
        )
        return per_loader, smis, y_true_arr

    # ---- val loaders (per-task) — OOF stacking 입력 ----
    # fold.val_df 는 outer-merged DataFrame ([Drug, task1, task2, ...]).
    # task 컬럼 dropna 하면 그 task 에 finite Y 가 있는 (Drug, Y) 행 묶음을 얻는다.
    val_loaders: dict[str, tuple[Any, list[str], np.ndarray]] = {}
    val_df_orig = fold.val_df  # log1p 적용 *전* — y_true 는 원본 스케일.
    for task_name in task_names:
        if task_name not in val_df_orig.columns:
            continue
        sub = val_df_orig[["Drug", task_name]].dropna(subset=[task_name])
        if sub.empty:
            continue
        smis = sub["Drug"].tolist()
        y_true = sub[task_name].to_numpy(np.float32)
        val_loaders[task_name] = _build_per_task_loader(smis, y_true)

    # ---- test loaders (per-task) ----
    test_loaders: dict[str, tuple[Any, list[str], np.ndarray]] = {}
    for task_name, test_df in fold.test_dfs.items():
        smis = test_df["Drug"].tolist()
        y_true = test_df["Y"].to_numpy(np.float32)
        test_loaders[task_name] = _build_per_task_loader(smis, y_true)

    return FoldLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        modality_dims=modality_dims,
        modality_keys_order=modality_keys_order,
        descriptor_dim_total=descriptor_dim_total,
        scaler_state=scaler_state,
        output_scaler=output_scaler,
        log_tasks=frozenset(log_tasks),
    )


# ---------------------------------------------------------------------------- STL loaders


@dataclass
class StlLoaders:
    """STL (단일 task) 학습용 dataloader 묶음.

    Why:
        ``FoldLoaders`` 는 22-task MTL 가정으로 ``test_loaders`` 가 dict 였으나
        STL 은 task 1 개 학습 → val/test 단일 loader 로 충분하다. LoRA backbone
        (MoLFormer/UniMol2) 은 chemprop 그래프가 필요 없어 별도 dataset/collate
        를 사용한다.

    Attributes:
        train_loader / val_loader / test_loader: PyTorch DataLoader.
        val_smiles / test_smiles: 입력 순서 SMILES 리스트 (val 은 OOF stacking 용).
        val_y_true / test_y_true: log1p 적용 *전* 원본 y (예측 후 비교용).
        output_scaler: regression Y normalize ``(StandardScaler, slice)`` 또는 None.
        log_task: 본 task 가 ``log1p`` 변환 대상인지 (예측 후 expm1 역변환).
    """

    train_loader: Any
    val_loader: Any
    test_loader: Any
    val_smiles: list[str]
    val_y_true: np.ndarray
    test_smiles: list[str]
    test_y_true: np.ndarray
    output_scaler: Any | None
    log_task: bool


def filter_fold_to_task(fold: TdcAdmetFold, task_name: str) -> TdcAdmetFold:
    """MTL fold → STL fold (한 task 만 남김, NaN 행 제거).

    Why:
        MTL fold 의 train/val 은 outer-merged DataFrame 으로 다른 task NaN 가 많다.
        STL 은 본 task target 이 finite 인 행만 학습 대상이므로 ``dropna`` 한다.

    Args:
        fold: ``load_fold`` 결과.
        task_name: 추출할 task 컬럼 이름.

    Returns:
        ``TdcAdmetFold`` — train/val 은 (Drug, task_name) 만, test_dfs 는 해당 task 만.
    """
    if task_name not in fold.train_df.columns:
        raise KeyError(
            f"task {task_name!r} 가 fold.train_df 컬럼에 없음: "
            f"{list(fold.train_df.columns)}"
        )
    train_df = (
        fold.train_df[["Drug", task_name]]
        .dropna(subset=[task_name])
        .reset_index(drop=True)
    )
    val_df = (
        fold.val_df[["Drug", task_name]]
        .dropna(subset=[task_name])
        .reset_index(drop=True)
    )
    return TdcAdmetFold(
        train_df=train_df,
        val_df=val_df,
        test_dfs={task_name: fold.test_dfs[task_name]},
        samples_per_task={task_name: len(train_df)},
    )


# ---------------------------------------------------------------------------- token (MoLFormer) STL loaders


def _token_collate_fn():
    """MoLFormer-LoRA STL 용 collate.

    TensorDataset (token_ids, attention_mask, y) → ``MultitaskLitModule`` batch dict.
    """
    import torch

    def _collate(items):
        token_ids = torch.stack([x[0] for x in items], dim=0)
        attention_mask = torch.stack([x[1] for x in items], dim=0)
        targets = torch.stack([x[2] for x in items], dim=0)
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "modalities": {},
            "targets": targets,
            "sample_weights": None,
        }

    return _collate


def _lookup_tokens(smiles: Sequence[str], parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """molformer_tokens parquet 에서 (token_ids, attention_mask) 를 lookup.

    Returns:
        - token_ids (N, L) int64.
        - attention_mask (N, L) int64.
    누락 SMILES 는 0 (pad) 으로 채움 — backbone 이 attention_mask=0 으로 무시.
    """
    from qadmet.features.builders.molformer_tokens import MolFormerTokensBuilder

    builder = MolFormerTokensBuilder()
    cols = builder.columns  # tok_000..tok_{L-1}, msk_000..msk_{L-1}
    L = builder.max_length
    store = ParquetStore(parquet_path)
    arr = store.lookup(list(smiles), cols)  # (N, 2L) — int 가 NaN-safe 로 float
    arr = np.nan_to_num(arr, nan=0.0).astype(np.int64)
    token_ids = arr[:, :L]
    attention_mask = arr[:, L:]
    return token_ids, attention_mask


def prepare_stl_molformer_loaders(
    fold: TdcAdmetFold,
    *,
    task_name: str,
    task_type: str,
    tokens_parquet: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 0,
    apply_log_transform: bool = True,
) -> StlLoaders:
    """MoLFormer-LoRA STL 학습용 loader 빌더.

    옛 ``run_stl_molformer_lora.py`` 의 token-only 학습 경로를 ``MultitaskLitModule``
    인터페이스로 옮긴다.

    Args:
        fold: MTL ``load_fold`` 결과 (전체 22 task 컬럼).
        task_name / task_type: 학습 대상 task 와 type.
        tokens_parquet: ``molformer_tokens`` builder 산출물 parquet 경로.
        batch_size / num_workers / seed: DataLoader 옵션.
        apply_log_transform: regression task 가 LOG_TRANSFORM_TASKS 에 속하면 log1p
            적용 (학습/검증 입력만; ``y_true`` 는 원본 보존).

    Returns:
        ``StlLoaders``.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    sub = filter_fold_to_task(fold, task_name)
    train_smiles = sub.train_df["Drug"].tolist()
    val_smiles = sub.val_df["Drug"].tolist()
    test_df = sub.test_dfs[task_name]
    test_smiles = test_df["Drug"].tolist()

    train_y = sub.train_df[task_name].to_numpy(dtype=np.float32)
    val_y = sub.val_df[task_name].to_numpy(dtype=np.float32)
    test_y_true = test_df["Y"].to_numpy(dtype=np.float32)

    # --- log1p (옛 코드 동일) ---
    log_task = apply_log_transform and task_type == "regression" and task_name in LOG_TRANSFORM_TASKS
    if log_task:
        train_y_in = np.log1p(train_y)
        val_y_in = np.log1p(val_y)
    else:
        train_y_in = train_y
        val_y_in = val_y

    # --- output scaler (regression only) ---
    output_scaler = None
    if task_type == "regression":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train_y_in.reshape(-1, 1))
        train_y_scaled = scaler.transform(train_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        val_y_scaled = scaler.transform(val_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        output_scaler = (scaler, slice(0, 1))
    else:
        train_y_scaled = train_y_in.astype(np.float32)
        val_y_scaled = val_y_in.astype(np.float32)

    # --- token lookup ---
    train_tok, train_attn = _lookup_tokens(train_smiles, tokens_parquet)
    val_tok, val_attn = _lookup_tokens(val_smiles, tokens_parquet)
    test_tok, test_attn = _lookup_tokens(test_smiles, tokens_parquet)

    # --- TensorDataset (y shape (B, 1)) ---
    def _ds(tok, attn, y):
        return TensorDataset(
            torch.from_numpy(tok),
            torch.from_numpy(attn),
            torch.from_numpy(y).float().unsqueeze(-1),
        )

    train_ds = _ds(train_tok, train_attn, train_y_scaled)
    val_ds = _ds(val_tok, val_attn, val_y_scaled)
    test_ds = _ds(test_tok, test_attn, np.zeros(len(test_smiles), dtype=np.float32))

    pin_memory = torch.cuda.is_available()
    collate = _token_collate_fn()

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=pin_memory, collate_fn=collate, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )

    return StlLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        val_smiles=val_smiles,
        val_y_true=val_y,
        test_smiles=test_smiles,
        test_y_true=test_y_true,
        output_scaler=output_scaler,
        log_task=log_task,
    )


# ---------------------------------------------------------------------------- GraphMVP STL loaders


class _SmilesYDataset:
    """``(smiles, y)`` 인덱싱 — graphmvp STL collate 의 입력.

    feature 변환은 collate 단계 (``smiles_batch_to_pyg``) 에서 일괄 수행하므로
    ``__getitem__`` 은 메타만 반환한다.
    """

    def __init__(self, smiles: Sequence[str], y: np.ndarray) -> None:
        self._smis = list(smiles)
        self._y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self._smis)

    def __getitem__(self, idx: int):
        return (self._smis[idx], float(self._y[idx]))


def _graphmvp_stl_collate_fn():
    """GraphMVP STL 용 collate.

    items: list of ``(smiles, y)``. SMILES 묶음을 ``smiles_batch_to_pyg`` 로
    PyG Batch + presence mask 로 변환. invalid SMILES 는 1-atom placeholder +
    presence False — graphmvp adapter 가 그대로 backbone 에 흘려보내고 head
    학습 시 sample 단위 영향력을 낮추는 효과는 없지만 (STL 단일 modality 라
    presence mask 활용 fusion 이 없음), backbone forward 자체는 안전.
    """
    import torch

    from qadmet.modules.backbones._graphmvp_features import smiles_batch_to_pyg

    def _collate(items):
        smis = [it[0] for it in items]
        ys = torch.tensor([it[1] for it in items], dtype=torch.float32).unsqueeze(-1)
        graphmvp_batch, _present = smiles_batch_to_pyg(smis)
        return {
            "graphmvp_data": graphmvp_batch,
            "modalities": {},
            "targets": ys,
            "sample_weights": None,
        }

    return _collate


def prepare_stl_graphmvp_loaders(
    fold: TdcAdmetFold,
    *,
    task_name: str,
    task_type: str,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 0,
    apply_log_transform: bool = True,
) -> StlLoaders:
    """GraphMVP STL 학습용 loader 빌더.

    synthetic modality — 사전 계산물(parquet/LMDB) 불필요. collate 가 SMILES 를
    PyG Batch (OGB 9-d atom / 3-d bond feat) 로 즉석 변환.

    Args:
        fold: MTL ``load_fold`` 결과.
        task_name / task_type: 학습 대상 task.
        batch_size / num_workers / seed: DataLoader 옵션.
        apply_log_transform: regression + LOG_TRANSFORM_TASKS 일 때 log1p
            (학습/검증 입력만; ``y_true`` 는 원본 보존).

    Returns:
        ``StlLoaders``.
    """
    import torch
    from torch.utils.data import DataLoader

    sub = filter_fold_to_task(fold, task_name)
    train_smiles = sub.train_df["Drug"].tolist()
    val_smiles = sub.val_df["Drug"].tolist()
    test_df = sub.test_dfs[task_name]
    test_smiles = test_df["Drug"].tolist()

    train_y = sub.train_df[task_name].to_numpy(dtype=np.float32)
    val_y = sub.val_df[task_name].to_numpy(dtype=np.float32)
    test_y_true = test_df["Y"].to_numpy(dtype=np.float32)

    log_task = (
        apply_log_transform
        and task_type == "regression"
        and task_name in LOG_TRANSFORM_TASKS
    )
    if log_task:
        train_y_in = np.log1p(train_y)
        val_y_in = np.log1p(val_y)
    else:
        train_y_in = train_y
        val_y_in = val_y

    output_scaler = None
    if task_type == "regression":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train_y_in.reshape(-1, 1))
        train_y_scaled = (
            scaler.transform(train_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        )
        val_y_scaled = (
            scaler.transform(val_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        )
        output_scaler = (scaler, slice(0, 1))
    else:
        train_y_scaled = train_y_in.astype(np.float32)
        val_y_scaled = val_y_in.astype(np.float32)

    train_ds = _SmilesYDataset(train_smiles, train_y_scaled)
    val_ds = _SmilesYDataset(val_smiles, val_y_scaled)
    test_ds = _SmilesYDataset(
        test_smiles, np.zeros(len(test_smiles), dtype=np.float32),
    )

    pin_memory = torch.cuda.is_available()
    collate = _graphmvp_stl_collate_fn()
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=pin_memory, collate_fn=collate, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )

    return StlLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        val_smiles=val_smiles,
        val_y_true=val_y,
        test_smiles=test_smiles,
        test_y_true=test_y_true,
        output_scaler=output_scaler,
        log_task=log_task,
    )


# ---------------------------------------------------------------------------- UniMol2 STL loaders


def _unimol2_collate_fn():
    """UniMol2-LoRA STL 용 collate.

    items: list of feat dict (vendored ``mol2unimolv2`` 결과 + ``__y__``).
    None (featurize 실패) 항목은 제거. 빈 batch 는 ``None`` 반환 (Lightning skip).
    """
    import torch

    from qadmet.modules.backbones._vendor import unimol_v2 as uv2

    def _collate(items):
        items = [s for s in items if s is not None]
        if not items:
            return None
        ys = torch.tensor([s["__y__"] for s in items], dtype=torch.float32).unsqueeze(-1)
        feat_only = [{k: v for k, v in s.items() if not k.startswith("__")} for s in items]
        unimol2_batch = uv2.collate_unimolv2(feat_only)
        return {
            "unimol2_batch": unimol2_batch,
            "modalities": {},
            "targets": ys,
            "sample_weights": None,
        }

    return _collate


class _UniMol2LmdbDataset:
    """LMDB 기반 Uni-Mol2 feature dict dataset.

    옛 ``run_stl_unimol2_lora.UniMol2Dataset`` 와 동등 — 워커 별 lazy LMDB open.
    필요한 키만 lookup 하고 ``__y__`` / ``__smi__`` 메타를 함께 반환한다.
    """

    def __init__(self, smiles_list: Sequence[str], y: np.ndarray, lmdb_path: Path) -> None:
        self.smiles_list = list(smiles_list)
        self.y = y.astype(np.float32)
        self.lmdb_path = str(lmdb_path)
        self._env = None
        self._txn = None

    def _ensure_open(self) -> None:
        if self._env is not None:
            return
        import lmdb  # type: ignore

        self._env = lmdb.open(
            self.lmdb_path, subdir=False, lock=False, readonly=True, max_readers=1024,
        )
        self._txn = self._env.begin()

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        import pickle

        self._ensure_open()
        smi = self.smiles_list[idx]
        v = self._txn.get(smi.encode("utf-8"))  # type: ignore[union-attr]
        if v is None:
            return None
        feat = pickle.loads(v)
        feat["__y__"] = self.y[idx]
        feat["__smi__"] = smi
        return feat


def prepare_stl_unimol2_loaders(
    fold: TdcAdmetFold,
    *,
    task_name: str,
    task_type: str,
    lmdb_path: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 0,
    apply_log_transform: bool = True,
) -> StlLoaders:
    """UniMol2-LoRA STL 학습용 loader 빌더.

    옛 ``run_stl_unimol2_lora.py`` 의 LMDB 기반 학습 경로를 그대로 옮긴다.
    LMDB 는 옛 precompute 산출물 (``data/embeddings/unimol2_features.lmdb``) 를 그대로
    재사용한다 — vendored ``mol2unimolv2`` 결과 dict 를 pickle 로 저장한 것.

    Args:
        fold: MTL load_fold 결과.
        task_name / task_type: 학습 대상 task.
        lmdb_path: LMDB 파일 경로.
        batch_size / num_workers / seed.
        apply_log_transform: regression + LOG_TRANSFORM_TASKS 일 때 log1p.

    Returns:
        ``StlLoaders``.
    """
    import torch
    from torch.utils.data import DataLoader

    sub = filter_fold_to_task(fold, task_name)
    train_smiles = sub.train_df["Drug"].tolist()
    val_smiles = sub.val_df["Drug"].tolist()
    test_df = sub.test_dfs[task_name]
    test_smiles = test_df["Drug"].tolist()

    train_y = sub.train_df[task_name].to_numpy(dtype=np.float32)
    val_y = sub.val_df[task_name].to_numpy(dtype=np.float32)
    test_y_true = test_df["Y"].to_numpy(dtype=np.float32)

    log_task = apply_log_transform and task_type == "regression" and task_name in LOG_TRANSFORM_TASKS
    if log_task:
        train_y_in = np.log1p(train_y)
        val_y_in = np.log1p(val_y)
    else:
        train_y_in = train_y
        val_y_in = val_y

    output_scaler = None
    if task_type == "regression":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train_y_in.reshape(-1, 1))
        train_y_scaled = scaler.transform(train_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        val_y_scaled = scaler.transform(val_y_in.reshape(-1, 1)).flatten().astype(np.float32)
        output_scaler = (scaler, slice(0, 1))
    else:
        train_y_scaled = train_y_in.astype(np.float32)
        val_y_scaled = val_y_in.astype(np.float32)

    train_ds = _UniMol2LmdbDataset(train_smiles, train_y_scaled, lmdb_path)
    val_ds = _UniMol2LmdbDataset(val_smiles, val_y_scaled, lmdb_path)
    test_ds = _UniMol2LmdbDataset(
        test_smiles, np.zeros(len(test_smiles), dtype=np.float32), lmdb_path,
    )

    pin_memory = torch.cuda.is_available()
    collate = _unimol2_collate_fn()
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=pin_memory, collate_fn=collate, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate,
    )

    return StlLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        val_smiles=val_smiles,
        val_y_true=val_y,
        test_smiles=test_smiles,
        test_y_true=test_y_true,
        output_scaler=output_scaler,
        log_task=log_task,
    )
