"""DataLoader builder for training + collate adapter.

This module takes ``TdcAdmetFold`` and ``FeatureSpec``, and
builds a PyTorch DataLoader satisfying the batch dict contract of ``MultitaskLitModule``
builds it.

Core flow:
    1. ``build_modality_block`` — in the order features.descriptors + features.embeddings
       stacks precomputed parquet files by SMILES to build ``X_d_combined (N, ΣD)`` and
       a per-modality slice index map.
    2. passes ``X_d_combined`` wholesale into the ``X_d`` slot of chemprop ``MoleculeDatapoint``
       → chemprop dataloader turns it into ``TrainingBatch.X_d (B, ΣD)`` per batch and
       returns it.
    3. ``make_collate_fn`` adds one more layer on top of chemprop collate_batch,
       splitting the descriptor slice into ``batch["X_d"]`` and the embedding slice into
       the ``batch["modalities"]`` dict → the dict that ``MultitaskLitModule`` receives
       is yielded.

Why this structure (Step B design, 2026-05-01):
    descriptor (rdkit2d / xtb / gfn_missing_vector) keeps the old chemprop X_d concat
    path as-is (DMPNN ``MPNN.fingerprint`` concats X_d after fp → output
    dim = d_h + d_xd). embedding (molformer / unimol2 frozen CLS, or multi-backbone's
    token modality from the LoRA path) alone is sent to the fusion module. As a result:
        - recipes using only descriptors (mtl_dmpnn_rdkit / rdkit_xtb / +missing)
          don't use fusion → single-backbone training follows the same path as the old phase1.3a/b.
        - recipes with embeddings added (mtl_with_molformer / mtl_dmpnn_molformer_lora)
          have fusion input in the form ``[H_dmpnn(d_h+d_xd), *embeddings]``.
    chemprop dataloader's shuffle/batching mechanism is reused as-is.

Lazy import:
    ``chemprop`` / ``torch`` are imported only within functions.
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
"""Set of tasks that apply ``log1p`` to the regression target (old ``pipeline_mtl.LOG_TRANSFORM_TASKS``)."""

_XTB_KEY = "xtb"
_XTB_MASK_KEY = "gfn_missing_vector"

#: Modality with no precomputed parquet/lmdb, generated on-the-fly from SMILES at the collate stage.
#: graphmvp_data: collate converts SMILES → ``torch_geometric.data.Batch`` → graphmvp
#: backbone consumes it. on-the-fly conversion cost is low, so LMDB precompute is unnecessary.
SYNTHETIC_MODALITY_KEYS: frozenset[str] = frozenset({"graphmvp_data"})


def expand_descriptors(descriptors: Sequence[str]) -> list[str]:
    """When xtb is included, automatically insert gfn_missing_vector right after it.

    Even if the recipe doesn't explicitly specify gfn_missing_vector, it always flows together with xtb.
    No-op if it's already included.
    """
    result = list(descriptors)
    if _XTB_KEY in result and _XTB_MASK_KEY not in result:
        idx = result.index(_XTB_KEY)
        result.insert(idx + 1, _XTB_MASK_KEY)
    return result


# ---------------------------------------------------------------------------- modality block


@dataclass
class ModalityBlock:
    """The (N, dim) tensor + metadata for a single modality.

    Attributes:
        name: modality key (name registered in FEATURE_BUILDERS).
        dim: number of channels.
        array: (N, dim) float32. Rows with missing SMILES are filled with 0 (zeros).
    """

    name: str
    dim: int
    array: np.ndarray


def _builder_columns(name: str) -> tuple[list[str], int]:
    """Fetch columns + dim from the builder class (instantiated only once)."""
    cls = FEATURE_BUILDERS.get(name)
    builder = cls()
    return list(builder.columns), int(builder.output_dim)


def build_modality_block(
    smiles: Sequence[str],
    name: str,
    parquet_path: Path,
) -> ModalityBlock:
    """Build the (N, dim) tensor for a single modality in SMILES order.

    Args:
        smiles: SMILES list that determines row order (unsorted, preserves the fold's original order).
        name: FEATURE_BUILDERS key.
        parquet_path: location of the precomputed parquet for that modality (``ParquetStore``).

    Returns:
        ``ModalityBlock(name, dim, array)``. Missing SMILES are filled with zeros.
    """
    cols, dim = _builder_columns(name)
    store = ParquetStore(parquet_path)
    arr = store.lookup(list(smiles), cols)  # (N, dim), missing values are NaN.
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return ModalityBlock(name=name, dim=dim, array=arr)


# ---------------------------------------------------------------------------- modality scaler


@dataclass
class ModalityScalerState:
    """Per-modality scaler state learned from the train fold.

    NaN-safe StandardScaler — computes mean/std only from train_arr and applies them at transform time.
    Integer-token modalities like ``molformer_tokens`` are bypassed via ``skip_keys``.
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


# ``molformer_tokens`` is token_id (integer) + attention mask (0/1) — not a scaling target.
# ``gfn_missing_vector`` is a binary 0/1 mask — standardizing destroys its meaning.
DEFAULT_SCALE_SKIP: tuple[str, ...] = ("molformer_tokens", "gfn_missing_vector")


def fit_modality_scalers(
    blocks: Mapping[str, ModalityBlock],
    *,
    clip: float = 10.0,
    skip_keys: tuple[str, ...] = DEFAULT_SCALE_SKIP,
) -> ModalityScalerState:
    """Compute mean/std from train modality blocks. clip ±10 (old code convention).

    xTB block: if ``gfn_missing_vector`` is present, fit using only rows where all channels
    are observed (QW-MTL observed-only strategy). Prevents 0-imputed missing values from distorting the distribution.
    Falls back to all rows if observed rows < 2 (same as the old ``fit_xd_scalers``).
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
    """modality block list → (X_d_combined, slices, dims).

    Returns:
        - X_d_combined: ``(N, ΣD)`` float32. ``None`` if blocks is empty.
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


# LMDB env cache per process + path — opening the same LMDB twice within the same process
# causes ``lmdb.Error: environment is already open in this process``, so it's shared.
# After a DataLoader worker fork, ``os.getpid()`` differs, so it opens anew (expected behavior).
_LMDB_ENV_CACHE: dict[tuple[int, str], Any] = {}


def _get_shared_lmdb_txn(path: str):
    """Caches LMDB env/txn per process. Opens on first call for a given path, reuses thereafter."""
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
    """Simultaneous indexing of chemprop dataset + LMDB feat (for the MTL UniMol2 LoRA path).

    base chemprop dataset's ``__getitem__`` result and LMDB's (smiles → feat dict)
    lookup result bundled into a tuple and returned. Shares env cache per (process, path).

    None feat (LMDB miss) is returned as-is — collate fills in the placeholder and presence
    mask blocks the corresponding modality at the fusion stage.
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
    """Collate that injects LMDB feat (UniMol2 LoRA MTL path).

    Same responsibility as base ``make_collate_fn`` + merges feat dicts received from LMDB
    via vendored ``collate_unimolv2`` and injects them into ``batch["modalities"][lmdb_modality]``
    as a dict. The unimol2_lora input adapter in ``lit_module`` reads the same dict from the ``__consumed__``
    slot.

    Args:
        descriptor_slices / embedding_slices: same as ``make_collate_fn``.
        lmdb_modality: LMDB modality name (e.g. ``"unimol2_lmdb"``). Since it is consumed,
            it is excluded from fusion input, so the dimension is meaningless.

    Note:
        Items with feat None (LMDB miss / featurize failure) are removed from the batch. If
        all are removed, returns ``None`` → Lightning skip.
    """
    import torch

    from chemprop.data.collate import collate_batch

    def _collate(items):
        # items: list of (chemprop_item, feat_dict_or_None)
        # missing handling: keep missing samples in the batch too (pass forward with placeholder feat)
        # then invalidate the corresponding modality alpha via the ``unimol2_present`` mask at the fusion stage.
        present_flags = [f is not None for (_cp, f) in items]
        if not any(present_flags):
            return None  # all missing → skip the batch itself
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
        modalities[lmdb_modality] = unimol2_batch  # dict — flows only through the consumes path.

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
    """Indexes chemprop dataset + SMILES simultaneously (for synthetic modality path).

    items: ``(chemprop_item, smiles)``. collate generates PyG Batch etc.
    synthetic modality payload on the fly from SMILES.
    """

    def __init__(self, chemprop_dset, smiles_list: Sequence[str]) -> None:
        self.chemprop = chemprop_dset
        self._smis = list(smiles_list)

    def __len__(self) -> int:
        return len(self.chemprop)

    def __getitem__(self, idx: int):
        return (self.chemprop[idx], self._smis[idx])


class _PairedChempropLmdbSmilesDataset(_PairedChempropLmdbDataset):
    """Indexes LMDB + SMILES simultaneously (for simultaneous use of LMDB modality + synthetic modality)."""

    def __getitem__(self, idx: int):
        cp_item, feat = super().__getitem__(idx)
        return (cp_item, feat, self._smis[idx])


def make_collate_fn_with_graphmvp(
    descriptor_slices: Mapping[str, slice],
    embedding_slices: Mapping[str, slice],
    lmdb_modality: str | None,
    graphmvp_key: str = "graphmvp_data",
):
    """Collate that supports graphmvp synthetic modality + (optional) LMDB modality simultaneously.

    Performs SMILES → PyG Batch conversion at the collate stage (no precompute needed). When combined
    with the LMDB path, also merges the unimol2 feat dict and exposes it under another key of the ``modalities`` dict.

    Args:
        descriptor_slices / embedding_slices: same as ``make_collate_fn``.
        lmdb_modality: LMDB modality name (e.g. "unimol2_lmdb"). If None, bypasses the LMDB path.
        graphmvp_key: synthetic modality key — must match the backbone consumes.

    Items shape (depends on how loader wraps the dataset):
        - No LMDB:    ``(chemprop_item, smiles)`` — ``_ChempropSmilesDataset``.
        - With LMDB:    ``(chemprop_item, lmdb_feat_or_None, smiles)`` —
          ``_PairedChempropLmdbSmilesDataset``.

    Note:
        graphmvp invalid SMILES is treated as a 1-atom placeholder + presence False, which
        blocks alpha at the fusion stage. LMDB miss is placeholder feat + presence False.
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
                # batch where all are LMDB miss → no meaning for training. skip.
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
            modalities[lmdb_modality] = unimol2_batch  # dict — only via consumes path.
            modality_present[lmdb_modality] = torch.tensor(
                lmdb_present_flags, dtype=torch.bool
            )

        # ---- graphmvp synthetic modality ----
        graphmvp_batch, graphmvp_present = smiles_batch_to_pyg(smis)
        modalities[graphmvp_key] = graphmvp_batch  # PyG Batch — only via consumes path.
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

    Both descriptor and embedding modality arrive stacked into chemprop's single ``X_d`` channel,
    so this collate splits them into two groups.

    Args:
        descriptor_slices: ``{name: slice(start, end)}`` range of descriptor within X_d.
            If empty, ``batch["X_d"] = None`` so that the DMPNN backbone
            operates without a descriptor.
        embedding_slices: ``{name: slice(start, end)}`` range of embedding within X_d.
            Becomes a key of the ``batch["modalities"]`` dict (fusion input).

    Returns:
        ``collate_fn`` callable. Used for the ``collate_fn`` argument of PyTorch DataLoader.

    Note:
        Since descriptor is placed before embedding when stacked (stacked in that order in
        loaders), it is usually a contiguous slice. However, this function does not assume the slice is contiguous
        and explicitly does ``torch.cat`` — safe even for old non-sequential stacking.
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
                # ensure slice result is a float tensor (token modality also passes through as float — backbone adapter casts via .long() etc.).
                modalities[name] = x_d[:, sl].float()
        return {
            "bmg": tb.bmg,
            "V_d": tb.V_d,
            "X_d": x_d_for_backbone,
            "modalities": modalities,
            "targets": tb.Y,
            "sample_weights": None,  # per-batch weight is handled by the LitModule.task_weights buffer.
        }

    return _collate


# ---------------------------------------------------------------------------- Unified builder


@dataclass
class FoldLoaders:
    """Bundle of dataloaders for one seed fold.

    Attributes:
        train_loader / val_loader: PyTorch DataLoader (collate returns a dict).
            ``val_loader`` is for EarlyStopping monitoring during training (outer-merged val).
        val_loaders: ``{task_name: (DataLoader, smiles_list, y_true)}``.
            for per-task val prediction — input to OOF stacking (Step 4.2 InverseVariance).
            Same pattern as ``test_loaders``: selects only finite val rows of one task into single-task
            loader and applies the same modality scaler.
        test_loaders: ``{task_name: (DataLoader, smiles_list, y_true)}``.
            test has a separate dataloader per task (old train_mtl_mpnn pattern).
        modality_dims: ``{name: D}`` dimension of all modalities (descriptors + embeddings).
            used by the runner to separately compute the sum of descriptors (= DMPNN d_xd) and the sum of embeddings (= fusion input)
            .
        modality_keys_order: descriptors order + embeddings order.
        descriptor_dim_total: sum of descriptor dimensions (= DMPNN backbone's ``d_xd``).
            0 if descriptors is empty.
        scaler_state: ModalityScalerState fit on train (val/test loader also transform with the same state).
        output_scaler: sklearn StandardScaler for regression normalization (result of the old train_dset.normalize_targets). Can be None.
        log_tasks: set of regression task names with log1p applied (needed for expm1 inverse transform after prediction).
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
    """Wraps chemprop ``make_datapoints`` → ``make_dataset`` calls."""
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
    """Pure PyTorch DataLoader — bypasses chemprop build_dataloader (to override collate_fn)."""
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
    """Builds train/val/test DataLoader for a single fold in one batch.

    Args:
        fold: result of ``load_fold``.
        feature_spec: ``FeatureSpec`` (descriptors + embeddings order).
        task_names: target column order.
        task_types: length ``len(task_names)``, each task's ``"regression"`` / ``"classification"``.
        parquet_root: parent directory of precompute output files.
            assumes each modality is at ``{parquet_root}/{name}.parquet``.
        parquet_paths: explicit path map per modality. Missing keys fall back to ``parquet_root/{name}.parquet``
            convention. Used when using legacy data (non-convention names such as ``data/rdkit_2d.parquet``)
            directly.
        batch_size / num_workers / seed: chemprop dataloader options.
        apply_log_transform: applies ``log1p`` to columns included in ``LOG_TRANSFORM_TASKS`` among
            regression tasks (legacy code ``pipeline_mtl.py:468-473``).

    Returns:
        ``FoldLoaders`` — train/val/test loader + metadata.
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

    # LMDB modality is not included in the parquet block stack — injected via separate collate.
    # synthetic modality (e.g. graphmvp_data) is generated on-the-fly from SMILES during the collate step.
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
            f"Using multiple LMDB modalities simultaneously is not supported (currently {lmdb_modality_keys}). "
            f"Only one at a time is allowed."
        )
    if len(synthetic_modality_keys) > 1:
        raise NotImplementedError(
            f"Using multiple synthetic modalities simultaneously is not supported (currently {synthetic_modality_keys})."
        )

    def _resolve(name: str) -> Path:
        return pp_map.get(name, parquet_root / f"{name}.parquet")

    # ---- modality blocks (train/val/test each, same scaler applied) ----
    train_blocks_dict: dict[str, ModalityBlock] = {}
    val_blocks_dict: dict[str, ModalityBlock] = {}
    train_smiles = train_df["Drug"].tolist()
    val_smiles = val_df["Drug"].tolist()

    for name in parquet_modality_keys:
        parquet = _resolve(name)
        if not parquet.exists():
            raise FileNotFoundError(
                f"Missing precomputed parquet for modality {name!r}: {parquet}. "
                f"Build it first with `qadmet precompute --recipe recipes/precompute/{name}.yaml`."
            )
        train_blocks_dict[name] = build_modality_block(train_smiles, name, parquet)
        val_blocks_dict[name] = build_modality_block(val_smiles, name, parquet)

    # Verify existence of LMDB modality.
    for name in lmdb_modality_keys:
        if not lmdb_map[name].exists():
            raise FileNotFoundError(
                f"Missing LMDB for modality {name!r}: {lmdb_map[name]}.\n"
                f"  Legacy precompute LMDB (atomic feature dict) is required."
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
    # LMDB modality is also registered as a sentinel in modality_dims (to pass lit_module validation — not used as fusion input).
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

    # ---- regression target normalize (legacy train_dset.normalize_targets) ----
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
        # graphmvp_data — collate converts SMILES → PyG Batch. Can be combined with LMDB.
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
        """Builder for per-task val/test single-task loader.

        Why:
            chemprop dataloader does not preserve SMILES. In OOF stacking,
            to combine base predictions by (seed, task, smiles) key, the input order = SMILES list
            must be preserved directly when building the loader.

        Reuses the same modality scaler / collate / chemprop builder.
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

    # ---- val loaders (per-task) — OOF stacking input ----
    # fold.val_df is an outer-merged DataFrame ([Drug, task1, task2, ...]).
    # dropna on the task column yields the set of (Drug, Y) rows with finite Y for that task.
    val_loaders: dict[str, tuple[Any, list[str], np.ndarray]] = {}
    val_df_orig = fold.val_df  # *before* log1p is applied — y_true is in the original scale.
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
    """Bundle of dataloaders for STL (single task) training.

    Why:
        ``FoldLoaders`` assumed 22-task MTL, so ``test_loaders`` was a dict,
        but STL trains a single task → a single val/test loader suffices. LoRA backbone
        (MoLFormer/UniMol2) does not need chemprop graphs, so a separate dataset/collate
        is used.

    Attributes:
        train_loader / val_loader / test_loader: PyTorch DataLoader.
        val_smiles / test_smiles: SMILES list in input order (val is for OOF stacking).
        val_y_true / test_y_true: original y *before* log1p is applied (for comparison after prediction).
        output_scaler: regression Y normalize ``(StandardScaler, slice)`` or None.
        log_task: whether this task is subject to ``log1p`` transform (expm1 inverse transform after prediction).
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
    """MTL fold → STL fold (keeps only one task, removes NaN rows).

    Why:
        train/val of MTL fold is an outer-merged DataFrame with many NaNs for other tasks.
        STL trains only on rows where this task's target is finite, so ``dropna`` is applied.

    Args:
        fold: result of ``load_fold``.
        task_name: name of the task column to extract.

    Returns:
        ``TdcAdmetFold`` — train/val have only (Drug, task_name), test_dfs only that task.
    """
    if task_name not in fold.train_df.columns:
        raise KeyError(
            f"task {task_name!r} not found in fold.train_df columns: "
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
    """Collate for MoLFormer-LoRA STL.

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
    """Lookup (token_ids, attention_mask) from molformer_tokens parquet.

    Returns:
        - token_ids (N, L) int64.
        - attention_mask (N, L) int64.
    Missing SMILES filled with 0 (pad) — backbone ignores via attention_mask=0.
    """
    from qadmet.features.builders.molformer_tokens import MolFormerTokensBuilder

    builder = MolFormerTokensBuilder()
    cols = builder.columns  # tok_000..tok_{L-1}, msk_000..msk_{L-1}
    L = builder.max_length
    store = ParquetStore(parquet_path)
    arr = store.lookup(list(smiles), cols)  # (N, 2L) — int becomes float for NaN-safety
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
    """Loader builder for MoLFormer-LoRA STL training.

    Moves the token-only training path from the old ``run_stl_molformer_lora.py`` to the ``MultitaskLitModule``
    interface.

    Args:
        fold: MTL ``load_fold`` result (all 22 task columns).
        task_name / task_type: target task and its type for training.
        tokens_parquet: path to the parquet output of the ``molformer_tokens`` builder.
        batch_size / num_workers / seed: DataLoader options.
        apply_log_transform: if the regression task belongs to LOG_TRANSFORM_TASKS, apply log1p
            (training/validation inputs only; ``y_true`` keeps the original values).

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

    # --- log1p (same as old code) ---
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
    """``(smiles, y)`` indexing — input for graphmvp STL collate.

    Since feature conversion is performed in batch at the collate stage (``smiles_batch_to_pyg``),
    ``__getitem__`` returns only the metadata.
    """

    def __init__(self, smiles: Sequence[str], y: np.ndarray) -> None:
        self._smis = list(smiles)
        self._y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self._smis)

    def __getitem__(self, idx: int):
        return (self._smis[idx], float(self._y[idx]))


def _graphmvp_stl_collate_fn():
    """Collate for GraphMVP STL.

    items: list of ``(smiles, y)``. Converts the batch of SMILES via ``smiles_batch_to_pyg``
    into a PyG Batch + presence mask. Invalid SMILES become a 1-atom placeholder +
    presence False — the graphmvp adapter passes it straight through to the backbone, and the head
    has no effect of reducing per-sample influence during training (since STL is a single modality,
    there's no fusion using the presence mask), but the backbone forward pass itself is safe.
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
    """Loader builder for GraphMVP STL training.

    synthetic modality — no precomputed artifacts (parquet/LMDB) needed. The collate converts SMILES
    into a PyG Batch (OGB 9-d atom / 3-d bond feat) on the fly.

    Args:
        fold: MTL ``load_fold`` result.
        task_name / task_type: target task for training.
        batch_size / num_workers / seed: DataLoader options.
        apply_log_transform: apply log1p when regression + LOG_TRANSFORM_TASKS
            (training/validation inputs only; ``y_true`` keeps the original values).

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
    """Collate for UniMol2-LoRA STL.

    items: list of feat dict (vendored ``mol2unimolv2`` result + ``__y__``).
    Removes None entries (featurize failures). Returns ``None`` for an empty batch (Lightning skip).
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
    """LMDB-based Uni-Mol2 feature dict dataset.

    Equivalent to the old ``run_stl_unimol2_lora.UniMol2Dataset`` — lazy LMDB open per worker.
    Looks up only the needed keys and returns them together with ``__y__`` / ``__smi__`` metadata.
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
    """Loader builder for UniMol2-LoRA STL training.

    Carries over the LMDB-based training path from the old ``run_stl_unimol2_lora.py`` as-is.
    The LMDB reuses the old precompute output (``data/embeddings/unimol2_features.lmdb``) as-is
    — it is the vendored ``mol2unimolv2`` result dict saved via pickle.

    Args:
        fold: MTL load_fold result.
        task_name / task_type: target task for training.
        lmdb_path: path to the LMDB file.
        batch_size / num_workers / seed.
        apply_log_transform: apply log1p when regression + LOG_TRANSFORM_TASKS.

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
