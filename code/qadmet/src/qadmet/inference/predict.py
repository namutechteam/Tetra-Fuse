"""Predict 22 TDC ADMET endpoints for arbitrary SMILES with the Tetra-Fuse ensemble.

Pipeline per (member, task, seed):

    SMILES -> modality features -> chemprop batch -> model.forward -> post-process

and the k=5 ensemble prediction is the uniform (unweighted) mean over members —
the same rule used throughout the paper.

Two details make new molecules work, and both are easy to get silently wrong:

1. `molformer_tokens` is normally read from a precomputed parquet keyed by SMILES,
   and `build_modality_block` fills *missing* SMILES with **zeros**. An unseen
   molecule would therefore be fed an all-zero token sequence and produce a
   confident-looking but meaningless prediction. We tokenize the input SMILES on
   the fly (`MolFormerTokensBuilder`) into a temporary parquet, so the lookup hits
   real tokens. Affects MoLFormer·MTL and MAP·STL.
2. Regression outputs are normalized during training. We invert with the exported
   per-(task, seed) scaler and apply `expm1` for the four log1p endpoints, exactly
   as `runners/train.py::_predict_per_task` does. Classification outputs get a
   sigmoid.

Cost note: a single prediction rebuilds each member from its recipe, so a call
covering all 22 tasks x 5 seeds loads 5 x 22 x 5 models for the STL members. Use
`seeds=(0,)` for a quick single-seed answer, or pass `tasks=[...]` to limit scope;
the full 5-seed ensemble reproduces the paper's seed-averaged track.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .loader import ENSEMBLE, MEMBER_RECIPES, load_member, _task_meta  # noqa: F401


def _tokens_parquet(smiles: list[str], dest: Path) -> Path:
    """Tokenize `smiles` with the MoLFormer tokenizer into a ParquetStore layout.

    The stored table is keyed by the SMILES string, matching what
    `qadmet precompute --recipe recipes/precompute/molformer_tokens.yaml` writes,
    so the normal lookup path finds these rows.
    """
    from qadmet.features.builders.molformer_tokens import MolFormerTokensBuilder

    builder = MolFormerTokensBuilder()
    cols = builder.columns
    # compute_batch returns {smiles: concat([token_ids, attention_mask])}, one
    # row-vector per molecule — not a column-keyed dict.
    out = builder.compute_batch(smiles)
    mat = np.stack([out[s] for s in smiles], axis=0)
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "smiles", smiles)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def _predict_graphmvp_stl(lm, smiles: list[str], *, batch_size: int = 32) -> np.ndarray:
    """Predictions for GraphMVP·STL, reusing the training loader's dataset/collate.

    `_SmilesYDataset` + `_graphmvp_stl_collate_fn` are the exact objects
    `prepare_stl_graphmvp_loaders` uses, so featurization matches training. Labels
    are unused at inference, so we pass zeros.
    """
    from qadmet.datasets.loaders import (  # type: ignore[attr-defined]
        _build_torch_loader, _graphmvp_stl_collate_fn, _SmilesYDataset,
    )

    dset = _SmilesYDataset(smiles, np.zeros(len(smiles), dtype=np.float32))
    loader = _build_torch_loader(dset, batch_size=batch_size, num_workers=0,
                                shuffle=False, seed=None, pin_memory=False,
                                collate_fn=_graphmvp_stl_collate_fn())
    device = next(lm.model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            outs.append(lm.model(_to_device(batch, device)).float().cpu().numpy())
    raw = np.concatenate(outs, axis=0)[:, lm.task_index].astype(np.float64)
    return _postprocess(raw, lm)


def _postprocess(raw: np.ndarray, lm) -> np.ndarray:
    """Invert target normalization / apply sigmoid, per runners/train.py."""
    if lm.task_type == "regression":
        raw = raw * lm.scaler_scale + lm.scaler_mean
        if lm.log1p:
            raw = np.expm1(raw)
    else:
        raw = 1.0 / (1.0 + np.exp(-raw))
    return raw.astype(np.float32)


def _to_device(obj, device):
    """Move a collated batch to `device`, preserving its nested structure.

    Batches mix plain tensors, dicts (`modalities`), chemprop's BatchMolGraph and
    PyG Batch objects; the latter two carry their own `.to()`.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if hasattr(obj, "to") and not isinstance(obj, (str, bytes)):
        # chemprop's BatchMolGraph moves in place and returns None; PyG's Batch
        # returns a new object. Keep the original when `.to()` gives nothing back.
        moved = obj.to(device)
        return obj if moved is None else moved
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _predict_one(lm, smiles: list[str], *, recipes_root: Path, repo_root: Path,
                 tokens_parquet: Path | None, batch_size: int = 32) -> np.ndarray:
    """Raw (post-processed) predictions for one loaded member over `smiles`."""
    from qadmet.core.recipe import load_recipe
    from qadmet.datasets.loaders import build_modality_block, stack_modality_blocks
    from qadmet.datasets.loaders import _build_chemprop_dataset  # type: ignore[attr-defined]

    recipe_stem, is_mtl = MEMBER_RECIPES[lm.member]
    spec = load_recipe(Path(recipes_root) / "train" / f"{recipe_stem}.yaml")
    embeddings = tuple(getattr(spec.features, "embeddings", ()) or ())

    # GraphMVP·STL has its own data path in training: `prepare_stl_graphmvp_loaders`
    # builds (smiles, y) items and converts them to a PyG Batch in collate, with no
    # chemprop graph and no modality stack (`embeddings: []` in the recipe). Routing
    # it through the chemprop path produces a batch without 'graphmvp_data' and the
    # adapter raises. Reuse the training helpers instead.
    if spec.model.backbone.type == "graphmvp" and not is_mtl:
        return _predict_graphmvp_stl(lm, smiles, batch_size=batch_size)

    # ---- feature blocks (parquet-backed modalities only) -------------------
    # `stack_modality_blocks` returns (X_d, slices, dims); the slices tell the
    # collate function where each modality sits inside chemprop's single X_d
    # channel, so they must come from the same call that built X_d.
    blocks = []
    for name in embeddings:
        if name == "graphmvp_data":
            continue                       # synthesized in collate from SMILES
        if name == "molformer_tokens":
            if tokens_parquet is None:
                raise RuntimeError("molformer_tokens required but no tokens parquet built")
            blocks.append(build_modality_block(smiles, name, tokens_parquet))
        else:
            raise NotImplementedError(f"modality {name!r} not supported at inference")
    if blocks:
        X_d, slices, _dims = stack_modality_blocks(blocks)
    else:
        X_d, slices = None, {}
    # These recipes carry no descriptors (all ten models use descriptors: []), so
    # every stacked block is an embedding.
    embedding_slices = dict(slices)
    descriptor_slices: dict = {}

    n_out = len(lm.ordered_tasks)
    dummy_y = np.full((len(smiles), n_out), np.nan, dtype=np.float32)
    dset = _build_chemprop_dataset(smiles, dummy_y, X_d, n_workers=0)

    from qadmet.datasets.loaders import (  # type: ignore[attr-defined]
        _build_torch_loader, _ChempropSmilesDataset, make_collate_fn,
        make_collate_fn_with_graphmvp,
    )

    # Mirror prepare_fold_loaders' collate selection: a synthetic modality
    # (graphmvp_data) needs the SMILES-carrying dataset wrapper and the PyG collate.
    if "graphmvp_data" in embeddings:
        collate = make_collate_fn_with_graphmvp(descriptor_slices, embedding_slices,
                                                None, "graphmvp_data")
        dset = _ChempropSmilesDataset(dset, smiles)
    else:
        collate = make_collate_fn(descriptor_slices, embedding_slices)

    loader = _build_torch_loader(dset, batch_size=batch_size, num_workers=0,
                                shuffle=False, seed=None, pin_memory=False,
                                collate_fn=collate)

    device = next(lm.model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            outs.append(lm.model(_to_device(batch, device)).float().cpu().numpy())
    raw = np.concatenate(outs, axis=0)[:, lm.task_index].astype(np.float64)
    return _postprocess(raw, lm)


def predict_smiles(smiles: list[str], *, weights_root: str | Path,
                   recipes_root: str | Path, repo_root: str | Path = ".",
                   tasks: list[str] | None = None, seeds=(0, 1, 2, 3, 4),
                   members=ENSEMBLE, device: str = "cpu",
                   batch_size: int = 32) -> pd.DataFrame:
    """Ensemble predictions for `smiles`.

    Returns a long DataFrame with one row per (smiles, task): the uniform-mean
    ensemble value plus each member's own seed-averaged value, so a surprising
    ensemble number can be traced to the member that produced it.
    """
    names, types, _ = _task_meta()
    type_map = dict(zip(names, types))
    tasks = list(tasks) if tasks else names

    tmpdir = Path(tempfile.mkdtemp(prefix="tetrafuse_infer_"))
    tokens = None
    if any("molformer_tokens" in _member_embeddings(m, recipes_root) for m in members):
        tokens = _tokens_parquet(smiles, tmpdir / "molformer_tokens.parquet")

    rows = []
    for task in tasks:
        per_member = {}
        for member in members:
            vals = []
            for seed in seeds:
                lm = load_member(member, task, seed, weights_root=weights_root,
                                 recipes_root=recipes_root, repo_root=repo_root,
                                 device=device)
                vals.append(_predict_one(lm, smiles, recipes_root=Path(recipes_root),
                                         repo_root=Path(repo_root),
                                         tokens_parquet=tokens, batch_size=batch_size))
                del lm
            per_member[member] = np.mean(vals, axis=0)
        ens = np.mean([per_member[m] for m in members], axis=0)
        for i, smi in enumerate(smiles):
            row = dict(smiles=smi, task=task, task_type=type_map[task],
                       tetrafuse=float(ens[i]))
            for m in members:
                row[m] = float(per_member[m][i])
            rows.append(row)
    return pd.DataFrame(rows)


def _member_embeddings(member: str, recipes_root: str | Path) -> tuple:
    from qadmet.core.recipe import load_recipe

    stem = MEMBER_RECIPES[member][0]
    spec = load_recipe(Path(recipes_root) / "train" / f"{stem}.yaml")
    return tuple(getattr(spec.features, "embeddings", ()) or ())
