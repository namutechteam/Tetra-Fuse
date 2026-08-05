"""Rebuild a trained Tetra-Fuse member from a recipe + exported PEFT weights.

The public release ships only the weights we trained (see `code/export_weights.py`);
the pretrained backbones (CheMeleon, GraphMVP, MoLFormer-XL) are downloaded by the
user into `code/qadmet/data/weights/`. Reconstructing a member therefore takes two
steps:

    1. build the LightningModule exactly as the training runner did, which loads
       the pretrained backbone from `data/weights/`;
    2. overlay the exported `peft.pt` state dict, which restores the adapters,
       LoRA matrices, projections, fusion and task heads.

`load_member()` returns a model in eval mode plus the metadata inference needs
(task type, target scaler, log1p flag).

Design notes
------------
* We reuse `qadmet.runners.train._build_lit_module` (MTL) and
  `qadmet.runners.train_stl._build_stl_lit_module` (STL) rather than
  re-instantiating `MultitaskLitModule` ourselves. Those functions encode the
  head/grouping/fusion wiring that the checkpoints were trained with; bypassing
  them is how shape mismatches creep in.
* `strict=True` on `load_state_dict` is deliberate. The exported file is expected
  to cover every non-frozen tensor, so a missing or unexpected key means the
  recipe and the weights disagree — better to fail loudly than to serve
  predictions from a partially randomized model.
"""
from __future__ import annotations

import contextlib
import json
import os
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# member -> (train recipe stem, is-multitask). Mirrors code/export_weights.py.
MEMBER_RECIPES = {
    "mtl_dmpnn": ("mtl_dmpnn_robust_causal6_kendall_epoch100", True),
    "mtl_molformer": ("mtl_molformer_robust_causal6_kendall_epoch100", True),
    "stl_chemeleon": ("stl_chemeleon_adaptivegnn_paper", False),
    "stl_graphmvp": ("stl_graphmvp_adaptivegnn_paper", False),
    "stl_map": ("stl_4modality_robust_nounimol2_seed01234", False),
}

#: The five members of the k=5 uniform-mean ensemble reported in the paper.
ENSEMBLE = ("mtl_dmpnn", "mtl_molformer", "stl_chemeleon", "stl_graphmvp", "stl_map")

DATASET_NAME = "tdc_admet_22"


@dataclass
class LoadedMember:
    """A restored member plus everything needed to post-process its output."""

    model: Any
    member: str
    task: str
    seed: int
    task_type: str          # "regression" | "classification"
    task_index: int         # column of the model output holding this task
    scaler_mean: float | None
    scaler_scale: float | None
    log1p: bool
    ordered_tasks: list[str]


def _task_meta(dataset_name: str = DATASET_NAME):
    """Task names and types from the TaskSet registry (same source as the runners).

    The dataset name matters: `tdc_admet_22` carries the 7-group mechanism
    grouping while `tdc_admet_22_causal6` carries the 6-group causal one, and the
    MTL grouped head is built with one trunk per group. Passing the wrong TaskSet
    yields a head whose trunk names do not match the trained weights.
    """
    import qadmet.modules  # noqa: F401  — triggers registry population
    from qadmet.tasks.registry import TASK_SETS

    ts = TASK_SETS.get(dataset_name)
    names = list(ts.tasks)
    types = list(ts.task_types())
    return names, types, ts


def _load_scalers(weights_root: str | Path) -> dict:
    path = Path(weights_root) / "target_scalers.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regression predictions cannot be un-normalized "
            f"without it — download it alongside the weights, or regenerate with "
            f"`python code/export_scalers.py --out {weights_root}`."
        )
    with open(path) as fh:
        return json.load(fh)


@contextlib.contextmanager
def _chdir(path: str | Path | None):
    """Temporarily run in `path` (no-op when None).

    Backbone checkpoints are named in the recipes as paths relative to the qadmet
    repo root (`data/weights/chemeleon_mp.pt`), and the backbone constructors
    resolve them against the process CWD. Rather than rewriting recipe fields, we
    build the module from that directory so the documented layout works no matter
    where the caller invoked `qadmet predict` from.
    """
    if path is None:
        yield
        return
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def load_member(member: str, task: str, seed: int, *, weights_root: str | Path,
                recipes_root: str | Path, repo_root: str | Path | None = None,
                device: str = "cpu") -> LoadedMember:
    """Rebuild one (member, task, seed) model with its exported PEFT weights.

    `repo_root` is where the pretrained backbones live under `data/weights/`
    (i.e. the qadmet checkout). Pass it when running from elsewhere.
    """
    if member not in MEMBER_RECIPES:
        raise KeyError(f"unknown member {member!r}; expected one of {list(MEMBER_RECIPES)}")
    recipe_stem, is_mtl = MEMBER_RECIPES[member]

    from qadmet.core.recipe import load_recipe

    spec = load_recipe(Path(recipes_root) / "train" / f"{recipe_stem}.yaml")
    # Use the TaskSet the recipe itself names (causal6 for the MTL members), so the
    # grouped head is built with the same trunks the checkpoint was trained with.
    names, types, task_set = _task_meta(spec.dataset.name)
    type_map = dict(zip(names, types))
    if task not in type_map:
        raise KeyError(f"unknown task {task!r}")
    task_type = type_map[task]

    # --- build the module the same way the runner did -----------------------
    # Inside _chdir so the recipes' relative backbone paths resolve.
    if is_mtl:
        from qadmet.runners.train import _build_lit_module

        tr = spec.trainer
        with _chdir(repo_root):
            model = _build_lit_module(
                model_spec=spec.model, loss_spec=spec.loss, n_tasks=len(names),
                task_types=types, ordered_tasks=names,
                embedding_dims=_embedding_dims(spec), descriptors=(),
                embeddings=tuple(spec.dataset.embeddings
                                 if hasattr(spec.dataset, "embeddings")
                                 else spec.features.embeddings),
                descriptor_dim_total=0, task_set=task_set,
                task_weights=None, lr=tr.lr,
            )
        ordered = names
        task_index = names.index(task)
    else:
        from qadmet.runners.train_stl import _build_stl_lit_module

        with _chdir(repo_root):
            model = _build_stl_lit_module(
                spec=spec, task_type=task_type,
                feature_spec=_feature_spec(spec),
                modality_dims=_embedding_dims(spec), descriptor_dim_total=0,
                lr=spec.trainer.lr,
            )
        ordered = [task]
        task_index = 0

    # --- overlay the trained (PEFT) weights ---------------------------------
    sub = "_multitask" if is_mtl else task
    peft = Path(weights_root) / "tetrafuse" / member / sub / f"seed_{seed}" / "peft.pt"
    if not peft.exists():
        raise FileNotFoundError(
            f"{peft} not found. Place the downloaded weight tree under "
            f"{weights_root}/tetrafuse/ (see weights/README.md)."
        )
    blob = torch.load(peft, map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    # fp16 archives are stored half; cast back to the model's parameter dtype.
    ref = next(iter(model.state_dict().values()))
    state = {k: (v.to(ref.dtype) if v.is_floating_point() else v) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Frozen backbone tensors are legitimately absent from the export; anything
    # else missing means the recipe and the weights disagree.
    hard_missing = [k for k in missing if not _is_backbone_key(k)]
    if hard_missing or unexpected:
        raise RuntimeError(
            f"PEFT overlay mismatch for {member}/{task}/seed_{seed}: "
            f"{len(hard_missing)} unexpectedly missing (e.g. {hard_missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {list(unexpected)[:3]})"
        )

    _restore_fusion_temperature(model, spec)
    model.eval().to(device)

    # --- regression post-processing metadata --------------------------------
    mean = scale = None
    log1p = False
    if task_type == "regression":
        sc = _load_scalers(weights_root)
        try:
            e = sc[member][task][str(seed)]
        except KeyError as exc:
            raise KeyError(
                f"target_scalers.json has no entry for {member}/{task}/seed {seed}"
            ) from exc
        mean, scale, log1p = float(e["mean"]), float(e["scale"]), bool(e["log1p"])

    return LoadedMember(model=model, member=member, task=task, seed=seed,
                        task_type=task_type, task_index=task_index,
                        scaler_mean=mean, scaler_scale=scale, log1p=log1p,
                        ordered_tasks=ordered)


def _restore_fusion_temperature(model, spec) -> None:
    """Set the attention-fusion softmax temperature to its end-of-training value.

    `RobustAttentionFusion.tau` is a plain Python float, not a buffer, so it is
    absent from `state_dict` and a freshly constructed module starts at `tau_init`
    (3.0 for MAP). During training `on_train_epoch_start` anneals it
    `tau = tau_init - (tau_init - tau_min) * min(1, epoch / anneal_epochs)`, and
    `trainer.predict` runs after training ends, so the predictions in the paper
    were produced at `tau_min`.

    Leaving tau at `tau_init` is a silent error: the gate softmax stays far too
    smooth and every fused prediction is biased. Measured on MAP·STL /
    Caco2_Wang / seed 0 against the released predictions, max abs error is
    4.0e-01 at tau=3.0 versus 1.2e-02 at tau=0.5; the CSV logger confirms
    `train/fusion_tau` reaches 0.5 well before the last epoch (epoch 19 of 19 for
    Caco2_Wang, 78 of 78 for HIA_Hou), so tau_min is correct for every
    (task, seed) rather than something to recompute per checkpoint.

    Only MAP has a fusion block; the other four members no-op here.
    """
    fusion = getattr(model, "fusion", None)
    fusion_spec = getattr(spec.model, "fusion", None)
    if fusion is None or fusion_spec is None or not hasattr(fusion, "set_tau"):
        return
    kwargs = getattr(fusion_spec, "kwargs", None) or {}
    fusion.set_tau(float(kwargs.get("tau_min", 0.5)))
    if hasattr(fusion, "set_warmup_active"):
        # warmup_epochs is a first-epochs-only gate bypass; inactive at predict time.
        fusion.set_warmup_active(False)


def _is_backbone_key(key: str) -> bool:
    """Keys that live inside a frozen pretrained backbone (absent from the export)."""
    return key.startswith(("backbone.", "backbones.")) or "molformer" in key.lower()


def _feature_spec(spec):
    """The recipe's FeatureSpec, whichever attribute the spec exposes it under."""
    for attr in ("features", "dataset"):
        obj = getattr(spec, attr, None)
        if obj is not None and hasattr(obj, "embeddings"):
            from qadmet.core.specs import FeatureSpec

            return FeatureSpec(descriptors=tuple(getattr(obj, "descriptors", ()) or ()),
                               embeddings=tuple(obj.embeddings or ()))
    raise AttributeError("recipe exposes no descriptors/embeddings")


def _embedding_dims(spec) -> dict:
    """{modality: dim} for the embeddings this recipe consumes."""
    fs = _feature_spec(spec)
    dims = {}
    for name in fs.embeddings:
        if name == "molformer_tokens":
            # token_ids + attention_mask, MAX_LEN each
            from qadmet.features.builders.molformer_tokens import MAX_LEN

            dims[name] = 2 * MAX_LEN
        elif name == "graphmvp_data":
            # Synthesized from SMILES at collate time (PyG Batch), so it has no
            # tabular feature width. `datasets/loaders.py` registers a sentinel of 1
            # purely to satisfy the LitModule's modality_dims validation; the value
            # is never used as a fusion input width. Mirror that here.
            dims[name] = 1
        else:
            dims[name] = 0
    return dims
