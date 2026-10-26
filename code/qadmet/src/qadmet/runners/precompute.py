"""precompute runner: FeatureSpec → builder → parquet artifact.

recipe example::

    name: precompute_rdkit2d
    kind: precompute
    features:
      descriptors: [rdkit2d]
    notes: 200-dim RDKit 2D descriptor of TDC ADMET 22 union SMILES

Listing multiple descriptors/embeddings in one recipe runs them sequentially (each separate
artifact). If you want to create the same modality with multiple options (e.g. token len),
it is clearer to separate the recipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qadmet.core.artifacts import Artifact, ArtifactStore
from qadmet.core.experiment import RUNNER_REGISTRY
from qadmet.core.logging import get_logger
from qadmet.core.specs import ExperimentSpec
from qadmet.features.registry import FEATURE_BUILDERS

logger = get_logger("qadmet.runners.precompute")


def _collect_smiles(seeds: Sequence[int] = (0,), task_set: str = "tdc_admet_22") -> list[str]:
    """task_set union SMILES. Requires external calls (internet + tdc/HF cache).

    PXR branch: if prefixed with ``pxr_*``, returns train+test SMILES from the PXR HF dataset.
    """
    if task_set.startswith("polaris_fang"):
        from qadmet.datasets.polaris_fang import collect_polaris_fang_smiles
        return collect_polaris_fang_smiles(task_set, seeds=seeds)
    if task_set.startswith("pxr_"):
        from qadmet.datasets.pxr_activity import collect_pxr_smiles
        return collect_pxr_smiles(task_set)
    from qadmet.datasets.tdc_admet import collect_smiles
    return collect_smiles(task_set, seeds=seeds)


@RUNNER_REGISTRY.register("precompute")
def run_precompute(
    *, spec: ExperimentSpec, store: ArtifactStore, repo_root: Path
) -> Artifact:
    spec.validate()
    if spec.features is None:  # __post_init__ catches this but explicit guard.
        raise ValueError("precompute recipe requires a features section.")

    modalities = list(spec.features.descriptors) + list(spec.features.embeddings)
    if not modalities:
        raise ValueError("features.descriptors or features.embeddings is empty.")

    artifact = store.create(spec, repo_root=repo_root)

    # Branch task_set by dataset.name (PXR vs TDC)
    task_set_name = spec.dataset.name if spec.dataset is not None else "tdc_admet_22"
    smiles = _collect_smiles(task_set=task_set_name)
    logger.info(
        "precompute start",
        extra={"payload": {"modalities": modalities, "n_smiles": len(smiles)}},
    )

    for mod in modalities:
        builder = FEATURE_BUILDERS.get(mod)()
        if getattr(builder, "output_format", "parquet") == "lmdb":
            # LMDB builder: outputs to fixed path (outside artifact dir)
            rel = getattr(builder, "default_output_relpath", f"data/embeddings/{mod}.lmdb")
            out_path = repo_root / rel
            builder.run(smiles, out_path, resume=True)
        else:
            out_parquet = artifact.data_dir / f"{mod}.parquet"
            builder.run(smiles, out_parquet, resume=True)
            store.register_output(artifact, f"{mod}.parquet")

    return artifact
