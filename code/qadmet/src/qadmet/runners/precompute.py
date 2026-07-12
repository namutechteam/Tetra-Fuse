"""precompute runner: FeatureSpec → builder → parquet artifact.

recipe 예::

    name: precompute_rdkit2d
    kind: precompute
    features:
      descriptors: [rdkit2d]
    notes: TDC ADMET 22 union SMILES 의 200-dim RDKit 2D descriptor

여러 descriptors/embeddings 를 한 recipe 에 나열하면 순차 실행한다 (각각 별도
artifact 로 저장). 같은 modality 를 여러 옵션(예: token len) 으로 만들고 싶다면
recipe 를 분리하는 게 명확하다.
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
    """task_set union SMILES. 외부 호출(인터넷 + tdc/HF 캐시) 필요.

    PXR 분기: ``pxr_*`` 접두사면 PXR HF dataset 의 train+test SMILES 를 반환.
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
    if spec.features is None:  # __post_init__ 가 잡지만 명시 가드.
        raise ValueError("precompute recipe 에 features 섹션이 필요합니다.")

    modalities = list(spec.features.descriptors) + list(spec.features.embeddings)
    if not modalities:
        raise ValueError("features.descriptors 또는 features.embeddings 가 비어있습니다.")

    artifact = store.create(spec, repo_root=repo_root)

    # dataset.name 으로 task_set 분기 (PXR vs TDC)
    task_set_name = spec.dataset.name if spec.dataset is not None else "tdc_admet_22"
    smiles = _collect_smiles(task_set=task_set_name)
    logger.info(
        "precompute start",
        extra={"payload": {"modalities": modalities, "n_smiles": len(smiles)}},
    )

    for mod in modalities:
        builder = FEATURE_BUILDERS.get(mod)()
        if getattr(builder, "output_format", "parquet") == "lmdb":
            # LMDB builder: 고정 경로에 출력 (artifact dir 밖)
            rel = getattr(builder, "default_output_relpath", f"data/embeddings/{mod}.lmdb")
            out_path = repo_root / rel
            builder.run(smiles, out_path, resume=True)
        else:
            out_parquet = artifact.data_dir / f"{mod}.parquet"
            builder.run(smiles, out_parquet, resume=True)
            store.register_output(artifact, f"{mod}.parquet")

    return artifact
