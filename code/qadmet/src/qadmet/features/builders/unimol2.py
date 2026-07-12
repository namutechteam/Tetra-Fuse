"""Uni-Mol2 (84M) frozen CLS embedding builder (768-dim).

옛 ``scripts-old/precompute_unimol2_embeddings.py`` 의 RDKit ETKDG → Uni-Mol2 forward
파이프라인을 ``PrecomputeBase`` 위에 재구성.

구조:
    multiprocessing pool (CPU) 가 ETKDG + MMFF + featurize 를 병렬 처리하고,
    main thread 는 GPU forward 로 cls_repr 를 받는다. 모델은 첫 ``compute_batch`` 호출
    시 lazy 로드 후 재사용.

Note:
    - vendored ``qadmet.modules.backbones._vendor.unimol_v2`` 를 사용한다 (scripts-old 의
      qw/backbones 와 byte-identical).
    - ETKDG 가 실패하면 모든 채널 0 으로 채운다 (옛 동작과 동일). 하류 missing_policy
      가 mask/drop 처리.
    - ckpt 경로는 ``data/weights/unimol2_84m/modelzoo/84M/checkpoint.pt`` 가 기본.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

logger = logging.getLogger(__name__)

EMBED_DIM = 768
MODEL_SIZE = "84m"
DEFAULT_CKPT = "data/weights/unimol2_84m/modelzoo/84M/checkpoint.pt"


def _featurize_one(smi_idx: tuple[int, str]) -> tuple[int, dict | None]:
    """Worker: SMILES → Uni-Mol2 feature dict (None on failure).

    Why:
        ETKDG embedding + MMFF optimize + featurize 는 분자별로 시간 편차가 크다 —
        worker process 분산으로 대기시간 평균화. featurize 는 vendored unimol_v2 의
        ``mol2unimolv2`` 를 사용한다.
    """
    idx, smi = smi_idx
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        from qadmet.modules.backbones._vendor import unimol_v2 as uv2

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return idx, None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 30
        if AllChem.EmbedMolecule(mol, params) < 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) < 0:
                return idx, None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass
        feat = uv2.mol2unimolv2(mol)
        return idx, feat
    except Exception:
        return idx, None


def _init_worker() -> None:
    """SIGINT 무시 — Ctrl+C 가 main 으로만 전파되도록."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@FEATURE_BUILDERS.register("unimol2")
class UniMol2Builder(PrecomputeBase):
    """Uni-Mol2 84M frozen CLS embedding (768-dim)."""

    name = "unimol2"
    column_prefix = "um_"
    output_dim = EMBED_DIM
    default_batch_size = 4096
    default_checkpoint_every = 4096

    def __init__(
        self,
        ckpt: str = DEFAULT_CKPT,
        gpu_batch_size: int = 32,
        n_workers: int = 8,
        device: str = "cuda:0",
        model_size: str = MODEL_SIZE,
    ) -> None:
        self.ckpt = ckpt
        self.gpu_batch_size = gpu_batch_size
        self.n_workers = n_workers
        self.device = device
        self.model_size = model_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        from qadmet.modules.backbones._vendor import unimol_v2 as uv2

        logger.info("building Uni-Mol2 %s model", self.model_size)
        model = uv2.UniMolV2Model(model_size=self.model_size).eval()
        info = uv2.load_unimol2_weights(model, self.ckpt)
        logger.info(
            "loaded weights: missing=%s unexpected=%s",
            len(info["missing"]), len(info["unexpected"]),
        )
        self._model = model.to(torch.device(self.device))

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        import torch

        from qadmet.modules.backbones._vendor import unimol_v2 as uv2

        self._ensure_loaded()
        smiles = list(smiles)
        out: dict[str, np.ndarray] = {}
        ctx = mp.get_context("spawn")
        device = torch.device(self.device)

        with ctx.Pool(processes=self.n_workers, initializer=_init_worker) as pool:
            it = pool.imap_unordered(_featurize_one, list(enumerate(smiles)), chunksize=4)
            pending: list[tuple[str, dict]] = []

            def flush_gpu() -> None:
                if not pending:
                    return
                feats = [p[1] for p in pending]
                batch = uv2.collate_unimolv2(feats)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    res = self._model(**batch)
                cls = res["cls_repr"].detach().cpu().numpy().astype(np.float32)
                for j, (smi, _) in enumerate(pending):
                    out[smi] = cls[j]
                pending.clear()

            for gi, feat in it:
                if feat is None:
                    # 실패 분자는 0 벡터 (옛 동작 유지). missing_policy 가 마스킹 담당.
                    out[smiles[gi]] = np.zeros(EMBED_DIM, dtype=np.float32)
                    continue
                pending.append((smiles[gi], feat))
                if len(pending) >= self.gpu_batch_size:
                    flush_gpu()
            flush_gpu()
        return out
