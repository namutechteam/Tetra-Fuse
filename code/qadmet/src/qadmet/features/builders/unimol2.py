"""Uni-Mol2 (84M) frozen CLS embedding builder (768-dim).

Legacy ``scripts-old/precompute_unimol2_embeddings.py``'s RDKit ETKDG → Uni-Mol2 forward
pipeline rebuilt on top of ``PrecomputeBase``.

Structure:
    multiprocessing pool (CPU) processes ETKDG + MMFF + featurize in parallel, and
    main thread receives cls_repr via GPU forward. The model, on the first ``compute_batch`` call,
    is lazily loaded and then reused.

Note:
    - uses vendored ``qadmet.modules.backbones._vendor.unimol_v2`` (byte-identical to scripts-old's
      qw/backbones).
    - if ETKDG fails, all channels are filled with 0 (same as legacy behavior). downstream missing_policy
      handles mask/drop.
    - ckpt path defaults to ``data/weights/unimol2_84m/modelzoo/84M/checkpoint.pt``.
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
        ETKDG embedding + MMFF optimize + featurize have large per-molecule time variance —
        worker process distribution averages out wait time. featurize uses vendored unimol_v2's
        ``mol2unimolv2``.
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
    """Ignore SIGINT — so that Ctrl+C only propagates to main."""
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
                    # failed molecules get a 0 vector (preserving legacy behavior). missing_policy handles masking.
                    out[smiles[gi]] = np.zeros(EMBED_DIM, dtype=np.float32)
                    continue
                pending.append((smiles[gi], feat))
                if len(pending) >= self.gpu_batch_size:
                    flush_gpu()
            flush_gpu()
        return out
