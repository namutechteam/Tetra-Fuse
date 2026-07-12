"""RDKit 2D descriptor builder (descriptastorus 200-dim).

옛 ``scripts-old/precompute_rdkit2d.py`` 의 핵심 로직을 ``PrecomputeBase`` 위에 재구성.
worker subprocess 격리 (한 분자가 worker 를 crash 시켜도 main 생존) 패턴은 유지.
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

RDKIT_DIM = 200

# 모듈-전역: subprocess worker 가 fork 시 공유. main 에서는 lazy init.
_GENERATOR = None


def _worker_init() -> None:
    """spawn worker 에서 descriptastorus generator 한 번만 만든다.

    Why:
        descriptastorus 는 main 에서 import 시 multiprocess 경고 출력. worker 에서만 import.
    """
    global _GENERATOR
    from descriptastorus.descriptors import rdDescriptors  # type: ignore

    _GENERATOR = rdDescriptors.RDKit2D()


def _worker_compute(smi: str) -> tuple[str, np.ndarray]:
    """SMILES → (smi, 200-dim vector). 실패 시 zeros."""
    global _GENERATOR
    try:
        out = _GENERATOR.process(smi)
        if out is None or len(out) < RDKIT_DIM + 1:
            return smi, np.zeros(RDKIT_DIM, dtype=np.float32)
        vec = np.asarray(out[1 : RDKIT_DIM + 1], dtype=np.float64)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, -1e6, 1e6)
        return smi, vec.astype(np.float32)
    except Exception:
        return smi, np.zeros(RDKIT_DIM, dtype=np.float32)


@FEATURE_BUILDERS.register("rdkit2d")
class RDKit2DBuilder(PrecomputeBase):
    """descriptastorus.RDKit2D 200-dim. spawn-pool 격리."""

    name = "rdkit2d"
    column_prefix = "rd_"
    output_dim = RDKIT_DIM

    def __init__(self, n_jobs: int = 16, chunksize: int = 100) -> None:
        self.n_jobs = n_jobs
        self.chunksize = chunksize

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        ctx = get_context("spawn")
        out: dict[str, np.ndarray] = {}
        with ctx.Pool(self.n_jobs, initializer=_worker_init) as pool:
            for smi, vec in pool.imap_unordered(
                _worker_compute, list(smiles), chunksize=self.chunksize
            ):
                out[smi] = vec
        return out
