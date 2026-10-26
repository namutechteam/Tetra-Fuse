"""RDKit 2D descriptor builder (descriptastorus 200-dim).

Rebuild the core logic of old ``scripts-old/precompute_rdkit2d.py`` on top of ``PrecomputeBase``.
Maintain the worker subprocess isolation pattern (main survives even if a molecule crashes the worker).
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

RDKIT_DIM = 200

# Module-global: shared when subprocess worker forks. Lazy init in main.
_GENERATOR = None


def _worker_init() -> None:
    """Create the descriptastorus generator only once in the spawn worker.

    Why:
        Importing descriptastorus in main prints a multiprocess warning. Import only in worker.
    """
    global _GENERATOR
    from descriptastorus.descriptors import rdDescriptors  # type: ignore

    _GENERATOR = rdDescriptors.RDKit2D()


def _worker_compute(smi: str) -> tuple[str, np.ndarray]:
    """SMILES → (smi, 200-dim vector). Returns zeros on failure."""
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
    """descriptastorus.RDKit2D 200-dim. Isolated via spawn-pool."""

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
