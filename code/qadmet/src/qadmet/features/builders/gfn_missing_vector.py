"""Binary missing-mask builder for molecules where GFN2-xTB failed.

Molecules that fail in the xtb builder due to ETKDG 3D embed failure / xtb SCF divergence / timeout, etc.
have all 4 channels of the xtb parquet (dipole/homo_lumo_gap/electrons/total_energy) as NaN
This builder exposes that NaN pattern directly as a 4-dim binary mask — 1.0 means the corresponding
xtb channel is missing (per molecule, all 4 channels being either all 1 or all 0 is typical, but for partial
failure cases, it is preserved per channel).

Why:
    The old phase1.3b (D-MPNN + RDKit2D + xTB) trained by filling xtb-failed molecules with zeros, but
    the model has no knowledge of the information "it is 0 because xtb failed". Concatenating this mask
    into the descriptor axis lets the backbone receive "this molecule's xtb information is unreliable"
    as an explicit signal (a variant of the old missing_policy="mask_token").

Data dependency:
    The xtb parquet must be precomputed first. The default location is
    ``{QADMET_DATA_DIR}/embeddings/xtb.parquet`` (= ``data/embeddings/xtb.parquet``).
    Other locations can be overridden via the ``xtb_parquet_path`` kwarg when instantiating the builder
    (for tests or non-standard layouts).

Note:
    Since the precompute runner instantiates the builder as ``FEATURE_BUILDERS.get(name)()`` (no kwargs)
    the default path is determined lazily in ``__init__`` (at module import
    time, the path is not checked — registry registration only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

# Output columns of the xtb builder — align this mask's channel order with them.
XTB_DESCRIPTOR_COLS: tuple[str, ...] = (
    "dipole",
    "homo_lumo_gap",
    "electrons",
    "total_energy",
)
MASK_COLUMNS: tuple[str, ...] = (
    "gfn_miss_dipole",
    "gfn_miss_gap",
    "gfn_miss_electrons",
    "gfn_miss_total_energy",
)


@FEATURE_BUILDERS.register("gfn_missing_vector")
class GfnMissingVectorBuilder(PrecomputeBase):
    """4-dim mask builder exposing NaN status of the 4 xtb channels as 0/1 float.

    Args:
        xtb_parquet_path: Parquet output of the xtb builder. If None,
            ``{QADMET_DATA_DIR}/embeddings/xtb.parquet`` — the standard precompute location.

    Example:
        >>> builder = GfnMissingVectorBuilder()
        >>> result = builder.compute_batch(["CCO", "FAILED_SMI"])
        >>> result["FAILED_SMI"]  # All 1 (missing) if not present in xtb.
        array([1., 1., 1., 1.], dtype=float32)
    """

    name = "gfn_missing_vector"
    column_prefix = ""  # Not used — columns specified directly.
    output_dim = len(MASK_COLUMNS)
    default_batch_size = 4096
    default_checkpoint_every = 50000

    def __init__(self, xtb_parquet_path: Path | str | None = None) -> None:
        if xtb_parquet_path is None:
            import os

            env = os.environ.get("QADMET_XTB_PARQUET")
            if env:
                xtb_parquet_path = env
            else:
                from qadmet.core import paths

                xtb_parquet_path = paths.data_dir() / "embeddings" / "xtb.parquet"
        self.xtb_parquet_path = Path(xtb_parquet_path)

    @property
    def columns(self) -> list[str]:
        return list(MASK_COLUMNS)

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        """SMILES batch → {smi: 4-dim 0/1 float vector}.

        In the ParquetStore.lookup result, NaN → 1.0 (missing), finite → 0.0.
        Cases where the SMILES itself is missing from the xtb parquet are also returned as NaN, hence automatically 1.0.
        """
        from qadmet.features.stores import ParquetStore

        if not self.xtb_parquet_path.exists():
            raise FileNotFoundError(
                f"xtb parquet missing: {self.xtb_parquet_path}\n"
                f"  First, using ``qadmet precompute --recipe recipes/precompute/xtb.yaml``, "
                f"precompute xtb."
            )
        store = ParquetStore(self.xtb_parquet_path)
        arr = store.lookup(list(smiles), list(XTB_DESCRIPTOR_COLS))  # (N, 4) float64.
        mask = np.isnan(arr).astype(np.float32)
        out: dict[str, np.ndarray] = {}
        for i, smi in enumerate(smiles):
            out[smi] = mask[i].copy()
        return out
