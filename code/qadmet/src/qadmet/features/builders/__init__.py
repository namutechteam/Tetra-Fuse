"""Each modality builder module registers itself to ``FEATURE_BUILDERS`` at import-time.

Importing this package registers all 7 builders (rdkit2d / xtb / gfn_missing_vector / molformer
/ molformer_tokens / unimol2 / unimol2_lmdb). heavy dependencies
(descriptastorus / xtb binary / transformers / unimol_v2) are only imported inside the builder class's
methods, so the import-time cost is light.
"""

from qadmet.features.builders import (  # noqa: F401  — import-time registration
    chemeleon_frozen,
    gfn_missing_vector,
    graphmvp_frozen,
    molformer,
    molformer_tokens,
    rdkit2d,
    unimol2,
    unimol2_lmdb,
    xtb,
)
