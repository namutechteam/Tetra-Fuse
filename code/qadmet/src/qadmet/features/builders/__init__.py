"""각 modality builder 모듈은 import-time 에 ``FEATURE_BUILDERS`` 에 등록된다.

이 패키지를 import 하면 7 개 builder (rdkit2d / xtb / gfn_missing_vector / molformer
/ molformer_tokens / unimol2 / unimol2_lmdb) 가 모두 등록된다. heavy 의존성
(descriptastorus / xtb 바이너리 / transformers / unimol_v2)은 builder 클래스의
메서드 안에서만 import 하므로 import-time cost 는 가볍다.
"""

from qadmet.features.builders import (  # noqa: F401  — import-time 등록
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
