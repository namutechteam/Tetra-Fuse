"""GFN2-xTB 실패 분자에 대한 binary missing-mask builder.

xtb builder 가 ETKDG 3D embed 실패 / xtb SCF 발산 / 타임아웃 등으로 실패한 분자는
xtb parquet 의 4 개 채널 (dipole/homo_lumo_gap/electrons/total_energy) 가 모두 NaN
이다. 본 builder 는 그 NaN 패턴을 그대로 4-dim binary 마스크로 노출 — 1.0 이면 해당
xtb 채널이 누락 (분자 단위로는 4 채널 모두 1 또는 모두 0 인 것이 일반적이지만, 부분
실패 케이스에 대비해 채널별로 보존).

Why:
    옛 phase1.3b (D-MPNN + RDKit2D + xTB) 는 xtb 실패 분자를 zeros 로 채워 학습했지만,
    "xtb 가 실패해서 0 이다" 라는 정보 자체를 모델이 모른다. 본 마스크를 descriptor
    축에 함께 concat 하면 backbone 이 "이 분자의 xtb 정보는 신뢰할 수 없다" 를
    명시적 신호로 받을 수 있다 (옛 missing_policy="mask_token" 의 변종).

데이터 의존성:
    xtb parquet 가 먼저 사전 계산돼야 한다. 기본 위치는
    ``{QADMET_DATA_DIR}/embeddings/xtb.parquet`` (= ``data/embeddings/xtb.parquet``).
    다른 위치는 builder 인스턴스화 시 ``xtb_parquet_path`` kwarg 로 override 가능
    (테스트 또는 비-표준 레이아웃 용).

Note:
    precompute runner 는 builder 를 ``FEATURE_BUILDERS.get(name)()`` (kwargs 없음)
    으로 인스턴스화하므로, 기본 path 는 ``__init__`` 에서 lazy 결정한다 (모듈 import
    시점에는 path 검사 안 함 — registry 등록만).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

# xtb builder 의 출력 컬럼 — 본 마스크의 채널 순서를 일치시킨다.
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
    """xtb 채널 4 종의 NaN 여부를 0/1 float 으로 노출하는 4-dim mask builder.

    Args:
        xtb_parquet_path: xtb builder 산출물 parquet. None 이면
            ``{QADMET_DATA_DIR}/embeddings/xtb.parquet`` 를 사용 — precompute 표준 위치.

    Example:
        >>> builder = GfnMissingVectorBuilder()
        >>> result = builder.compute_batch(["CCO", "FAILED_SMI"])
        >>> result["FAILED_SMI"]  # xtb 에 없으면 모두 1 (missing).
        array([1., 1., 1., 1.], dtype=float32)
    """

    name = "gfn_missing_vector"
    column_prefix = ""  # 사용 안 함 — columns 직접 지정.
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
        """SMILES 배치 → {smi: 4-dim 0/1 float vector}.

        ParquetStore.lookup 결과에서 NaN → 1.0 (missing), finite → 0.0.
        xtb parquet 에 SMILES 자체가 누락된 경우도 NaN 으로 반환되므로 자동으로 1.0.
        """
        from qadmet.features.stores import ParquetStore

        if not self.xtb_parquet_path.exists():
            raise FileNotFoundError(
                f"xtb parquet 누락: {self.xtb_parquet_path}\n"
                f"  먼저 ``qadmet precompute --recipe recipes/precompute/xtb.yaml`` 로 "
                f"xtb 를 사전 계산하세요."
            )
        store = ParquetStore(self.xtb_parquet_path)
        arr = store.lookup(list(smiles), list(XTB_DESCRIPTOR_COLS))  # (N, 4) float64.
        mask = np.isnan(arr).astype(np.float32)
        out: dict[str, np.ndarray] = {}
        for i, smi in enumerate(smiles):
            out[smi] = mask[i].copy()
        return out
