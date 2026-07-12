"""블록별 StandardScaler + clipping.

옛 ``fit_xd_scalers`` / ``apply_xd_scalers`` (scripts-old/pipeline_mtl.py:334-365) 의
일반화. RDKit/xTB/MoLFormer/UniMol2 가 한 X_d 행렬에 concat 되어 있을 때, 각 블록
인덱스 범위를 받아 블록 단위로 fit·transform·clip 한다.

Why:
    block 단위 scaling 이 필요한 이유는 (a) 각 모달리티의 분포가 매우 다르고
    (b) mask 채널은 0/1 이므로 scaling 대상이 아니기 때문. 옛 코드는 슬라이스가
    하드코딩되어 있었으나 여기서는 Spec 으로 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BlockSpec:
    """X_d 의 한 슬라이스 정의.

    Attributes:
        name: 블록 이름 (예: "rdkit", "xtb", "molformer").
        start: 시작 인덱스 (포함).
        end: 끝 인덱스 (제외).
        clip: 학습 안정성을 위한 ±σ clip 값. None 이면 clip 안 함.
        observed_only: 관측 행만으로 fit (예: xTB 처럼 결측 row 가 0 으로 imputed 된 경우).
    """

    name: str
    start: int
    end: int
    clip: float | None = 10.0
    observed_only: bool = False
    mask_block: tuple[int, int] | None = None  # observed_only=True 일 때 mask 슬라이스.


class BlockScaler:
    """블록별 StandardScaler 컨테이너. fit / transform 두 단계 분리."""

    def __init__(self, blocks: list[BlockSpec]) -> None:
        self.blocks = blocks
        self._scalers: dict[str, StandardScaler] = {}

    def fit(self, train_xd: np.ndarray) -> "BlockScaler":
        for b in self.blocks:
            block = train_xd[:, b.start : b.end]
            if b.observed_only:
                if b.mask_block is None:
                    raise ValueError(f"BlockSpec {b.name!r}: observed_only=True 면 mask_block 필수")
                ms, me = b.mask_block
                mask = train_xd[:, ms:me]
                is_obs = (mask == 0).all(axis=1)
                if is_obs.sum() < 2:
                    # fallback: 전체로 fit (옛 코드와 동일).
                    s = StandardScaler().fit(block)
                else:
                    s = StandardScaler().fit(block[is_obs])
            else:
                s = StandardScaler().fit(block)
            self._scalers[b.name] = s
        return self

    def transform(self, xd: np.ndarray) -> np.ndarray:
        out = xd.astype(np.float64).copy()
        for b in self.blocks:
            s = self._scalers[b.name]
            out[:, b.start : b.end] = s.transform(xd[:, b.start : b.end])
            if b.clip is not None:
                out[:, b.start : b.end] = np.clip(out[:, b.start : b.end], -b.clip, b.clip)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)

    def fit_transform(self, train_xd: np.ndarray) -> np.ndarray:
        return self.fit(train_xd).transform(train_xd)

    def state(self) -> Mapping[str, dict]:
        """체크포인트 동봉용 직렬화 dict."""
        return {
            name: {"mean": s.mean_.tolist(), "scale": s.scale_.tolist(), "var": s.var_.tolist()}
            for name, s in self._scalers.items()
        }


def fit_block_scalers(train_xd: np.ndarray, blocks: list[BlockSpec]) -> BlockScaler:
    """편의 함수: BlockScaler 한 줄 생성."""
    return BlockScaler(blocks).fit(train_xd)
