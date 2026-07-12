"""MoLFormer-XL token_ids + attention_mask 사전 계산 builder.

옛 ``scripts-old/precompute_molformer_tokens.py`` 의 로직을 ``PrecomputeBase`` 위에 재구성.
LoRA fine-tune 시 매 batch tokenize 비용을 없애기 위해 (token_ids, attention_mask) 를
parquet 으로 저장한다.

출력 컬럼: ``[smiles, tok_000..tok_{L-1}, msk_000..msk_{L-1}]`` (L = ``MAX_LEN``).
값은 int32 (tokenizer pad_token_id 와 0/1 mask).

Note:
    GPU 가 필요 없다. tokenize 만 — 멀티프로세스도 굳이 필요 없어 단일 스레드.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

logger = logging.getLogger(__name__)

MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
MAX_LEN = 202


@FEATURE_BUILDERS.register("molformer_tokens")
class MolFormerTokensBuilder(PrecomputeBase):
    """MoLFormer 토큰 lookup parquet 빌더.

    한 SMILES 당 출력 벡터는 ``[token_ids (MAX_LEN,), attention_mask (MAX_LEN,)]`` 의
    int32 concat. 컬럼은 ``tok_xxx`` + ``msk_xxx`` 로 직접 지정한다.
    """

    name = "molformer_tokens"
    column_prefix = ""  # 사용 안 함 — columns 직접 지정.
    output_dim = 2 * MAX_LEN
    default_batch_size = 1024
    default_checkpoint_every = 4096

    def __init__(self, max_length: int = MAX_LEN, tokenize_batch: int = 256) -> None:
        if max_length != MAX_LEN:
            # parquet 컬럼 이름은 클래스 상수에 묶여 있다. max_length 변경은 신중히.
            logger.warning("max_length=%s != MAX_LEN(%s) — 컬럼 이름이 어긋날 수 있음", max_length, MAX_LEN)
        self.max_length = max_length
        self.tokenize_batch = tokenize_batch
        self._tokenizer = None

    @property
    def columns(self) -> list[str]:
        return [f"tok_{i:03d}" for i in range(self.max_length)] + [
            f"msk_{i:03d}" for i in range(self.max_length)
        ]

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer  # type: ignore

        logger.info("loading tokenizer %s", MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        out: dict[str, np.ndarray] = {}
        smiles = list(smiles)
        for start in range(0, len(smiles), self.tokenize_batch):
            chunk = smiles[start : start + self.tokenize_batch]
            enc = self._tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            tok = enc["input_ids"].astype(np.int32)
            msk = enc["attention_mask"].astype(np.int32)
            for i, s in enumerate(chunk):
                out[s] = np.concatenate([tok[i], msk[i]], axis=0)
        return out
