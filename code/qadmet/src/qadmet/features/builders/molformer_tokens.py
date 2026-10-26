"""MoLFormer-XL token_ids + attention_mask precompute builder.

Restructures the logic of the old ``scripts-old/precompute_molformer_tokens.py`` on top of ``PrecomputeBase``.
To eliminate per-batch tokenize cost during LoRA fine-tune, (token_ids, attention_mask) is
saved as parquet.

Output columns: ``[smiles, tok_000..tok_{L-1}, msk_000..msk_{L-1}]`` (L = ``MAX_LEN``).
Values are int32 (tokenizer pad_token_id and 0/1 mask).

Note:
    No GPU needed. Tokenize only — multiprocessing isn't really needed either, so single thread.
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
    """MoLFormer token lookup parquet builder.

    The output vector per SMILES is an int32 concat of ``[token_ids (MAX_LEN,), attention_mask (MAX_LEN,)]``.
    Columns are directly specified as ``tok_xxx`` + ``msk_xxx``.
    """

    name = "molformer_tokens"
    column_prefix = ""  # Not used — columns specified directly.
    output_dim = 2 * MAX_LEN
    default_batch_size = 1024
    default_checkpoint_every = 4096

    def __init__(self, max_length: int = MAX_LEN, tokenize_batch: int = 256) -> None:
        if max_length != MAX_LEN:
            # The parquet column names are tied to class constants. Change max_length with care.
            logger.warning("max_length=%s != MAX_LEN(%s) — column names may become misaligned", max_length, MAX_LEN)
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
