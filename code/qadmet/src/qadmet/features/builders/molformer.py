"""MoLFormer-XL frozen embedding builder (768-dim pooler_output).

옛 ``scripts-old/precompute_molformer.py`` 의 핵심 로직을 ``PrecomputeBase`` 위에 재구성.
- transformers ``AutoModel.from_pretrained(..., deterministic_eval=True, trust_remote_code=True)``.
- 모델·토크나이저는 첫 ``compute_batch`` 호출 시 lazy 초기화 (멀티 batch 재사용).

Reference:
    Ross et al., "Large-Scale Chemical Language Representations Capture
    Molecular Structure and Properties", Nature Machine Intelligence 2022.
    HuggingFace: ibm/MoLFormer-XL-both-10pct
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from qadmet.features.registry import FEATURE_BUILDERS
from qadmet.precompute.base import PrecomputeBase

logger = logging.getLogger(__name__)

MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
EMBED_DIM = 768
MAX_LEN = 202


@FEATURE_BUILDERS.register("molformer")
class MolFormerBuilder(PrecomputeBase):
    """MoLFormer-XL frozen 768-dim pooler_output."""

    name = "molformer"
    column_prefix = "mf_"
    output_dim = EMBED_DIM
    default_batch_size = 4096  # 외부 batch — 내부에서 다시 gpu_batch_size 로 쪼갬.
    default_checkpoint_every = 4096

    def __init__(
        self,
        gpu_batch_size: int = 64,
        device: str = "cuda",
        max_length: int = MAX_LEN,
    ) -> None:
        self.gpu_batch_size = gpu_batch_size
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  — device 결정에 필요.
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        logger.info("loading %s", MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID, deterministic_eval=True, trust_remote_code=True,
        ).eval()
        self._model = model.to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True,
        )

    def compute_batch(self, smiles: Sequence[str]) -> dict[str, np.ndarray]:
        import torch

        self._ensure_loaded()
        out: dict[str, np.ndarray] = {}
        smiles = list(smiles)
        for start in range(0, len(smiles), self.gpu_batch_size):
            chunk = smiles[start : start + self.gpu_batch_size]
            inputs = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                fwd = self._model(**inputs)
            arr = fwd.pooler_output.detach().cpu().numpy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for s, vec in zip(chunk, arr):
                out[s] = vec
        return out
