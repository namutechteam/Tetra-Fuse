"""UniMol2LoRABackbone — Uni-Mol2 84M + PEFT LoRA fine-tune.

옛 ``run_stl_unimol2_lora.UniMol2STL`` 의 LoRA 설정을 단일 backbone 으로 분리한다.
vendored ``_vendor.unimol_v2`` (옛 ``scripts-old/qw/backbones/unimol_v2.py`` 와 byte-identical)
를 사용한다.

Why target_modules=["linear_q", "linear_v"]:
    Uni-Mol2 의 attention 모듈명은 ``linear_q`` / ``linear_v`` 이다 (MoLFormer 와 다름).

Lazy import:
    peft 는 ``__init__`` 안에서만 import. vendored unimol_v2 는 도메인 특화 코드라
    이미 모듈 import 시 같이 로드되어도 큰 비용은 없으나 일관성을 위해 함수 안에서
    import 한다.

Checkpoint:
    실제 가중치 로딩은 ``ckpt_path`` 가 None 이 아닐 때만 수행된다. 테스트나 random
    초기화 비교에는 ``ckpt_path=None`` 으로 인스턴스화 가능.
    옛 경로: ``data/weights/unimol2_84m/modelzoo/84M/checkpoint.pt``.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

from .registry import BACKBONES

logger = logging.getLogger(__name__)


UNIMOL2_DIM = 768


@BACKBONES.register("unimol2_lora")
class UniMol2LoRABackbone(nn.Module):
    """Uni-Mol2 base + LoRA wrapper. forward 는 vendored 모델 dict 입력을 받아 cls_repr 반환.

    Args:
        ckpt_path: 사전학습 가중치 ``.pt`` 경로. None 이면 랜덤 초기화 (테스트용).
        model_size: vendored ``UniMolV2Model`` 의 size key (기본 ``"84m"``).
        lora_rank: LoRA r (기본 16).
        lora_alpha: LoRA alpha (기본 32).
        lora_dropout: LoRA dropout (기본 0.1).
        target_modules: LoRA 적용 모듈 (기본 ``("linear_q", "linear_v")``).
        strict_load: ``load_unimol2_weights(strict=...)`` 인자.

    Note:
        forward 는 ``collate_unimolv2`` 가 만든 dict 를 그대로 받는다.
        키 중 ``__y__`` / ``__smi__`` 같이 leading double-underscore 는 제외된다.
    """

    output_dim: int = UNIMOL2_DIM

    def __init__(
        self,
        ckpt_path: str | None = None,
        model_size: str = "84m",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: tuple[str, ...] = ("linear_q", "linear_v"),
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model

        from ._vendor import unimol_v2 as uv2

        base = uv2.UniMolV2Model(model_size=model_size)
        if ckpt_path is not None:
            info = uv2.load_unimol2_weights(base, ckpt_path, strict=strict_load)
            missing = info.get("missing", []) if isinstance(info, dict) else []
            if missing:
                logger.warning(
                    "UniMol2LoRABackbone: %d missing weight keys when loading %s",
                    len(missing), ckpt_path,
                )

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.unimol2 = get_peft_model(base, lora_cfg)

    def forward(self, **batch_dict) -> Tensor:
        kwargs = {k: v for k, v in batch_dict.items() if not k.startswith("__")}
        out = self.unimol2(**kwargs)
        return out["cls_repr"]  # (B, 768)
