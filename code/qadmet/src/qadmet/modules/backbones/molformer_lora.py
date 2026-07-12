"""MoLFormerLoRABackbone — IBM MoLFormer-XL + PEFT LoRA fine-tune.

옛 ``run_stl_molformer_lora.MolFormerSTL`` 과 ``GroupedTaskUncertaintyMPNN``
(molformer_lora_enabled 분기, pipeline_mtl line 933-948) 에서 동일하게 사용된 LoRA
설정을 단일 backbone 으로 분리한다.

Why LoRA target_modules=["query", "value"]:
    legacy 옛 코드와 동일. attention 의 q/v projection 만 학습하고 base 는 frozen.

Lazy import:
    transformers / peft 는 import 시 큰 의존성 로딩이 일어나므로 ``__init__`` 안에서만
    import 한다. 등록 데코레이터 동작 시점에는 import 되지 않는다.
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


MOLFORMER_MODEL_ID_DEFAULT = "ibm/MoLFormer-XL-both-10pct"
# HF hub 최신 커밋의 modeling_molformer.py 가 transformers>=신버전 전용
# create_bidirectional_mask 를 import → 4.57.x 와 비호환. 검증된 커밋으로 pin.
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
MOLFORMER_DIM = 768


@BACKBONES.register("molformer_lora")
class MoLFormerLoRABackbone(nn.Module):
    """MoLFormer-XL pretrained + LoRA wrapper. forward → (B, 768) pooler.

    Args:
        model_id: HuggingFace Hub id 또는 로컬 경로. 기본 ``ibm/MoLFormer-XL-both-10pct``.
        lora_rank: LoRA rank r (기본 16).
        lora_alpha: LoRA alpha (기본 32).
        lora_dropout: LoRA dropout (기본 0.1).
        target_modules: LoRA 적용 모듈 이름 (기본 ``("query", "value")``).
        deterministic_eval: HF 모델 deterministic mode (기본 True).

    Note:
        token_ids/attention_mask shape 은 (B, L). MoLFormer 옛 토큰 길이는 202
        (``MOLFORMER_TOKEN_LEN``). 본 backbone 은 길이에 의존하지 않는다.
    """

    output_dim: int = MOLFORMER_DIM

    def __init__(
        self,
        model_id: str = MOLFORMER_MODEL_ID_DEFAULT,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: tuple[str, ...] = ("query", "value"),
        deterministic_eval: bool = True,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        base = AutoModel.from_pretrained(
            model_id,
            revision=MOLFORMER_REVISION,
            deterministic_eval=deterministic_eval,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.molformer = get_peft_model(base, lora_cfg)

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.molformer(input_ids=token_ids, attention_mask=attention_mask)
        return out.pooler_output  # (B, 768)
