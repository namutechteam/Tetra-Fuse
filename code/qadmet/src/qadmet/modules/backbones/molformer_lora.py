"""MoLFormerLoRABackbone — IBM MoLFormer-XL + PEFT LoRA fine-tune.

Old ``run_stl_molformer_lora.MolFormerSTL`` and ``GroupedTaskUncertaintyMPNN``
(molformer_lora_enabled branch, pipeline_mtl line 933-948) LoRA used identically in
configuration separated into a single backbone.

Why LoRA target_modules=["query", "value"]:
    legacy, same as old code. Only attention's q/v projection is trained; base is frozen.

Lazy import:
    transformers / peft trigger heavy dependency loading on import, so only inside ``__init__``
    is the import performed. It is not imported when the registration decorator executes.
"""

from __future__ import annotations

from torch import Tensor, nn

from .registry import BACKBONES


MOLFORMER_MODEL_ID_DEFAULT = "ibm/MoLFormer-XL-both-10pct"
# HF hub modeling_molformer.py at the latest commit is exclusive to transformers>=new version
# imports create_bidirectional_mask → incompatible with 4.57.x. Pin to a verified commit.
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
MOLFORMER_DIM = 768


@BACKBONES.register("molformer_lora")
class MoLFormerLoRABackbone(nn.Module):
    """MoLFormer-XL pretrained + LoRA wrapper. forward → (B, 768) pooler.

    Args:
        model_id: HuggingFace Hub id or local path. Default ``ibm/MoLFormer-XL-both-10pct``.
        lora_rank: LoRA rank r (default 16).
        lora_alpha: LoRA alpha (default 32).
        lora_dropout: LoRA dropout (default 0.1).
        target_modules: Names of modules to apply LoRA to (default ``("query", "value")``).
        deterministic_eval: HF model deterministic mode (default True).

    Note:
        token_ids/attention_mask shape is (B, L). MoLFormer's old token length is 202
        (``MOLFORMER_TOKEN_LEN``). This backbone does not depend on length.
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
