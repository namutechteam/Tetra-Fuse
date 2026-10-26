"""UniMol2LoRABackbone — Uni-Mol2 84M + PEFT LoRA fine-tune.

Separates the LoRA config of the old ``run_stl_unimol2_lora.UniMol2STL`` into a single backbone.
vendored ``_vendor.unimol_v2`` (byte-identical to the old ``scripts-old/qw/backbones/unimol_v2.py``)
uses it.

Why target_modules=["linear_q", "linear_v"]:
    Uni-Mol2's attention module names are ``linear_q`` / ``linear_v`` (different from MoLFormer).

Lazy import:
    peft is imported only inside ``__init__``. Since vendored unimol_v2 is domain-specific code,
    there is little cost even if it is loaded together at module import time, but for consistency it is
    imported inside the function.

Checkpoint:
    Actual weight loading is performed only when ``ckpt_path`` is not None. For tests or random
    initialization comparisons, it can be instantiated with ``ckpt_path=None``.
    Old path: ``data/weights/unimol2_84m/modelzoo/84M/checkpoint.pt``.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

from .registry import BACKBONES

logger = logging.getLogger(__name__)


UNIMOL2_DIM = 768


@BACKBONES.register("unimol2_lora")
class UniMol2LoRABackbone(nn.Module):
    """Uni-Mol2 base + LoRA wrapper. forward takes vendored model dict input and returns cls_repr.

    Args:
        ckpt_path: Pretrained weights ``.pt`` path. If None, random initialization (for testing).
        model_size: size key of vendored ``UniMolV2Model`` (default ``"84m"``).
        lora_rank: LoRA r (default 16).
        lora_alpha: LoRA alpha (default 32).
        lora_dropout: LoRA dropout (default 0.1).
        target_modules: Modules to apply LoRA to (default ``("linear_q", "linear_v")``).
        strict_load: Argument for ``load_unimol2_weights(strict=...)``.

    Note:
        forward receives the dict created by ``collate_unimolv2`` as-is.
        Keys with a leading double-underscore, such as ``__y__`` / ``__smi__``, are excluded.
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
