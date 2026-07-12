"""multitask Lightning 모듈.

Spec(``BackboneSpec + FusionSpec + HeadSpec + LossSpec``) → ``LightningModule`` 조립.
``lightning`` 은 import 시 비교적 무겁기 때문에 본 sub-package 는 ``qadmet.modules``
의 ``__init__`` 에서는 import 하지 않는다 — 등록할 registry 가 없어 import-time
부작용도 없다. 사용 측에서 명시적으로::

    from qadmet.modules.multitask import MultitaskLitModule

으로 가져온다.
"""

from __future__ import annotations

from .lit_module import (
    BackboneAdapter,
    MultitaskLitModule,
    register_input_adapter,
)

__all__ = ["MultitaskLitModule", "BackboneAdapter", "register_input_adapter"]
