"""multitask Lightning module.

Spec(``BackboneSpec + FusionSpec + HeadSpec + LossSpec``) → ``LightningModule`` assembly.
``lightning`` is relatively heavy on import, so this sub-package's ``qadmet.modules``
``__init__`` does not import it — since there's no registry to register, import-time
side effect is also absent. Explicitly on the usage side::

    from qadmet.modules.multitask import MultitaskLitModule

import it as.
"""

from __future__ import annotations

from .lit_module import (
    BackboneAdapter,
    MultitaskLitModule,
    register_input_adapter,
)

__all__ = ["MultitaskLitModule", "BackboneAdapter", "register_input_adapter"]
