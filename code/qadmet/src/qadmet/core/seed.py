"""Seed standardization. Delegate to lightning seed_everything first if available.

``seed_everything(seed)`` sets up random / numpy / torch (cpu+cuda) all at once with a single call.
The deterministic option can incur significant cost during training, so let the caller choose it explicitly.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Initializes all RNGs with the given seed."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int: {type(seed)}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    return seed
