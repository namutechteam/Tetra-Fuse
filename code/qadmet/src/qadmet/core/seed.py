"""seed 표준화. lightning seed_everything 이 있으면 우선 위임.

``seed_everything(seed)`` 한 번 호출로 random / numpy / torch (cpu+cuda) 모두 설정.
deterministic 옵션은 학습 시 큰 비용이 들 수 있어 호출자가 명시적으로 선택하게 한다.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """주어진 seed 로 모든 RNG 를 초기화한다."""
    if not isinstance(seed, int):
        raise TypeError(f"seed 는 int 여야 합니다: {type(seed)}")

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
