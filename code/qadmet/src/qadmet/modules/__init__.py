"""nn 모듈 패키지: backbones / fusion / heads / losses / multitask.

Step 3 진행과 함께 import-time registry 등록 모듈을 추가한다. 각 sub-package
``__init__`` 가 자체 등록 트리거를 담당하므로 본 ``__init__`` 은 Lightweight
re-export 만 수행 (heavy import 회피).
"""

from . import losses  # noqa: F401  — LOSSES / PER_TASK_LOSSES 등록
from . import heads  # noqa: F401  — HEADS 등록
from . import fusion  # noqa: F401  — FUSIONS 등록
from . import backbones  # noqa: F401  — BACKBONES 등록 (heavy deps lazy)

