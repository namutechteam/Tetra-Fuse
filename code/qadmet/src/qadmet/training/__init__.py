"""학습 루프 빌더 — Lightning Trainer / Callback / seed loop.

본 패키지는 ``TrainerSpec`` 를 받아 lightning ``Trainer`` 인스턴스와 표준
콜백을 만든다. 학습 데이터·모델 빌드는 ``qadmet.runners.train`` 의 책임.

Lazy import:
    ``lightning`` 은 sub-module 안에서만 import. 본 ``__init__`` 자체는
    ``qadmet`` 다른 곳에서 import 해도 lightning 을 끌어들이지 않는다.
"""

from qadmet.training.callbacks import build_callbacks, extract_best_val_loss
from qadmet.training.seed_loop import SeedRun, run_seed_loop
from qadmet.training.trainer import build_trainer

__all__ = [
    "SeedRun",
    "build_callbacks",
    "build_trainer",
    "extract_best_val_loss",
    "run_seed_loop",
]
