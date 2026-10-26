"""Training loop builder — Lightning Trainer / Callback / seed loop.

This package takes ``TrainerSpec`` and builds a lightning ``Trainer`` instance and standard
callbacks. Building training data/model is the responsibility of ``qadmet.runners.train``.

Lazy import:
    ``lightning`` is imported only within sub-modules. This ``__init__`` itself
    does not pull in lightning even when imported from elsewhere in ``qadmet``.
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
