"""Runner modules. Automatically registered to ``RUNNER_REGISTRY`` upon import."""

from qadmet.runners import evaluate as _evaluate  # noqa: F401
from qadmet.runners import precompute as _precompute  # noqa: F401
from qadmet.runners import stack as _stack  # noqa: F401
from qadmet.runners import train as _train  # noqa: F401
# train_stl is imported by train.py at dispatch time (registration uses the single key "train" in train.py).

__all__ = ["_evaluate", "_precompute", "_stack", "_train"]
