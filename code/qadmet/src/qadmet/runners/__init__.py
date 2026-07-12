"""runner 모듈들. import 시 ``RUNNER_REGISTRY`` 에 자동 등록된다."""

from qadmet.runners import evaluate as _evaluate  # noqa: F401
from qadmet.runners import precompute as _precompute  # noqa: F401
from qadmet.runners import stack as _stack  # noqa: F401
from qadmet.runners import train as _train  # noqa: F401
# train_stl 은 train.py 가 dispatch 시점에 import 한다 (등록은 train.py 단일 키 "train").

__all__ = ["_evaluate", "_precompute", "_stack", "_train"]
