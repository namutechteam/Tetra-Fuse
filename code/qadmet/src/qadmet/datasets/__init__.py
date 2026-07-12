"""원본 데이터셋 로더 + splitter + transform.

본 패키지는 chemprop 객체에 직접 의존하지 않는다 (학습 코드와 분리). 학습
파이프라인은 ``qadmet.training`` 에서 본 모듈의 출력 (pd.DataFrame) 을 받아
chemprop datapoint 로 변환한다.
"""

from qadmet.datasets.registry import DATASETS
from qadmet.datasets import tdc_admet  # noqa: F401  — import-time 등록
from qadmet.datasets import pxr_activity  # noqa: F401  — PXR challenge

__all__ = ["DATASETS"]
