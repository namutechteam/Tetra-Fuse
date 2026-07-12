"""evaluation: 신규 artifact 의 metric 계산·비교 도구.

세 모듈로 구성:

- ``tdc_benchmark``: ``predictions.parquet`` → 옛 phase result CSV 와 같은 7 컬럼
  metric 표 (`load_predictions`, `per_task_metric_df`, `evaluate_artifact`).
  TDC 인터넷 의존 없이 sklearn 기반 self-contained.
- ``compare``: 신규 artifact + 옛 phase CSV 를 wide pivot 으로 합치고 W/L/T
  요약을 계산. ``runners.evaluate`` 가 사용.
- ``leaderboard``: TDC 웹 리더보드 스크레이프 + 캐시 (선택). 옛
  ``compare_leaderboard.py`` 흡수. requests/bs4 lazy import.

Why:
    parity 디렉토리는 옛 phase 결과 ↔ 신규 artifact 의 1회성 회귀 방지 도구.
    evaluation 패키지는 사용자 향(向) — recipe 에서 호출되어 artifact 를 만들고
    lineage 그래프에 비교 결과를 남긴다. ``parity._metrics`` 는 본 모듈의
    ``tdc_benchmark`` 함수를 그대로 재사용하는 호환 wrapper 가 된다.

Note:
    본 패키지는 registry 를 사용하지 않으므로 import-time 부수효과가 없다.
    사용자는 ``from qadmet.evaluation.tdc_benchmark import evaluate_artifact``
    형태로 직접 호출.
"""

from qadmet.evaluation.tdc_benchmark import (
    evaluate_artifact,
    load_predictions,
    metric_direction,
    per_task_metric_df,
)

__all__ = [
    "evaluate_artifact",
    "load_predictions",
    "metric_direction",
    "per_task_metric_df",
]
