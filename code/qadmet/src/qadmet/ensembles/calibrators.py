"""``Calibrator`` ABC + Platt / Isotonic 구현.

본 모듈은 분류 task 의 prob 예측을 보정하는 도구다. 옛 ``run_stacking_v6_hybrid`` 의
v6 hybrid blend 단계가 base 출력 분포 정렬을 필요로 했고, 신규 ``HybridBlendStrategy``
및 future stacking 파이프라인 (per-base prob 보정) 에서 사용된다.

상태 (Step 4.6):
    - ``Calibrator`` ABC + ``CALIBRATORS`` registry.
    - ``PlattCalibrator`` (sklearn ``LogisticRegression`` 1D) — sigmoid 보정.
    - ``IsotonicCalibrator`` (sklearn ``IsotonicRegression``) — 단조 piecewise constant.
    - sklearn 은 lazy import (``import qadmet.ensembles`` 시 sys.modules 미등록).

사용 패턴:
    >>> cal = CALIBRATORS.build_with_kwargs("platt")
    >>> cal.fit(probs_val, y_val)
    >>> probs_test_calibrated = cal.transform(probs_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qadmet.ensembles.registry import CALIBRATORS


class Calibrator(ABC):
    """분류 prob 보정기의 공통 인터페이스.

    Why:
        v6 hybrid blend 와 stacking 파이프라인은 base 별로 prob 분포가 다른
        (sigmoid 출력 / soft-label / 학습된 prob) 점을 정렬해야 한다. 회귀 task
        는 보정이 필요 없으므로 본 ABC 는 분류 전용으로 가정한다.

    구현 책임:
        - ``fit(p, y)``: 1D prob 배열 ``p`` (값 ∈ [0, 1]) + 0/1 라벨 ``y`` 로 학습.
        - ``transform(p)``: 입력 prob 를 보정된 prob 로 변환 (출력 ∈ [0, 1]).

    공통 검증:
        - ``p``, ``y`` 1D + 길이 일치.
        - ``y`` 는 0/1 정수 (또는 정수 캐스트 가능).
        - ``p`` 는 [0, 1] 범위 (NaN 허용 — fit 시 finite mask, transform 시 NaN 통과).
    """

    @abstractmethod
    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        ...

    @abstractmethod
    def transform(self, p: np.ndarray) -> np.ndarray:
        ...


def _validate_fit_inputs(
    p: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(p, y)`` 1D ndarray 변환 + 형태/값 검증.

    Returns:
        finite mask 적용 전의 ``(p_arr, y_arr)``. NaN 처리는 호출부 책임.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y).ravel()
    if p_arr.shape != y_arr.shape:
        raise ValueError(
            f"p, y 길이 불일치: p={p_arr.shape} y={y_arr.shape}"
        )
    if p_arr.size == 0:
        raise ValueError("빈 입력 (p/y 길이 0).")
    # y 0/1 검증.
    y_int = y_arr.astype(np.int64)
    if not np.array_equal(y_int, y_arr):
        raise ValueError(
            "Calibrator: y 가 0/1 정수가 아님 (분류 라벨 필요)."
        )
    uniq = set(np.unique(y_int).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"Calibrator: y 는 {{0, 1}} 만 허용. 실제={sorted(uniq)}"
        )
    # p 범위 — NaN 은 통과시키고 fit 시 finite mask 로 처리.
    finite = np.isfinite(p_arr)
    if finite.any():
        if (p_arr[finite] < 0.0).any() or (p_arr[finite] > 1.0).any():
            raise ValueError(
                "Calibrator: p 는 [0, 1] 범위여야 합니다 (NaN 허용)."
            )
    return p_arr, y_int.astype(np.float64)


@CALIBRATORS.register("platt")
class PlattCalibrator(Calibrator):
    """Platt scaling — ``sigmoid(a * logit(p) + b)`` 학습.

    Why:
        sklearn ``CalibratedClassifierCV(method='sigmoid')`` 와 동등 — base prob 의
        logit 을 단일 feature 로 LogisticRegression 1D 학습. base 가 학습 셋에서
        과확신/저확신일 때 (`a` 가 1 이 아닐 때) 보정 효과가 크다.

    Args:
        eps: ``logit`` 변환 시 0/1 클립용 마진. 기본 ``1e-6``.
        C: sklearn LogisticRegression 의 역-정규화 강도. 기본 ``1e10`` — 거의 비-정규화
            (Platt 의 원본 의도). 작게 두면 약한 정규화로 보정 곡선이 평탄해진다.

    학습 산출물:
        ``coef_``, ``intercept_`` — sklearn 모델 파라미터 사본 (디버그용).

    Note:
        - 학습 데이터에 한 클래스만 있으면 LogisticRegression 이 실패 → fit 이
          명시 ValueError.
        - transform 시 입력 NaN 은 NaN 그대로 통과.
    """

    def __init__(self, *, eps: float = 1e-6, C: float = 1e10) -> None:
        if not (0.0 < float(eps) < 0.5):
            raise ValueError(f"eps 는 (0, 0.5) 범위여야 합니다: {eps!r}")
        if float(C) <= 0.0:
            raise ValueError(f"C 는 양수여야 합니다: {C!r}")
        self.eps = float(eps)
        self.C = float(C)
        self._model: Any | None = None
        self._coef: float | None = None
        self._intercept: float | None = None

    def _logit(self, p: np.ndarray) -> np.ndarray:
        """``logit(p)`` — ``p`` 를 ``[eps, 1-eps]`` 로 clip 후 변환."""
        clipped = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p_arr, y_arr = _validate_fit_inputs(p, y)
        finite = np.isfinite(p_arr)
        if finite.sum() < 2:
            raise ValueError(
                "PlattCalibrator.fit: finite (p, y) 행이 2 미만입니다."
            )
        p_fin = p_arr[finite]
        y_fin = y_arr[finite].astype(np.int64)
        if len(set(y_fin.tolist())) < 2:
            raise ValueError(
                "PlattCalibrator.fit: y 에 한 클래스만 존재해 학습 불가."
            )
        z = self._logit(p_fin).reshape(-1, 1)
        # solver='lbfgs' default, max_iter 보수적.
        model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000)
        model.fit(z, y_fin)
        self._model = model
        self._coef = float(model.coef_.ravel()[0])
        self._intercept = float(model.intercept_.ravel()[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "PlattCalibrator: fit 을 먼저 호출하세요."
            )
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        out = np.full_like(p_arr, np.nan)
        finite = np.isfinite(p_arr)
        if finite.any():
            z = self._logit(p_arr[finite]).reshape(-1, 1)
            # predict_proba 가 (n, 2) → P(y=1) 두번째 열.
            out[finite] = self._model.predict_proba(z)[:, 1]
        return out

    @property
    def coef_(self) -> float | None:
        return self._coef

    @property
    def intercept_(self) -> float | None:
        return self._intercept


@CALIBRATORS.register("isotonic")
class IsotonicCalibrator(Calibrator):
    """Isotonic regression 보정 — 단조증가 piecewise constant 매핑.

    Why:
        sklearn ``CalibratedClassifierCV(method='isotonic')`` 와 동등. Platt 보다
        유연하나 over-fit 위험 — 일반적으로 데이터 ≥ 1000 일 때 권장.

    Args:
        out_of_bounds: 학습 범위 밖 입력 처리.
            - ``"clip"`` (기본): 학습 prob 의 [min, max] 로 clip 후 매핑.
            - ``"nan"``: 범위 밖은 NaN 으로 출력.
        y_min: 출력 하한. 기본 ``0.0``.
        y_max: 출력 상한. 기본 ``1.0``.

    학습 산출물:
        ``X_thresholds_``, ``y_thresholds_`` — sklearn IsotonicRegression 의 학습된
        매핑점 사본 (디버그용).

    Note:
        - transform 시 입력 NaN 은 NaN 그대로 통과.
        - 학습 데이터에 한 클래스만 있으면 IsotonicRegression 자체는 학습되나
          매핑이 의미 없으므로 명시 ValueError 로 거부.
    """

    def __init__(
        self,
        *,
        out_of_bounds: str = "clip",
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> None:
        if out_of_bounds not in {"clip", "nan"}:
            raise ValueError(
                f"out_of_bounds 는 'clip' | 'nan' 중 하나: {out_of_bounds!r}"
            )
        if not (0.0 <= float(y_min) < float(y_max) <= 1.0):
            raise ValueError(
                f"y_min < y_max 이고 [0, 1] 안: y_min={y_min!r} y_max={y_max!r}"
            )
        self.out_of_bounds = out_of_bounds
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self._model: Any | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p_arr, y_arr = _validate_fit_inputs(p, y)
        finite = np.isfinite(p_arr)
        if finite.sum() < 2:
            raise ValueError(
                "IsotonicCalibrator.fit: finite (p, y) 행이 2 미만입니다."
            )
        p_fin = p_arr[finite]
        y_fin = y_arr[finite]
        if len(set(y_fin.astype(np.int64).tolist())) < 2:
            raise ValueError(
                "IsotonicCalibrator.fit: y 에 한 클래스만 존재해 학습 불가."
            )
        model = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=True,
            out_of_bounds=self.out_of_bounds,
        )
        model.fit(p_fin, y_fin)
        self._model = model
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "IsotonicCalibrator: fit 을 먼저 호출하세요."
            )
        p_arr = np.asarray(p, dtype=np.float64).ravel()
        out = np.full_like(p_arr, np.nan)
        finite = np.isfinite(p_arr)
        if finite.any():
            out[finite] = self._model.predict(p_arr[finite])
        return out

    @property
    def X_thresholds_(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return np.asarray(self._model.X_thresholds_, dtype=np.float64).copy()

    @property
    def y_thresholds_(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return np.asarray(self._model.y_thresholds_, dtype=np.float64).copy()


__all__ = ["Calibrator", "PlattCalibrator", "IsotonicCalibrator"]
