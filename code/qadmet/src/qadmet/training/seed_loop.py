"""다중 시드 학습 루프.

옛 ``run_mtl_*`` (scripts-old/pipeline_mtl.py:1502-1839) 의 시드 루프를 spec
기반으로 정리한다. 시드별로 (a) seed_everything (b) 모델 빌드 (c) Trainer.fit
(d) test 예측을 수행하고, ``SeedRun`` 결과를 누적 반환한다.

본 모듈은 학습/예측 자체 로직을 가지지 않고, 콜백 함수(build/train/predict)
형태로 외부 책임을 받는다. 실제 호출은 ``qadmet.runners.train`` 이 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from qadmet.core.logging import get_logger
from qadmet.core.seed import seed_everything

logger = get_logger("qadmet.training.seed_loop")


@dataclass
class SeedRun:
    """한 시드 학습 결과 묶음."""

    seed: int
    ckpt_path: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    val_predictions: Any = None  # 자유 형식 (DataFrame / dict 등). runner 가 정의.
    test_predictions: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def run_seed_loop(
    *,
    seeds: Sequence[int],
    fit_one: Callable[[int, Path], SeedRun],
    ckpt_root: Path,
) -> list[SeedRun]:
    """seed 별로 ``fit_one(seed, ckpt_dir)`` 를 호출하고 결과를 모은다.

    Args:
        seeds: 학습할 시드 목록.
        fit_one: 한 시드의 학습+예측을 책임지는 콜백.
            인자 (seed, seed_ckpt_dir) → ``SeedRun`` 반환.
        ckpt_root: 시드 별 체크포인트 디렉토리의 부모. 자동으로 ``seed_{N}`` 하위
            디렉토리를 만든다.

    Returns:
        ``SeedRun`` 리스트 (입력 순).

    Note:
        ``seed_everything`` 은 fit_one 내부가 부르도록 노출만 한다 — 학습 외에
        여러 단계가 시드 영향을 받기 때문에 호출 시점을 fit_one 이 결정한다.
    """
    runs: list[SeedRun] = []
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_dir = ckpt_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        logger.info("seed start", extra={"payload": {"seed": seed, "ckpt_dir": str(seed_dir)}})
        run = fit_one(seed, seed_dir)
        runs.append(run)
        logger.info(
            "seed done",
            extra={
                "payload": {
                    "seed": seed,
                    "ckpt": str(run.ckpt_path) if run.ckpt_path else None,
                    "metrics": run.metrics,
                }
            },
        )
        # seed 간 메모리 해제: 직전 seed 의 모델/optimizer 가 남긴 CUDA reserved
        # 메모리를 비워 다음 seed 빌드 시 단편화 누적 OOM 방지. 4-modality 처럼
        # GPU ceiling 근접 recipe 는 이게 없으면 seed 1+ 에서 OOM (관측됨).
        _release_between_seeds()
    return runs


def _release_between_seeds() -> None:
    """seed 경계에서 Python GC + CUDA cache 해제 (가용 시)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
