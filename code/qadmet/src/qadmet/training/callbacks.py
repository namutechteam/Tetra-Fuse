"""표준 Lightning 콜백 빌더.

옛 ``train_mtl_mpnn`` (scripts-old/pipeline_mtl.py:1386-1388) 의 콜백 조합
``ModelCheckpoint(monitor=val_loss, mode=min, save_top_k=1)`` +
``EarlyStopping(monitor=val_loss, mode=min, patience=...)`` 을 그대로 따른다.

Lazy import:
    lightning 콜백은 호출 시점에만 import 한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_callbacks(
    *,
    ckpt_dir: Path | str,
    monitor: str = "val_loss",
    mode: str = "min",
    save_top_k: int = 1,
    early_stopping: bool = True,
    patience: int = 20,
    extra: tuple[str, ...] = (),
    save_last: bool = False,
) -> list[Any]:
    """표준 콜백 조합을 만든다.

    Args:
        ckpt_dir: ModelCheckpoint 저장 디렉토리. 부모 포함 자동 생성.
        monitor: 모니터링 metric 이름 (LitModule 이 ``self.log`` 한 키).
        mode: ``"min"`` 또는 ``"max"``.
        save_top_k: ModelCheckpoint top-k 저장 수.
        early_stopping: True 면 EarlyStopping 추가.
        patience: EarlyStopping patience.
        extra: 등록된 추가 콜백 이름들 (현재는 미지원, 추후 registry 추가 여지).

    Returns:
        Trainer ``callbacks`` 인자에 그대로 전달 가능한 리스트.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename="best",
            save_last=save_last,
            # validation epoch end 직후에만 save 판단 (manual_optimization + DDP
            # 조합에서 train_epoch_end 직후 save 가 stale metric 을 capture 해
            # best.ckpt 가 진짜 best 보다 1 epoch 늦은 weight 를 저장하던 이슈 차단).
            save_on_train_epoch_end=False,
        )
    ]
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, mode=mode, patience=patience, check_finite=True,
            )
        )

    if extra:
        # TODO(연구방향): 추가 콜백 registry — alpha 진단, gradient norm 로깅 등.
        raise NotImplementedError(
            f"extra_callbacks={extra} 은 아직 미지원 (Step 3.6 단계). "
            "callback registry 도입 시 활성화."
        )

    return callbacks


def extract_best_val_loss(trainer: Any, monitor: str = "val_loss") -> float:
    """Trainer 의 EarlyStopping / ModelCheckpoint 에서 진짜 best metric 을 꺼낸다.

    ``trainer.callback_metrics[monitor]`` 는 학습 종료 시점의 *마지막* validation
    값이라 best 가 아니다. EarlyStopping.best_score 가 monitor 의 단조 최저값을
    그대로 들고 있으므로 1순위. 그게 없으면 ModelCheckpoint.best_model_score,
    마지막 fallback 으로 callback_metrics 의 last 값.
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    for cb in getattr(trainer, "callbacks", []) or []:
        if (
            isinstance(cb, EarlyStopping)
            and cb.monitor == monitor
            and cb.best_score is not None
        ):
            return float(cb.best_score)
    for cb in getattr(trainer, "callbacks", []) or []:
        if (
            isinstance(cb, ModelCheckpoint)
            and cb.monitor == monitor
            and cb.best_model_score is not None
        ):
            return float(cb.best_model_score)
    return float(trainer.callback_metrics.get(monitor, float("nan")))
