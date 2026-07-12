"""``TrainerSpec`` → lightning ``Trainer`` 빌더.

옛 ``train_mtl_mpnn`` (scripts-old/pipeline_mtl.py:1409-1413) 의 Trainer 인자
조합을 그대로 가져온다::

    pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1,
        callbacks=callbacks, enable_progress_bar=True,
        logger=False, enable_model_summary=False,
    )

Why dedicated builder:
    Trainer 인자 (precision, deterministic, gradient_clip 등) 를 한 곳에 모아
    spec 변경에 일관되게 대응한다. 옛 코드는 함수 내부에 인자가 흩어져 있어
    실험 비교 시 hard-coded 값이 새는 문제가 있었다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qadmet.core.specs import TrainerSpec
from qadmet.training.callbacks import build_callbacks


def build_trainer(
    spec: TrainerSpec,
    *,
    ckpt_dir: Path | str,
    log_dir: Path | str | None = None,
    monitor: str = "val_loss",
    mode: str = "min",
    accelerator: str | None = None,
    devices: int = 1,
    strategy: str | None = None,
    num_nodes: int = 1,
    find_unused_parameters: bool = False,
    ddp_static_graph: bool = True,
    deterministic: bool = False,
    save_last: bool = False,
) -> Any:
    """``TrainerSpec`` → lightning ``Trainer`` 인스턴스.

    Args:
        spec: TrainerSpec (max_epochs / patience / precision / early_stopping 등).
        ckpt_dir: ModelCheckpoint 저장 디렉토리 (seed 별로 다르게 줄 것).
        log_dir: CSVLogger 출력 디렉토리. None 이면 logger 비활성.
        monitor / mode: 콜백·체크포인트 공통 모니터.
        accelerator: ``"gpu"`` / ``"cpu"`` / None (None 이면 CUDA 자동 감지).
        devices: 사용 device 수.
        strategy: ``"ddp"`` / None. None 이면 Lightning auto (single → SingleDevice).
            ``"ddp"`` 면 DDPStrategy 인스턴스를 빌드 (find_unused_parameters 반영).
        num_nodes: 다중 노드 수 (보통 1).
        find_unused_parameters: DDPStrategy 옵션. False 가 기본 (성능 우선).
            forward path 가 batch 마다 다른 파라미터를 건드릴 때만 True.
        ddp_static_graph: DDPStrategy 의 ``static_graph`` 옵션. True 가 기본
            (find_unused_parameters=True 와 함께 쓸 때 NCCL hang 회피용 — 첫
            iteration 의 used-param set 을 캐싱해서 rank 간 일관성 보장).
            PyTorch 공식 권장: ``find_unused_parameters=True`` 만 단독 사용 시
            grad 준비 순서가 rank 마다 어긋나 NCCL allreduce 가 spin-wait deadlock
            가능. 본 모델 (GroupedHead) 은 forward 에서 모든 trunk + head 호출이
            보장돼 static_graph 가정 만족.
        deterministic: True 면 ``Trainer(deterministic=True)`` 로 재현성 강화.

    Returns:
        lightning.pytorch.Trainer 인스턴스.

    Note:
        Trainer 인자 우선순위: defaults < ``spec.kwargs`` < 본 함수의 명시 인자.
        ``devices/strategy/num_nodes`` 등 CLI 에서 흘러온 값이 recipe 의
        ``trainer.kwargs`` 값을 이긴다.
    """
    import lightning as L

    if accelerator is None:
        try:
            import torch

            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            accelerator = "cpu"

    callbacks = build_callbacks(
        ckpt_dir=ckpt_dir,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        early_stopping=spec.early_stopping,
        patience=spec.patience,
        extra=spec.extra_callbacks,
        save_last=save_last,
    )

    pl_logger: Any = False
    if log_dir is not None:
        from lightning.pytorch.loggers import CSVLogger

        log_path = Path(log_dir)
        pl_logger = CSVLogger(save_dir=str(log_path.parent), name=log_path.name)

    # strategy 값에 따라 DDPStrategy 인스턴스 또는 문자열 처리.
    resolved_strategy: Any = "auto"
    if strategy is not None:
        if strategy == "ddp":
            from lightning.pytorch.strategies import DDPStrategy

            resolved_strategy = DDPStrategy(
                find_unused_parameters=find_unused_parameters,
                static_graph=ddp_static_graph,
            )
        else:
            # ddp_spawn / fsdp / deepspeed 등 — 문자열 그대로 전달, Lightning 이 해석.
            resolved_strategy = strategy

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": spec.max_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "callbacks": callbacks,
        "logger": pl_logger,
        "precision": spec.precision,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "deterministic": deterministic,
    }
    # 우선순위: defaults < spec.kwargs < 명시 인자 (CLI).
    trainer_kwargs.update(dict(spec.kwargs))
    trainer_kwargs["devices"] = devices
    trainer_kwargs["num_nodes"] = num_nodes
    trainer_kwargs["strategy"] = resolved_strategy
    return L.Trainer(**trainer_kwargs)
