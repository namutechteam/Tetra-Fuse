"""ExperimentRunner: spec → 실행 → artifact 생성.

본 모듈은 dispatch 만 한다. kind 별 실제 로직은 ``qadmet.runners.*`` 에서
제공하며 ``RUNNER_REGISTRY`` 에 등록된다.

순환 import 방지를 위해 runner 등록은 import-time 사이드이펙트로 처리한다
(``qadmet.runners`` 패키지의 ``__init__.py`` 가 모든 서브 runner 를 import).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from qadmet.core.artifacts import Artifact, ArtifactStore
from qadmet.core.paths import root
from qadmet.core.registry import Registry
from qadmet.core.specs import ExperimentSpec

# kind 문자열 → callable(spec, store, repo_root, [runtime]) -> Artifact
RUNNER_REGISTRY: Registry[Callable] = Registry("runners")


class ExperimentRunner:
    """spec 한 건을 실행하고 artifact 를 반환한다.

    Args:
        store / repo_root: 일반 의존성.
        runtime: 실행 환경 메타 (DDP devices/strategy/world_size 등). manifest 의
            ``runtime`` 섹션에 기록되며 run_id 해시에는 영향이 없다. runner 시그니처에
            ``runtime`` 키워드를 받을 수 있을 때만 전달된다 (opt-in) — 그 외 runner 는
            기존 시그니처 유지.
        force: CLI ``--force`` 의 값. ``runtime`` 과 동일한 opt-in 방식 — runner
            시그니처에 ``force`` 키워드가 있는 경우만 전달.
    """

    def __init__(
        self,
        store: ArtifactStore | None = None,
        repo_root: Path | None = None,
        runtime: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        self.store = store or ArtifactStore()
        self.repo_root = Path(repo_root) if repo_root else root()
        self.runtime = runtime
        self.force = force

    def run(self, spec: ExperimentSpec) -> Artifact:
        spec.validate()
        try:
            runner = RUNNER_REGISTRY.get(spec.kind)
        except KeyError as e:
            raise NotImplementedError(
                f"ExperimentRunner: kind={spec.kind!r} 에 대한 runner 가 등록되어 있지 않습니다. "
                f"qadmet.runners 패키지를 import 했는지 확인하세요."
            ) from e
        params = inspect.signature(runner).parameters
        extra: dict[str, Any] = {}
        if self.runtime is not None and "runtime" in params:
            extra["runtime"] = self.runtime
        if "force" in params:
            extra["force"] = self.force
        return runner(spec=spec, store=self.store, repo_root=self.repo_root, **extra)
