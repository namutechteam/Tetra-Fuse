"""ExperimentRunner: spec → execution → artifact generation.

This module only does dispatch. The actual logic per kind lives in ``qadmet.runners.*``
and is registered in ``RUNNER_REGISTRY``.

To avoid circular imports, runner registration is handled as an import-time side effect
(the ``__init__.py`` of the ``qadmet.runners`` package imports all sub runners).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from qadmet.core.artifacts import Artifact, ArtifactStore
from qadmet.core.paths import root
from qadmet.core.registry import Registry
from qadmet.core.specs import ExperimentSpec

# kind string → callable(spec, store, repo_root, [runtime]) -> Artifact
RUNNER_REGISTRY: Registry[Callable] = Registry("runners")


class ExperimentRunner:
    """Executes a single spec and returns an artifact.

    Args:
        store / repo_root: common dependencies.
        runtime: execution environment metadata (DDP devices/strategy/world_size, etc.). Recorded in the manifest's
            ``runtime`` section; does not affect the run_id hash. Passed to the runner only when its signature
            can accept the ``runtime`` keyword (opt-in) — other runners
            keep their existing signature.
        force: value of CLI ``--force``. Same opt-in approach as ``runtime`` — passed only when the runner
            signature has a ``force`` keyword.
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
                f"ExperimentRunner: kind={spec.kind!r} has no registered runner. "
                f"Check whether the qadmet.runners package has been imported."
            ) from e
        params = inspect.signature(runner).parameters
        extra: dict[str, Any] = {}
        if self.runtime is not None and "runtime" in params:
            extra["runtime"] = self.runtime
        if "force" in params:
            extra["force"] = self.force
        return runner(spec=spec, store=self.store, repo_root=self.repo_root, **extra)
