"""Artifact 저장소.

레이아웃::

    artifacts/{kind}/{name}/{run_id}/
        ├── manifest.yaml      # ★ recipe + 입력 의존성 + 코드/환경 정보
        ├── data/              # 실제 산출물 (model.ckpt, predictions.parquet, ...)
        └── log.jsonl          # 실행 로그

run_id::
    sha1(recipe_normalized + git_sha)[:12]
    같은 recipe + 같은 코드 → 동일 run_id (재현성). seed 등 다중 실행 정보는
    manifest 안에 outputs 로 누적된다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from qadmet.core.paths import artifacts_dir
from qadmet.core.specs import ExperimentSpec, spec_to_dict


# ---------------------------------------------------------------------------- 해시·환경

def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def normalize_for_hash(obj: Any) -> str:
    """spec dict 를 결정적 JSON 문자열로 변환 (해시 입력)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compute_run_id(spec: ExperimentSpec, git_sha: str) -> str:
    payload = normalize_for_hash({"spec": spec_to_dict(spec), "git": git_sha})
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def git_info(repo_root: Path) -> dict[str, Any]:
    """현재 git SHA + dirty 여부. git 없으면 unknown."""
    info: dict[str, Any] = {"sha": "unknown", "dirty": False}
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        info["sha"] = sha
        info["dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pass
    return info


# ---------------------------------------------------------------------------- 데이터모델

@dataclass
class InputRef:
    """다른 artifact 참조 (lineage 의존성)."""

    kind: str
    name: str
    run_id: str
    sha256: str | None = None  # 입력 데이터 파일 해시 (선택)


@dataclass
class OutputFile:
    """산출 파일 한 건."""

    relpath: str
    sha256: str
    bytes: int


@dataclass
class Artifact:
    """저장된 artifact 핸들."""

    kind: str
    name: str
    run_id: str
    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.yaml"


# ---------------------------------------------------------------------------- 저장소

class ArtifactStore:
    """Artifact CRUD.

    Note:
        같은 (kind, name, run_id) 가 이미 존재하면 기본적으로 기존 artifact 를
        반환한다 (멱등). ``force=True`` 로 덮어쓰기.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else artifacts_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- locate
    def path_for(self, kind: str, name: str, run_id: str) -> Path:
        return self.root / kind / name / run_id

    def exists(self, kind: str, name: str, run_id: str) -> bool:
        return (self.path_for(kind, name, run_id) / "manifest.yaml").exists()

    def load(self, kind: str, name: str, run_id: str) -> Artifact:
        root = self.path_for(kind, name, run_id)
        mp = root / "manifest.yaml"
        if not mp.exists():
            raise FileNotFoundError(f"artifact 없음: {root}")
        with mp.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        return Artifact(kind=kind, name=name, run_id=run_id, root=root, manifest=manifest)

    def find_by_name(self, kind: str, name: str) -> list[Artifact]:
        """이름으로 모든 run 조회 (최신순)."""
        base = self.root / kind / name
        if not base.exists():
            return []
        runs = sorted(
            [p for p in base.iterdir() if (p / "manifest.yaml").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [self.load(kind, name, p.name) for p in runs]

    def latest(self, kind: str, name: str) -> Artifact | None:
        runs = self.find_by_name(kind, name)
        return runs[0] if runs else None

    # ----------------------------------------------------------------- create
    def create(
        self,
        spec: ExperimentSpec,
        *,
        repo_root: Path,
        inputs: list[InputRef] | None = None,
        force: bool = False,
        runtime: dict[str, Any] | None = None,
    ) -> Artifact:
        """run 디렉토리 + 초기 manifest 작성. 산출 파일은 이후 finalize 에서 등록.

        ``runtime`` 은 해시에 들어가지 않는 실행 환경 메타 (DDP devices/strategy 등).
        ``compute_run_id`` 는 spec + git_sha 만 보므로 runtime 값이 바뀌어도 run_id 는
        동일하다 — 같은 recipe 를 다른 GPU 수로 돌려도 같은 artifact 에 누적된다.
        """
        gi = git_info(repo_root)
        run_id = compute_run_id(spec, gi["sha"])
        root = self.path_for(spec.kind, spec.name, run_id)
        if root.exists() and not force:
            return self.load(spec.kind, spec.name, run_id)
        if root.exists() and force:
            shutil.rmtree(root)
        (root / "data").mkdir(parents=True, exist_ok=True)

        manifest = {
            "artifact": {
                "kind": spec.kind,
                "name": spec.name,
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "recipe": spec_to_dict(spec),
            "inputs": [
                {"kind": r.kind, "name": r.name, "run_id": r.run_id, "sha256": r.sha256}
                for r in (inputs or [])
            ],
            "code": {
                "git_sha": gi["sha"],
                "git_dirty": gi["dirty"],
                "python": _python_version(),
                "packages": _key_package_versions(),
                "cwd": str(repo_root),
                "host": os.uname().nodename,
            },
            "runtime": runtime or {},
            "outputs": [],
            "metrics": {},
        }
        artifact = Artifact(
            kind=spec.kind, name=spec.name, run_id=run_id, root=root, manifest=manifest,
        )
        self._write_manifest(artifact)
        return artifact

    # ----------------------------------------------------------------- finalize
    def register_output(self, artifact: Artifact, relpath: str) -> OutputFile:
        """data/ 안에 새로 만든 파일을 manifest 에 기록한다."""
        full = artifact.data_dir / relpath
        if not full.exists():
            raise FileNotFoundError(f"등록할 출력 파일 없음: {full}")
        out = OutputFile(relpath=relpath, sha256=sha256_file(full), bytes=full.stat().st_size)
        outputs = artifact.manifest.setdefault("outputs", [])
        outputs.append({"path": f"data/{relpath}", "sha256": out.sha256, "bytes": out.bytes})
        self._write_manifest(artifact)
        return out

    def set_metrics(self, artifact: Artifact, metrics: dict[str, Any]) -> None:
        artifact.manifest["metrics"] = metrics
        self._write_manifest(artifact)

    def _write_manifest(self, artifact: Artifact) -> None:
        with artifact.manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                _yaml_safe(artifact.manifest),
                f,
                sort_keys=False,
                allow_unicode=True,
            )


def _yaml_safe(obj):
    """tuple/MappingProxyType/Path/torch.TorchVersion 등 yaml.safe_dump 비호환 타입을 정규화.

    ``torch.__version__`` 같은 str 서브클래스는 SafeDumper 가 거부하므로
    원시 타입으로 캐스팅한다.
    """
    from types import MappingProxyType
    if obj is None:
        return None
    if isinstance(obj, bool):  # bool 은 int 하위 → 먼저 분기.
        return bool(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (tuple, list)):
        return [_yaml_safe(x) for x in obj]
    if isinstance(obj, (dict, MappingProxyType)):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------- 환경 메타

def _python_version() -> str:
    import sys
    return sys.version.split()[0]


def _key_package_versions() -> dict[str, str]:
    """학습 재현성에 영향이 큰 패키지 버전만 기록."""
    versions: dict[str, str] = {}
    for mod in ("torch", "lightning", "chemprop", "transformers", "peft", "rdkit",
                "numpy", "pandas", "sklearn", "xgboost"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return versions
