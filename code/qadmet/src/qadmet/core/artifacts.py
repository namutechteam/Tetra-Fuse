"""Artifact storage.

Layout::

    artifacts/{kind}/{name}/{run_id}/
        ├── manifest.yaml      # ★ recipe + input dependencies + code/environment info
        ├── data/              # actual outputs (model.ckpt, predictions.parquet, ...)
        └── log.jsonl          # execution log

run_id::
    sha1(recipe_normalized + git_sha)[:12]
    Same recipe + same code → same run_id (reproducibility). Multiple run info such as seed is
    accumulated as outputs inside the manifest.
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


# ---------------------------------------------------------------------------- hash/environment

def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def normalize_for_hash(obj: Any) -> str:
    """Convert spec dict into a deterministic JSON string (hash input)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compute_run_id(spec: ExperimentSpec, git_sha: str) -> str:
    payload = normalize_for_hash({"spec": spec_to_dict(spec), "git": git_sha})
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def git_info(repo_root: Path) -> dict[str, Any]:
    """Current git SHA + dirty status. unknown if git is unavailable."""
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


# ---------------------------------------------------------------------------- data model

@dataclass
class InputRef:
    """Reference to another artifact (lineage dependency)."""

    kind: str
    name: str
    run_id: str
    sha256: str | None = None  # input data file hash (optional)


@dataclass
class OutputFile:
    """A single output file."""

    relpath: str
    sha256: str
    bytes: int


@dataclass
class Artifact:
    """Handle to a saved artifact."""

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


# ---------------------------------------------------------------------------- storage

class ArtifactStore:
    """Artifact CRUD.

    Note:
        If the same (kind, name, run_id) already exists, by default the existing artifact is
        returned (idempotent). Use ``force=True`` to overwrite.
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
            raise FileNotFoundError(f"artifact not found: {root}")
        with mp.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        return Artifact(kind=kind, name=name, run_id=run_id, root=root, manifest=manifest)

    def find_by_name(self, kind: str, name: str) -> list[Artifact]:
        """Look up all runs by name (most recent first)."""
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
        """Create run directory + write initial manifest. Output files are registered later via finalize.

        ``runtime`` is execution environment metadata (DDP devices/strategy, etc.) not included in the hash.
        ``compute_run_id`` only looks at spec + git_sha, so even if runtime values change, run_id stays
        the same — running the same recipe with a different number of GPUs accumulates into the same artifact.
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
        """Record newly created files in data/ into the manifest."""
        full = artifact.data_dir / relpath
        if not full.exists():
            raise FileNotFoundError(f"no output file to register: {full}")
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
    """Normalize types incompatible with yaml.safe_dump such as tuple/MappingProxyType/Path/torch.TorchVersion.

    str subclasses like ``torch.__version__`` are rejected by SafeDumper, so
    cast them to primitive types.
    """
    from types import MappingProxyType
    if obj is None:
        return None
    if isinstance(obj, bool):  # bool is a subtype of int → branch first.
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


# ---------------------------------------------------------------------------- environment metadata

def _python_version() -> str:
    import sys
    return sys.version.split()[0]


def _key_package_versions() -> dict[str, str]:
    """Record only package versions that significantly affect training reproducibility."""
    versions: dict[str, str] = {}
    for mod in ("torch", "lightning", "chemprop", "transformers", "peft", "rdkit",
                "numpy", "pandas", "sklearn", "xgboost"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return versions
